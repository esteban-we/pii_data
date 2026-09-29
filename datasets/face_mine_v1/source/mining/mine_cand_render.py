"""Run the two-model triage on real never-detected data and render everything it flags.

Everything measured so far about the two-model criterion came off the GT bench, and worse, off the
subset of GT frames already known to contain a missed face. Neither says what the queue looks like
when it runs over the corpus: what fraction is junk, what a candidate actually looks like, whether a
human reviewing it would learn anything. That is what this produces.

No ground truth exists here, deliberately -- these are chunks production has never detected, drawn
from pilot_pool.json, which already excludes every GT, complaint and face10k session at SESSION
level. Sessions touched by the earlier pilot are excluded too so the frames are fresh.

Two things make a candidate meaningful rather than merely present:

  the size gate -- a head box more than 3x the armW box's long side is a person or a torso, not a
  face, and CrowdHuman's class 0 clears that bar for free (median 7.0x). Only class 1 is used.

  the continuation check -- a face armW scores at 0.6 in one frame is NOT missed if the same face
  scores >=0.75 anywhere in its track, because continuation() would blur it. Sparse 1fps sampling
  cannot answer that (consecutive samples are 30 frames apart, past max_gap=15, so no track can
  form). So every flagged frame gets a +/-150 frame window decoded at FULL rate, armW run over it,
  and production's continuation applied. A candidate counts as 遗漏 only if no track covers it.
  The window is finite: a track seeded further than 150 frames away and chained continuously would
  be missed by this check, which biases 遗漏 upward. Stated rather than hidden.

Scene mix follows the floor-adjusted proportional allocation, not pure expected yield -- the density
estimates behind it come from very unequal pilot samples.
"""
from __future__ import annotations

import json
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

T = Path("/data/liangzhenghao/train")
ATLAS = Path("/data/liangzhenghao/code/atlas")
ENGINE = T / "trt_out/scrfd_1024_fp16_dynb16_5090_armW.engine"
HEAD_MODEL = T / "mine_models/crowdhuman.onnx"
SCRATCH = T / "mine_cache"
OUT = T / "cand_mine"

STRIDE = 30                      # 1 fps at 30 fps capture
KEEP, DET_SIZE = 0.25, 1024
T_HIGH, T_LOW, MAX_GAP, MIN_LONG = 0.75, 0.50, 15, 40.0
BATCH = 8
HEAD_CLS, HEAD_SIZE, HEAD_CONF, HEAD_NMS = 1, 640, 0.15, 0.45
MAX_HEAD_RATIO = 3.0
# T0 is not a mining tier -- armW already scores it above the blur threshold, so it is counted
# only to keep the tier assignment total and to show how much of the head model's output is
# redundant with what already ships.
TIERS = [("T0", 0.75, 1.01), ("T1", 0.45, 0.75), ("T2", 0.25, 0.45), ("T3", -1.0, 0.25)]
WINDOW = 150                     # frames each side for the continuation check
CELL, PAD = 260, 2.2
N_CHUNKS = 24
ALLOC = {"工厂": 0.792, "医疗/健康": 0.057, "零售/超市": 0.044,
         "家庭": 0.042, "餐饮/酒店": 0.034, "手作/工位": 0.032}

C_HEAD = (255, 160, 0)
C_ARMW = (0, 220, 255)


def norm_scene(s):
    p = (s or "").split("_")
    raw = re.sub(r"[^a-z]", "", (p[3] if len(p) > 3 else "").lower())
    for pat, name in [(r"factor|facatory|manufactur|electronicsfact|toyfact", "工厂"),
                      (r"home|livingroom", "家庭"),
                      (r"supermarket|fruitshop|petstore|retail|store", "零售/超市"),
                      (r"workbench|craft|draft|marquetry|mosaic|handcraft|studio", "手作/工位"),
                      (r"logistic|storage", "仓储物流"),
                      (r"health|hospital|beauty", "医疗/健康"),
                      (r"office|offic|school", "办公/学校"),
                      (r"restaurant|coffee|hotel|hotal|resort", "餐饮/酒店"),
                      (r"repair|mobilerepair|applianc", "维修"),
                      (r"outdoor", "户外")]:
        if re.search(pat, raw):
            return name
    return "其他/未标注" if raw else "(无 scene_id)"


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _iof(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return 0.0 if a <= 0 else (iw * ih) / a


def _cdist(a, b):
    return (((a[0] + a[2]) - (b[0] + b[2])) ** 2
            + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2


def _gate(a, b):
    return _iou(a, b) >= 0.2 or _cdist(a, b) < max(long_side(a), long_side(b))


def _greedy(dets, tracks):
    pairs = []
    for di, d in enumerate(dets):
        for t in tracks:
            if _gate(d, t["last"]):
                pairs.append((_iou(d, t["last"]), _cdist(d, t["last"]), di, t))
    pairs.sort(key=lambda p: (-p[0], p[1]))
    out, ud, ut = {}, set(), set()
    for _s, _c, di, t in pairs:
        if di in ud or id(t) in ut:
            continue
        ud.add(di)
        ut.add(id(t))
        out[di] = t
    return out


def tracked(dets, frames_sorted):
    """Production continuation() over a contiguous run of frames."""
    tracks = []
    for f in frames_sorted:
        d = dets.get(f)
        if not d:
            continue
        hi = [b for b, s in d if s >= T_HIGH and long_side(b) >= MIN_LONG]
        lo = [b for b, s in d if T_LOW <= s < T_HIGH and long_side(b) >= MIN_LONG]
        if not hi and not lo:
            continue
        op = [t for t in tracks if 0 < f - t["last_f"] <= MAX_GAP]
        mh = _greedy(hi, op)
        for di, t in mh.items():
            t["frames"][f] = hi[di]
            t["last"] = hi[di]
            t["last_f"] = f
        used = {id(t) for t in mh.values()}
        ml = _greedy(lo, [t for t in op if id(t) not in used])
        for di, t in ml.items():
            t["frames"][f] = lo[di]
            t["last"] = lo[di]
            t["last_f"] = f
        for di, b in enumerate(hi):
            if di not in mh:
                tracks.append({"frames": {f: b}, "last": b, "last_f": f})
    out = {}
    for t in tracks:
        for f, b in t["frames"].items():
            out.setdefault(f, []).append(b)
    return out


def nms(dets, thr):
    dets = sorted(dets, key=lambda x: -x[1])
    keep = []
    for b, s, c in dets:
        if all(_iou(b, kb) < thr for kb, _ks, kc in keep if kc == c):
            keep.append((b, s, c))
    return keep


def run_yolo(sess, iname, img, cv2):
    H, W = img.shape[:2]
    r = min(HEAD_SIZE / H, HEAD_SIZE / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    canvas = np.full((HEAD_SIZE, HEAD_SIZE, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {iname: blob})[0]
    pred = out[0] if out.ndim == 3 else out
    if pred.shape[0] < pred.shape[1] and pred.shape[0] < 100:
        pred = pred.T
    pred = pred[pred[:, 4] >= HEAD_CONF]
    if not len(pred):
        return []
    cs = pred[:, 5:]
    cid = cs.argmax(1)
    conf = pred[:, 4] * cs.max(1)
    ok = conf >= HEAD_CONF
    pred, cid, conf = pred[ok], cid[ok], conf[ok]
    dets = [([float((cx - w / 2) / r), float((cy - h / 2) / r),
              float((cx + w / 2) / r), float((cy + h / 2) / r)], float(s), int(c))
            for (cx, cy, w, h), c, s in zip(pred[:, :4], cid, conf)]
    return nms(dets, HEAD_NMS)


def crop(cv2, img, hb, extra):
    H, W = img.shape[:2]
    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
    half = max(long_side(hb) * PAD / 2, 34.0)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = cv2.resize(img[y0:y1, x0:x1].copy(), (CELL, CELL))
    sx, sy = CELL / (x1 - x0), CELL / (y1 - y0)
    for b, col, th in extra:
        cv2.rectangle(sub, (int((b[0] - x0) * sx), int((b[1] - y0) * sy)),
                      (int((b[2] - x0) * sx), int((b[3] - y0) * sy)), col, th)
    return sub


def cluster(items):
    """Collapse per-frame candidates into distinct faces.

    A candidate is emitted per sampled frame, so one person standing in view for a whole chunk
    produces ~360 of them. Annotation cost is driven by distinct faces, not frames, and quoting the
    frame count as a queue size overstates it by whatever that ratio is. Two candidates join the
    same face when they are within 3 samples (90 frames) and their centres are closer than the
    larger box's long side -- the same gate continuation() uses to extend a track.

    Returns a list of clusters, each a list of the input indices. Pairing is O(n^2); above 6000
    candidates in one tier of one chunk it is skipped rather than left to stall, and the caller
    sees the raw count.
    """
    if len(items) > 6000:
        return [[i] for i in range(len(items))]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for k in range(i + 1, len(items)):
            a, b = items[i], items[k]
            if abs(a["f"] - b["f"]) > 3 * STRIDE:
                continue
            ba, bb = a["box"], b["box"]
            if _cdist(ba, bb) < max(long_side(ba), long_side(bb)):
                ra, rb = find(i), find(k)
                if ra != rb:
                    parent[ra] = rb
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    return list(groups.values())


def pick_chunks():
    """Floor-adjusted proportional draw, pilot sessions excluded so the frames are fresh."""
    pool = json.load(open(T / "pilot_pool.json"))["pool"]
    used = set()
    for p in (T / "pilot_out").glob("res_B_*.json"):
        m = re.match(r"res_B_(20\d{6}_\d{6}_[A-Z]{6})_c", p.name)
        if m:
            used.add(m.group(1))
    by = defaultdict(list)
    for e in pool:
        if e.get("nch") and e["sid"] not in used:
            by[norm_scene(e["scene"])].append(e)
    out = []
    for scene, frac in ALLOC.items():
        quota = max(1, round(N_CHUNKS * frac))
        eps = sorted(by.get(scene, []), key=lambda x: x["sid"])
        if not eps:
            print(f"  ⚠️ {scene} 池内无可用 episode，跳过")
            continue
        step = max(1, len(eps) // quota)
        for e in eps[::step][:quota]:
            ci = zlib.crc32(e["sid"].encode()) % max(1, e["nch"])
            out.append({"sid": e["sid"], "chunk": ci, "scene": scene})
    return out[:N_CHUNKS + 2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import cv2
    import onnxruntime as ort
    import torch
    from torchcodec.decoders import VideoDecoder
    from face_pii.detect import FaceDetector
    from we_io import R2Bucket

    OUT.mkdir(exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    for p in OUT.glob("*.jpg"):
        p.unlink()

    det = FaceDetector(str(ENGINE), det_thresh=0.1, nms_thresh=0.4, size=DET_SIZE)
    sess = ort.InferenceSession(str(HEAD_MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    bucket = R2Bucket(str(SCRATCH), bucket="we-fpv-sh-ns", storage_profile="aliyun_oss")

    jobs = pick_chunks()
    mix = defaultdict(int)
    for j in jobs:
        mix[j["scene"]] += 1
    print(f"选中 {len(jobs)} 个未检 chunk（已排除 GT/投诉/face10k/试点 session）")
    print("  场景配比 " + "  ".join(f"{k} {v}" for k, v in sorted(mix.items())), flush=True)

    def armw_only(dec, frame_ids):
        """armW over a frame list, retaining no pixels -- for the dense continuation window.

        The first version kept every decoded frame on the GPU so the crops could be drawn later;
        a merged +/-150 window runs to thousands of frames and that OOMed at 31GB. Nothing here
        needs the pixels, so they are dropped as soon as the boxes are out.
        """
        out = {}
        for i in range(0, len(frame_ids), BATCH):
            ids = frame_ids[i:i + BATCH]
            fr = dec.get_frames_at(ids)
            data = fr.data if hasattr(fr, "data") else fr
            for j, b in enumerate(det.detect_batch(data)):
                out[ids[j]] = [([float(x) for x in bb], float(ss))
                               for bb, ss in b if ss >= KEEP]
            del data, fr
        return out

    def sparse_pass(dec, frame_ids):
        """Both models in one traversal; keeps only the frames that produced a candidate.

        Retaining all ~340 sampled frames as host arrays costs ~2GB per chunk for nothing -- only
        frames with a candidate are ever cropped.
        """
        sparse, cand, keep = {}, [], {}
        for i in range(0, len(frame_ids), BATCH):
            ids = frame_ids[i:i + BATCH]
            fr = dec.get_frames_at(ids)
            data = fr.data if hasattr(fr, "data") else fr
            res = det.detect_batch(data)
            for j, f in enumerate(ids):
                sparse[f] = [([float(x) for x in bb], float(ss))
                             for bb, ss in res[j] if ss >= KEEP]
                img = data[j].permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()  # RGB->BGR
                hits = []
                for hb in [b for b, s, c in run_yolo(sess, iname, img, cv2) if c == HEAD_CLS]:
                    m = [(b, s) for b, s in sparse[f]
                         if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5]
                    ab, best = max(m, key=lambda x: x[1]) if m else (None, 0.0)
                    if ab is not None and long_side(hb) > MAX_HEAD_RATIO * long_side(ab):
                        continue                  # person-sized box wearing a head label
                    tier = next((nm for nm, lo, hi in TIERS if lo <= best < hi), "T0")
                    hits.append({"f": f, "head": hb, "armw_box": ab,
                                 "armw": best, "tier": tier})
                if hits:
                    keep[f] = img
                    cand.extend(hits)
            del data, fr
        return sparse, cand, keep

    rows, per_chunk = [], []
    st = defaultdict(int)
    for ji, j in enumerate(jobs):
        key = f"{j['sid']}/chunk_{j['chunk']:03d}/vst_left/vst_left_video.mp4"
        t0 = time.time()
        try:
            path = Path(str(bucket.get(key)))
        except Exception as e:
            print(f"  [{ji+1}] {j['sid'][-6:]} c{j['chunk']} 下载失败 {type(e).__name__}",
                  flush=True)
            continue
        t_dl, mb = time.time() - t0, path.stat().st_size / 1e6

        t1 = time.time()
        try:
            dec = VideoDecoder(str(path), device="cuda:0")
            n_frames = int(dec.metadata.num_frames or 0)
        except Exception as e:
            print(f"  [{ji+1}] 解码器打不开 {type(e).__name__}", flush=True)
            path.unlink(missing_ok=True)
            continue
        sampled = list(range(0, n_frames, STRIDE))
        # both models on every sampled frame -- all tiers, no pre-filter by armW score
        sparse, cand, imgs = sparse_pass(dec, sampled)
        t_armw = time.time() - t1
        t_head = 0.0

        # continuation check: dense window around every flagged frame
        t3 = time.time()
        win = set()
        for c in cand:
            win |= set(range(max(0, c["f"] - WINDOW), min(n_frames, c["f"] + WINDOW + 1)))
        blurred = {}
        if win:
            wl = sorted(win)
            blurred = tracked(armw_only(dec, wl), wl)
        t_track = time.time() - t3

        n_by_tier = defaultdict(int)
        missed_by_tier = defaultdict(list)
        for c in cand:
            probe = c["armw_box"] or c["head"]
            covered = any(_iof(probe, b) >= 0.5 or _iou(probe, b) >= 0.4
                          for b in blurred.get(c["f"], []))
            kind = "已打码" if covered else "遗漏"
            # production's policy never blurs below min_long_side=40, so a sub-40px candidate is
            # out of scope by design rather than a detector failure. Flagged, not filtered.
            small = long_side(probe) < MIN_LONG
            st[f"{c['tier']}_{kind}"] += 1
            if kind == "遗漏" and small:
                st[f"{c['tier']}_遗漏_小于40px"] += 1
            n_by_tier[c["tier"]] += 1
            if kind == "遗漏":
                missed_by_tier[c["tier"]].append({"f": c["f"], "box": probe, "small": small})
            if kind == "遗漏" and c["tier"] != "T0":
                img = imgs[c["f"]]                       # already BGR host array
                extra = [(c["head"], C_HEAD, 2)]
                if c["armw_box"]:
                    extra.append((c["armw_box"], C_ARMW, 2))
                sub = crop(cv2, img, c["head"], extra)
                if sub is not None:
                    name = (f"{c['tier']}_{j['scene'].replace('/', '')}_"
                            f"{j['sid'][-6:]}_c{j['chunk']:03d}_f{c['f']:06d}.jpg")
                    cv2.imwrite(str(OUT / name), sub, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    rows.append({"file": name, "tier": c["tier"], "scene": j["scene"],
                                 "sid": j["sid"], "chunk": j["chunk"], "frame": c["f"],
                                 "armw": round(c["armw"], 3), "small": small,
                                 "probe_long": round(long_side(probe), 1),
                                 "head_long": round(long_side(c["head"]), 1),
                                 "box": [round(x, 1) for x in c["head"]]})

        # distinct faces, not frames -- the number that actually sizes the annotation queue
        faces = {}
        for tier, items in missed_by_tier.items():
            cl = cluster(items)
            faces[tier] = len(cl)
            big = [g for g in cl if any(not items[i]["small"] for i in g)]
            faces[f"{tier}_≥40px"] = len(big)
            st[f"{tier}_遗漏_面孔"] += len(cl)
            st[f"{tier}_遗漏_面孔_≥40px"] += len(big)
        path.unlink(missing_ok=True)
        imgs.clear()
        del dec
        torch.cuda.empty_cache()
        per_chunk.append({**j, "frames": n_frames, "sampled": len(sampled),
                          "cand": len(cand), "by_tier": dict(n_by_tier), "faces": faces,
                          "mb": round(mb, 1), "t_dl": round(t_dl, 1),
                          "t_armw": round(t_armw, 1), "t_head": round(t_head, 1),
                          "t_track": round(t_track, 1), "window_frames": len(win)})
        json.dump({"rows": rows, "stats": dict(st), "per_chunk": per_chunk},
                  open(OUT / "index.json", "w"), ensure_ascii=False)
        print(f"  [{ji+1}/{len(jobs)}] {j['scene']:8} {j['sid'][-6:]} c{j['chunk']:03d} "
              f"{n_frames}帧 抽{len(sampled)} 候选{len(cand)} "
              f"(T1 {n_by_tier['T1']}/T2 {n_by_tier['T2']}/T3 {n_by_tier['T3']}) | "
              f"下载{t_dl:.0f}s {mb:.0f}MB 双模型{t_armw:.0f}s 续轨{t_track:.0f}s",
              flush=True)

    tot_s = sum(c["sampled"] for c in per_chunk)
    print(f"\n=== 无条件产出（{len(per_chunk)} 个未检 chunk，{tot_s} 个抽样帧）===")
    print(f"{'层':6}{'候选(帧)':>10}{'仍遗漏(帧)':>12}{'已打码':>8}"
          f"{'遗漏面孔':>10}{'其中≥40px':>11}{'帧/面孔':>9}{'面孔/chunk':>12}")
    nc = max(1, len(per_chunk))
    for nm, _lo, _hi in TIERS:
        a, b = st[f"{nm}_遗漏"], st[f"{nm}_已打码"]
        fa, fb = st[f"{nm}_遗漏_面孔"], st[f"{nm}_遗漏_面孔_≥40px"]
        print(f"{nm:6}{a+b:>10}{a:>12}{b:>8}{fa:>10}{fb:>11}"
              f"{a/max(1,fa):>9.1f}{fb/nc:>12.1f}")
    print(f"\n  抽样帧合计 {tot_s}")
    print(f"\n  渲染 {len(rows)} 张「遗漏」候选 -> {OUT}")


if __name__ == "__main__":
    main()

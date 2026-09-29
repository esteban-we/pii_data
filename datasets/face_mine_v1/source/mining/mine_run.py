"""Overnight T1+T2 mining sweep: both detectors on TensorRT, sharded across GPUs, resumable.

What changed from the 24-chunk probe (mine_cand_render.py): the head model moved from onnxruntime-CPU
(~16 f/s, the entire bottleneck) to a TensorRT FP16 engine fed straight off the GPU decoder (~565
f/s, gate in head_parity.py: batch-invariance 100.00%, and the >=0.25 score bands this criterion
consumes match the FP32 reference at 100.0%).

Design decisions that are not obvious:

ORDER, NOT QUOTA. The run stops on a wall clock, so it will process a PREFIX of the job list. A list
grouped by scene would therefore deliver whatever scene happened to sort first. Scenes are instead
interleaved by their target share, so every prefix carries the intended mix and truncation costs
coverage evenly. Shards then take a stride through that same order, so each shard is representative
too and they all reach a similar depth.

ONE CHUNK PER SESSION. The pool holds 16,612 eligible sessions and 167,175 chunks; ten hours on six
GPUs is ~15k chunks. Spending that budget on 15k different sessions rather than on more chunks from
the same sessions buys diversity, which is what a training set needs -- two chunks of one session are
the same room, the same people and the same light.

A CONTROL CHUNK, RUN BY EVERY SHARD. The failure this project keeps hitting is not a crash, it is a
run that finishes and is quietly wrong (a protobuf bug once dropped 80% of frames while the totals
still looked sane). So one fixed chunk is processed by every shard at startup and compared box-for-box
against a reference computed once up front; a shard whose control does not match refuses to run.

A FREE SELF-CHECK EVERY CHUNK. Head boxes whose armW score is already >=0.75 (tier T0) must almost
all be covered by continuation -- measured 2118/2128 = 99.5% on the probe. That rate is computed on
every chunk at no extra cost and is a known answer: if it drifts, the tracker, the engine or the
decode is broken, and it says so in the log rather than in the results.

Outputs are per-shard (a shared directory would have shards wiping each other's crops), per-chunk
JSON is written before the next chunk starts, and the mp4 is unlinked in a finally -- eight workers
leaking 360MB per failed chunk would fill the 3.5T volume in a night.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/liangzhenghao/train")

T = Path("/data/liangzhenghao/train")
ARMW_ENGINE = T / "trt_out/scrfd_1024_fp16_dynb16_5090_armW.engine"
HEAD_ENGINE = T / "trt_out/crowdhuman_640_fp16_dynb16_5090.engine"
RUN = T / "mine_run"
SCRATCH = T / "mine_run_cache"

STRIDE = 30                          # 1 fps at 30 fps capture
KEEP, DET_SIZE = 0.25, 1024
T_HIGH, T_LOW, MAX_GAP, MIN_LONG = 0.75, 0.50, 15, 40.0
BATCH = 8
MAX_HEAD_RATIO = 3.0
TIERS = [("T0", 0.75, 1.01), ("T1", 0.45, 0.75), ("T2", 0.25, 0.45), ("T3", -1.0, 0.25)]
MINE_TIERS = ("T1", "T2")            # what this run is for; T3 was ~90% junk by eye on the probe
WINDOW = 150                         # frames each side for the continuation check
DENSE_SLICE = 4096                   # cap frames per dense inference call
CELL, PAD = 260, 2.2
CROPS_PER_CHUNK = 12                 # one per distinct face, capped
CLUSTER_GATE = 3.0                   # x long side, middle of the measured sensitivity range
CONTROL = ("20260803_054831_BJHFBR", 0)     # small, candidate-rich, from the probe
T0_COVER_EXPECTED = 0.995            # measured 2118/2128 on the probe

# Floor-adjusted proportional shares (expected yield, with a 3% floor for scenes whose density
# estimate rests on a very small pilot sample). Anything not listed is not drawn.
ALLOC = {"工厂": 0.792, "医疗/健康": 0.057, "零售/超市": 0.044,
         "家庭": 0.042, "餐饮/酒店": 0.034, "手作/工位": 0.032}

C_HEAD = (255, 160, 0)               # cv2 is BGR: this draws blue
C_ARMW = (0, 220, 255)               # ... and this draws yellow


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
    """Production continuation(): HIGH seeds a track, LOW only extends, forward only."""
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


def cluster(items, gate=CLUSTER_GATE, gap=3 * STRIDE):
    """Collapse per-frame candidates into distinct faces (union-find on proximity)."""
    if len(items) > 4000:
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
            if abs(a["f"] - b["f"]) > gap:
                continue
            if _cdist(a["box"], b["box"]) < gate * max(long_side(a["box"]),
                                                       long_side(b["box"])):
                ra, rb = find(i), find(k)
                if ra != rb:
                    parent[ra] = rb
    g = defaultdict(list)
    for i in range(len(items)):
        g[find(i)].append(i)
    return list(g.values())


def build_order():
    """Episode list ordered so that EVERY PREFIX carries the target scene mix.

    A wall-clock deadline truncates this list, so grouping by scene would hand the whole budget to
    whichever scene sorted first. Scenes are drawn round-robin weighted by ALLOC via a largest-
    remainder counter, which is deterministic and needs no RNG.
    """
    pool = json.load(open(T / "pilot_pool.json"))["pool"]
    used = set()
    for p in (T / "pilot_out").glob("res_B_*.json"):
        m = re.match(r"res_B_(20\d{6}_\d{6}_[A-Z]{6})_c", p.name)
        if m:
            used.add(m.group(1))
    try:                                  # the 24-chunk probe -- already reviewed, do not repeat
        for r in json.load(open(T / "cand_mine/index.json"))["per_chunk"]:
            used.add(r["sid"])
    except Exception:
        pass

    by = defaultdict(list)
    for e in pool:
        if not e.get("nch") or e["sid"] in used:
            continue
        sc = norm_scene(e["scene"])
        if sc in ALLOC:
            by[sc].append(e)
    for v in by.values():
        v.sort(key=lambda e: e["sid"])

    scenes = [s for s in ALLOC if by.get(s)]
    idx = {s: 0 for s in scenes}
    credit = {s: 0.0 for s in scenes}
    order = []
    total = sum(len(by[s]) for s in scenes)
    while len(order) < total:
        for s in scenes:
            credit[s] += ALLOC[s]
        # emit from whichever scene has the most accumulated credit and still has episodes
        avail = [s for s in scenes if idx[s] < len(by[s])]
        if not avail:
            break
        s = max(avail, key=lambda x: credit[x])
        credit[s] -= 1.0
        e = by[s][idx[s]]
        idx[s] += 1
        # Alternate the view per episode. Production detects on BOTH vst views independently, and a
        # left/right comparison over 60 chunks (stereo_gap.py) found the detector behaves the same on
        # each (T1 candidates -6.9%, T2 +12.1% -- noise, no direction) but only 68.0% of
        # candidate-bearing frames overlap: 16.0% carry a face the left view does not show at all.
        # Mining one fixed view is therefore an unbiased sample of detector behaviour but a permanent
        # blind spot for disparity-hidden faces.
        #
        # !! THIS HAS NEVER RUN. It was added 2026-08-26 17:28, after the 00:04-08:54 sweep that
        # produced handoff_2026-08 finished. That delivered mining subset is 100% left eye: all
        # 15,852 chunk records on disk carry no `view` field and no _left/_right tag suffix.
        # Two things must be fixed before a re-run buys anything:
        #   (a) recover_boxes.py -- the script that extracts the frames actually shipped -- hardcodes
        #       vst_left and takes no view parameter, so it would re-emit left frames regardless;
        #   (b) resume is broken across this change: `done` is built from chunk-file stems, and the
        #       new tag adds a _left/_right suffix, so all 15,852 existing records read as not-done.
        # Note also that alternating does NOT give a session full coverage -- one eye captures ~84%
        # of that session's stereo union either way. What it removes is the corpus-level systematic
        # bias toward left-camera-visible faces. Full coverage needs both eyes per session.
        #
        # crc32 of a salted sid keeps it deterministic, so a resumed run picks the same view.
        view = "right" if zlib.crc32((e["sid"] + "|view").encode()) % 2 else "left"
        order.append({"sid": e["sid"], "scene": s, "view": view,
                      "chunk": zlib.crc32(e["sid"].encode()) % max(1, e["nch"]),
                      "prov": e.get("prov"), "hw": e.get("hw")})
    return order


class Worker:
    def __init__(self, out_dir: Path, scratch: Path | None = None):
        import cv2
        import torch
        from torchcodec.decoders import VideoDecoder
        from face_pii.detect import FaceDetector
        from head_trt import HeadDetector
        from we_io import R2Bucket

        self.cv2, self.torch, self.VideoDecoder = cv2, torch, VideoDecoder
        self.out = out_dir
        (self.out / "chunks").mkdir(parents=True, exist_ok=True)
        (self.out / "crops").mkdir(parents=True, exist_ok=True)
        self.armw = FaceDetector(str(ARMW_ENGINE), det_thresh=0.1, nms_thresh=0.4,
                                 size=DET_SIZE)
        self.head = HeadDetector(HEAD_ENGINE)
        # Per-worker download dir. R2Bucket's mutual exclusion is threading-only and its tmp path
        # is a deterministic function of the key, so two PROCESSES fetching the same key race on
        # the same <key>.tmp and one of them dies renaming a file the other already moved. Shards
        # draw disjoint chunks, but every shard fetches the same control chunk at startup -- which
        # is exactly what happened on the first launch: six of seven shards failed their canary on
        # a download collision, not on a model difference.
        sc = Path(scratch) if scratch else SCRATCH
        sc.mkdir(parents=True, exist_ok=True)
        self.bucket = R2Bucket(str(sc), bucket="we-fpv-sh-ns",
                               storage_profile="aliyun_oss")

    def _sparse(self, dec, ids):
        """Both models over the sampled frames; retains only frames that produced a candidate."""
        sparse, cand, keep = {}, [], {}
        for i in range(0, len(ids), BATCH):
            blk = ids[i:i + BATCH]
            fr = dec.get_frames_at(blk)
            data = fr.data if hasattr(fr, "data") else fr
            face = self.armw.detect_batch(data)
            heads = self.head.heads(data)
            for j, f in enumerate(blk):
                sparse[f] = [([float(x) for x in bb], float(ss))
                             for bb, ss in face[j] if ss >= KEEP]
                hits = []
                for hb in heads[j]:
                    m = [(b, s) for b, s in sparse[f]
                         if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5]
                    ab, best = max(m, key=lambda x: x[1]) if m else (None, 0.0)
                    if ab is not None and long_side(hb) > MAX_HEAD_RATIO * long_side(ab):
                        continue
                    tier = next((nm for nm, lo, hi in TIERS if lo <= best < hi), "T0")
                    hits.append({"f": f, "head": hb, "armw_box": ab,
                                 "armw": best, "tier": tier})
                if hits:
                    keep[f] = data[j].permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
                    cand.extend(hits)
            del data, fr
        return sparse, cand, keep

    def _dense(self, dec, ids):
        """armW at full rate over the continuation window; keeps no pixels."""
        out = {}
        for s in range(0, len(ids), DENSE_SLICE):
            sl = ids[s:s + DENSE_SLICE]
            for i in range(0, len(sl), BATCH):
                blk = sl[i:i + BATCH]
                fr = dec.get_frames_at(blk)
                data = fr.data if hasattr(fr, "data") else fr
                for j, b in enumerate(self.armw.detect_batch(data)):
                    out[blk[j]] = [([float(x) for x in bb], float(ss))
                                   for bb, ss in b if ss >= KEEP]
                del data, fr
        return out

    def _crop(self, img, hb, extra):
        cv2 = self.cv2
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

    def process(self, job, *, write_crops=True, view="left"):
        """One chunk of one view. Always deletes the mp4. Returns the record dict.

        ``view`` exists because production detects on BOTH vst views independently
        (chunk_detect.py: ``VIEWS = ("lview", "rview")``) while every measurement here has been
        left-only. The sweep ran with the default; the stereo comparison drives both.
        """
        cv2, torch = self.cv2, self.torch
        tag = f"{job['sid']}_c{job['chunk']:03d}_{view}"
        key = (f"{job['sid']}/chunk_{job['chunk']:03d}/"
               f"vst_{view}/vst_{view}_video.mp4")
        rec = {**job, "tag": tag, "view": view}
        path = None
        dec = None
        t0 = time.time()
        try:
            path = Path(str(self.bucket.get(key)))
            rec["mb"] = round(path.stat().st_size / 1e6, 1)
            rec["t_dl"] = round(time.time() - t0, 1)

            t1 = time.time()
            dec = self.VideoDecoder(str(path), device="cuda:0")
            n_frames = int(dec.metadata.num_frames or 0)
            sampled = list(range(0, n_frames, STRIDE))
            sparse, cand, imgs = self._sparse(dec, sampled)
            # invariant: every sampled frame must come back, or the decoder silently skipped work
            if len(sparse) != len(sampled):
                rec["error"] = f"decoded {len(sparse)} of {len(sampled)} sampled frames"
                return rec
            rec["frames"], rec["sampled"] = n_frames, len(sampled)
            rec["t_infer"] = round(time.time() - t1, 1)

            t2 = time.time()
            win = set()
            for c in cand:
                win |= set(range(max(0, c["f"] - WINDOW), min(n_frames, c["f"] + WINDOW + 1)))
            blurred = {}
            if win:
                wl = sorted(win)
                blurred = tracked(self._dense(dec, wl), wl)
            rec["window_frames"], rec["t_track"] = len(win), round(time.time() - t2, 1)

            by_tier = defaultdict(int)
            missed = defaultdict(list)
            t0_seen = t0_cov = 0
            for c in cand:
                probe = c["armw_box"] or c["head"]
                covered = any(_iof(probe, b) >= 0.5 or _iou(probe, b) >= 0.4
                              for b in blurred.get(c["f"], []))
                by_tier[c["tier"]] += 1
                if c["tier"] == "T0":
                    t0_seen += 1
                    t0_cov += bool(covered)
                if not covered:
                    missed[c["tier"]].append({"f": c["f"], "box": probe, "head": c["head"],
                                              "armw": c["armw"],
                                              "small": long_side(probe) < MIN_LONG})
            # sanity check with a known answer: T0 is armW >= 0.75, which seeds a track by
            # definition, so it should be ~99.5% covered. Anything else means something upstream
            # is wrong, and the run should say so rather than bank the numbers.
            rec["t0_seen"], rec["t0_covered"] = t0_seen, t0_cov
            rec["cand"] = len(cand)
            rec["by_tier"] = dict(by_tier)

            faces, shown = {}, 0
            for tier in MINE_TIERS:
                items = missed.get(tier, [])
                cl = cluster(items)
                big = [g for g in cl if any(not items[i]["small"] for i in g)]
                faces[tier] = len(cl)
                faces[f"{tier}_≥40px"] = len(big)
                if not write_crops:
                    continue
                # one crop per distinct face, best-scoring member, capped
                order = sorted(big, key=lambda g: -max(items[i]["armw"] for i in g))
                for g in order:
                    if shown >= CROPS_PER_CHUNK:
                        break
                    i = max(g, key=lambda k: items[k]["armw"])
                    it = items[i]
                    img = imgs.get(it["f"])
                    if img is None:
                        continue
                    extra = [(it["head"], C_HEAD, 2), (it["box"], C_ARMW, 2)]
                    sub = self._crop(img, it["head"], extra)
                    if sub is None:
                        continue
                    name = (f"{tier}_{job['scene'].replace('/', '')}_{tag}"
                            f"_f{it['f']:06d}.jpg")
                    cv2.imwrite(str(self.out / "crops" / name), sub,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    rec.setdefault("crops", []).append(
                        {"file": name, "tier": tier, "frame": it["f"],
                         "armw": round(it["armw"], 3),
                         "probe_long": round(long_side(it["box"]), 1),
                         "n_frames_in_face": len(g)})
                    shown += 1
            rec["faces"] = faces
            # frame indices carrying a mining-tier miss -- lets a left/right comparison ask
            # "did the other view see anything here at all", which counts is what disparity
            # would move between the views
            rec["cand_frames"] = sorted({it["f"] for t in MINE_TIERS
                                         for it in missed.get(t, [])})
            rec["missed"] = {k: len(v) for k, v in missed.items()}
            rec["missed_small"] = {k: sum(1 for x in v if x["small"])
                                   for k, v in missed.items()}
        except Exception as exc:                       # one bad chunk must not end the shard
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if dec is not None:
                del dec
            if path is not None:
                path.unlink(missing_ok=True)
            self.torch.cuda.empty_cache()
        rec["t_total"] = round(time.time() - t0, 1)
        return rec


def control_signature(rec):
    """The part of a control result that must be identical across shards."""
    return {"cand": rec.get("cand"), "by_tier": rec.get("by_tier"),
            "faces": rec.get("faces"), "missed": rec.get("missed"),
            "sampled": rec.get("sampled"), "t0_seen": rec.get("t0_seen"),
            "t0_covered": rec.get("t0_covered"), "error": rec.get("error")}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--mode", choices=("run", "control"), default="run")
    args = ap.parse_args()

    RUN.mkdir(exist_ok=True)
    SCRATCH.mkdir(exist_ok=True)
    ref_path = RUN / "control_ref.json"

    if args.mode == "control":
        w = Worker(RUN / "control", SCRATCH / "control")
        rec = w.process({"sid": CONTROL[0], "chunk": CONTROL[1], "scene": "工厂"},
                        write_crops=False)
        json.dump(control_signature(rec), open(ref_path, "w"), ensure_ascii=False, indent=1)
        print(f"控制样本参考写出 {ref_path}")
        print(json.dumps(control_signature(rec), ensure_ascii=False))
        return

    out = RUN / f"shard{args.shard}"
    w = Worker(out, SCRATCH / f"shard{args.shard}")

    # Canary: this shard must reproduce the reference exactly before it is allowed to contribute.
    # A control that ERRORED is a different fact from one that produced different numbers -- the
    # first is transient (a download race, a bad connection) and deserves a retry; only the second
    # means this GPU or engine is producing different results and must not run.
    if ref_path.exists():
        ref = json.load(open(ref_path))
        got = None
        for attempt in range(3):
            got = control_signature(w.process({"sid": CONTROL[0], "chunk": CONTROL[1],
                                               "scene": "工厂"}, write_crops=False))
            if not got.get("error"):
                break
            print(f"控制样本第 {attempt+1} 次出错：{got['error']}，重试", flush=True)
            time.sleep(5 * (attempt + 1))
        if got.get("error"):
            print(f"控制样本连续出错，本分片放弃：{got['error']}", flush=True)
            sys.exit(3)
        if got != ref:
            print("控制样本数值不一致，本分片拒绝运行：", flush=True)
            print(f"  期望 {json.dumps(ref, ensure_ascii=False)}")
            print(f"  实得 {json.dumps(got, ensure_ascii=False)}")
            sys.exit(2)
        print(f"控制样本一致（cand={ref['cand']}），分片 {args.shard} 启动", flush=True)
    else:
        print("⚠️ 无控制样本参考，跳过自检", flush=True)

    order = build_order()
    jobs = [j for i, j in enumerate(order) if i % args.nshard == args.shard]
    done = {p.stem for p in (out / "chunks").glob("*.json")}
    deadline = time.time() + args.hours * 3600
    mix = defaultdict(int)
    for j in jobs:
        mix[j["scene"]] += 1
    print(f"分片 {args.shard}/{args.nshard}  待处理 {len(jobs)}（已完成 {len(done)}）  "
          f"截止 {time.strftime('%H:%M', time.localtime(deadline))}", flush=True)
    print("  配比 " + "  ".join(f"{k} {v}" for k, v in sorted(mix.items())), flush=True)

    n_ok = n_err = 0
    agg = defaultdict(int)
    t_start = time.time()
    for i, j in enumerate(jobs):
        if time.time() > deadline:
            print(f"到达截止时间，停在第 {i} 个", flush=True)
            break
        view = j.get("view", "left")
        tag = f"{j['sid']}_c{j['chunk']:03d}_{view}"
        if tag in done:
            continue
        rec = w.process(j, view=view)
        json.dump(rec, open(out / "chunks" / f"{tag}.json", "w"), ensure_ascii=False)
        if rec.get("error"):
            n_err += 1
            print(f"  [{i}] {tag} 失败 {rec['error']}", flush=True)
            continue
        n_ok += 1
        for k in ("cand", "t0_seen", "t0_covered", "sampled"):
            agg[k] += rec.get(k, 0) or 0
        for t in MINE_TIERS:
            agg[f"faces_{t}"] += rec.get("faces", {}).get(f"{t}_≥40px", 0) or 0
        if n_ok % 25 == 0:
            cov = agg["t0_covered"] / max(1, agg["t0_seen"])
            rate = n_ok / max(1e-9, time.time() - t_start) * 3600
            flag = "" if cov >= T0_COVER_EXPECTED - 0.03 or agg["t0_seen"] < 200 else "  ⚠️自检偏低"
            print(f"  [{n_ok} 成功 / {n_err} 失败] {rate:.0f} chunk/h  "
                  f"T1面孔 {agg['faces_T1']} T2面孔 {agg['faces_T2']}  "
                  f"T0自检 {cov*100:.1f}%{flag}", flush=True)
    cov = agg["t0_covered"] / max(1, agg["t0_seen"])
    print(f"\n分片 {args.shard} 结束：成功 {n_ok}  失败 {n_err}  "
          f"抽样帧 {agg['sampled']}  候选 {agg['cand']}  "
          f"T1面孔 {agg['faces_T1']}  T2面孔 {agg['faces_T2']}  "
          f"T0自检 {cov*100:.2f}%（期望 {T0_COVER_EXPECTED*100:.1f}%）", flush=True)


if __name__ == "__main__":
    main()

"""Is the 95.4% coverage real, or is a whole-person box trivially containing the face?

CrowdHuman has two classes and both scored ~94% here, which is the tell: a person box contains the
face by construction, so intersection-over-the-face-box is 1.0 for free and says nothing. A mining
signal has to LOCALISE the face, not merely contain it.

So: measure the size of each class's boxes relative to the GT face they cover. A head box should be
roughly 1-3x the face's long side; a person box will be 5-15x. Then recompute coverage keeping only
boxes within a head-like size ratio -- that number is the one worth quoting, and the gap between the
two is how much of the original 95.4% was an artefact of my own metric.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

T = Path("/data/liangzhenghao/train")
CACHE = T / "pipeline_eval2"
MODEL = T / "mine_models/crowdhuman.onnx"
MIN_LONG, IOF_HIT = 40.0, 0.5
T_HIGH, T_LOW, MAX_GAP = 0.75, 0.50, 15
DOLL = ("BTJPSP", "NQYJPB")
SIZE = 640
CONF, NMS_THR = 0.15, 0.45
RATIOS = [2.0, 3.0, 5.0]        # max head-box long side as a multiple of the face's


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / (
        (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _iof(gt, b):
    ix1, iy1 = max(gt[0], b[0]), max(gt[1], b[1])
    ix2, iy2 = min(gt[2], b[2]), min(gt[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return 0.0 if a <= 0 else (iw * ih) / a


def _cdist(a, b):
    return (((a[0]+a[2])-(b[0]+b[2]))**2 + ((a[1]+a[3])-(b[1]+b[3]))**2) ** 0.5 / 2


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
        ud.add(di); ut.add(id(t)); out[di] = t
    return out


def tracked(dets, n):
    tracks = []
    for f in range(n):
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
            t["frames"][f] = hi[di]; t["last"] = hi[di]; t["last_f"] = f
        used = {id(t) for t in mh.values()}
        ml = _greedy(lo, [t for t in op if id(t) not in used])
        for di, t in ml.items():
            t["frames"][f] = lo[di]; t["last"] = lo[di]; t["last_f"] = f
        for di, b in enumerate(hi):
            if di not in mh:
                tracks.append({"frames": {f: b}, "last": b, "last_f": f})
    out = {}
    for t in tracks:
        for f, b in t["frames"].items():
            out.setdefault(f, []).append(b)
    return out


def load(tag, model):
    z = np.load(CACHE / f"dets_{tag}_{model}.npz")
    frames, counts, dets = z["frames"], z["counts"], z["dets"]
    out, off = {}, 0
    for f, c in zip(frames, counts):
        blk = dets[off:off + c]
        off += c
        out[int(f)] = [([float(x) for x in b[:4]], float(b[4])) for b in blk]
    return out, int(z["n_frames"])


def nms(dets, thr):
    dets = sorted(dets, key=lambda x: -x[1])
    keep = []
    for b, s, c in dets:
        if all(_iou(b, kb) < thr for kb, _ks, kc in keep if kc == c):
            keep.append((b, s, c))
    return keep


def run_yolo(sess, iname, img, size, cv2):
    H, W = img.shape[:2]
    r = min(size / H, size / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {iname: blob})[0]
    pred = out[0] if out.ndim == 3 else out
    if pred.shape[0] < pred.shape[1] and pred.shape[0] < 100:
        pred = pred.T
    pred = pred[pred[:, 4] >= CONF]
    if not len(pred):
        return []
    cs = pred[:, 5:]
    cid = cs.argmax(1)
    conf = pred[:, 4] * cs.max(1)
    ok = conf >= CONF
    pred, cid, conf = pred[ok], cid[ok], conf[ok]
    dets = []
    for (cx, cy, w, h), c, s in zip(pred[:, :4], cid, conf):
        dets.append(([float((cx - w / 2) / r), float((cy - h / 2) / r),
                      float((cx + w / 2) / r), float((cy + h / 2) / r)],
                     float(s), int(c)))
    return nms(dets, NMS_THR)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import cv2
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    gt = defaultdict(dict)
    for e in json.load(open(T / "gt_eval.json")):
        f = int(re.search(r"_f(\d+)\.jpg", e["path"]).group(1))
        gt[(e["session"][-6:], int(e["chunk"]))][f] = (e["boxes"], e["path"])

    targets, frames_of, n_target = {}, {}, 0
    for p in sorted(CACHE.glob("dets_*_armW.npz")):
        tag = p.name[len("dets_"):-len("_armW.npz")]
        ss, ck = tag.rsplit("_c", 1)
        if any(d in ss for d in DOLL) or (ss, int(ck)) not in gt:
            continue
        dw, n = load(tag, "armW")
        dp, _ = load(tag, "prod")
        tw, tp = tracked(dw, n), tracked(dp, n)
        for f, (boxes, path) in gt[(ss, int(ck))].items():
            tg = [g for g in boxes if long_side(g) >= MIN_LONG
                  and not any(_iof(g, b) >= IOF_HIT for b in tw.get(f, []))
                  and not any(_iof(g, b) >= IOF_HIT for b in tp.get(f, []))]
            if tg:
                targets[(tag, f)] = tg
                frames_of[(tag, f)] = (path, [g for g in boxes if long_side(g) >= MIN_LONG])
                n_target += len(tg)

    combo = defaultdict(int)
    sizes_by_cls = defaultdict(list)
    hit = {(c, r): 0 for c in (0, 1, "any") for r in RATIOS + [None]}
    det_tot = defaultdict(int)
    det_on_face = defaultdict(int)
    for i, (key, tg) in enumerate(sorted(targets.items())):
        path, truth = frames_of[key]
        img = cv2.imread(path)
        if img is None:
            continue
        dets = run_yolo(sess, iname, img, SIZE, cv2)
        for b, _s, c in dets:
            det_tot[c] += 1
            covered = [g for g in truth if _iof(g, b) >= IOF_HIT]
            if covered:
                det_on_face[c] += 1
                sizes_by_cls[c].append(long_side(b) / long_side(covered[0]))
        for g in tg:
            gl = long_side(g)
            for c in (0, 1):
                cd = [b for b, _s, cc in dets if cc == c and _iof(g, b) >= IOF_HIT]
                if cd:
                    hit[(c, None)] += 1
                for rr in RATIOS:
                    if any(long_side(b) <= rr * gl for b in cd):
                        hit[(c, rr)] += 1
            allc = [b for b, _s, _c in dets if _iof(g, b) >= IOF_HIT]
            if allc:
                hit[("any", None)] += 1
            for rr in RATIOS:
                if any(long_side(b) <= rr * gl for b in allc):
                    hit[("any", rr)] += 1
        # the actual mining rule: a head box where armW is absent or weak
        tag, fr = key
        dw, _n = load(tag, "armW")
        aw = dw.get(fr, [])
        heads = [b for b, _s, c in dets if c == 1]
        for hb in heads:
            best = max((s for b, s in aw if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5),
                       default=0.0)
            on_face = any(_iof(g, hb) >= IOF_HIT for g in truth)
            is_tgt = any(_iof(g, hb) >= IOF_HIT for g in tg)
            for thr in (0.75, 0.50, 0.25):
                if best < thr:
                    combo[f"cand_{thr}"] += 1
                    combo[f"onface_{thr}"] += on_face
                    combo[f"tgt_{thr}"] += is_tgt
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(targets)}", flush=True)

    print(f"\n=== 每个类别的框，相对它盖住的 GT 人脸有多大 ===")
    for c in sorted(sizes_by_cls):
        v = sizes_by_cls[c]
        print(f"  类别 {c}: {len(v)} 个盖住人脸的框，长边倍数 "
              f"中位 {st.median(v):.1f}×  p10 {np.quantile(v,0.1):.1f}×  p90 {np.quantile(v,0.9):.1f}×"
              f"   (共检出 {det_tot[c]})")
    print("  头框应在 1~3×，整人框会到 5~15×")

    print(f"\n=== 目标召回（{n_target} 个目标框）===")
    print(f"{'限制':22}{'类别0':>10}{'类别1':>10}{'任意':>10}")
    print(f"{'不限尺寸（会被整人框刷高）':22}"
          f"{hit[(0,None)]/n_target*100:>9.1f}%{hit[(1,None)]/n_target*100:>9.1f}%"
          f"{hit[('any',None)]/n_target*100:>9.1f}%")
    for rr in RATIOS:
        print(f"{f'框长边 ≤ {rr}× 人脸':22}"
              f"{hit[(0,rr)]/n_target*100:>9.1f}%{hit[(1,rr)]/n_target*100:>9.1f}%"
              f"{hit[('any',rr)]/n_target*100:>9.1f}%")
    print(f"\n  对照：分数带判据 A@≥0.45 的目标召回 62.2%、精确率 69.4%")
    print("")
    print("=== 组合判据：头框 且 armW 分数低于门槛 ===")
    print(f"{'门槛':16}{'候选头框':>10}{'落在GT脸上':>12}{'精确率':>9}{'覆盖目标':>10}{'目标召回':>10}")
    for thr in (0.75, 0.50, 0.25):
        c = combo[f"cand_{thr}"]; o = combo[f"onface_{thr}"]; t = combo[f"tgt_{thr}"]
        if c:
            print(f"{f'armW < {thr}':16}{c:>10}{o:>12}{o/c*100:>8.1f}%{t:>10}"
                  f"{t/n_target*100:>9.1f}%")
    json.dump({"combo": dict(combo), "hit": {f"{k[0]}_{k[1]}": v for k, v in hit.items()},
               "n_target": n_target,
               "size_ratio_median": {str(c): st.median(v) for c, v in sizes_by_cls.items()},
               "det_tot": {str(k): v for k, v in det_tot.items()},
               "det_on_face": {str(k): v for k, v in det_on_face.items()}},
              open(T / "head_probe2.json", "w"))


if __name__ == "__main__":
    main()

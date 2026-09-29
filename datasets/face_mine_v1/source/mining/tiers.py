"""Precision and recall of the two-tier candidate queue, the one number the plan still lacks.

Measured separately so far: the score band alone (armW in [0.45,0.75) -- 69.4% precision, 62.2%
target recall) and the head detector alone (90.1% target recall under a <=2x size constraint, but a
measured precision of only 20.6%, which is a lower bound because GT labels only faces the annotators
marked at >=40px). Their INTERSECTION has not been measured, and multiplying the two numbers would be
wrong -- the errors are not independent, that is the entire reason for using two models.

So bucket every head detection by what armW scored at the same place:

  T1  armW in [0.45, 0.75)   both models point at the same spot -- expected highest precision
  T2  armW in [0.25, 0.45)   head model leads, armW barely responds
  T3  armW < 0.25            head model alone; back-of-head and sub-40px faces live here

Precision counts a candidate as correct when it covers a labelled GT face; recall counts the 283
boxes neither shipped model covered. Both are computed only on GT frames, where the labelling is
complete.
"""
from __future__ import annotations

import json
import re
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
SIZE, CONF, NMS_THR = 640, 0.15, 0.45
HEAD_CLS = 1                    # class 1 is the head (median 1.5x the face's long side)
MAX_HEAD_RATIO = 3.0            # reject person-sized boxes masquerading as coverage
TIERS = [("T1  armW 0.45~0.75", 0.45, 0.75),
         ("T2  armW 0.25~0.45", 0.25, 0.45),
         ("T3  armW < 0.25", -1.0, 0.25)]


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


def run_yolo(sess, iname, img, cv2):
    H, W = img.shape[:2]
    r = min(SIZE / H, SIZE / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
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
    dets = [([float((cx - w / 2) / r), float((cy - h / 2) / r),
              float((cx + w / 2) / r), float((cy + h / 2) / r)], float(s), int(c))
            for (cx, cy, w, h), c, s in zip(pred[:, :4], cid, conf)]
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

    work, n_target = {}, 0
    armw_cache = {}
    for p in sorted(CACHE.glob("dets_*_armW.npz")):
        tag = p.name[len("dets_"):-len("_armW.npz")]
        ss, ck = tag.rsplit("_c", 1)
        if any(d in ss for d in DOLL) or (ss, int(ck)) not in gt:
            continue
        dw, n = load(tag, "armW")
        dp, _ = load(tag, "prod")
        armw_cache[tag] = dw
        tw, tp = tracked(dw, n), tracked(dp, n)
        for f, (boxes, path) in gt[(ss, int(ck))].items():
            truth = [g for g in boxes if long_side(g) >= MIN_LONG]
            tg = [g for g in truth
                  if not any(_iof(g, b) >= IOF_HIT for b in tw.get(f, []))
                  and not any(_iof(g, b) >= IOF_HIT for b in tp.get(f, []))]
            if tg:
                work[(tag, f)] = (path, truth, tg)
                n_target += len(tg)
    print(f"目标框 {n_target}，帧 {len(work)}", flush=True)

    st = defaultdict(lambda: defaultdict(int))
    hit_targets = defaultdict(set)
    for i, ((tag, f), (path, truth, tg)) in enumerate(sorted(work.items())):
        img = cv2.imread(path)
        if img is None:
            continue
        aw = armw_cache[tag].get(f, [])
        heads = [b for b, _s, c in run_yolo(sess, iname, img, cv2) if c == HEAD_CLS]
        for hb in heads:
            best = max((s for b, s in aw if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5),
                       default=0.0)
            covered = [g for g in truth
                       if _iof(g, hb) >= IOF_HIT and long_side(hb) <= MAX_HEAD_RATIO * long_side(g)]
            for name, lo, hi in TIERS:
                if lo <= best < hi:
                    st[name]["cand"] += 1
                    st[name]["on_face"] += bool(covered)
                    for g in covered:
                        if any(g is t or g == t for t in tg):
                            hit_targets[name].add((tag, f, tuple(g)))
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(work)}", flush=True)

    print(f"\n=== 两级候选队列（头框，且按 armW 分数分层）===")
    print(f"{'层':22}{'候选':>8}{'落在GT脸':>10}{'精确率':>9}{'覆盖目标':>10}{'目标召回':>10}")
    cum_c = cum_t = 0
    for name, _lo, _hi in TIERS:
        s = st[name]
        t = len(hit_targets[name])
        cum_c += s["cand"]; cum_t += t
        print(f"{name:22}{s['cand']:>8}{s['on_face']:>10}"
              f"{s['on_face']/max(1,s['cand'])*100:>8.1f}%{t:>10}{t/n_target*100:>9.1f}%")
    print(f"{'累计':22}{cum_c:>8}{'':>10}{'':>9}{cum_t:>10}{cum_t/n_target*100:>9.1f}%")
    print(f"\n  对照：分数带判据 A@≥0.45 单用 = 556 候选 / 69.4% 精确率 / 62.2% 目标召回")
    print(f"  注：精确率是下界 —— GT 只标了 ≥40px 且标注员认定的脸，")
    print(f"      头框打在后脑、更小的脸、未标注的人身上都会被算成「不在 GT 脸上」")
    json.dump({k: dict(v) for k, v in st.items()}, open(T / "tiers.json", "w"))


if __name__ == "__main__":
    main()

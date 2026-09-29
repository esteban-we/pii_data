"""Validate the hard-positive mining criterion on the GT bench, before spending GPU on 1.9M chunks.

The plan needs a rule that finds faces armW misses. The obvious rule -- "a second model sees it" --
has no second model available: sapiens5_2d predicts the WEARER's own shoulders/elbows/hip (it feeds
the 3D upper-body solver), EgoBlur and RTMO weights are not on this host, and the SCRFD arms are all
finetunes of the same official checkpoint (on this bench armW-only is 193 boxes while prod-only is 6,
so production contributes almost nothing armW cannot already see).

So use TIME as the second opinion instead: within one track, some frames score high and others dip
below t_high. A dipped frame is a hard positive whose "this is a face" label is vouched for by the
high-scoring frames of the same track -- no second model, no extra inference, and it targets exactly
armW's failure mode (cannot seed a track at 0.75).

Measured here against the bench's 1,541 human-labelled boxes, and specifically against the 283 that
NEITHER armW nor production covered -- those are the mining target. Precision is only scored on GT
frames, where the labelling is complete; candidates on unlabelled frames cannot be judged and are
excluded rather than assumed correct.
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
GT = T / "gt_eval.json"
MIN_LONG, IOF_HIT = 40.0, 0.5
T_HIGH, T_LOW, MAX_GAP = 0.75, 0.50, 15
LINK_GAP = 15          # frames a chain may skip while still being the same face
VOUCH_W = 90           # a high-scoring frame this far away still vouches for the dip
DOLL = ("BTJPSP", "NQYJPB")
BANDS = [0.25, 0.35, 0.45, 0.50, 0.60, 0.70]


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


def production_tracked(dets, n):
    """Exactly the shipped policy, to reproduce which GT boxes end up uncovered."""
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


def chains(dets, n, floor):
    """Link every detection above `floor` into chains, regardless of score.

    Deliberately NOT the production tracker: production only seeds at 0.75, which is the very
    behaviour being worked around. Here any detection can start a chain, so a face that never
    once crosses 0.75 still forms one -- and a face that does gives its dips a voucher.
    """
    tracks = []
    for f in range(n):
        d = [(b, s) for b, s in dets.get(f, []) if s >= floor and long_side(b) >= MIN_LONG]
        if not d:
            continue
        boxes = [b for b, _s in d]
        op = [t for t in tracks if 0 < f - t["last_f"] <= LINK_GAP]
        m = _greedy(boxes, op)
        for di, t in m.items():
            t["frames"][f] = d[di]; t["last"] = boxes[di]; t["last_f"] = f
            t["best"] = max(t["best"], d[di][1])
            t["hi_f"].append(f) if d[di][1] >= T_HIGH else None
        for di, (b, s) in enumerate(d):
            if di not in m:
                tracks.append({"frames": {f: (b, s)}, "last": b, "last_f": f,
                               "best": s, "hi_f": [f] if s >= T_HIGH else []})
    return tracks


def load(tag, model):
    z = np.load(CACHE / f"dets_{tag}_{model}.npz")
    frames, counts, dets = z["frames"], z["counts"], z["dets"]
    out, off = {}, 0
    for f, c in zip(frames, counts):
        blk = dets[off:off + c]
        off += c
        out[int(f)] = [([float(x) for x in b[:4]], float(b[4])) for b in blk]
    return out, int(z["n_frames"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gt = defaultdict(dict)
    for e in json.load(open(GT)):
        f = int(re.search(r"_f(\d+)\.jpg", e["path"]).group(1))
        gt[(e["session"][-6:], int(e["chunk"]))][f] = e["boxes"]

    # target set: GT boxes that neither model's shipped output covered
    targets, all_gt = [], []
    cand_stats = {b: {"cand": 0, "tp": 0, "hit_target": 0} for b in BANDS}
    vouch_stats = {b: {"cand": 0, "tp": 0, "hit_target": 0} for b in BANDS}
    raw_score_of_target = []

    for p in sorted(CACHE.glob("dets_*_armW.npz")):
        tag = p.name[len("dets_"):-len("_armW.npz")]
        sess, ck = tag.rsplit("_c", 1)
        if any(d in sess for d in DOLL):
            continue
        gtf = gt.get((sess, int(ck)), {})
        if not gtf:
            continue
        dw, n = load(tag, "armW")
        dp, _ = load(tag, "prod")
        tw, tp = production_tracked(dw, n), production_tracked(dp, n)

        miss = defaultdict(list)
        for f, boxes in gtf.items():
            for g in boxes:
                if long_side(g) < MIN_LONG:
                    continue
                all_gt.append(1)
                hw = any(_iof(g, b) >= IOF_HIT for b in tw.get(f, []))
                hp = any(_iof(g, b) >= IOF_HIT for b in tp.get(f, []))
                if not hw and not hp:
                    miss[f].append(g)
                    targets.append((tag, f, g))
                    best = max((s for b, s in dw.get(f, []) if _iof(g, b) >= 0.3), default=0.0)
                    raw_score_of_target.append(best)

        ch = chains(dw, n, min(BANDS))
        # index: frame -> list of (box, score, vouched)
        by_frame = defaultdict(list)
        for t in ch:
            hi = sorted(t["hi_f"])
            for f, (b, s) in t["frames"].items():
                vouched = any(abs(f - h) <= VOUCH_W for h in hi)
                by_frame[f].append((b, s, vouched))

        for f, boxes in gtf.items():
            truth = [g for g in boxes if long_side(g) >= MIN_LONG]
            tgt = miss.get(f, [])
            for b, s, vouched in by_frame.get(f, []):
                if s >= T_HIGH:
                    continue                       # already blurred, not a mining candidate
                is_face = any(_iof(g, b) >= IOF_HIT for g in truth)
                is_tgt = any(_iof(g, b) >= IOF_HIT for g in tgt)
                for lo in BANDS:
                    if s < lo:
                        continue
                    cand_stats[lo]["cand"] += 1
                    cand_stats[lo]["tp"] += is_face
                    cand_stats[lo]["hit_target"] += is_tgt
                    if vouched:
                        vouch_stats[lo]["cand"] += 1
                        vouch_stats[lo]["tp"] += is_face
                        vouch_stats[lo]["hit_target"] += is_tgt
        print(f"  {tag} 完成", flush=True)

    NT = len(targets)
    print(f"\nGT 框 {len(all_gt)}；两模型都没打上的目标框 {NT}")
    import statistics as stt
    z = sum(1 for s in raw_score_of_target if s == 0)
    print(f"目标框上 armW 的原始分：完全无响应 {z} = {z/NT*100:.0f}%；"
          f"有响应的中位 {stt.median([s for s in raw_score_of_target if s>0] or [0]):.3f}")
    for lo in (0.25, 0.5):
        c = sum(1 for s in raw_score_of_target if s >= lo)
        print(f"  ≥{lo}: {c}/{NT} = {c/NT*100:.0f}%")

    print(f"\n{'判据':30}{'候选框':>8}{'是脸':>7}{'精确率':>8}{'捞回目标':>9}{'目标召回':>9}")
    for name, stats in (("A 仅低分带", cand_stats), ("B 低分带+同轨高分背书", vouch_stats)):
        for lo in BANDS:
            s = stats[lo]
            if not s["cand"]:
                continue
            print(f"{name+f'  ≥{lo}':30}{s['cand']:>8}{s['tp']:>7}"
                  f"{s['tp']/s['cand']*100:>7.1f}%{s['hit_target']:>9}"
                  f"{s['hit_target']/NT*100:>8.1f}%")
    json.dump({"n_gt": len(all_gt), "n_target": NT,
               "A": {str(k): v for k, v in cand_stats.items()},
               "B": {str(k): v for k, v in vouch_stats.items()},
               "target_scores": raw_score_of_target},
              open(T / "mine_validate.json", "w"))


if __name__ == "__main__":
    main()

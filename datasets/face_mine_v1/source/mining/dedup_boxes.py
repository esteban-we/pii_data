"""Drop the second box on a face: containment dedup, which IoU-NMS structurally cannot do.

armW emits, on the same face, a tight high-score box and a looser low-score one covering face plus
hair. Containment is 1.00 -- the small box sits entirely inside the big one -- but IoU is only
0.26-0.29, under the 0.4 NMS threshold, so both survive. On the gt_bench right view, where neither
the head-corroboration filter (mining) nor human boxes (left) provide a second line of dedup, that is
13.8% of boxes against 0.5-0.6% everywhere else.

Precedence, so the fix cannot destroy anything that matters:

  a human 已标注 box is never dropped, whatever it overlaps;
  a 候选 box (the mining review target) is never dropped in favour of a mere 参考/已打码;
  otherwise the higher model score wins.

Pure post-process on the shipped file -- no re-decode, and the counts are re-derived afterwards.
"""
from __future__ import annotations

import json
import shutil
import statistics as st
import sys
from collections import Counter
from pathlib import Path

PKG = Path("/data/liangzhenghao/train/handoff_pkg")
SRC = PKG / "annotations.jsonl"
CONTAIN = 0.7
IOU_DUP = 0.3
RANK = {"已标注": 0, "候选": 1, "模型预标": 2, "疑似漏标": 2, "已打码": 3, "参考": 4}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def contain(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return 0.0 if m <= 0 else inter / m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    shutil.copy(SRC, PKG / "annotations.predup.bak")

    dropped = Counter()
    kept_roles = Counter()
    for r in recs:
        # best first: human boxes, then review targets, then by model score
        order = sorted(r["boxes"],
                       key=lambda b: (RANK.get(b["role"], 9), -(b["armw_score"] or 1.0)))
        keep = []
        for b in order:
            dup = False
            for k in keep:
                if k["role"] == "已标注" and b["role"] == "已标注":
                    continue          # two human boxes are the labeller's call, not ours
                if (contain(b["xyxy"], k["xyxy"]) >= CONTAIN
                        or iou(b["xyxy"], k["xyxy"]) >= IOU_DUP):
                    dup = True
                    break
            if dup:
                dropped[(r["dataset"] + ("/" + r["view"] if r["dataset"] == "gt_bench"
                                         else ""), b["role"])] += 1
            else:
                keep.append(b)
        keep.sort(key=lambda b: (not b["need_review"], -(b["armw_score"] or 1.0)))
        r["boxes"] = keep
        r["n_need_review"] = sum(1 for b in keep if b["need_review"])
        for b in keep:
            kept_roles[(r["dataset"], b["role"])] += 1

    with open(SRC, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== 删掉的重复框 ===")
    for k, v in sorted(dropped.items()):
        print(f"  {k[0]:18}{k[1]:10}{v:>7,}")
    print(f"  合计 {sum(dropped.values()):,}")
    print("\n=== 剩余 ===")
    need = Counter()
    for r in recs:
        need[r["dataset"]] += r["n_need_review"]
    for k, v in sorted(kept_roles.items()):
        print(f"  {k[0]:10}{k[1]:10}{v:>9,}")
    print(f"\n待判框 mining {need['mining']:,} + gt_bench {need['gt_bench']:,} = "
          f"{sum(need.values()):,}")
    for ds in ("mining", "gt_bench"):
        per = sorted(len(r["boxes"]) for r in recs if r["dataset"] == ds)
        print(f"  {ds:10} 每帧框数 中位 {st.median(per):.0f}  p90 {per[int(len(per)*0.9)]}"
              f"  最大 {max(per)}  0框帧 {sum(1 for x in per if x == 0)}")
    print("\n去重前已备份 annotations.predup.bak")


if __name__ == "__main__":
    main()

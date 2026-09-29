"""Raise the right-view proposal floor to 0.25, matching the call already made on the left.

The 0.15-0.25 band held 47% of the right view's proposals at 15.1% head corroboration -- the same
class dropped from the left as too dirty to be worth rejecting. Cutting it costs about 3% of
proposal coverage (measured on the left: 0.25 covers 94.5% of known human boxes, 0.15 covers 97.7%)
for 47% of the volume.

Frames that end up with no proposal at all still have to be looked at -- that is what the
scan-the-frame requirement is for -- so their count is reported rather than hidden.
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
FLOOR = 0.25
ROLE = "模型预标"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    shutil.copy(SRC, PKG / "annotations.right015.bak")

    before = after = 0
    empty = {"left": 0, "right": 0}
    for r in recs:
        if r["dataset"] != "gt_bench":
            continue
        keep = []
        for b in r["boxes"]:
            if b["role"] == ROLE:
                before += 1
                if b["armw_score"] < FLOOR:
                    continue
                after += 1
            keep.append(b)
        r["boxes"] = keep
        r["n_need_review"] = sum(1 for b in keep if b["need_review"])
        if not keep:
            empty[r["view"]] += 1
    with open(SRC, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    roles = Counter()
    need = Counter()
    for r in recs:
        need[r["dataset"]] += r["n_need_review"]
        for b in r["boxes"]:
            roles[(r["dataset"], b["role"])] += 1
    gt = [r for r in recs if r["dataset"] == "gt_bench"]
    per = sorted(len(r["boxes"]) for r in gt)

    print(f"{ROLE}  {before:,} -> {after:,}  （砍掉 {before-after:,}）")
    print(f"\n{'子集':10}{'role':10}{'数量':>10}")
    for k, v in sorted(roles.items()):
        print(f"{k[0]:10}{k[1]:10}{v:>10,}")
    print(f"\n待判框  mining {need['mining']:,} + gt_bench {need['gt_bench']:,} = "
          f"{sum(need.values()):,}")
    print(f"gt_bench 每帧框数 中位 {st.median(per):.0f}  p90 {per[int(len(per)*0.9)]}  "
          f"最大 {max(per)}")
    print(f"0 框的帧 {sum(1 for x in per if x == 0)}"
          f"（左目 {empty['left']} / 右目 {empty['right']}）—— 仍需人工看一眼")
    print(f"\n上一版（右目 0.15）已备份 annotations.right015.bak")


if __name__ == "__main__":
    main()

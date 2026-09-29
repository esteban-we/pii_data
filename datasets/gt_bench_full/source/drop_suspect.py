"""Drop 疑似漏标 from gt_bench, and report what the same threshold leaves on the right view.

The left view already had a human pass, so a low-threshold proposal there mostly adds noise on top of
work already done -- that is the call being applied. The right view never had one, so its proposals
are the only scaffold an annotator has, which is why they stay.

But they are the same kind of box: model output at 0.15 where no human has drawn anything. The left
set was judged too dirty to be worth the rejections; the right set is 60% larger and differs only in
having slightly more head-model corroboration (51% vs 37%). So this also measures the right view by
score band and by corroboration, so the same decision can be made there on evidence rather than by
analogy.

Pure filter on the merged annotations.jsonl -- every box already carries its score, so tightening a
threshold needs no re-decode.
"""
from __future__ import annotations

import json
import shutil
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

PKG = Path("/data/liangzhenghao/train/handoff_pkg")
SRC = PKG / "annotations.jsonl"
DROP_ROLE = "疑似漏标"
BANDS = [(0.15, 0.25), (0.25, 0.45), (0.45, 0.75), (0.75, 1.01)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]

    # ---- what the right view looks like, before touching anything -------
    right = [b for r in recs if r["dataset"] == "gt_bench"
             for b in r["boxes"] if b["role"] == "模型预标"]
    print(f"=== 右目「模型预标」构成（{len(right):,} 个）===")
    print(f"{'分数带':12}{'数量':>9}{'占比':>8}{'有头框佐证':>12}{'佐证率':>9}{'长边中位':>10}")
    for lo, hi in BANDS:
        sel = [b for b in right if lo <= b["armw_score"] < hi]
        if not sel:
            continue
        hh = sum(1 for b in sel if b["head_xyxy"])
        print(f"{f'{lo}~{hi}':12}{len(sel):>9,}{len(sel)/len(right)*100:>7.1f}%"
              f"{hh:>12,}{hh/len(sel)*100:>8.1f}%"
              f"{st.median([b['long_side_px'] for b in sel]):>9.0f}px")
    for name, keep in (("若也提到 ≥0.25", lambda b: b["armw_score"] >= 0.25),
                       ("若要求头框佐证", lambda b: b["head_xyxy"] is not None),
                       ("若 ≥0.25 且有头框", lambda b: b["armw_score"] >= 0.25
                        and b["head_xyxy"] is not None)):
        n = sum(1 for b in right if keep(b))
        print(f"  {name:20} 剩 {n:>7,}  （砍掉 {len(right)-n:,} = "
              f"{(len(right)-n)/len(right)*100:.0f}%）")

    # ---- apply the drop -------------------------------------------------
    shutil.copy(SRC, PKG / "annotations.with_suspect.bak")
    st_before = Counter()
    st_after = Counter()
    empty_left = 0
    for r in recs:
        for b in r["boxes"]:
            st_before[(r["dataset"], b["role"])] += 1
        if r["dataset"] == "gt_bench":
            r["boxes"] = [b for b in r["boxes"] if b["role"] != DROP_ROLE]
            r["n_need_review"] = sum(1 for b in r["boxes"] if b["need_review"])
            if r["view"] == "left" and not r["boxes"]:
                empty_left += 1
        for b in r["boxes"]:
            st_after[(r["dataset"], b["role"])] += 1
    with open(SRC, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== 已删除 gt_bench 的「{DROP_ROLE}」===")
    print(f"{'子集':10}{'role':10}{'原':>10}{'现':>10}")
    for k in sorted(set(st_before) | set(st_after)):
        print(f"{k[0]:10}{k[1]:10}{st_before.get(k,0):>10,}{st_after.get(k,0):>10,}")
    gt = [r for r in recs if r["dataset"] == "gt_bench"]
    mn = [r for r in recs if r["dataset"] == "mining"]
    n_gt = sum(r["n_need_review"] for r in gt)
    n_mn = sum(r["n_need_review"] for r in mn)
    per = sorted(len(r["boxes"]) for r in gt)
    print(f"\n图片 {len(recs):,}   待判框 mining {n_mn:,} + gt_bench {n_gt:,} = "
          f"{n_mn + n_gt:,}")
    print(f"gt_bench 每帧框数 中位 {st.median(per):.0f}  p90 {per[int(len(per)*0.9)]}  "
          f"最大 {max(per)}   0 框的帧 {sum(1 for x in per if x == 0)}"
          f"（其中左目 {empty_left}）")
    print(f"\n原始（含疑似漏标）已备份 annotations.with_suspect.bak")


if __name__ == "__main__":
    main()

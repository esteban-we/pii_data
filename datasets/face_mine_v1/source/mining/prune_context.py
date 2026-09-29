"""Thin the context boxes so the box being asked about is still findable.

Shipping every armW detection as context was the right instinct and the wrong threshold. Verifying
the package on its busiest frames showed the failure: a perforated ceiling panel, a black moulding
tray and a supermarket shelf each carried dozens of low-score boxes on texture. Those do not serve
the purpose context boxes exist for -- telling the annotator "we already know about this one" -- and
they bury the green box that actually needs a decision.

Two rules, both conservative in the direction that costs least:

  A context box must clear the same two-model bar the candidates do: armW alone is not enough, the
  head model has to agree. 45% of 参考 boxes have no head support and that is where the texture
  noise lives.

  Then a hard cap per frame, highest score first. A frame needing more than this many context boxes
  is a crowd scene, and the annotator does not need all of them enumerated to understand it.

Dropping a context box on a real face is cheap: the annotator may draw it, and a face neither model
was confident about is exactly what we asked them to add. Keeping a false one is not cheap. 已打码
boxes (armW >= 0.75) are never dropped -- those are real and production blurs them, which is the one
thing the annotator must not second-guess.

Pure post-processing on annotations.jsonl. Nothing is re-decoded, no box moves, and the 候选 set is
untouched -- verified by asserting the need_review total is unchanged.
"""
from __future__ import annotations

import json
import shutil
import statistics as st
import sys
from pathlib import Path

PKG = Path("/data/liangzhenghao/train/handoff_pkg")
SRC = PKG / "annotations.jsonl"
MAX_CONTEXT = 8


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    before_boxes = sum(len(r["boxes"]) for r in recs)
    before_need = sum(r["n_need_review"] for r in recs)
    before_n = [len(r["boxes"]) for r in recs]

    for r in recs:
        keep, ctx = [], []
        for b in r["boxes"]:
            if b["need_review"] or b["role"] == "已打码":
                keep.append(b)
            elif b["head_xyxy"] is not None:
                ctx.append(b)
        ctx.sort(key=lambda b: -b["armw_score"])
        keep.extend(ctx[:MAX_CONTEXT])
        keep.sort(key=lambda b: (not b["need_review"], -b["armw_score"]))
        r["boxes"] = keep

    after_boxes = sum(len(r["boxes"]) for r in recs)
    after_need = sum(r["n_need_review"] for r in recs)
    after_real = sum(1 for r in recs for b in r["boxes"] if b["need_review"])
    assert after_need == before_need == after_real, (
        f"候选集被动到了: {before_need} -> {after_need} / {after_real}")
    after_n = [len(r["boxes"]) for r in recs]

    shutil.copy(SRC, PKG / "annotations_full_context.jsonl")
    with open(SRC, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"框总数 {before_boxes:,} -> {after_boxes:,}  （删掉 {before_boxes-after_boxes:,} 个上下文框）")
    print(f"待判框 {before_need:,} -> {after_need:,}  （必须不变）")
    for nm, n in (("修前", before_n), ("修后", after_n)):
        s = sorted(n)
        print(f"  {nm} 每帧框数 中位 {st.median(n):.0f}  p90 {s[int(len(s)*0.9)]}  "
              f"p99 {s[int(len(s)*0.99)]}  最大 {max(n)}  "
              f">10框的帧 {sum(1 for x in n if x > 10):,}")
    print(f"\n  原始（含全部上下文）已另存 annotations_full_context.jsonl")


if __name__ == "__main__":
    main()

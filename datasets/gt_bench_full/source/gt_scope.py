"""Size the both-views GT re-review set before building it.

Three counts differ and picking the wrong one misstates the ask by a factor of several:

  the frames gt_eval.json actually holds (the recall bench -- hand-labelled, left view only);
  the doll sequences, which are a separate bench measuring OVER-blur and carry no face labels;
  the total frames in those chunks, if someone means "label the whole sequence".

Also reports how many frames the model finds a candidate on in each view, since the right view has
never been labelled at all and that is what a reviewer would be handed there.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

T = Path("/data/liangzhenghao/train")
GTV = Path("/data/liangzhenghao/face_pii/gt_videos")
DOLL = ("BTJPSP", "NQYJPB")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.load(open(T / "gt_eval.json"))
    per = defaultdict(lambda: {"frames": 0, "boxes": 0})
    for e in d:
        k = (e["session"], int(e["chunk"]))
        per[k]["frames"] += 1
        per[k]["boxes"] += len(e["boxes"])

    print("=== gt_eval.json（召回评测台，人工标注，仅左目）===")
    print(f"{'session':26}{'chunk':>7}{'标注帧':>8}{'已标框':>8}")
    tf = tb = 0
    for (s, ck), v in sorted(per.items()):
        tf += v["frames"]
        tb += v["boxes"]
        print(f"{s:26}{ck:>7}{v['frames']:>8}{v['boxes']:>8}")
    print(f"{'合计':26}{len(per):>7}{tf:>8}{tb:>8}")

    print("\n=== gt_videos 目录里的全部序列 ===")
    seqs = sorted(p.name for p in GTV.iterdir() if p.is_dir())
    in_eval = {s for s, _ in per}
    for s in seqs:
        chunks = sorted(c.name for c in (GTV / s).iterdir() if c.is_dir())
        kind = ("玩偶（过打码台，无人脸标注）" if any(x in s for x in DOLL)
                else "召回台" if s in in_eval else "未在 gt_eval 中")
        print(f"  {s:26} {','.join(chunks):14} {kind}")

    print("\n=== 双目复核集的规模 ===")
    print(f"  A 只取已标注帧，左右各一份")
    print(f"      左目 {tf:,} 张（已有人工框 {tb}，本轮复核）")
    print(f"      右目 {tf:,} 张（从未标注，全新）")
    print(f"      合计 {tf*2:,} 张")
    doll_seqs = [s for s in seqs if any(x in s for x in DOLL)]
    print(f"  B 若把 {len(doll_seqs)} 个玩偶序列也纳入，需要另定抽帧口径"
          f"（它们在 gt_eval.json 里没有帧）")


if __name__ == "__main__":
    main()

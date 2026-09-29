"""Fold the GT bench frames into handoff_pkg as one batch, tagged so they can be pulled back out.

Shipping two batches costs the annotation side a second setup, so they go in together. But the two
subsets must never be confused afterwards: the mining frames are training data, and the GT frames are
the evaluation bench -- if bench labels reach the training set, every recall number this project has
produced stops meaning anything. The plan is to extract them after annotation, so the job here is to
make that extraction impossible to get wrong.

Three independent markers, so no single mistake merges them:

  ``dataset`` field on every record -- "mining" or "gt_bench". The primary mechanism.
  a ``gt_`` filename prefix -- survives even if only the image files are kept.
  disjoint ``role`` vocabularies -- mining uses 候选/已打码/参考, GT uses 已标注/疑似漏标/模型预标.

The box schema is harmonised across both so a consumer sees one format: every box carries the union
of keys, with nulls where a field does not apply (GT boxes have no tier, model boxes have no human
provenance). Filename collisions are checked rather than assumed -- the mining pool excluded GT
sessions by construction, but "by construction" is what the check is for.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

T = Path("/data/liangzhenghao/train")
MINE = T / "handoff_pkg"
GT = T / "gt_bench_v2"

BOX_KEYS = ("xyxy", "role", "need_review", "source", "tier", "armw_score",
            "long_side_px", "head_xyxy", "preview_crop")


def norm_box(b, *, default_source):
    return {k: b.get(k, None) if k != "source" else b.get("source", default_source)
            for k in BOX_KEYS}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    mine = [json.loads(l) for l in open(MINE / "annotations.jsonl", encoding="utf-8")
            if l.strip()]
    gt = [json.loads(l) for l in open(GT / "annotations.jsonl", encoding="utf-8")
          if l.strip()]
    print(f"挖掘 {len(mine):,} 行   GT {len(gt):,} 行")

    # collision check before anything is copied
    gt_names = {f"gt_{Path(r['image']).name}" for r in gt}
    mine_names = {Path(r["image"]).name for r in mine}
    clash = gt_names & mine_names
    if clash:
        raise SystemExit(f"文件名冲突 {len(clash)} 个，例如 {sorted(clash)[:3]}")
    print(f"文件名冲突检查: 0（挖掘 {len(mine_names):,} / GT {len(gt_names):,}）")

    for r in mine:
        r["dataset"] = "mining"
        r.setdefault("view", "left")
        r["boxes"] = [norm_box(b, default_source="模型") for b in r["boxes"]]

    img_dst = MINE / "images"
    copied = 0
    for r in gt:
        src = GT / r["image"]
        name = f"gt_{Path(r['image']).name}"
        dst = img_dst / name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1
        r["dataset"] = "gt_bench"
        r["image"] = f"images/{name}"
        r["boxes"] = [norm_box(b, default_source=b.get("source", "模型"))
                      for b in r["boxes"]]
    print(f"GT 图片已并入 {copied:,} 张")

    all_recs = mine + gt
    all_recs.sort(key=lambda r: (r["dataset"], r["image"]))
    with open(MINE / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    st = Counter()
    roles = Counter()
    for r in all_recs:
        st[r["dataset"]] += 1
        st[f"{r['dataset']}_boxes"] += len(r["boxes"])
        st[f"{r['dataset']}_need"] += sum(1 for b in r["boxes"] if b["need_review"])
        for b in r["boxes"]:
            roles[(r["dataset"], b["role"])] += 1
    missing = [r["image"] for r in all_recs if not (MINE / r["image"]).exists()]
    if missing:
        raise SystemExit(f"缺图 {len(missing)}，例如 {missing[:3]}")

    size = sum(f.stat().st_size for f in MINE.rglob("*") if f.is_file())
    manifest = {
        "图片数": len(all_recs),
        "两个子集": {
            "mining（挖掘，可作训练数据）": {
                "图片": st["mining"], "预标框": st["mining_boxes"],
                "待判框": st["mining_need"]},
            "gt_bench（评测台，标注结果不可进训练集）": {
                "图片": st["gt_bench"], "预标框": st["gt_bench_boxes"],
                "待判框": st["gt_bench_need"]},
        },
        "如何分离": {
            "首选": 'annotations.jsonl 每行的 "dataset" 字段',
            "备用": '图片文件名前缀 gt_',
            "再备用": 'role 取值不重叠（挖掘=候选/已打码/参考，GT=已标注/疑似漏标/模型预标）',
        },
        "按 role": {f"{d} / {r}": n for (d, r), n in sorted(roles.items())},
        "分辨率": [2328, 1748],
        "坐标口径": "xyxy 像素坐标，直接对应 images/ 下同名 jpg，无需缩放",
        "包大小GB": round(size / 1e9, 2),
    }
    json.dump(manifest, open(MINE / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n合计 {len(all_recs):,} 张   {size/1e9:.1f} GB")
    print(f"  mining   图 {st['mining']:,}  框 {st['mining_boxes']:,}  "
          f"待判 {st['mining_need']:,}")
    print(f"  gt_bench 图 {st['gt_bench']:,}  框 {st['gt_bench_boxes']:,}  "
          f"待判 {st['gt_bench_need']:,}")
    for (d, r), n in sorted(roles.items()):
        print(f"    {d:9} {r:8} {n:,}")


if __name__ == "__main__":
    main()

"""Consolidate the sweep's 15,852 per-chunk JSONs into one index the annotation side can read.

Per-chunk files are convenient for a resumable run and useless for anyone downstream: nobody wants
to open fifteen thousand files to find out what is here. This writes one CSV (openable in Excel) and
one JSON, each row a crop that exists on disk, carrying everything needed to locate the source frame
in OSS: session, chunk, frame index, scene, tier and the armW score.

Two honest gaps are recorded in the manifest rather than left for someone to trip over:

  No box coordinates. The crops have the boxes drawn on, but the coordinates in the ORIGINAL frame
  were never persisted -- so these are reviewable as images, but cannot be imported as pre-drawn
  boxes without a recovery pass.

  792 of the 63,755 crop records point at a filename that a later crop overwrote. The crop name is
  tier+chunk+frame, and two distinct faces in one frame collide on it. Those rows are marked so the
  count is not quietly wrong.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

T = Path("/data/liangzhenghao/train")
RUN = T / "mine_run"
OUT = T / "handoff"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    seen = Counter()
    rows = []
    per_scene = defaultdict(Counter)
    for p in glob.glob(str(RUN / "shard*/chunks/*.json")):
        r = json.load(open(p))
        if r.get("error"):
            continue
        crops_dir = Path(p).parent.parent / "crops"
        for c in r.get("crops", []):
            seen[c["file"]] += 1
            f = crops_dir / c["file"]
            rows.append({
                "file": c["file"],
                "tier": c["tier"],
                "scene": r["scene"],
                "session": r["sid"],
                "chunk": r["chunk"],
                "frame": c["frame"],
                "armw_score": c["armw"],
                "box_long_px": c["probe_long"],
                "frames_in_this_face": c["n_frames_in_face"],
                "shard_dir": Path(p).parent.parent.name,
                "overwritten": "",
            })
            per_scene[r["scene"]][c["tier"]] += 1
    for row in rows:
        if seen[row["file"]] > 1:
            row["overwritten"] = "同名被覆盖"

    rows.sort(key=lambda r: (r["tier"], r["scene"], r["session"], r["frame"]))
    cols = list(rows[0].keys())
    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(OUT / "candidates.json", "w"), ensure_ascii=False)

    dup = sum(1 for r in rows if r["overwritten"])
    manifest = {
        "生成时间口径": "2026-08-26 夜间粗挖 mine_run",
        "候选记录数": len(rows),
        "磁盘上实际裁图数": len(seen),
        "同名被覆盖的记录数": dup,
        "覆盖范围": {
            "chunk": len(glob.glob(str(RUN / "shard*/chunks/*.json"))),
            "session": len({r["session"] for r in rows}),
            "说明": "每个 session 只取了一个 chunk；每个 chunk 最多渲 12 张，每张脸只出一次",
        },
        "分层定义": {
            "T1": "头检测框 ∧ armW 分数 0.45~0.75，且头框长边 ≤ 3× armW 框长边",
            "T2": "同上，armW 分数 0.25~0.45",
            "共同前提": "该框在 ±150 帧全率续轨模拟后仍未被生产策略覆盖，即今天不会被打码",
        },
        "裁图规格": "260×260，围绕头框放大 2.2 倍裁出；蓝框=头检测，黄框=armW 人脸框",
        "已知限制": [
            "裁图上的框是画上去的，原始帧坐标系里的框坐标没有保存 —— "
            "可以按图人工判断，不能直接作为预标框导入平台（需要一次恢复重跑，约 1 小时）",
            f"{dup} 条记录（占 {dup/max(1,len(rows))*100:.1f}%）的文件名被同 chunk 同帧的另一张脸覆盖",
            "「面孔」是空间+时间邻近合并后的候选簇，是人数上界不是人数",
        ],
        "按场景": {k: dict(v) for k, v in sorted(per_scene.items(),
                                                key=lambda x: -sum(x[1].values()))},
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)

    print(f"候选记录 {len(rows):,}   磁盘裁图 {len(seen):,}   同名覆盖 {dup}")
    print(f"{'场景':12}{'T1':>9}{'T2':>9}")
    for k, v in sorted(per_scene.items(), key=lambda x: -sum(x[1].values())):
        print(f"{k:12}{v['T1']:>9,}{v['T2']:>9,}")
    print(f"\n写出 {OUT}/candidates.csv, candidates.json, manifest.json")


if __name__ == "__main__":
    main()

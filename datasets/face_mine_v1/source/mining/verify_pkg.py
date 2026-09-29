"""Draw the exported xyxy onto the exported image and look at it.

The whole package is worthless if the coordinates do not land on the faces, and no summary statistic
catches that -- a consistent off-by-a-scale-factor produces perfectly plausible numbers. So the boxes
are read back from annotations.jsonl exactly as a consumer would, drawn onto images/<name>.jpg
exactly as a consumer would, and the result is looked at.

Green = the box we are asking about (role 候选). Grey = context boxes shipped so the annotator is not
told that the other faces in the frame are not faces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path("/data/liangzhenghao/train/handoff_pkg")
OUT = PKG / "_verify_final.jpg"
COLS, CELL = 4, 420
ROLE_COLOR = {"候选": (0, 255, 0), "已打码": (160, 160, 160), "参考": (0, 200, 255)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import random
    merged = PKG / "annotations.jsonl"
    src = [merged] if merged.exists() else sorted((PKG / "parts").glob("annotations_*.jsonl"))
    recs = []
    for p in src:
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                recs.append(json.loads(ln))
    print(f"记录 {len(recs):,}")
    if not recs:
        return
    # Busiest frames drawn from a random slice, one per session: busy frames are the ones that
    # actually exercise the 候选/已打码/参考 split, and the one-per-session rule keeps a single
    # talkative session from filling the sheet -- the same mistake an earlier review sheet made.
    random.seed(20260826)
    pool = random.sample(recs, min(4000, len(recs)))
    pool.sort(key=lambda r: -len(r["boxes"]))
    seen, picks = set(), []
    for r in pool:
        if r["session"] in seen:
            continue
        seen.add(r["session"])
        picks.append(r)
        if len(picks) == COLS * 3:
            break

    tiles = []
    for r in picks:
        img = cv2.imread(str(PKG / r["image"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != (r["width"], r["height"]):
            print(f"  ⚠️ {r['image']} 尺寸不符: 文件 {w}x{h} vs 记录 {r['width']}x{r['height']}")
        for b in r["boxes"]:
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            c = ROLE_COLOR.get(b["role"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
            cv2.putText(img, f"{b['tier']} {b['armw_score']:.2f}", (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3, cv2.LINE_AA)
        s = CELL / max(w, h)
        tile = cv2.resize(img, (int(w * s), int(h * s)))
        pad = np.full((CELL, CELL, 3), 26, np.uint8)
        pad[:tile.shape[0], :tile.shape[1]] = tile
        cv2.putText(pad, Path(r["image"]).stem[-22:], (6, CELL - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(pad)

    rows = (len(tiles) + COLS - 1) // COLS
    canvas = np.full((rows * CELL, COLS * CELL, 3), 26, np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // COLS) * CELL:(i // COLS + 1) * CELL,
               (i % COLS) * CELL:(i % COLS + 1) * CELL] = t
    cv2.imwrite(str(OUT), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])

    nb = sum(len(r["boxes"]) for r in recs)
    nt = sum(r["n_need_review"] for r in recs)
    sizes = [(PKG / r["image"]).stat().st_size for r in recs]
    print(f"框 {nb}  待判 {nt}  上下文 {nb-nt}")
    print(f"单图中位 {np.median(sizes)/1e3:.0f} KB  → 62,587 张约 "
          f"{np.median(sizes)*62587/1e9:.0f} GB")
    print(f"验证图 {OUT}")
    print("\n样例记录：")
    print(json.dumps(picks[0], ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()

"""Pull out the overlapping pairs of HUMAN boxes for the original labeller to look at.

The containment dedup deliberately never drops a 已标注 box: two human boxes on the same spot are the
labeller's call, not a script's. But they are also the signature of a slip -- an 11px box sitting
inside a 94px one is almost certainly a stray click, not two faces. Small enough to render every case
rather than sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path("/data/liangzhenghao/train/handoff_pkg")
OUT = Path("/data/liangzhenghao/train/human_overlap.jpg")
CELL = 300
COLS = 5


def contain(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return 0.0 if m <= 0 else inter / m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    recs = [json.loads(l) for l in open(PKG / "annotations.jsonl", encoding="utf-8")
            if '"gt_bench"' in l]
    pairs = []
    for r in recs:
        hb = [b for b in r["boxes"] if b["role"] == "已标注"]
        for i in range(len(hb)):
            for j in range(i + 1, len(hb)):
                c = contain(hb[i]["xyxy"], hb[j]["xyxy"])
                if c >= 0.7:
                    pairs.append((r, hb[i], hb[j], c))
    pairs.sort(key=lambda p: -p[3])
    print(f"人工框互相套的对数: {len(pairs)}")
    for r, a, b, c in pairs:
        print(f"  {r['session'][-6:]} c{r['chunk']:03d} f{r['frame']}  包含 {c:.2f}  "
              f"{a['long_side_px']:.0f}px vs {b['long_side_px']:.0f}px")
    if not pairs:
        return

    tiles = []
    for r, a, b, c in pairs[:COLS * 4]:
        img = cv2.imread(str(PKG / r["image"]))
        big, small = (a, b) if a["long_side_px"] >= b["long_side_px"] else (b, a)
        x = big["xyxy"]
        cx, cy = (x[0] + x[2]) / 2, (x[1] + x[3]) / 2
        half = max(big["long_side_px"] * 1.9 / 2, 60)
        H, W = img.shape[:2]
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        sub = img[y0:y1, x0:x1].copy()
        sx, sy = CELL / max(1, x1 - x0), CELL / max(1, y1 - y0)
        sub = cv2.resize(sub, (CELL, CELL))
        for bx, col, th in ((big["xyxy"], (0, 255, 0), 3),
                            (small["xyxy"], (0, 80, 255), 3)):
            cv2.rectangle(sub, (int((bx[0] - x0) * sx), int((bx[1] - y0) * sy)),
                          (int((bx[2] - x0) * sx), int((bx[3] - y0) * sy)), col, th)
        for cc, t in (((0, 0, 0), 3), ((235, 235, 235), 1)):
            cv2.putText(sub, f"{r['session'][-6:]} f{r['frame']} "
                             f"{big['long_side_px']:.0f}/{small['long_side_px']:.0f}px",
                        (5, CELL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cc, t, cv2.LINE_AA)
        tiles.append(sub)
    rows = (len(tiles) + COLS - 1) // COLS
    canvas = np.full((rows * CELL, COLS * CELL, 3), 26, np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // COLS) * CELL:(i // COLS + 1) * CELL,
               (i % COLS) * CELL:(i % COLS + 1) * CELL] = t
    cv2.imwrite(str(OUT), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(f"\n绿=较大的框  橙=较小的框   写出 {OUT}")


if __name__ == "__main__":
    main()

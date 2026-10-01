#!/usr/bin/env python3
"""Rebuild .knuth/pages/media/face-mine/boxes.json from human labels (WOR-37).

Source of truth: /data/esteban/pii_backup/face-mine_labeled.csv (UTF-8 with BOM;
the PII-1315 store never took this file, so it stays in the legacy tree),
one row per human box; rows with empty box_i are zero-box images. Coordinates
x,y,w,h are normalized top-left+size in the 2328x1748 frame.

Each human box is joined to the previous machine dump (boxes_machine.json,
created here as a backup of the machine boxes.json on first run) by best
IoU >= 0.5 within the same image; matched boxes carry the machine score
(origin "human_kept"), unmatched get score 0.0 (origin "human_new").
Provenance lives in a parallel boxes_meta array; the page tolerates the
extra key.
"""

import csv
import json
import shutil
import sys
import time
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import images as dataset_images, pages_media  # noqa: E402
# Provenance (PII-1682): the pre-PII-1315 tree this script read, now retired. The path below
# records where the input came from; it is not a tree to read today.
LEGACY_ROOT = Path("/data/esteban/pii_backup")

MEDIA = pages_media("face-mine")
CSV_PATH = LEGACY_ROOT / "face-mine_labeled.csv"
IMAGES = dataset_images("face_mine_v1")
COORD_W, COORD_H = 2328, 1748
IOU_THRESHOLD = 0.5


def iou(a, b):
    """IoU of two absolute xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_machine_boxes():
    """Backup the machine boxes.json (once) and return {file: [box, ...]}."""
    backup = MEDIA / "boxes_machine.json"
    current = MEDIA / "boxes.json"
    if not backup.exists():
        shutil.copy2(current, backup)
        print(f"backed up {current} -> {backup}")
    with open(backup, encoding="utf-8") as f:
        machine = json.load(f)
    return {img["file"]: img.get("boxes", []) for img in machine["images"]}


def main():
    machine_by_file = load_machine_boxes()

    # file -> list of (abs_xyxy, meta) in CSV order; zero-box files map to [].
    images = {}
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fname = (
                f"{row['session']}_c{row['chunk']}"
                f"_f{int(row['frame_idx']):06d}.jpg"
            )
            entry = images.setdefault(fname, [])
            if row["box_i"] == "":
                continue  # zero-box image: the empty entry is the record
            x, y = float(row["x"]), float(row["y"])
            w, h = float(row["w"]), float(row["h"])
            box = (x * COORD_W, y * COORD_H, (x + w) * COORD_W, (y + h) * COORD_H)
            meta = {
                "labeled_by": row["labeled_by"],
                "review_round": int(row["review_round"]),
            }
            entry.append((box, meta))

    n_boxes = n_kept = n_new = n_zero = 0
    out_images = []
    for fname in sorted(images):
        boxes_out, meta_out = [], []
        machine_boxes = machine_by_file.get(fname, [])
        for box, meta in images[fname]:
            best_iou, best_score = 0.0, 0.0
            for mb in machine_boxes:
                v = iou(box, mb[:4])
                if v > best_iou:
                    best_iou, best_score = v, mb[4]
            if best_iou >= IOU_THRESHOLD:
                score, origin = best_score, "human_kept"
                n_kept += 1
            else:
                score, origin = 0.0, "human_new"
                n_new += 1
            boxes_out.append([round(c, 1) for c in box] + [round(score, 4)])
            meta_out.append({"origin": origin, **meta})
        n_boxes += len(boxes_out)
        if not boxes_out:
            n_zero += 1
        out_images.append({"file": fname, "boxes": boxes_out, "boxes_meta": meta_out})

    out = {
        "dataset": "face_mine_v1",
        "coord_space": [COORD_W, COORD_H],
        "images": out_images,
    }
    boxes_path = MEDIA / "boxes.json"
    with open(boxes_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    manifest = {
        "dataset": "face_mine_v1",
        "coord_space": [COORD_W, COORD_H],
        "n_images": len(out_images),
        "n_thumbs": 62587,
        "n_boxes": n_boxes,
        "n_images_zero_box": n_zero,
        "n_boxes_human_kept": n_kept,
        "n_boxes_human_new": n_new,
        "images_dir": str(IMAGES),
        "boxes_source": (
            f"{CSV_PATH} (human review), scores "
            "joined by IoU>=0.5 to boxes_machine.json (former machine dump)"
        ),
        "thumb": {"width": 256, "format": "webp", "tool": "ffmpeg scale=256:-1"},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "mining/face_mine_human_boxes.py",
    }
    with open(MEDIA / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(
        f"images={len(out_images)} boxes={n_boxes} "
        f"kept={n_kept} new={n_new} zero_box_images={n_zero}"
    )


if __name__ == "__main__":
    main()

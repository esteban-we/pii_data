"""WOR-17: export face_mine_v1 boxes to boxes.json and build 256px webp thumbnails.

The review page (WOR-18) consumes three artifacts under .knuth/pages/media/face-mine/:
boxes.json (contract: {"dataset","coord_space":[2328,1748],"images":[{"file","boxes":
[[x1,y1,x2,y2,score],...]}]}), thumbs/<stem>.webp (width 256), and manifest.json.

Stages are independent subcommands so the slow one (62,587 ffmpeg invocations) can run in
the background and be re-run after a kill: thumbs skips any output that already exists and
writes via tmp+rename, so a partial file never masquerades as done.

boxes reads the per-image annotation rows (one JSON object per line, the handoff_pkg
annotations.jsonl shape: image/width/height/session/chunk/frame/scene/source/boxes[]) and
emits exactly the page contract. Box-level extras (role, need_review, tier, head box,
preview crop) cannot ride inside the 5-number box arrays without forking the schema, so
they travel in a parallel per-image "boxes_meta" list, index-aligned with "boxes".
Coordinates are xyxy in the raw 2328x1748 frame (torchcodec decode, no resize) per
mining/recover_boxes.py, so no scaling happens here.

The gate is loud, not silent: every annotation row must match an existing jpg by name and
every jpg must have a row; mismatches are listed, never dropped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from pii_root import images, pages_media  # noqa: E402  (needs the sys.path line above)

# PII-1449: the live store, datasets/face_mine_v1/images.
IMAGES = images("face_mine_v1")
OUT = pages_media("face-mine")
THUMBS = OUT / "thumbs"
COORD_SPACE = [2328, 1748]
THUMB_W = 256
EXPECT = 62587


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


# ---------------------------------------------------------------- thumbs

def _one_thumb(jpg: Path) -> str | None:
    """Returns None on success, an error string on failure."""
    dst = THUMBS / (jpg.stem + ".webp")
    if dst.exists():
        return None
    tmp = THUMBS / (jpg.stem + ".webp.tmp.webp")   # ffmpeg needs a real extension
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(jpg), "-vf", f"scale={THUMB_W}:-1", "-frames:v", "1", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            return f"{jpg.name}: ffmpeg rc={r.returncode} {r.stderr.strip()[:200]}"
        os.replace(tmp, dst)
        return None
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return f"{jpg.name}: {type(exc).__name__}: {exc}"


def cmd_thumbs(args) -> int:
    THUMBS.mkdir(parents=True, exist_ok=True)
    jpgs = sorted(IMAGES.glob("*.jpg"))
    have = {p.stem for p in THUMBS.glob("*.webp")}
    todo = [p for p in jpgs if p.stem not in have]
    print(f"{len(jpgs):,} images, {len(have):,} thumbs exist, {len(todo):,} to do", flush=True)
    errs: list[str] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, err in enumerate(ex.map(_one_thumb, todo)):
            if err:
                errs.append(err)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1:,}/{len(todo):,}  {rate:.0f}/s  errs {len(errs)}", flush=True)
    n = sum(1 for _ in THUMBS.glob("*.webp"))
    print(f"done: {n:,} thumbs on disk, {len(errs)} errors", flush=True)
    for e in errs[:50]:
        print("  ERR", e, flush=True)
    return 1 if errs or n != len(jpgs) else 0


# ---------------------------------------------------------------- boxes

def cmd_boxes(args) -> int:
    src = Path(args.src)
    files_on_disk = {p.name for p in IMAGES.glob("*.jpg")}
    rows = []
    with open(src, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("dataset") not in (None, "mining"):   # exclude gt_bench rows if mixed file
                continue
            rows.append(r)

    images = []
    row_files = set()
    dup, bad_dim = [], []
    for r in rows:
        name = r["image"].split("/")[-1]
        if name in row_files:
            dup.append(name)
        row_files.add(name)
        if (r.get("width"), r.get("height")) != (COORD_SPACE[0], COORD_SPACE[1]):
            bad_dim.append(name)
        boxes, meta = [], []
        for b in r.get("boxes", []):
            x1, y1, x2, y2 = b["xyxy"]
            boxes.append([x1, y1, x2, y2, b.get("armw_score", 0.0)])
            meta.append({k: v for k, v in b.items() if k not in ("xyxy", "armw_score")})
        entry = {"file": name, "boxes": boxes, "boxes_meta": meta}
        for k, v in r.items():   # carry every extra row field rather than dropping it
            if k not in ("image", "width", "height", "boxes", "dataset"):
                entry[k] = v
        images.append(entry)

    missing_file = sorted(row_files - files_on_disk)     # row without jpg
    missing_row = sorted(files_on_disk - row_files)      # jpg without row
    print(f"rows {len(rows):,}  files {len(files_on_disk):,}  "
          f"row-without-file {len(missing_file)}  file-without-row {len(missing_row)}  "
          f"dup-rows {len(dup)}  bad-dims {len(bad_dim)}", flush=True)
    for lst, tag in ((missing_file, "ROW-NO-FILE"), (missing_row, "FILE-NO-ROW"),
                     (dup, "DUP"), (bad_dim, "BAD-DIM")):
        for x in lst[:20]:
            print(f"  {tag} {x}", flush=True)

    images.sort(key=lambda e: e["file"])
    out = {"dataset": "face_mine_v1", "coord_space": COORD_SPACE, "images": images}
    _atomic_write_json(OUT / "boxes.json", out)
    print(f"wrote {OUT/'boxes.json'} ({len(images):,} entries)", flush=True)

    gate_ok = (len(rows) == EXPECT and len(files_on_disk) == EXPECT
               and not missing_file and not missing_row and not dup)
    print(f"gate: {'PASS' if gate_ok else 'FAIL'}", flush=True)
    return 0 if gate_ok else 1


# ---------------------------------------------------------------- manifest

def cmd_manifest(args) -> int:
    boxes = json.load(open(OUT / "boxes.json", encoding="utf-8"))
    n_thumbs = sum(1 for _ in THUMBS.glob("*.webp"))
    n_boxes = sum(len(e["boxes"]) for e in boxes["images"])
    n_review = sum(sum(1 for m in e.get("boxes_meta", []) if m.get("need_review"))
                   for e in boxes["images"])
    manifest = {
        "dataset": "face_mine_v1",
        "coord_space": COORD_SPACE,
        "n_images": len(boxes["images"]),
        "n_thumbs": n_thumbs,
        "n_boxes": n_boxes,
        "n_need_review": n_review,
        "images_dir": str(IMAGES),
        "boxes_source": args.src,
        "thumb": {"width": THUMB_W, "format": "webp",
                  "tool": "ffmpeg scale=256:-1"},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "mining/face_mine_dump.py",
    }
    _atomic_write_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thumbs", help="build 256px webp thumbnails (resumable)")
    t.add_argument("--workers", type=int, default=48)
    b = sub.add_parser("boxes", help="export boxes.json from an annotations jsonl")
    b.add_argument("--src", required=True, help="annotations.jsonl (handoff_pkg shape)")
    m = sub.add_parser("manifest", help="write manifest.json from finished outputs")
    m.add_argument("--src", default="", help="recorded boxes source (provenance)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rc = {"thumbs": cmd_thumbs, "boxes": cmd_boxes, "manifest": cmd_manifest}[args.cmd](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

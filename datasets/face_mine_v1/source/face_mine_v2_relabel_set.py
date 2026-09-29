"""WOR-93: face_mine_v2 relabel set, frames where one model has a false positive on the eval slice.

Reads a WOR-64 box dump (.knuth/pages/media/face-mine-eval/<model>.json, contract in
.knuth/docs/face-mine-eval-dump.md: frames[].dets = [x1,y1,x2,y2,score,kind,gt_idx,subtype],
kind 1 TP / 0 FP / -1 ignored; gt and gt_ignored are pixel xyxy in a 2328x1748 frame),
selects every frame with at least one det of kind 0 and score >= THR, copies the JPEG
byte-for-byte from face_mine_v1/images into OUT/images (same basename, no symlink) and writes
one JSONL line per frame:

  {"file","session","width":2328,"height":1748,
   "boxes":[{"x1","y1","x2","y2","source":"gt"} for every GT box (gt + gt_ignored, no size tag),
            {"x1","y1","x2","y2","score","source":"model_fp"} for each FP det with score >= THR]}

No TP boxes, no FN markers. Coordinates as in the dump (1 decimal), scores 4 decimals.

The dump is read-only (a dev server on :47321 streams it); this script never writes to it.

Build:     python mining/face_mine_v2_relabel_set.py --dump DUMP --thr 0.5 --out OUT
Validate:  python mining/face_mine_v2_relabel_set.py --dump DUMP --thr 0.5 --out OUT --validate
  re-derives the selection from the dump and asserts: jsonl lines == images in OUT/images ==
  selected frames; per-line GT count == len(gt)+len(gt_ignored); per-line FP count == dump FP
  count at THR and the sum equals the dump's FP@THR; every file exists in the source images
  dir and its session is in the eval session list; md5 of N sampled copies equals the
  source; renders 3 lines with cv2 to --render-dir for eyeballing. Exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import dataset as pii_dataset, images as pii_images, pages_media  # noqa: E402

DUMP = pages_media("face-mine-eval", "armY.json")
SRC_IMAGES = pii_images("face_mine_v1")
# PII-1449: face_mine_v2 is no longer its own dataset; it is boxes v2 of face_mine_v1.
OUT = pii_dataset("face_mine_v1") / "boxes" / "v2"
EVAL_SESSIONS = STORE / "data" / "splits" / "eval_sessions_v1.txt"
WIDTH, HEIGHT = 2328, 1748
DATE = "2026-09-09"


def thr_tag(thr: float) -> str:
    return f"{thr:g}"


def load_dump(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def select_frames(dump: dict, thr: float) -> list[dict]:
    """Frames with >= 1 det of kind 0 (FP) and score >= thr, in dump order."""
    return [fr for fr in dump["frames"] if any(d[5] == 0 and d[4] >= thr for d in fr["dets"])]


def frame_record(fr: dict, thr: float) -> dict:
    boxes = []
    for x1, y1, x2, y2 in list(fr["gt"]) + list(fr["gt_ignored"]):
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": "gt"})
    for x1, y1, x2, y2, score, kind, _gt_idx, _subtype in fr["dets"]:
        if kind == 0 and score >= thr:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "score": round(score, 4), "source": "model_fp"})
    return {"file": fr["file"], "session": fr["session"],
            "width": WIDTH, "height": HEIGHT, "boxes": boxes}


def in_frame(b: dict) -> bool:
    return b["x1"] >= 0 and b["y1"] >= 0 and b["x2"] <= WIDTH and b["y2"] <= HEIGHT


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_readme(out: Path, dump: dict, thr: float, jsonl_name: str, n_frames: int, n_gt: int,
                 n_gt_eval: int, n_gt_ign: int, n_fp: int, n_sessions: int, n_zero_gt: int,
                 n_outside: int) -> None:
    model = dump["model"]
    text = f"""# face_mine_v2: {model} false-positive frames from the face_mine_v1 eval slice, for relabeling

Written {DATE} by `mining/face_mine_v2_relabel_set.py` (fpv_face_pii repo, WOR-93).

## What this is

Every frame of the frozen face_mine_v1 eval slice (12,754 frames, 1,858 sessions,
`face_mine_v1/splits/eval_sessions_v1.txt`) on which the `{model}` detector at score
threshold {thr_tag(thr)} produces at least one detection that matches no human box
(a false positive under the eval matching). The set exists so relabelers can decide,
box by box, whether each red model box is a face the labelers missed or a genuine
model mistake. Nothing here is a new label: the JSONL carries the current human boxes
plus the model's unmatched proposals, for review.

Images are byte-identical copies of `face_mine_v1/images/<file>` (same basename), not
symlinks. This directory is a relabel package, not a training set; the face_mine_v1
rule that eval sessions are never trained on applies to every frame in it.

## Source dump and its parameters

- Dump: `.knuth/pages/media/face-mine-eval/{model}.json` (WOR-64, 2026-09-08),
  model `{dump.get("path", "")}`.
- det_size {dump["det_size"]}, matching IoU {dump["iou"]}, greedy 1-1 by descending score
  (the matching `evaluation/eval_onnx.py` uses for its precision numbers).
- min_side {dump["min_side"]:g}: human boxes with a side under {dump["min_side"]:g} px are ignore
  regions in the eval (they cannot be TP or FN; dets touching them are neither TP nor FP).
  They ARE included here as GT boxes (see box rule), because relabelers need to see them.
- Selection threshold: {thr_tag(thr)} on the dump's scores (floored to 4 decimals).

## Counts

| quantity | value |
|---|---|
| frames (JSONL lines, images) | {n_frames:,} |
| sessions | {n_sessions:,} |
| GT boxes, total (`source: gt`) | {n_gt:,} |
| of which eval GT (side >= {dump["min_side"]:g} px) | {n_gt_eval:,} |
| of which ignored GT (side < {dump["min_side"]:g} px) | {n_gt_ign:,} |
| model FP boxes at thr {thr_tag(thr)} (`source: model_fp`) | {n_fp:,} |
| frames with no GT at all | {n_zero_gt:,} |

The FP count equals the dump's FP@{thr_tag(thr)} for `{model}` (the precision denominator
minus TPs), so every false positive the eval counted at this threshold is in this set.

## Files

- `images/<file>.jpg`: {n_frames:,} JPEGs, raw fisheye {WIDTH}x{HEIGHT}, `vst_left` only.
- `{jsonl_name}`: one JSON object per line, dump order:

```
{{"file": "<basename>.jpg", "session": "<YYYYMMDD_HHMMSS_XXXXXX>",
 "width": {WIDTH}, "height": {HEIGHT},
 "boxes": [
   {{"x1", "y1", "x2", "y2", "source": "gt"}},                  // one per human box
   {{"x1", "y1", "x2", "y2", "score", "source": "model_fp"}}    // one per FP det >= thr
 ]}}
```

## Box rule

- `gt`: EVERY human box of the frame, i.e. the dump's `gt` and `gt_ignored` lists
  concatenated, all with `source: "gt"` and no size tag. Sub-{dump["min_side"]:g} px boxes are
  therefore present and indistinguishable from the rest by source; recompute the
  size from the coordinates if you need it.
- `model_fp`: each detection of kind 0 (FP) with score >= {thr_tag(thr)}, with its score.
- Not included: TP detections (they overlap a GT box already listed), ignored
  detections (kind -1, overlapping a sub-{dump["min_side"]:g} px GT), dets below the threshold,
  FN markers. A frame is in the set only if it has >= 1 `model_fp` box; it may have
  zero `gt` boxes.

## Coordinate convention

Pixel xyxy (`x1 < x2`, `y1 < y2`) in the {WIDTH}x{HEIGHT} raw frame, origin top-left,
as in the dump: 1 decimal. Scores are the model's, 4 decimals. No normalization;
this differs from `face-mine_labeled.csv` (normalized x,y,w,h).

Human boxes always lie inside the frame. Model boxes are the detector's raw output and
are NOT clipped: {n_outside} of the {n_fp:,} `model_fp` boxes start above or left of the
frame edge (negative `y1` or `x1`, by up to about 115 px; none exceed the right or bottom
edge). They are kept exactly as the dump has them so the set stays reconcilable with the
eval; clip to [0, {WIDTH}] x [0, {HEIGHT}] at render time.

## Reproduce and check

```
python mining/face_mine_v2_relabel_set.py --dump <dump> --thr {thr_tag(thr)} --out {out}
python mining/face_mine_v2_relabel_set.py --dump <dump> --thr {thr_tag(thr)} --out {out} --validate
```

PII: frames show people. Everything stays on this box; never upload the images or
the JSONL to an external service.
"""
    (out / "README.md").write_text(text)


def build(args: argparse.Namespace) -> int:
    dump = load_dump(args.dump)
    frames = select_frames(dump, args.thr)
    out: Path = args.out
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    jsonl_name = f"{dump['model']}_fp_thr{thr_tag(args.thr)}.jsonl"
    n_gt = n_gt_eval = n_gt_ign = n_fp = n_zero = n_outside = 0
    sessions = set()
    tmp = out / (jsonl_name + ".tmp")
    with open(tmp, "w") as fo:
        for i, fr in enumerate(frames):
            src = args.src_images / fr["file"]
            if not src.is_file():
                print(f"FAIL: source image missing: {src}", file=sys.stderr)
                return 1
            dst = images / fr["file"]
            shutil.copyfile(src, dst)  # plain byte copy, follows no symlink semantics of cp -P
            rec = frame_record(fr, args.thr)
            n_gt_eval += len(fr["gt"])
            n_gt_ign += len(fr["gt_ignored"])
            n_gt += len(fr["gt"]) + len(fr["gt_ignored"])
            n_fp += sum(1 for b in rec["boxes"] if b["source"] == "model_fp")
            n_outside += sum(1 for b in rec["boxes"] if b["source"] == "model_fp" and not in_frame(b))
            n_zero += int(len(fr["gt"]) + len(fr["gt_ignored"]) == 0)
            sessions.add(fr["session"])
            fo.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(frames)}", flush=True)
    os.replace(tmp, out / jsonl_name)
    write_readme(out, dump, args.thr, jsonl_name, len(frames), n_gt, n_gt_eval, n_gt_ign, n_fp,
                 len(sessions), n_zero, n_outside)
    print(f"built {out}: {len(frames)} frames, {len(sessions)} sessions, {n_gt} gt "
          f"({n_gt_eval} eval + {n_gt_ign} ignored), {n_fp} model_fp at thr {args.thr}")
    return 0


def render(rec: dict, img_path: Path, out_path: Path) -> None:
    import cv2  # eval_venv

    img = cv2.imread(str(img_path))
    assert img is not None, img_path
    assert img.shape[1] == WIDTH and img.shape[0] == HEIGHT, img.shape
    for b in rec["boxes"]:
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        if b["source"] == "gt":
            cv2.rectangle(img, p1, p2, (0, 200, 0), 3)
        else:
            cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
            cv2.putText(img, f"{b['score']:.2f}", (p1[0], max(p1[1] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def validate(args: argparse.Namespace) -> int:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    dump = load_dump(args.dump)
    frames = select_frames(dump, args.thr)
    by_file = {fr["file"]: fr for fr in frames}
    dump_fp = sum(1 for fr in dump["frames"] for d in fr["dets"] if d[5] == 0 and d[4] >= args.thr)
    out: Path = args.out
    jsonl = out / f"{dump['model']}_fp_thr{thr_tag(args.thr)}.jsonl"
    check(jsonl.is_file(), f"missing {jsonl}")
    check((out / "README.md").is_file(), "missing README.md")
    eval_sessions = set(EVAL_SESSIONS.read_text().split()) if args.eval_sessions is None \
        else set(args.eval_sessions.read_text().split())

    lines = [json.loads(l) for l in open(jsonl)] if jsonl.is_file() else []
    image_files = sorted(p.name for p in (out / "images").iterdir()) if (out / "images").is_dir() else []
    print(f"jsonl lines {len(lines)}  images {len(image_files)}  selected frames {len(frames)}  "
          f"dump FP@{args.thr} {dump_fp}")
    check(len(lines) == len(image_files) == len(frames), "line/image/frame counts differ")
    check(set(image_files) == set(by_file), "image set != selected frame set")
    check(len({l["file"] for l in lines}) == len(lines), "duplicate files in jsonl")
    check(set(l["file"] for l in lines) == set(by_file), "jsonl file set != selected frame set")

    sum_fp = 0
    n_outside = 0
    for l in lines:
        fr = by_file.get(l["file"])
        if fr is None:
            fails.append(f"{l['file']}: not a selected frame")
            continue
        gt = [b for b in l["boxes"] if b["source"] == "gt"]
        fp = [b for b in l["boxes"] if b["source"] == "model_fp"]
        check(len(gt) + len(fp) == len(l["boxes"]), f"{l['file']}: unknown source")
        check(len(gt) == len(fr["gt"]) + len(fr["gt_ignored"]),
              f"{l['file']}: gt count {len(gt)} != {len(fr['gt']) + len(fr['gt_ignored'])}")
        exp_fp = [d for d in fr["dets"] if d[5] == 0 and d[4] >= args.thr]
        check(len(fp) == len(exp_fp), f"{l['file']}: fp count {len(fp)} != {len(exp_fp)}")
        check(len(fp) >= 1, f"{l['file']}: no model_fp box")
        exp_gt_coords = [tuple(b) for b in list(fr["gt"]) + list(fr["gt_ignored"])]
        check([(b["x1"], b["y1"], b["x2"], b["y2"]) for b in gt] == exp_gt_coords,
              f"{l['file']}: gt coords differ from dump")
        check([(b["x1"], b["y1"], b["x2"], b["y2"], b["score"]) for b in fp]
              == [tuple(d[:5]) for d in exp_fp], f"{l['file']}: fp coords/scores differ from dump")
        check(all("score" not in b for b in gt), f"{l['file']}: gt box carries a score")
        check(all(b["x1"] < b["x2"] and b["y1"] < b["y2"] for b in l["boxes"]),
              f"{l['file']}: degenerate box")
        # GT is always inside the frame; model boxes are the detector's raw output and may
        # start above/left of the frame edge (the dump does not clip). Counted, not failed.
        check(all(in_frame(b) for b in gt), f"{l['file']}: gt box outside frame")
        n_outside += sum(not in_frame(b) for b in fp)
        check(l["width"] == WIDTH and l["height"] == HEIGHT, f"{l['file']}: bad width/height")
        check(l["session"] == fr["session"] and l["file"].startswith(l["session"] + "_"),
              f"{l['file']}: session mismatch")
        check(l["session"] in eval_sessions, f"{l['file']}: session not in eval slice")
        check((args.src_images / l["file"]).is_file(), f"{l['file']}: not in {args.src_images}")
        check((out / "images" / l["file"]).is_file(), f"{l['file']}: copy missing")
        sum_fp += len(fp)
    print(f"sum model_fp {sum_fp}  (dump FP@{args.thr} {dump_fp});  model_fp boxes extending past a "
          f"frame edge (kept as in the dump): {n_outside}")
    check(sum_fp == dump_fp, f"sum model_fp {sum_fp} != dump FP {dump_fp}")

    rng = random.Random(args.seed)
    sample = rng.sample(lines, min(args.n_md5, len(lines))) if lines else []
    n_md5_ok = 0
    for l in sample:
        a, b = md5(args.src_images / l["file"]), md5(out / "images" / l["file"])
        if a == b:
            n_md5_ok += 1
        else:
            fails.append(f"{l['file']}: md5 differs ({a} vs {b})")
    print(f"md5 identical on {n_md5_ok}/{len(sample)} sampled copies (seed {args.seed})")

    args.render_dir.mkdir(parents=True, exist_ok=True)
    for l in rng.sample(lines, min(args.n_render, len(lines))) if lines else []:
        dst = args.render_dir / f"wor93_{Path(l['file']).stem}.jpg"
        render(l, out / "images" / l["file"], dst)
        n_gt = sum(b["source"] == "gt" for b in l["boxes"])
        fps = [b["score"] for b in l["boxes"] if b["source"] == "model_fp"]
        print(f"rendered {dst}  gt {n_gt}  model_fp {len(fps)} scores {fps}")

    if fails:
        print(f"VALIDATE FAILED: {len(fails)} problems", file=sys.stderr)
        for f in fails[:50]:
            print("  " + f, file=sys.stderr)
        return 1
    print(f"VALIDATE OK: {len(lines)} frames, {sum_fp} model_fp boxes, "
          f"{sum(sum(b['source'] == 'gt' for b in l['boxes']) for l in lines)} gt boxes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DUMP, help="WOR-64 dump json (read-only)")
    ap.add_argument("--thr", type=float, default=0.5, help="score threshold on FP dets")
    ap.add_argument("--out", type=Path, default=OUT, help="output dataset dir")
    ap.add_argument("--src-images", type=Path, default=SRC_IMAGES)
    ap.add_argument("--eval-sessions", type=Path, default=None, help=f"default {EVAL_SESSIONS}")
    ap.add_argument("--validate", action="store_true", help="check an existing build instead of building")
    ap.add_argument("--n-md5", type=int, default=20)
    ap.add_argument("--n-render", type=int, default=3)
    ap.add_argument("--render-dir", type=Path, default=Path("/tmp"))
    ap.add_argument("--seed", type=int, default=93)
    args = ap.parse_args()
    return validate(args) if args.validate else build(args)


if __name__ == "__main__":
    sys.exit(main())

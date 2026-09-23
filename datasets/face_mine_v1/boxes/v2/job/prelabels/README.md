# face_mine_v2: armY false-positive frames from the face_mine_v1 eval slice, for relabeling

Written 2026-09-09 by `mining/face_mine_v2_relabel_set.py` (fpv_face_pii repo, WOR-93).

## What this is

Every frame of the frozen face_mine_v1 eval slice (12,754 frames, 1,858 sessions,
`face_mine_v1/splits/eval_sessions_v1.txt`) on which the `armY` detector at score
threshold 0.5 produces at least one detection that matches no human box
(a false positive under the eval matching). The set exists so relabelers can decide,
box by box, whether each red model box is a face the labelers missed or a genuine
model mistake. Nothing here is a new label: the JSONL carries the current human boxes
plus the model's unmatched proposals, for review.

Images are byte-identical copies of `face_mine_v1/images/<file>` (same basename), not
symlinks. This directory is a relabel package, not a training set; the face_mine_v1
rule that eval sessions are never trained on applies to every frame in it.

## Source dump and its parameters

- Dump: `.knuth/pages/media/face-mine-eval/armY.json` (WOR-64, 2026-09-08),
  model `/data/esteban/pii/runs/train/armY_det34g.onnx`.
- det_size 1024, matching IoU 0.4, greedy 1-1 by descending score
  (the matching `evaluation/eval_onnx.py` uses for its precision numbers).
- min_side 40: human boxes with a side under 40 px are ignore
  regions in the eval (they cannot be TP or FN; dets touching them are neither TP nor FP).
  They ARE included here as GT boxes (see box rule), because relabelers need to see them.
- Selection threshold: 0.5 on the dump's scores (floored to 4 decimals).

## Counts

| quantity | value |
|---|---|
| frames (JSONL lines, images) | 3,600 |
| sessions | 935 |
| GT boxes, total (`source: gt`) | 5,384 |
| of which eval GT (side >= 40 px) | 4,827 |
| of which ignored GT (side < 40 px) | 557 |
| model FP boxes at thr 0.5 (`source: model_fp`) | 4,768 |
| frames with no GT at all | 1,144 |

The FP count equals the dump's FP@0.5 for `armY` (the precision denominator
minus TPs), so every false positive the eval counted at this threshold is in this set.

## Files

- `images/<file>.jpg`: 3,600 JPEGs, raw fisheye 2328x1748, `vst_left` only.
- `armY_fp_thr0.5.jsonl`: one JSON object per line, dump order:

```
{"file": "<basename>.jpg", "session": "<YYYYMMDD_HHMMSS_XXXXXX>",
 "width": 2328, "height": 1748,
 "boxes": [
   {"x1", "y1", "x2", "y2", "source": "gt"},                  // one per human box
   {"x1", "y1", "x2", "y2", "score", "source": "model_fp"}    // one per FP det >= thr
 ]}
```

## Box rule

- `gt`: EVERY human box of the frame, i.e. the dump's `gt` and `gt_ignored` lists
  concatenated, all with `source: "gt"` and no size tag. Sub-40 px boxes are
  therefore present and indistinguishable from the rest by source; recompute the
  size from the coordinates if you need it.
- `model_fp`: each detection of kind 0 (FP) with score >= 0.5, with its score.
- Not included: TP detections (they overlap a GT box already listed), ignored
  detections (kind -1, overlapping a sub-40 px GT), dets below the threshold,
  FN markers. A frame is in the set only if it has >= 1 `model_fp` box; it may have
  zero `gt` boxes.

## Coordinate convention

Pixel xyxy (`x1 < x2`, `y1 < y2`) in the 2328x1748 raw frame, origin top-left,
as in the dump: 1 decimal. Scores are the model's, 4 decimals. No normalization;
this differs from `face-mine_labeled.csv` (normalized x,y,w,h).

Human boxes always lie inside the frame. Model boxes are the detector's raw output and
are NOT clipped: 120 of the 4,768 `model_fp` boxes start above or left of the
frame edge (negative `y1` or `x1`, by up to about 115 px; none exceed the right or bottom
edge). They are kept exactly as the dump has them so the set stays reconcilable with the
eval; clip to [0, 2328] x [0, 1748] at render time.

## Reproduce and check

```
python mining/face_mine_v2_relabel_set.py --dump <dump> --thr 0.5 --out /data/esteban/pii/datasets/face_mine_v2
python mining/face_mine_v2_relabel_set.py --dump <dump> --thr 0.5 --out /data/esteban/pii/datasets/face_mine_v2 --validate
```

PII: frames show people. Everything stays on this box; never upload the images or
the JSONL to an external service.

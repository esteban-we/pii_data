# gt_bench_sparse boxes v1: the human labels as delivered

## Why

The only label pass this set has: the face boxes of the ground-truth bench as delivered
by the previous owner (2026-08-27), restricted to the 889 frames of the sparse slice.
These boxes are the reference every recall number of the project was measured against
until the full set was staged.

## Rules the labels follow

- Source: `job/output/gt_eval_sparse.json`, a list of `{path, boxes, session, chunk}`
  entries produced by `evaluation/gt_extract.py` from `gt_bundle.json` (the full human
  labels); `boxes` are absolute pixel corners `[x1, y1, x2, y2]` in the raw 2328x1748
  `vst_left` fisheye frame, not `x, y, w, h`.
- Boxes with long side under 40 px were dropped by the extraction; 3 of the 889 frames
  have no box for that reason and are listed in `frames.csv` with no `boxes.csv` row.
- The labelers' written instructions, the labeling tool and the labelers are not on file
  here; the delivered README says only that chunks were labelled every 5th frame across
  the whole chunk and that a frame appears in the bundle only if it holds at least one face.
- Coordinates are written at one decimal (`%.1f` of the delivered floats). `ignore` is
  0 on every row.
- `frames.csv` has `reviewed=1` on all 889 frames: every frame carries a human answer,
  including the 3 box-free ones.

## Counts

| | count |
|---|---:|
| frames | 889 |
| boxes | 1,541 |
| frames with boxes | 886 |
| frames without boxes | 3 |

`boxes.csv` is byte for byte the pii-data file `datasets/gt_bench_sparse/boxes/v1.csv`
(md5 de1d589fe2e0c81f1181a74ef69ed2d0). On 5 frames it carries fewer boxes than
`gt_bench_full/boxes/v1` (PII-130, unresolved; see the dataset README).

## Files

- `job/output/gt_eval_sparse.json`: byte copy of
  `/data/esteban/pii/datasets/gt_bench_v1/labels/gt_eval_sparse.json`
  (md5 cfd17fff00f32a86e70285469c5f31dc, equal to the OSS ETag per PII-516).
- `job/prelabels/`: absent; no prelabel file exists for this set.
- `boxes.csv` is re-derived from the `job/output` copy by `build_gt_bench_pii2.py --set sparse verify`.

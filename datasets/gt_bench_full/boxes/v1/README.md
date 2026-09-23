# gt_bench_full boxes v1: the human labels as delivered

## Why

The only label pass this set has: the full human face labels of the ground-truth bench
as delivered by the previous owner (2026-08-27), on every labelled frame of the 9 chunks.

## Rules the labels follow

- Source: `job/output/gt_bundle.json`, `{<chunk_uuid>: {"image_size": [1748, 2328],
  "frames": {<frame_idx>: [[x1, y1, x2, y2], ...]}}}`; boxes are absolute pixel corners in
  the raw 2328x1748 `vst_left` fisheye frame, not `x, y, w, h`. `job/output/uuid_map.json`
  (`{uuid: [session, chunk, n_matching_sparse_frames]}`, written by
  `evaluation/stage_full_bench.py` on 2026-08-29) turns a uuid and frame index into the
  image name.
- Chunks were labelled every 5th frame across the whole chunk; a frame is present only
  if it holds at least one face (delivered README; both asserted or measured, see the
  dataset README). No size floor: the eval applies its own 40 px long-side floor.
- The labelers' written instructions, the labeling tool and the labelers are not on file
  here.
- Coordinates are written at one decimal (`%.1f` of the delivered floats). `ignore` is
  0 on every row.
- `frames.csv` has `reviewed=1` on all 9,636 frames.

## Counts

| | count |
|---|---:|
| frames | 9,636 |
| boxes | 20,054 |
| frames with boxes | 9,636 |
| boxes with long side under 40 px | 54 (PII-516; not stored separately) |

`boxes.csv` is byte for byte the pii-data file `datasets/gt_bench_full/boxes/v1.csv`
(md5 1a479d33eeb9488e790b2346ed2f254f). On 5 of the 889 frames shared with
`gt_bench_sparse` it carries more boxes than that set (PII-130, unresolved).

## Files

- `job/output/gt_bundle.json`: byte copy of
  `/data/esteban/pii/datasets/gt_bench_v1/labels/gt_bundle.json`
  (md5 47b67814ee9aee4480f0784fef170bcf, equal to the OSS ETag per PII-516).
- `job/output/uuid_map.json`: byte copy of `gt_bench_v1/labels/uuid_map.json`
  (md5 62f45bbff51d91095eeadf008c12ad91); the 10th bundle uuid, with 0 frames, is absent from it.
- `job/prelabels/`: absent; no prelabel file exists for this set.
- `boxes.csv` is re-derived from the `job/output` copies by `build_gt_bench_pii2.py --set full verify`.

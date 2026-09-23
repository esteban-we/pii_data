# face_mine_right boxes v2: armAG34-over-human relabel of every frame (pending)

Status 2026-09-21: at the vendor. This directory holds the prelabels only; `frames.csv`
and `boxes.csv` are written when the output lands. Until then v1 is the current
version of this dataset.

## Why

A second full pass over both eyes of face_mine (PII-1273; the left eye is
`face_mine_v1` v3). Every frame carries the armAG34 detections at score >= 0.3
plus the human boxes no detection overlaps, so the labelers see the model's view
of every frame and confirm, move or delete each box. Where a detection overlaps a
human box the model box replaced the human box in the prelabel (the user's call,
PII-1270).

## Rules

- Prelabels (`job/prelabels/import_right.jsonl`, one row per frame, all 62,423
  frames, normalised xywh at 4 decimals, each box with `source` `human` or `armAG34`
  and `score`): model `armAG34_det34g.onnx` (md5 fe1721d8cc93b17dd82e6aabb33a5266),
  det_size 1024, padded canvas, NMS 0.4, score cut 0.3, IoU gate 0.1, prefer model,
  dedup so that each human box is replaced by at most one model box. No size floor,
  no clipping to the frame.
- Human box source: the v1 human labels (`../v1/boxes.csv` in its original form,
  `/data/esteban/pii/datasets/face_mine_right/boxes.jsonl`).
- `changed_right.txt`: the 42,649 frames with at least one replaced or added box
  (the upload set). The verdict upload of 2026-09-18
  (`/data/esteban/tmp/pii1273/face-mine-v2-right/manifest.json`, not copied here)
  staged exactly these 42,649 frames with 139,971 boxes, 3,319 of them clipped to
  the frame at upload; the vendor did not receive the other 19,774 frames.
  `reviewed` will follow the output.
- Labeling rules for the vendor and the output format are those of v1 (full box
  set per frame, empty list = real negative, normalised xywh); the vendor's written
  instructions are not on file here.

## Counts (prelabels, right eye)

| | count |
|---|---:|
| frames | 62,423 |
| human boxes in | 90,902 |
| detections at >= 0.3 | 143,672 |
| human boxes replaced by a model box | 86,243 |
| additions (model, no human overlap) | 49,797 |
| boxes out | 140,599 |
| frames changed | 42,649 |
| frames with no box (before / after) | 26,663 / 19,238 |

Full tables (score and size bands, left eye) in `job/prelabels/README.md` and
`stats.json`.

## Files

- `job/prelabels/`: byte copies from `/data/esteban/pii/datasets/face_mine_relabel_v2/`
  (`import_right.jsonl`, `changed_right.txt`, `README.md`, `stats.json`). Not
  carried: the detection dumps `dets_*.jsonl`, the `render/` samples and the
  `*.predup` build before dedup.
- `job/output/`: none yet.

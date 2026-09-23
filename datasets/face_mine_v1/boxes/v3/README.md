# face_mine_v1 boxes v3: armAG34-over-human relabel of every frame (pending)

Status 2026-09-21: at the vendor. This directory holds the prelabels only; `frames.csv`
and `boxes.csv` are written when the output lands. Until then v2 is the current
version of this dataset.

## Why

A second full pass over both eyes of face_mine (PII-1273; the right eye is
`face_mine_right` v2). Every frame is sent with the armAG34 detections at score
>= 0.3 plus the human boxes no detection overlaps, so the labelers see the model's
view of every frame and confirm, move or delete each box. Where a detection
overlaps a human box the model box replaced the human box in the prelabel (the
user's call, PII-1270).

## Rules

- Prelabels (`job/prelabels/import_left.jsonl`, one row per frame, all 62,587
  frames, normalised xywh at 4 decimals, each box with `source` `human` or `armAG34`
  and `score`): model `armAG34_det34g.onnx` (md5 fe1721d8cc93b17dd82e6aabb33a5266),
  det_size 1024, padded canvas, NMS 0.4, score cut 0.3, IoU gate 0.1, prefer model,
  dedup so that each human box is replaced by at most one model box. No size floor,
  no clipping to the frame.
- Human box source: the v1 human labels (`../v1/boxes.csv` in its original form,
  `/data/esteban/pii/face-mine_labeled.csv`), not the v2 relabel (PII-142 has not
  promoted it).
- `changed_left.txt`: the 48,328 frames with at least one replaced or added box
  (the upload set). Whether the vendor received all 62,587 frames or only these is
  settled when the output lands; `reviewed` will follow the output.
- Labeling rules for the vendor and the output format are those of the earlier
  passes (full box set per frame, empty list = real negative, normalised xywh);
  the vendor's written instructions are not on file here.

## Counts (prelabels, left eye)

| | count |
|---|---:|
| frames | 62,587 |
| human boxes in | 76,859 |
| detections at >= 0.3 | 153,588 |
| human boxes replaced by a model box | 74,983 |
| additions (model, no human overlap) | 72,980 |
| boxes out | 149,807 |
| frames changed | 48,328 |
| frames with no box (before / after) | 21,508 / 13,509 |

Full tables (score and size bands, right eye) in `job/prelabels/README.md` and
`stats.json`.

## Files

- `job/prelabels/`: byte copies from `/data/esteban/pii/datasets/face_mine_relabel_v2/`
  (`import_left.jsonl`, `changed_left.txt`, `README.md`, `stats.json`). Not
  carried: the detection dumps `dets_*.jsonl`, the `render/` samples and the
  `*.predup` build before dedup.
- `job/output/`: none yet.

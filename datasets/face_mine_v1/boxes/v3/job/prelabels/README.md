# face_mine re-label prelabels, armAG34 over human (PII-1273)

Prelabels for a vendor pass over both eyes of face_mine. Every frame carries
the armAG34 detections at score >= 0.3 plus the human boxes no
detection overlaps. Where a detection overlaps a human box the MODEL box is kept
and the human box dropped (the user's call, PII-1270; PII-940 kept the human box).

## Model and rule

- Model: `/data/esteban/pii/runs/train/armAG34_det34g.onnx` (md5 fe1721d8cc93b17dd82e6aabb33a5266),
  det_size 1024, padded canvas, decode `mining/rview_detect.py::SCRFD`
  (box-for-box parity with `evaluation/eval_onnx.py::SCRFD`, PII-942, not re-run here).
- Dump: every box at score >= 0.3 in original pixels
  (`dets_<view>.jsonl`, one row per frame, boxes `[x1, y1, x2, y2, score]` sorted by
  score descending after NMS at 0.4).
- Merge, per frame, `--prefer model --score-cut 0.3 --iou-gate 0.1`:
  a model box at score >= 0.3 whose IoU (pixels, plain
  intersection over union) against any human box of the frame is >= 0.1
  REPLACES every human box it overlaps: those human boxes are dropped and the model
  box is emitted once. A model box overlapping no human box is an ADDITION. A human
  box no model box overlaps stays verbatim. No size floor, no clipping to the frame.
- Dedup (PII-1273): model boxes are taken in score order; one that overlaps a human
  box already claimed by a higher-scoring emitted model box is DROPPED (not emitted,
  not an addition), so each human box is replaced by at most one model box. A
  dropped box claims nothing: a human box only it overlapped stays verbatim.
- Every box carries `source` (`human` or `armAG34`) and `score` (null for human).
  Box order: surviving human boxes in source order, then model boxes by score.
  A frame with no box keeps `"boxes": []` (faceback_45 rule).

## Human box sources

- left: `/data/esteban/pii/face-mine_labeled.csv`, the HUMAN labels of face_mine_v1 (the only full human
  pass over the view). NOT `datasets/face_mine_v1/boxes.jsonl`, which is the miner's
  machine output (PII-132, PII-335). NOT pii-data `boxes/v2.csv`: that relabel covers
  3,600 eval frames only, is uncommitted in pii-data and PII-142 has not promoted it.
- right: `/data/esteban/pii/datasets/face_mine_right/boxes.jsonl`, human (box_src `human`, PII-194).
- Images: `/data/esteban/pii/datasets/face_mine_v1/images` and `/data/esteban/pii/datasets/face_mine_right/images`; the frame set is the image dir
  listing, and the dump, the human file and the prelabels all hold it exactly.

## Files

```
dets_left.jsonl / dets_right.jsonl      full detection dump, one row per frame
import_left.jsonl / import_right.jsonl  the prelabels, faceback_45 import shape
stats.json                              every count below, machine readable
render/                                 sampled frames: human green, model red,
                                        replaced human dashed; render/index.txt
changed_left.txt / changed_right.txt    frames with >= 1 replaced or added box
                                        (the upload set), one image name per line
*.predup                                the same build before dedup (PII-1272)
```

`import_<view>.jsonl` rows are `{session, chunk, view, frame_idx, width, height,
boxes, image}`; coordinates are normalised 0..1 top-left plus size at 4 decimals.

## Counts

| | left | right |
|---|---:|---:|
| Frames | 62,587 | 62,423 |
| Human boxes in | 76,859 | 90,902 |
| Frames with a human box | 41,079 | 35,760 |
| Detections at >= 0.3 | 153,588 | 143,672 |
| Human boxes replaced (dropped) | 74,983 | 86,243 |
| Model boxes replacing a human box | 74,951 | 86,143 |
| Model boxes matched to more than one human box | 32 | 99 |
| Human boxes hit by more than one candidate model box (before dedup) | 5,608 | 7,727 |
| Model boxes dropped by dedup | 5,657 | 7,732 |
| Additions (model, no human overlap) | 72,980 | 49,797 |
| Boxes out | 149,807 | 140,599 |
| Frames changed | 48,328 | 42,649 |
| Frames with an addition | 29,172 | 24,552 |
| Frames with a replacement | 40,118 | 35,007 |
| Frames with no box (before) | 21,508 | 26,663 |
| Frames with no box (after) | 13,509 | 19,238 |
| Model boxes reaching outside the frame | 3,740 | 3,319 |

### Model boxes by score

| band | additions L | additions R | replacing L | replacing R |
|---|---:|---:|---:|---:|
| 0.3-0.4 | 32,676 | 26,965 | 1,879 | 5,055 |
| 0.4-0.5 | 19,007 | 14,122 | 4,278 | 8,447 |
| 0.5-0.6 | 10,911 | 6,272 | 8,704 | 12,355 |
| 0.6-0.8 | 10,032 | 2,387 | 44,905 | 44,777 |
| 0.8+ | 354 | 51 | 15,185 | 15,509 |

### Additions by long side (px)

| band | left | right |
|---|---:|---:|
| < 20 | 22 | 13 |
| 20-40 | 29,317 | 18,222 |
| 40-60 | 22,114 | 15,109 |
| 60-100 | 13,890 | 10,537 |
| 100+ | 7,637 | 5,916 |

## Re-cutting

A different score cut or IoU gate is one `build --prefer model` call over the same
dumps (no inference); `build` without `--prefer` gives the PII-940 shape (human
boxes verbatim plus unmatched additions). `check --prefer model` re-verifies the
output against the sources; `render --prefer model` draws sampled frames.

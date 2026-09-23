# faceight_c re-label prelabels v1 (PII-980 / PII-990)

Prelabels for a second vendor pass over the faceight_c round-3 return. Every frame
carries the human boxes of that return verbatim plus the armAE34 detections that do
not match any of them, so the vendor only has to judge the additions.

## Model and cut

- Model: `/data/esteban/pii/runs/train/armAE34_det34g.onnx` (md5 3fa016c5dc7241e448ec101f4234e738).
- Inference: det_size 1024, padded canvas (the historical mode every
  eval in this project has used); decode is `mining/rview_detect.py::SCRFD`, verified
  box for box against `evaluation/eval_onnx.py::SCRFD` on 20 frames (220 box pairs,
  worst |dbox| 0.000000 px, worst |dscore| 0.0; that parity run is PII-942, on this
  model, and was not repeated here).
- A detection is an ADDITION when score >= 0.3 AND its max IoU
  against every human box of that frame is < 0.1. There is no size
  floor: tiny boxes are kept.
- Human boxes are copied verbatim: never modified, never dropped, even when a
  detection overlaps them. A frame with no box at all keeps `"boxes": []` (the
  faceback_45 rule: an empty list is "the machine looked and found nothing", which
  the portal shows differently from "no prelabel at all", and those frames are
  exactly where a genuine miss hides).
- Boxes are not clipped to the frame, following the faceback_45 precedent
  (additions reaching outside the frame: left 412, right 375).

## Human box sources

- Both views: `/data/esteban/pii/datasets/faceight_c/boxes.jsonl` (PII-981), the round-3 vendor return
  converted to the faceight_a box shape (box_src `human`). One file holds both
  views; rows are taken by their `view` field, and the frame index comes from the
  record's `frame` field.
- Images: `/data/esteban/pii/datasets/faceight_c/images_left` and `/data/esteban/pii/datasets/faceight_c/images_right` (symlinks into the faceight frame
  trees, which hold far more than these 40,000 frames; the frame set comes from
  the human file, not from a directory listing).

## Frame names

The faceight frame file is named by timestamp, `<session>_c<chunk>_<view>_t<ms>.jpg`,
but the verdict package the portal already ingested names frames
`<session>_c<chunk>_f<frame_idx:06d>.jpg` (chunk `%03d`). `image` in the rows below
is that verdict name, with `frame_idx` taken from the human record, so the two
packages line up; `dets_<view>.jsonl` keeps the timestamp file name instead.

## Files

```
dets_left.jsonl / dets_right.jsonl    full detection dump, one row per frame
import_left.jsonl / import_right.jsonl  the prelabels, faceback_45 import shape
stats.json                            every count below, machine readable
```

`import_<view>.jsonl` rows are `{session, chunk, view, frame_idx, width, height,
boxes, image}`; coordinates are normalised 0..1 top-left plus size at 4 decimals,
the same convention as the faceback_45 import files. Each box carries
`source`: `human` or `armAE34`; machine boxes also carry `score`.

## Counts

| | left | right |
|---|---:|---:|
| Frames | 18,994 | 21,006 |
| Human boxes | 12,712 | 15,330 |
| Frames with a human box | 7,356 | 8,540 |
| Additions | 6,688 | 7,664 |
| Frames with an addition | 4,661 | 5,264 |
| Frames with an addition and no human box | 2,379 | 2,596 |
| Frames with no box at all | 9,259 | 9,870 |
| Dump boxes (>= 0.1) | 101,809 | 116,732 |
| Detections at >= 0.3 | 18,208 | 21,525 |

### Additions by score

| band | left | right |
|---|---:|---:|
| 0.3-0.4 | 4,055 | 4,526 |
| 0.4-0.5 | 1,724 | 2,009 |
| 0.5-0.6 | 664 | 798 |
| 0.6-0.8 | 244 | 326 |
| 0.8+ | 1 | 5 |

### Additions by long side (px)

| band | left | right |
|---|---:|---:|
| < 20 | 24 | 28 |
| 20-40 | 2,696 | 3,161 |
| 40-60 | 1,184 | 1,365 |
| 60-100 | 1,649 | 1,968 |
| 100+ | 1,135 | 1,142 |

## Re-cutting without another inference pass

`dets_<view>.jsonl` holds EVERY box down to score 0.1 with its
score, in original 2328x1748 pixels, one row per frame:

```json
{"file": "<session>_c<chunk>_<view>_t<t_ms>.jpg", "session": "...",
 "chunk": "000", "frame_idx": null, "view": "lview", "t_ms": 22233,
 "w": 2328, "h": 1748, "boxes": [[x1, y1, x2, y2, score], ...]}
```

Boxes are sorted by score descending, after NMS at 0.4. A different score cut, IoU
gate or size floor is a filter over that file, not another inference pass:

```sh
mining/face_mine_relabel_v3.py build --out-dir . --ds faceight_c \
  --dump-left dets_left.jsonl --dump-right dets_right.jsonl \
  --score-cut 0.3 --iou-gate 0.1
```

`build` is deterministic: the same dump and the same human files give byte-identical
outputs. `mining/face_mine_relabel_v3.py check` re-asserts, over the whole output,
that every frame appears once, that the frame set equals the source frame set, that
the human box count matches the source file, and that no addition reaches the IoU
gate against a human box of its frame.

## PII

The frames are unblurred faces of real people. Everything here stays on this box.

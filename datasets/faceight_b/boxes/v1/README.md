# faceight_b boxes v1: the vendor human pass over every frame

## Why

First human labeling of round 2. All 60,000 frames went to the vendor's annotation portal
as the Verdict dataset `faceight_b` (both eyes in one dataset, the eye in the `view`
field), seeded with the WOR-195 cell prelabels, and came back 2026-09-16 as
`face_boxes_faceight_b.jsonl` (md5 2c1726c751a1499a476222bb60efdbfe). Validated and
converted by `training/ann_faceight_c_conv.py` in the fpv_face_pii repo (WOR-1016).

## Rules the labelers worked under

- Start from the prelabel boxes and make every visible face boxed: accept, move, delete
  or add. A frame that comes back with an empty box list is a real negative.
- Prelabel rule (PII-195, user decision 2026-09-12): per frame exactly ONE detector,
  chosen by the frame's cell. armW-only cells use `det_10g_armW`, the both and
  AA34-only cells use `det_34g_armAA34`, the faceless cell gets no box at all. Only that
  detector's boxes at or above the cell's band edge are kept (0.6+ cells at 0.6,
  [0.3,0.6) at 0.3, [0.1,0.3) at 0.1); no scores, no second detector, nothing else.
  68,980 prelabel boxes over the set, 9,635 frames sent with an empty list. The per cell
  table is in `job/prelabels/README.md`.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted here
  to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the faceback_45 convention). `ignore` is 0 on every row.
- No size floor is applied here; an eval applies its own floor.

The written instruction sheet the labelers were given is not on file; the rules above are
read off the prelabel package and the return.

## Counts

`boxes.csv`: 65,267 boxes on 24,514 images (35,486 frames are face-free); 51,894 boxes on
the train frames and 13,373 on the eval frames. `frames.csv`: `reviewed=1` on all 60,000
images, since the whole set was sent.

Per eye and transitions from the prelabels, from the WOR-1016 validation of the drop
(34 labelers, 2026-09-12 to 2026-09-16):

| | all | left | right |
|---|---:|---:|---:|
| frames | 60,000 | 28,503 | 31,497 |
| human boxes | 65,267 | 29,739 | 35,528 |
| frames with boxes | 24,514 | 11,288 | 13,226 |
| face-free frames | 35,486 | 17,215 | 18,271 |
| prelabel boxes | 68,980 | 32,680 | 36,300 |
| boxes unchanged | 4,731 | 2,154 | 2,577 |
| boxes moved (IoU >= 0.5) | 10,673 | 4,834 | 5,839 |
| boxes added | 49,863 | 22,751 | 27,112 |
| prelabel boxes deleted | 53,576 | 25,692 | 27,884 |
| frames edited | 48,384 | 23,005 | 25,379 |

The high edit rate is expected: the prelabel cut was one detector per frame at a band
edge as low as 0.1, so most cells were deliberately noisy.

## Files

- `job/prelabels/`: byte copies of `shang:/data/esteban/faceight/verdict_b/`
  (`import.jsonl`, one record per frame with the eye in `view`, and the `README.md` that
  states the cell rule, the per cell counts and the source md5s).
- `job/output/`: byte copies of the drop `/data/esteban/tmp/faceight_b_boxes/
  face_boxes_faceight_b.jsonl` and of the labels README from
  `shang:/data/esteban/pii/datasets/faceight_b/labels/README.md`.
- `boxes.csv` is derived from the drop by `data/build_faceight_pii2.py`, which also
  asserts that every record's `n_boxes_machine` equals the prelabel box count of that
  frame, and that the result is box for box equal to the independent conversion
  `/data/esteban/pii/datasets/faceight_b/boxes.jsonl` (md5
  4a4c5f04eff27312cc8c3113613b375f) on all 60,000 frames.

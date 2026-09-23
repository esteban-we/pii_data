# faceight_c boxes v1: the vendor human pass over every frame

## Why

First human labeling of round 3. All 40,000 frames went to the vendor's annotation portal
as the Verdict dataset `faceight_c` (both eyes in one dataset, the eye in the `view`
field), seeded with the WOR-195 cell prelabels, and came back 2026-09-16 as
`face_boxes_faceight_c.jsonl` (md5 e6aa449eeacf9624d3003dee6a6cde70). Validated and
converted by `training/ann_faceight_c_conv.py` in the fpv_face_pii repo (WOR-981).

## Rules the labelers worked under

- Start from the prelabel boxes and make every visible face boxed: accept, move, delete
  or add. A frame that comes back with an empty box list is a real negative.
- Prelabel rule (PII-195, user decision 2026-09-12): per frame exactly ONE detector,
  chosen by the frame's cell. armW-only cells use `det_10g_armW`, the both and
  AA34-only cells use `det_34g_armAA34`, the faceless cell gets no box at all. Only that
  detector's boxes at or above the cell's band edge are kept (0.6+ cells at 0.6,
  [0.3,0.6) at 0.3, [0.1,0.3) at 0.1); no scores, no second detector, nothing else.
  47,431 prelabel boxes over the set, 6,428 frames sent with an empty list. The per cell
  table is in `job/prelabels/README.md`.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted here
  to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the faceback_45 convention). `ignore` is 0 on every row.
- No size floor is applied here; an eval applies its own floor.

The written instruction sheet the labelers were given is not on file; the rules above are
read off the prelabel package and the return.

## Counts

`boxes.csv`: 28,042 boxes on 15,896 images (24,104 frames are face-free); 22,582 boxes on
the train frames and 5,460 on the eval frames. `frames.csv`: `reviewed=1` on all 40,000
images, since the whole set was sent.

Per eye and transitions from the prelabels, from the WOR-981 validation of the drop
(58 labelers, 2026-09-14 to 2026-09-16):

| | all | left | right |
|---|---:|---:|---:|
| frames | 40,000 | 18,994 | 21,006 |
| human boxes | 28,042 | 12,712 | 15,330 |
| frames with boxes | 15,896 | 7,356 | 8,540 |
| face-free frames | 24,104 | 11,638 | 12,466 |
| prelabel boxes | 47,431 | 22,326 | 25,105 |
| boxes unchanged | 15,948 | 7,309 | 8,639 |
| boxes moved (IoU >= 0.5) | 2,145 | 934 | 1,211 |
| boxes added | 9,949 | 4,469 | 5,480 |
| prelabel boxes deleted | 29,338 | 14,083 | 15,255 |
| frames edited | 25,297 | 12,001 | 13,296 |

The high delete rate is expected: the prelabel cut was one detector per frame at a band
edge as low as 0.1, so most cells were deliberately noisy.

## Files

- `job/prelabels/`: byte copies of `shang:/data/esteban/faceight/verdict_c/`
  (`import.jsonl`, one record per frame with the eye in `view`, and the `README.md` that
  states the cell rule, the per cell counts and the source md5s). A second copy of
  `import.jsonl` exists locally at `/data/esteban/tmp/pii/faceight_c/verdict_c_import.jsonl`;
  the build asserts the two are identical (md5 77392d0f141244930a09e3c4487b1fd2).
- `job/output/`: byte copy of the drop
  `/data/esteban/pii/datasets/faceight_c/labels/face_boxes_faceight_c.jsonl`. That labels
  directory has no README of its own.
- `boxes.csv` is derived from the drop by `data/build_faceight_pii2.py`, which also
  asserts that every record's `n_boxes_machine` equals the prelabel box count of that
  frame, and that the result is box for box equal to the independent conversion
  `/data/esteban/pii/datasets/faceight_c/boxes.jsonl` (md5
  c7a00a0bfac46d7b8000aa7b9fe26922) on all 40,000 frames.

# face_mine_right boxes v1: the human pass over every frame

## Why

First human labeling of the right eye. All 62,423 frames went to the vendor's
annotation portal as one dataset (`face-mine-right`), seeded with the armW face
detector's boxes, and came back 2026-09-11 with one human answer per frame
(validated by `training/ann_9_11_conv.py`, WOR-194).

## Rules the labelers worked under

- Start from the armW prelabels at score >= 0.5 (`machine_src`
  `face_mine_right/armw@1`, 70,337 boxes per the staging README); accept, move,
  delete or add boxes so that every visible face is boxed.
- A frame with no face comes back with an empty box list; that is a real negative,
  not a missing label.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted
  here to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the WOR-140 convention, the same as face_mine_v1). `ignore`
  is 0 on every row.
- No size floor is applied here; the eval applies its own 40 px long-side floor.
- The vendor's written instructions for this pass are not on file here.

## Prelabels

There is no prelabel file: the armW >= 0.5 boxes were generated inside the portal
and never exported. The drop carries only their count per frame (`n_boxes_machine`)
and their source tag (`machine_src`); the staging README reports 70,337 prelabel
boxes, 63,052 kept unchanged, 3,006 moved, 4,279 deleted, 24,844 boxes added by the
labelers (18,154 frames edited). `job/prelabels/` is therefore absent in this
version.

## Counts

Labeled 2026-09-04 to 2026-09-09 by 34 labelers, `review_round` 0 or 1 per row.

| | count |
|---|---:|
| frames | 62,423 |
| human boxes | 90,902 |
| frames with boxes | 35,760 |
| face-free frames | 26,663 |

`boxes.csv`: 90,902 rows on 35,760 images (72,786 on the 49,701 train frames, 18,116
on the 12,722 eval frames). `frames.csv`: `reviewed=1` on all 62,423 images.
No pii-data file exists for this eye; `build_facemine_pii2.py` pins the md5 of
`boxes.csv` as first built (5188364d192dc67faf52f91017cfb7ed, 2026-09-21) and
re-derives it from `job/output/` on every `verify`.

## Files

- `job/output/face_boxes_face-mine-right.jsonl`: byte copy of
  `/data/esteban/pii/datasets/face_mine_right/labels/face_boxes_face-mine-right.jsonl`,
  itself a byte copy of the drop `/data/esteban/tmp/ann_9_11/face_boxes_face-mine-right.jsonl`
  (md5 31f9541be247176ff1612ed835a20b59): 62,424 records plus an `{"kind": "end"}`
  trailer, normalised xywh, `labeled_by`, `labeled_ts`, `n_boxes_machine`,
  `machine_src`. One record has no image (`image_uri` null,
  `20260812_103909_AKGBXX_c000_f000000`, one box): the staging skipped that frame,
  so it has no row here. `build_facemine_pii2.py` derives `boxes.csv` from this copy.
- `job/output/README.md`: byte copy of the staging `labels/README.md`.
- `job/output/face-mine-right.boxes.csv` and `face-mine-right.frames.csv`: byte
  copies of `/data/esteban/tmp/faceback_anotation/` (2026-09-09), the vendor's csv
  exports of the same labels (one row per box with pixel `px,py,pw,ph`, and one row
  per frame with `boxes_json`); checked box for box equal to the jsonl.

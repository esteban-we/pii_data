# face_mine_v1 boxes v1: the human pass over every frame

## Why

First human labeling of the whole dataset. All 62,587 frames went to the vendor's
annotation portal as one dataset (`face-mine`), seeded with the miner's candidate
boxes, and came back 2026-09-01 with one human answer per frame.

## Rules the labelers worked under

- Start from the miner's boxes (`face_mine/two_model@1`: armW face candidates
  corroborated by the CrowdHuman head model, 83,536 boxes, at least one on every
  frame); accept, move, delete or add boxes so that every visible face is boxed.
- A frame with no face comes back with an empty box list; that is a real negative,
  not a missing label.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted
  here to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the WOR-140 convention). `ignore` is 0 on every row.
- No size floor is applied here; the eval applies its own 40 px long-side floor.

## Counts

Labeled 2026-09-01 (11:30 to 18:00) by 20 labelers, `review_round` 0 or 1 per row.

| | count |
|---|---:|
| frames | 62,587 |
| human boxes | 76,859 |
| frames with boxes | 41,079 |
| face-free frames | 21,508 |
| miner boxes offered | 83,536 |
| miner boxes kept | 67,944 |
| boxes added by humans | 8,915 |
| miner boxes rejected | 15,592 |

`boxes.csv`: 76,859 rows on 41,079 images (61,093 on the 49,833 train frames, 15,766
on the 12,754 eval frames). `frames.csv`: `reviewed=1` on all 62,587 images.
`boxes.csv` is byte for byte the pii-data file `datasets/face_mine_v1/boxes/v1.csv`
(md5 a6a35c2e37ad70b3fd525f91b4a61298).

## Files

- `job/prelabels/boxes.jsonl`: byte copy of
  `/data/esteban/pii/datasets/face_mine_v1/boxes.jsonl`, the miner's output (one row
  per frame, `box_src` `face_mine/two_model@1`, normalised xywh, plus `src_key`
  and `src_frame`, which the top-level `frames.csv` takes from here). These are
  machine boxes, not labels.
- `job/output/face-mine_labeled.csv`: byte copy of `/data/esteban/pii/face-mine_labeled.csv`,
  the portal export (one row per box, empty `box_i` = face-free frame, normalised
  xywh, `labeled_by`, `labeled_ts`). `build_facemine_pii2.py` derives `boxes.csv`
  from this copy and asserts the pinned md5.

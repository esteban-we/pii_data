# face_mine_v1 boxes v2: relabel of the armY false-positive eval frames

## Why

Under the eval matching (IoU 0.4, greedy, 40 px floor) the armY detector at score
0.5 produced false positives on 3,600 of the 12,754 eval frames. To settle, box by
box, whether each red model box was a face the v1 labelers missed or a genuine
model mistake, those frames went back to the vendor with the v1 human boxes and the
model's unmatched detections as proposals (WOR-93 built the package, delivered
2026-09-10; PII-140 measured the result, PII-142 tracks whether v2 becomes the eval
reference: still open at 2026-09-21). Frames not sent keep their v1 rows byte for
byte; train frames are untouched.

## Rules

- Candidate frames: eval frames on which armY (det_size 1024, dump of 2026-09-08)
  has at least one detection with score >= 0.5 that matches no human box under the
  eval matching. 3,600 frames over 935 sessions, all in the eval split.
  `frames.csv` has `reviewed=1` on exactly these frames.
- The labelers saw every v1 human box of the frame (including sub-40 px ones) and
  the model's false positives together (`job/prelabels/armY_fp_thr0.5.jsonl`,
  `source` = `gt` or `model_fp`; `machine_src` `face_mine_v2/armY_fp@1` in the
  output) and returned the full box set of each frame, so a v1 box could also be
  moved or dropped in this pass. The vendor's written instructions for this pass
  are not on file here.
- Coordinates and `ignore` as in v1 (normalised xywh in the output, converted to
  pixel xyxy at one decimal with the same convention; `ignore` 0 everywhere; the
  40 px floor lives in the eval).

## Counts

On the 3,600 reviewed frames the vendor (9 labelers, 2026-09-10) returned 10,088
boxes on 3,107 frames against 5,384 v1 boxes; 4,768 false positives were sent. Per
the PII-140 measurement: 3,824 v1 boxes kept, 588 moved, 5,676 added, 972 removed;
of the 4,768 sent false positives 2,427 became true positives, 1,405 became sub-40 px
ignore regions and 936 remain false positives.

Over the whole eval split: 20,470 boxes against 15,766 in v1.

`boxes.csv`: 81,563 rows on 41,730 images (61,093 train rows identical to v1 plus
the 20,470 eval rows). It is byte for byte the pii-data file
`datasets/face_mine_v1/boxes/v2.csv` (md5 7561137eb5c0c506310f2dc839de08d2); the
reviewed set equals pii-data `boxes/v2.frames.txt` (md5 b30c60df2820fae8429efdfa1d058fda).
`build_facemine_pii2.py` re-derives the reviewed frames' rows from `job/output/`,
takes v1 rows elsewhere and asserts the pinned md5.

## Files

- `job/prelabels/`: byte copies of `/data/esteban/pii/datasets/face_mine_v2/`
  (`armY_fp_thr0.5.jsonl`: one row per sent frame, pixel xyxy, v1 boxes as `gt`
  and the model's false positives as `model_fp` with scores; `README.md`: the
  selection and the box rule). The 3,600 image copies of that directory are not
  carried over (same bytes as `../../images/`).
- `job/output/face_boxes_face-mine-v2.jsonl`: byte copy of
  `/data/esteban/tmp/face_boxes_face-mine-v2.jsonl` (3,600 records plus an
  `{"kind": "end"}` trailer).

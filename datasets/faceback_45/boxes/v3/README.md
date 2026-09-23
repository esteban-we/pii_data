# faceback_45 boxes v3: round-2 review of the eval frames with armAE34 additions

## Why

Model armAE34 found faces on eval frames that the v2 (HQ) ground truth did not cover.
To settle whether those are misses of the labelers or false positives of the model,
the affected eval frames were sent back to the vendor with the v2 boxes plus the
model's uncovered detections as proposals (PII-946 built the prelabels, PII-960 ran
the review; delivered 2026-09-16). Frames without an addition were not sent and keep
their v2 rows byte for byte; train frames are untouched.

## Rules

- Candidate frames: the eval frames on which armAE34 (det_size 1024, dump down to
  score 0.02) has at least one detection with score >= 0.3 whose IoU to every v2 box
  of that frame (including boxes under the 40 px eval floor) is < 0.1. No size floor
  on the additions. 4,023 frames (2,110 left, 1,913 right) over 63 sessions; 6,713
  additions. `frames.csv` has `reviewed=1` on exactly these frames.
- The labelers saw the v2 boxes and the additions together
  (`job/prelabels/import.jsonl`, `source` = `gt` or `armAE34`; `machine_src`
  `faceback_hq_2/gt+armAE34@1` in the output) and returned the full box set of each
  frame, so a v2 box could also be moved or dropped in this pass (587 were). The
  vendor's written instructions for this pass are not on file here.
- Coordinates and `ignore` as in v1 and v2 (pixel xyxy at one decimal, `ignore` 0
  everywhere; the 40 px floor lives in the eval).

## Counts

On the 4,023 reviewed frames the vendor returned 12,786 boxes against 8,348 v2 rows;
3,055 of the reviewed frames came back with at least one box. Per the PII-960 import
into pii-data: 7,019 boxes coincide with a v2 box at 0.1 px, 742 moved (IoU >= 0.5),
5,025 are new, 587 v2 boxes are gone; of the 6,713 machine proposals the humans kept
3,829.

Over the whole eval split: 27,454 boxes (left 13,589, right 13,865) on 8,228 frames,
against 23,016 on 7,681 in v2.

`boxes.csv`: 103,453 rows on 36,498 images (75,999 train rows identical to v1 and v2
plus the 27,454 eval rows). The eval rows are byte for byte the pii-data file
`datasets/faceback_hq/boxes/v2.csv` (md5 1c44ee062e7d80703c31f1c9f2ce896f); the
reviewed set equals pii-data `faceback_hq/boxes/v2.frames.txt`.
`build_faceback45_pii2.py` re-derives the reviewed frames' rows from `job/output/`
and asserts them equal, and asserts that every unreviewed eval frame has the same
rows as v2.

## Files

- `job/prelabels/`: byte copies of `/data/esteban/pii/datasets/faceback_hq_relabel_v1/`
  (`import.jsonl`: what was imported, v2 boxes first then the additions;
  `additions.jsonl`: the additions alone in pixels with scores; `stats.json`;
  `README.md`: the cut and how to re-cut it).
- `job/output/`: byte copies of `/data/esteban/tmp/pii/fb_hq_2/face_boxes_faceback_hq_2-{left,right}.jsonl`
  (2,110 and 1,913 records plus an `{"kind": "end"}` trailer).

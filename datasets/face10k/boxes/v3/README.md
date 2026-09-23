# face10k boxes v3: facecheck2, a second independent vendor re-labelling of every frame

## Why

The second of two parallel check passes over all 11,507 face10k frames (the first is
`facecheck1`, v2 of this dataset). Delivered 2026-09-10 as the portal dataset
`facecheck2`; numbered v3 by delivery order (PII-1315). It is independent of v2, not a
refinement of it: it started from the same machine merge, not from v2's boxes.
Imported into pii-data as the dataset `facecheck2` sharing face10k's image bytes
(PII-152).

Against v1's 33,849 boxes this pass keeps 22,896 (0.1 px), moves 4,634 (IoU >= 0.5),
adds 28,416 and removes 6,319; against v2 it keeps 20,158, moves 13,678, adds 22,110
and removes 15,543, and 4,399 of the 11,507 frames have a different box count than in
v2 (PII-314). The two passes disagree with each other more than either disagrees with
v1; which pass, if either, becomes the training reference is an open decision.
Training still uses v1.

## Rules

- Every frame was sent: `frames.csv` has `reviewed=1` on all 11,507 frames.
- The labelers started from the same machine merge as v2 (`machine_src`
  `facecheck/merge@1`, `n_boxes_machine` 36,667) and returned the full box set of every
  frame; an empty list is a real negative (154 frames). The vendor's written
  instructions for this pass are not on file here, and the prelabel file imported into
  the portal was not retained on this box.
- Coordinates and `ignore` as in v2 (normalised xywh converted to pixel xyxy at one
  decimal, `ignore` 0 everywhere). Image names equal the face10k names.

## Counts

`boxes.csv`: 55,946 rows on 11,353 images (batch v3 45,208 rows on 8,187 images,
batch repair 10,738 on 3,166; the 16 v1-unlabeled frames carry 29 boxes). Labelled
2026-09-10 to 2026-09-11 by 26 labelers. `boxes.csv` is byte for byte the pii-data
file `datasets/facecheck2/boxes/v1.csv` (md5 7352c56b6a2909d8acd6035d641a73f2),
which `build_face10k_pii2.py` also re-derives from `job/output/` and asserts equal.

## Files

- `job/output/face_boxes_facecheck2.jsonl`: byte copy of
  `/data/esteban/tmp/face_boxes_facecheck2.jsonl` (11,507 records plus an
  `{"kind": "end", "frames": 11507}` trailer).
- `job/prelabels/`: none; see Rules.

# face10k boxes v2: facecheck1, an independent vendor re-labelling of every frame

## Why

A check on the v1 labels: all 11,507 face10k frames (both batches, including the 16
frames v1 never got back) were sent to the vendor's annotation portal again as the
dataset `facecheck1` and came back 2026-09-10 with one human answer per frame. A
second, independent pass over the same frames ran in parallel (`facecheck2`, v3 of
this dataset); the two are numbered by delivery order (PII-1315). Imported into
pii-data as the dataset `facecheck1` sharing face10k's image bytes (PII-152); here it
is v2 of face10k.

Against v1's 33,849 boxes this pass keeps 25,707 (coincide at 0.1 px), moves 3,491
(IoU >= 0.5 to a v1 box), adds 20,181 and removes 4,651 (PII-314). Whether the added
boxes are faces v1 missed or over-labelling is not settled; a visual audit is
PII-154. Training still uses v1.

## Rules

- Every frame was sent: `frames.csv` has `reviewed=1` on all 11,507 frames.
- The labelers started from a machine merge (`machine_src` `facecheck/merge@1` in the
  output, `n_boxes_machine` 36,667 over the set) and returned the full box set of every
  frame; an empty list is a real negative (178 frames). The vendor's written
  instructions for this pass are not on file here, and the prelabel file that was
  imported into the portal was not retained on this box.
- Coordinates: normalised top-left xywh at four decimals in the vendor output,
  converted to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one
  decimal (W x H = 2328 x 1748, the PII-140 convention). `ignore` is 0 on every row.
- The vendor's image names equal the face10k names (no eye mapping is needed).

## Counts

`boxes.csv`: 49,379 rows on 11,329 images (batch v3 38,650 rows on 8,171 images,
batch repair 10,729 on 3,158; the 16 v1-unlabeled frames carry 28 boxes). Labelled
2026-09-10 to 2026-09-11 by 65 labelers (`labeled_by` in the output). `boxes.csv` is
byte for byte the pii-data file `datasets/facecheck1/boxes/v1.csv` (md5
93339899800b5b14d29abf13afde01dd), which `build_face10k_pii2.py` also re-derives from
`job/output/` and asserts equal.

## Files

- `job/output/face_boxes_facecheck1.jsonl`: byte copy of
  `/data/esteban/tmp/face_boxes_facecheck1.jsonl` (11,507 records plus an
  `{"kind": "end", "frames": 11507}` trailer).
- `job/prelabels/`: none; see Rules.

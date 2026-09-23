# faceback_45 boxes v2: the HQ vendor pass over the eval split

## Why

A second, higher-quality vendor labelling pass over the 16,940 eval frames (the 67
sessions of the faceback eval split, 8,470 per eye), so that the eval slice has a
ground truth better than v1 (PII-176; the original v1 labels stay the training
labels). Delivered 2026-09-11 as two jsonl files (one per eye); in pii-data it
was imported as the separate dataset `faceback_hq` (WOR-176). Here it is v2 of
faceback_45: the eval frames take the HQ boxes, the 68,014 train frames keep their
v1 rows byte for byte.

## Rules

- Only the eval split was sent; `frames.csv` has `reviewed=1` on exactly those
  16,940 frames and 0 elsewhere.
- The labelers started from the v1 human boxes (`machine_src`
  `faceback45_hq_eval/human_labels@1`, `review_round` 2 in the output) and returned
  the full box set of every frame; an empty list is a real negative. The vendor's
  written instructions for this pass are not on file here.
- Coordinates: normalised xywh in the vendor output, converted to pixel xyxy at one
  decimal with the same convention as v1. `ignore` is 0 on every row; the 40 px
  long-side floor is applied by the eval, not stored here.
- The vendor's image names carry no eye (`<session>_c<chunk>_f<idx>.jpg`); the eye
  comes from the record's `view` (`lview` = left, `rview` = right).

## Counts

On the 16,940 eval frames: 23,016 boxes (left 11,572, right 11,444) on 7,681 frames,
against the 19,007 v1 boxes on 7,041 of the same frames. Per the WOR-176 import:
13,893 HQ boxes coincide with a v1 box at 0.1 px, 2,501 moved (IoU >= 0.5 to a v1
box), 6,622 were added, 2,613 v1 boxes were removed.

`boxes.csv`: 99,015 rows on 35,951 images (75,999 train rows identical to v1 plus
the 23,016 eval rows). The eval rows are byte for byte the pii-data file
`datasets/faceback_hq/boxes/v1.csv` (md5 1f50860639a223f11e90c754422add0c), which
`build_faceback45_pii2.py` also re-derives from `job/output/` and asserts equal.

## Files

- `job/output/`: byte copies of `/data/esteban/tmp/hq_eval/face_boxes_faceback-45-HQ-eval-{left,right}.jsonl`
  (8,470 records per eye plus an `{"kind": "end"}` trailer).
- `job/prelabels/`: none. The vendor started from the v1 human boxes, and no separate
  prelabel file for this pass was retained; the v1 boxes on the eval frames are the
  prelabels of this pass (`../v1/boxes.csv` restricted to `reviewed=1` frames of this
  version).

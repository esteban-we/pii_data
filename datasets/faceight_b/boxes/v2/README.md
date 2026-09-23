# faceight_b boxes v2: the armAE34 addition review

## Why

The v1 pass started from a deliberately thin prelabel (one detector per frame at its
cell's band edge) and deleted 53,576 of the 68,980 prelabel boxes, so a stronger detector
was run over the whole round to find faces the labelers had not boxed. Every armAE34
detection that did not match a v1 human box was offered back to the vendor as an addition
to judge; the 9,402 frames that carried at least one such addition were sent again as the
Verdict dataset `faceight_b_v2` (PII-1035, PII-1047) and came back 2026-09-18 as
`face_boxes_faceight_b_v2.jsonl` (md5 cce2d5b389c747faf9a4804590e91ff2).

v2 is a full replacement over all 60,000 frames: the 9,402 reviewed frames take their
boxes from this pass, the other 50,598 keep their v1 rows byte for byte. A consumer reads
one version, never a diff.

## Rules

- Prelabels for this pass (PII-1015, `mining/face_mine_relabel_v3.py`, model
  `armAE34_det34g.onnx` md5 3fa016c5dc7241e448ec101f4234e738, det_size 1024, padded
  canvas): every v1 human box of the frame verbatim, plus every detection at score >= 0.3
  whose best IoU against those human boxes is below 0.1. No size floor, no clipping to the
  frame. A frame with no box at all keeps an empty list.
- Frames sent: exactly those with at least one addition, 9,402 of 60,000 (4,549 left and
  4,853 right). They carried 24,234 prelabel boxes: 12,675 human and 11,559 additions.
  The build re-derives that frame set from the prelabel files and asserts it equals the
  frames in the return.
- The labelers judged the whole frame again, not only the additions: they could delete a
  box that came from v1. On the 9,402 reviewed frames the return carries 9,691 boxes on
  3,706 frames, against 12,675 v1 boxes on the same frames, so the pass is a net removal
  of 2,984 boxes. That is the intended outcome of a review, not a loss of labels.
- Coordinate convention and `ignore` as in v1.

The written instruction sheet for this pass is not on file.

## Counts

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 65,267 | 24,514 | 60,000 (all) |
| v2 | 62,283 | 24,184 | 9,402 (the frames with an armAE34 addition) |

v2 has 49,540 boxes on the train frames and 12,743 on the eval frames. The 50,598 frames
outside the reviewed set are identical to v1, which `data/build_faceight_pii2.py verify`
re-checks row by row.

## Files

- `job/prelabels/`: byte copies of
  `shang:/data/esteban/pii/datasets/faceight_b_relabel_v1/`: `import_{left,right}.jsonl`
  (what the vendor started from, each box tagged `source` human or armAE34, machine boxes
  also carrying `score`), `dets_{left,right}.jsonl` (the full armAE34 dump down to score
  0.1 in pixels, so a different cut needs no second inference pass), `README.md` and
  `stats.json` (the cut, the counts and the score and size bands).
- `job/output/`: byte copy of the drop `/data/esteban/tmp/pii/faceight_v2/
  face_boxes_faceight_b_v2.jsonl`.
- `boxes.csv` is derived from the v1 rows and this drop by `data/build_faceight_pii2.py`.

# faceight_c boxes v2: the armAE34 addition review

## Why

The v1 pass started from a deliberately thin prelabel (one detector per frame at its
cell's band edge) and deleted 29,338 of the 47,431 prelabel boxes, so a stronger detector
was run over the whole round to find faces the labelers had not boxed. Every armAE34
detection that did not match a v1 human box was offered back to the vendor as an addition
to judge; the 9,925 frames that carried at least one such addition were sent again as the
Verdict dataset `faceight_c_v2` (PII-980, PII-996) and came back 2026-09-17 as
`face_boxes_faceight_c_v2.jsonl` (md5 39da90d3ffde895bebe87a594a4d6041).

v2 is a full replacement over all 40,000 frames: the 9,925 reviewed frames take their
boxes from this pass, the other 30,075 keep their v1 rows byte for byte. A consumer reads
one version, never a diff.

## Rules

- Prelabels for this pass (PII-990, `mining/face_mine_relabel_v3.py`, model
  `armAE34_det34g.onnx` md5 3fa016c5dc7241e448ec101f4234e738, det_size 1024, padded
  canvas): every v1 human box of the frame verbatim, plus every detection at score >= 0.3
  whose best IoU against those human boxes is below 0.1. No size floor, no clipping to the
  frame. A frame with no box at all keeps an empty list.
- Frames sent: exactly those with at least one addition, 9,925 of 40,000 (4,661 left and
  5,264 right). They carried 24,792 prelabel boxes: 10,440 human and 14,352 additions.
  The build re-derives that frame set from the prelabel files and asserts it equals the
  frames in the return.
- The labelers judged the whole frame again, not only the additions: they could delete a
  box that came from v1. On the 9,925 reviewed frames the return carries 16,279 boxes on
  6,061 frames, against 10,440 v1 boxes on the same frames, so the pass is a net addition
  of 5,839 boxes.
- Coordinate convention and `ignore` as in v1.

The written instruction sheet for this pass is not on file.

## Counts

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 28,042 | 15,896 | 40,000 (all) |
| v2 | 33,881 | 17,007 | 9,925 (the frames with an armAE34 addition) |

v2 has 27,143 boxes on the train frames and 6,738 on the eval frames. The 30,075 frames
outside the reviewed set are identical to v1, which `data/build_faceight_pii2.py verify`
re-checks row by row.

## Files

- `job/prelabels/`: byte copies of
  `shang:/data/esteban/pii/datasets/faceight_c_relabel_v1/`: `import_{left,right}.jsonl`
  (what the vendor started from, each box tagged `source` human or armAE34, machine boxes
  also carrying `score`), `dets_{left,right}.jsonl` (the full armAE34 dump down to score
  0.1 in pixels, so a different cut needs no second inference pass), `README.md` and
  `stats.json` (the cut, the counts and the score and size bands).
- `job/output/`: byte copy of the drop `/data/esteban/tmp/pii/faceight_v2/
  face_boxes_faceight_c_v2.jsonl`.
- `boxes.csv` is derived from the v1 rows and this drop by `data/build_faceight_pii2.py`.

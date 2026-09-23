# pii_frames boxes v1: the original labels, recovered from the training manifests

## Why

The only label pass over this set, and the labels every training manifest of the
fpv_face_pii repo uses for its `pii/` blocks (19,579 boxes in `train_W.txt`,
`train_Z2.txt` and their successors).

## What is known of the pass

The original vendor drop is lost: no prelabel file, no returned file and no written
labeling instructions for this set are on disk on this box or in either repo, so
`job/prelabels/` and `job/output/` do not exist here. The boxes were recovered from the
`pii/` labelv2 blocks of the training manifests (fpv_face_pii `data/FRAME_TABLE.md`,
box sources table: `train_Z2.txt`), which is the only label source for this pass.
`build_pii_frames_pii2.py` asserts `boxes.csv` equal to the 10,249 `pii/` blocks of
both `training/manifests/train_Z2.txt` and `train_W.txt` (identical for this set).

An earlier manifest, `training/manifests/train_labelv2.txt`, carries the same 10,249
frames with 18,003 boxes: a strict subset of the 19,579 here (1,424 blocks differ, all
by added boxes, same 2,704 ignore rows). Which pass added the 1,576 boxes between the
two manifests, and whether it was a vendor or an in-house edit, is not recorded.

Rules the labelers worked under: not on file. What the data shows: one box per face,
no class field, and 2,704 rows flagged `ignore=1` (loaded as `gt_bboxes_ignore`, not as
positives; 467 images carry only ignore rows). The per-row reason for an ignore flag is
not recorded.

## Counts

`boxes.csv`: 19,579 rows on 8,761 images, 2,704 with `ignore=1`. Byte for byte the
pii-data file `datasets/pii_frames/boxes/v1.csv` (md5 3b8cb6d1a325aeefb59987e47fac588c);
per-image counts equal pii-data `frames.csv` `n_boxes`.

`frames.csv`: `reviewed=1` on all 10,249 frames. 1,488 frames have no box row; the
manifests list them with an empty block and the trainer takes them as face-free
negatives. That every one of them was looked at by a labeler is unverified: there is no
per-frame review record for this pass, only the manifest.

## Files

- `job/prelabels/`: none. Nothing the vendor started from is on disk.
- `job/output/`: none. The vendor's returned files are not on disk; the boxes come
  from the training manifests as described above.

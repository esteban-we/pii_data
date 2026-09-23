# face10k boxes v1: the original vendor labels of both batches

## Why

The first human labeling of the set, and the labels every training manifest of the
fpv_face_pii repo uses for face10k (33,849 boxes in every `train_*.txt` that carries
face10k). Batch v3 (8,339 frames) went to the vendor 2026-08-10 with machine
pre-boxes; batch repair (3,168 frames) followed 2026-08-11 with the strip-shaped boxes
of the rectified-to-fisheye mapping as "redraw these" hints and the sound boxes of the
same frames as "keep these" references. Both batches were labelled under
`job/prelabels/face10k_v3/ANNOTATION_SPEC.md` (v5, 2026-08-11).

## Rules the labelers worked under

- One binary decision per face, no class field: can you tell which specific individual
  this is? Yes (a live person, a real person on a screen, a printed photograph of a
  real person): draw a box. No (cartoon, drawing, logo, plush toy, mannequin): do not
  draw, and delete a pre-box that sits on one.
- The only per-box flag was "uncertain": when in doubt, box it and tick uncertain, for
  review on our side. That flag is not in this file.
- Batch v3 pre-boxes: the armW detections at score >= 0.50 (23,906 boxes, long side
  p10/p50/p90 = 45/72/124 px), suggestions only.
- Batch repair: `pre_boxes` are positions to redraw (location roughly right, shape
  wrong), `kept_boxes` are confirmed boxes on the same frame not to be touched.
- Acceptance: miss rate < 2%, "a printed or on-screen real face skipped as a cartoon"
  < 0.5% (from the spec; whether it was measured is not recorded here).

## Counts

`boxes.csv`: 33,849 rows on 11,266 images, batch v3 25,966 rows on 8,201 images (138
labelled frames face-free), batch repair 7,883 rows on 3,065 images (87 labelled
frames face-free). 964 rows have `ignore=1` (505 in batch v3, 459 in batch repair):
they load as `gt_bboxes_ignore`, not as positives. `docs/FACE_DATASETS.md` in the
fpv_face_pii repo attributes ignore regions in this set to the boxes that degenerated in
the rectified-to-fisheye mapping; the per-row reason is not recorded.

`frames.csv`: `reviewed=1` on 11,491 frames; 0 on the 16 repair frames of session
`20260725_055005_MPCWWA` chunk 000 that the vendor never returned (`unlabeled=1` in
the top-level `frames.csv`, PII-131). Those 16 have no box row and must not train.

The batch v3 rows are byte for byte the pii-data file
`datasets/face10k_v3/boxes/v1.csv` (md5 eab68f114ece62cf9deef487e1b60c34) and the
batch repair rows the file `datasets/face10k_repair/boxes/v1.csv` (md5
c204316dda8b7c3c4c3f803c27a13b6d); `build_face10k_pii2.py` merges the two, checks
the per-image counts against pii-data `frames.csv` `n_boxes`, and asserts the result
equal to the 11,491 face10k labelv2 blocks of `training/manifests/train_Z2.txt`.

## Files

- `job/prelabels/face10k_v3/`: byte copies of `/data/esteban/pii/datasets/face10k_v3/`
  `manifest.jsonl` (8,339 rows: name, session, chunk, view, selection bucket, `pre_boxes`
  with `pre_scores`, task/scene/environment/city), `README.md` (selection method, in
  Chinese), `ANNOTATION_SPEC.md` (the rules given to the vendor, v5, in Chinese) and
  `stats.json`.
- `job/prelabels/repair_v1/`: byte copies of
  `/data/esteban/pii/datasets/face10k_v3/repair_v1/` `manifest.jsonl` (3,168 rows:
  `pre_boxes` to redraw, `kept_boxes`, `reason`) and `README.md`.
- `job/output/`: none. The vendor's returned files are not on disk on this box; the
  boxes were recovered from the labelv2 blocks of `training/manifests/train_Z2.txt`
  (PII-125, PII-330), which is the only label source for this pass. The
  `repair_v1/manifest.jsonl` hints agree with the returned boxes on 5 of 3,152 frames
  only (PII-330); it is a prelabel, not a label.

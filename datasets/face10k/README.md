# face10k

The main human-labelled face training set: 11,507 raw fisheye frames (2328x1748,
`vst_left` head-mounted view for batch v3, both eyes for batch repair) selected for
faces the production detector nearly catches and drops (`docs/FACE_DATASETS.md`
section 3 in the fpv_face_pii repo; the selection method is in
`boxes/v1/job/prelabels/face10k_v3/README.md`). Delivered to the vendor in two
batches under one annotation spec (`boxes/v1/job/prelabels/face10k_v3/ANNOTATION_SPEC.md`, v5):

- batch `v3`: 8,339 frames, 333 sessions, 667 chunks, finalized 2026-08-10 as
  `oss://algorithm-datasets/face_pii/face10k_v3/`;
- batch `repair`: 3,168 frames, 28 sessions, 29 chunks, generated 2026-08-11 as
  `face_pii/face10k_v3/repair_v1/`: frames whose boxes had been drawn on the rectified
  1280x720 delivery video and degenerated into strips when mapped back to fisheye near
  the periphery, sent for redrawing directly on the fisheye frame.

Staged locally as `/data/esteban/pii/datasets/face10k_v3/` (`images/` and
`repair_v1/images/`). Rebuilt 2026-09-21 in this layout (PII-1315, PII-1320) by
`data/build_face10k_pii2.py` in the fpv_face_pii repo: the two batches are one image
set in one flat `images/` directory (no name is shared between the batches; the
build asserts it), every image a byte copy of the staged file (md5-checked by
`build_face10k_pii2.py verify`).

## Layout

```
images/                 11,507 jpg, 6,532,081,854 bytes, all 2328x1748
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, batch, unlabeled
split.csv               session, role; every one of the 361 sessions is train
boxes/v1/               the original vendor labels of both batches (33,849 boxes; what training uses)
boxes/v2/               facecheck1: an independent vendor re-labelling of every frame (49,379 boxes)
boxes/v3/               facecheck2: a second independent re-labelling of every frame (55,946 boxes)
```

Image names differ by batch and carry the batch's own fields:

```
v3:      <session>_<chunk>_f<frame_idx>.jpg                  e.g. 20260415_060144_KSXGVV_000_f001965.jpg
repair:  <session>_chunk_<chunk>_<lview|rview>_f<frame_idx>.jpg   e.g. 20260616_010704_GMTYRC_chunk_006_lview_f000030.jpg
```

`chunk` is three digits and `frame_idx` six in the name; `frames.csv` stores `chunk` as
the three-digit string and `frame_idx` as an integer. `eye` is `left` for every v3 frame
(the manifest's `vst_left`) and `left`/`right` from `lview`/`rview` for repair frames
(1,463 left, 1,705 right). `frames.csv` is sorted by image and is the source of truth
for which images exist. `size` (bytes) and `md5` are the image file's own; they come
from the pii-data repo `frames.csv` (datasets `face10k_v3` and `face10k_repair`, hashed
at OSS upload) and the build asserts `size` against the file on disk. `batch` is `v3`
or `repair`. Provenance beyond these columns (task, scene, city, selection bucket,
pre-box scores, repair reason) is in the two manifests under `boxes/v1/job/prelabels/`;
the frame time in ms is not recorded anywhere for this set.

`unlabeled` is 1 on exactly 16 repair frames, all of session `20260725_055005_MPCWWA`
chunk 000 (5 left, 11 right). The original vendor never returned a label for them
(PII-131, user decision 2026-09-09): they are not face-free negatives, they are frames
with an unknown label. They have no row in `boxes/v1/boxes.csv`, `reviewed=0` in
`boxes/v1/frames.csv`, and no training mix may use them under v1. The two facecheck
passes did label them (28 and 29 boxes), so under v2 and v3 they are ordinary frames.
`split.csv` is per session and their session is `train` like every other; the flag is
per frame and overrides it.

Each `boxes/vN/` holds:

```
frames.csv        image, reviewed   (1 = the frame was sent in this pass, even if it came back unchanged)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 fisheye frame, one decimal)
README.md         why the pass exists and its labeling rules
job/prelabels/    what the vendor started from (byte copies)
job/output/       what the vendor returned (byte copies)
```

A box file is a full replacement over all 11,507 images, never a diff: a consumer
reads one version. Rows are sorted by (image, x1, y1); an image with no box has no
row. `ignore=1` marks a region to load as `gt_bboxes_ignore` rather than a positive;
v1 has 964 such rows, v2 and v3 none.

| version | pass | boxes | ignore rows | images with a box | reviewed frames |
|---|---|---:|---:|---:|---:|
| v1 | original vendor labels, both batches | 33,849 | 964 | 11,266 | 11,491 (all but the 16 unlabeled) |
| v2 | facecheck1 (delivered 2026-09-10) | 49,379 | 0 | 11,329 | 11,507 (all) |
| v3 | facecheck2 (delivered 2026-09-10) | 55,946 | 0 | 11,353 | 11,507 (all) |

v2 and v3 are independent passes over the same frames, not a refinement chain; they
are numbered by delivery order (user decision, PII-1315). Which of them, if either,
should replace v1 for training is an open question (PII-314): every `train_*.txt`
manifest in the fpv_face_pii repo still carries the v1 boxes.

## Split

All 361 sessions are `train`. No face10k session appears in any eval split file of the
fpv_face_pii repo (`data/splits/*eval*`), checked at build time 2026-09-21. There is no
eval slice of this dataset; the eval benches are gt_bench and the eval splits of
face_mine_v1 and faceback_45.

## Provenance

- images: `/data/esteban/pii/datasets/face10k_v3/images/` (batch v3) and
  `/data/esteban/pii/datasets/face10k_v3/repair_v1/images/` (batch repair);
- `frames.csv` fields: the two `manifest.jsonl` files (copied under
  `boxes/v1/job/prelabels/`); `size`, `md5`, `unlabeled`: pii-data `frames.csv`;
- v1 boxes: pii-data `datasets/face10k_v3/boxes/v1.csv` and
  `datasets/face10k_repair/boxes/v1.csv`; v2, v3: pii-data
  `datasets/facecheck1/boxes/v1.csv` and `datasets/facecheck2/boxes/v1.csv`; see the
  per-version README for the job files.
- Not carried over: `_probe.txt`, `.complete` (staging markers); the OSS copies are
  unchanged.

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out
of this directory.

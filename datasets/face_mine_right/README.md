# face_mine_right

Right-eye (`vst_right`) frames of the face_mine_v1 sessions: 62,423 frames over
8,994 sessions, raw fisheye 2328x1748, the same (session, chunk, frame_idx) as the
left-eye dataset `face_mine_v1` minus the 164 frames the rview staging skipped
(WOR-53, `idx-out-of-range`, per the staging README; 160 sessions lose 1 to 5
frames, and session `20260812_103909_AKGBXX` (eval) is absent altogether). Staged
in `/data/esteban/pii/datasets/face_mine_right` (labels received 2026-09-11); rebuilt 2026-09-21 in this
layout (PII-1315, PII-1319) by `data/build_facemine_pii2.py --eye right` in the
fpv_face_pii repo. Every image is a byte copy of the staged file (md5 checked by
`build_facemine_pii2.py verify --eye right`). The left-eye frames of the same
sessions are the separate dataset `face_mine_v1` (PII-1318): different pixels, same
sessions, same split.

## Layout

```
images/                 62,423 jpg, 39,462,988,868 bytes, all 2328x1748, names <session>_c<chunk>_f<frame_idx>.jpg
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, src_key, src_frame
split.csv               session, role (train | eval); the split is per session
boxes/v1/               the human pass over every frame (2026-09-04 to 09-09, delivered 2026-09-11)
boxes/v2/               pending: armAG34-over-human relabel (PII-1273); prelabels only
```

Image names carry no eye; `eye` is `right` on every row. `frame_idx` is the frame
number inside the chunk video (`src_key`, the source video key; `src_frame` equals
`frame_idx`). `size` is the byte count of the image and `md5` the md5 of its bytes,
both computed from the staged bytes at build time (no earlier table carried them
for this eye; pii-data has no rows for face_mine_right). `frames.csv` is sorted by
image and is the source of truth for which images exist.

Each `boxes/vN/` holds:

```
frames.csv        image, reviewed   (1 = the frame was sent in this pass, even if it came back unchanged)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 frame, one decimal)
README.md         why the pass exists and its labeling rules
job/prelabels/    what the vendor started from (byte copies); absent in v1, see its README
job/output/       what the vendor returned (byte copies)
```

A box file is a full replacement over all 62,423 images, never a diff: a consumer
reads one version. Rows are sorted by (image, x1, y1); an image with no box has no
row. Frames not sent in a pass keep the previous version's rows byte for byte.
`ignore=1` would mark a region to load as `gt_bboxes_ignore` rather than a
positive; no version of this dataset has an ignore row. A pending version (still at
the vendor) has `job/prelabels/` and a README only; its `frames.csv` and `boxes.csv`
appear when the output lands.

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 90,902 | 35,760 | 62,423 (all) |
| v2 | pending | pending | (42,649 uploaded, see boxes/v2/README.md) |

## Train/eval split (v1, frozen)

`split.csv`: 7,137 train sessions (49,701 frames) and 1,857 eval sessions (12,722
frames), the face_mine_v1 assignment (`data/splits/{train,eval}_sessions_v1.txt` in
the repo) restricted to the sessions that have a right-eye frame. Split by session,
never by frame.

Rules:

- The eval sessions must never be trained on, from this or any later dataset; the
  assignment is shared with `face_mine_v1` and `faceback_45`.
- The split is frozen; additions to eval land as new named slices.

## Provenance

- images: `/data/esteban/pii/datasets/face_mine_right/images` (a real directory;
  the staging README calls it a symlink to `/data/esteban/face-mine-rview/frames`,
  which no longer exists). `src_key` and `src_frame`: the staged `boxes.jsonl`
  (WOR-194 conversion of the vendor drop). `size` and `md5`: computed from the
  staged bytes.
- v1 boxes: the vendor drop in `boxes/v1/job/output/`, asserted box for box equal to
  the staged `boxes.jsonl` by the build script, and checked once (2026-09-21) against
  the pixel boxes of `training/manifests/face_mine_right_labelv2.txt` (WOR-194).
- Not carried over: the staged `labels/` directory as such (its two files are the
  v1 `job/output/` copies) and the staged `boxes.jsonl` (its boxes are `boxes/v1/boxes.csv`,
  its provenance columns are in `frames.csv`).

## Handling

The frames are unblurred faces of real people. Frames stay on this box; usage is
materialized as views, never by copying images out of this directory.

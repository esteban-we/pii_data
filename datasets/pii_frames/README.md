# pii_frames

The first in-house training set: 10,249 raw fisheye frames (2328x1748, both eyes)
from 25 sessions and 27 chunks of our own delivery captures, sampled mostly every
30 frames along each chunk (8,383 of the 10,197 in-chunk steps are 30 frames; the rest
are 60 to 210). All 10,249 frames are train; every training manifest of the
fpv_face_pii repo from `train_labelv2.txt` onward carries this set as its `pii/` blocks
(`training/mk_mix_Z4.py` expects 10,249 frames and 19,579 boxes for it).

Staged locally as `/data/esteban/pii/datasets/pii_frames/<session>/chunk_<NNN>/<view>/f<NNNNNN>.jpg`
(a nested tree; no manifest, no marker files). The fpv_face_pii repo `data/README.md`
says the frames were extracted from the capture bucket; the extraction script, the
sampling rule and the date are not on file in either repo, so they are not stated
here. Rebuilt 2026-09-21 in this layout (PII-1315, PII-1338) by
`data/build_pii_frames_pii2.py` in the fpv_face_pii repo: the tree is flattened into one
`images/` directory, every image a byte copy of the staged file (md5-checked by
`build_pii_frames_pii2.py verify`).

## Layout

```
images/                 10,249 jpg, 3,459,387,268 bytes, all 2328x1748
frames.csv              one row per image: image, session, chunk, view, frame_idx, size, md5
split.csv               session, role; every one of the 25 sessions is train
boxes/v1/               the only label pass (19,579 boxes; what training uses)
```

Image names flatten the source path with the same rule as the OSS keys under
`oss://algorithm-datasets/pii/data/pii_frames/` (`data/oss_pii_keys.jsonl` in the
fpv_face_pii repo; the build asserts every name equals its OSS key basename and every
`local_path` equals the staged file):

```
<session>/chunk_<NNN>/<view>/f<NNNNNN>.jpg  ->  <session>_c<NNN>_<view>_f<NNNNNN>.jpg
e.g. 20260616_053528_BYEJKD/chunk_035/lview/f002100.jpg -> 20260616_053528_BYEJKD_c035_lview_f002100.jpg
```

`view` is the token exactly as on disk, `lview` (5,247 frames) or `rview` (5,002),
not normalized to `left`/`right`; this differs from the `eye` column of faceback_45,
face10k and face_mine_v1 and is deliberate (PII-1338), because the flattened image
names carry the same token and the OSS keys were built from it. `chunk` is the
three-digit string and `frame_idx` an integer, as in the other pii2 datasets. Two
chunks have only an `lview` directory (`20260621_194844_BTJPSP` chunks 007 and 025),
so there are 52 (session, chunk, view) groups over 27 chunks. `frames.csv` is sorted
by image and is the source of truth for which images exist. `size` (bytes) and `md5`
are the image file's own; they come from the pii-data repo `frames.csv` (dataset
`pii_frames`, hashed at OSS upload) and the build asserts `size` against the file on
disk and `md5` against the OSS key map. No frame time (ms) or capture path is recorded
for this set.

`boxes/v1/` holds:

```
frames.csv        image, reviewed   (1 on every frame)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 fisheye frame, one decimal)
README.md         where the pass comes from and what is known of its rules
```

The box file is a full replacement over all 10,249 images. Rows are sorted by
(image, x1, y1); an image with no box has no row. `ignore=1` marks a region to load
as `gt_bboxes_ignore` rather than a positive; v1 has 2,704 such rows.

| version | pass | boxes | ignore rows | images with a box | reviewed frames |
|---|---|---:|---:|---:|---:|
| v1 | original labels, recovered from the training manifests | 19,579 | 2,704 | 8,761 | 10,249 (all) |

1,488 frames carry no box at all. They are treated as labeled negatives (face-free
frames) because the training manifests list them with an empty block and the trainer
loads them as such; whether a labeler actually looked at each of them is unverified
(see `boxes/v1/README.md`). `training/patch_negweight.py` in the fpv_face_pii repo
describes the in-house fisheye annotation as roughly 10 percent incomplete, which is
why `neg_weight_pii=0.5` exists in the training recipes.

## Split

All 25 sessions are `train`. There is no eval slice of this dataset; the eval benches
are gt_bench and the eval splits of face_mine_v1 and faceback_45.

## Provenance

- images: `/data/esteban/pii/datasets/pii_frames/` (nested tree, read-only source);
- `frames.csv` fields: parsed from the source path; `size`, `md5`: pii-data `frames.csv`
  (dataset `pii_frames`), cross-checked against `data/oss_pii_keys.jsonl`;
- v1 boxes: pii-data `datasets/pii_frames/boxes/v1.csv` (md5
  3b8cb6d1a325aeefb59987e47fac588c), copied byte for byte; see `boxes/v1/README.md`.
- Nothing dropped: the source tree holds only the 10,249 jpg files.

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out
of this directory.

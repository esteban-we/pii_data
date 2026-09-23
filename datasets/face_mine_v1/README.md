# face_mine_v1

Left-eye (`vst_left`) frames mined from about 9k production sessions: 62,587 frames
over 8,995 sessions, raw fisheye 2328x1748, one frame per selection of the miner
`face_mine/two_model@1` (armW face model plus the CrowdHuman head model). Staged
2026-09-01 in `/data/esteban/pii/datasets/face_mine_v1`; rebuilt 2026-09-21 in this
layout (PII-1315, PII-1318) by `data/build_facemine_pii2.py --eye left` in the
fpv_face_pii repo. Every image is a byte copy of the staged file (md5 checked by
`build_facemine_pii2.py verify --eye left`). The right-eye frames of the same
sessions are the separate dataset `face_mine_right` (PII-1319).

## Layout

```
images/                 62,587 jpg, 28,557,034,942 bytes, all 2328x1748, names <session>_c<chunk>_f<frame_idx>.jpg
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, src_key, src_frame
split.csv               session, role (train | eval); the split is per session
boxes/v1/               the human pass over every frame (2026-09-01)
boxes/v2/               v1 plus the relabel of the 3,600 armY false-positive eval frames (2026-09-10)
boxes/v3/               pending: armAG34-over-human relabel of every frame (PII-1273); prelabels only
```

Image names carry no eye; `eye` is `left` on every row. `frame_idx` is the frame
number inside the chunk video (`src_key`, the source video key; `src_frame` equals
`frame_idx`). `size` is the byte count of the image and `md5` the md5 of its bytes;
both equal the pii-data `frames.csv` row (`dataset=face_mine_v1`) and the OSS ETag.
`frames.csv` is sorted by image and is the source of truth for which images exist.

Each `boxes/vN/` holds:

```
frames.csv        image, reviewed   (1 = the frame was sent in this pass, even if it came back unchanged)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 frame, one decimal)
README.md         why the pass exists and its labeling rules
job/prelabels/    what the vendor started from (byte copies)
job/output/       what the vendor returned (byte copies)
```

A box file is a full replacement over all 62,587 images, never a diff: a consumer
reads one version. Rows are sorted by (image, x1, y1); an image with no box has no
row. Frames not sent in a pass keep the previous version's rows byte for byte.
`ignore=1` would mark a region to load as `gt_bboxes_ignore` rather than a
positive; no version of this dataset has an ignore row. A pending version (still at
the vendor) has `job/prelabels/` and a README only; its `frames.csv` and `boxes.csv`
appear when the output lands.

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 76,859 | 41,079 | 62,587 (all) |
| v2 | 81,563 | 41,730 | 3,600 (eval frames with an armY false positive at 0.5) |
| v3 | pending | pending | (62,587 sent) |

Train rows (61,093 boxes on the 49,833 train frames) are identical in v1 and v2; the
versions differ only on the 3,600 reviewed eval frames.

## Train/eval split (v1, frozen)

`split.csv`: 7,137 train sessions (49,833 frames) and 1,858 eval sessions (12,754
frames), split by session, never by frame (frames within a session are temporal
near-duplicates). Sessions were stratified on capture month, machine-human
disagreement rate and small-face share; seed 20260902, generator
`mining/face_mine_split.py`. The lists are `data/splits/{train,eval}_sessions_v1.txt`
in the repo (byte-equal to the staged `splits/` files).

Rules:

- The eval sessions must never be trained on, from this or any later dataset; a
  session listed as eval here is excluded from training everywhere (faceback_45 and
  face_mine_right inherit the assignment).
- The split is frozen; additions to eval land as new named slices.

## Provenance

- images: `/data/esteban/pii/datasets/face_mine_v1/images`; `src_key` and
  `src_frame`: the staged `boxes.jsonl` (the miner's output, also the v1 prelabels);
  `size` and `md5`: computed from the staged bytes and asserted equal to pii-data.
- v1 boxes: `/data/esteban/pii/face-mine_labeled.csv` (byte-equal to pii-data
  `datasets/face_mine_v1/boxes/v1.csv`); v2 boxes: the vendor jsonl in
  `boxes/v2/job/output/` (byte-equal to pii-data `boxes/v2.csv`, which is not yet
  committed in pii-data and not yet promoted as the eval reference, PII-142).
- Not carried over: `_download.log`, `labels/eval_v1.json` (a derived eval bundle
  of the v1 boxes, rebuilt by `evaluation/face_mine_bundle.py`), the
  `face_mine_v2/images` copies (the same bytes as `images/`), and the in-tool relabel
  log of the pages site (out of scope, PII-1315).

## Handling

The frames are unblurred faces of real people. Frames stay on this box; usage is
materialized as views, never by copying images out of this directory.

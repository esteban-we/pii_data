# gt_bench_full

The full face-PII ground-truth bench: 9,636 frames from the 9 labelled chunks (one chunk
in each of 9 sessions, recorded 2026-05-15 to 2026-05-28), left eye only (`vst_left`),
raw fisheye 2328x1748, with human face boxes on every frame. It is an evaluation set: no
frame of these 9 sessions may be trained on, from this or any other dataset.

The 889 names of `gt_bench_sparse` are a subset of these 9,636, but the bytes differ:
this set is a re-extraction of every labelled frame from the source videos (2026-08-29),
the sparse set is the previous owner's encode, and 0 of the 889 shared names have an
equal md5. The two are kept as separate datasets for that reason (user decision
2026-09-21, PII-1339).

The labels (`labels/gt_bundle.json`) were delivered by the previous owner 2026-08-27 as
`oss://algorithm-datasets/face_pii/gt_bench_v1/` and staged 2026-08-28 as
`/data/esteban/pii/datasets/gt_bench_v1/` (PII-516 checked them md5-equal to the OSS
ETag). The delivery carried no full-frame images ("reproducible by decoding the source
videos"); `evaluation/stage_full_bench.py` (repo, run 2026-08-29) mapped each bundle
chunk-uuid to its (session, chunk) by matching boxes against the sparse file, wrote
`labels/uuid_map.json`, downloaded `oss://we-fpv-sh-ns/<session>/chunk_<chunk>/vst_left/vst_left_video.mp4`
and decoded the labelled frame indices with ffmpeg (`select=eq(n,idx)`, `-q:v 2`) into
`images_full/<session>_<chunk>/f<idx>.jpg`. Rebuilt 2026-09-21 in this layout
(PII-1315, PII-1339) by `data/build_gt_bench_pii2.py --set full`; every image is a byte
copy of the staged file (md5-checked by `build_gt_bench_pii2.py --set full verify`).

## Layout

```
images/                 9,636 jpg, 3,201,611,182 bytes, all 2328x1748
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, src_key, src_frame
split.csv               session, role; every one of the 9 sessions is eval
boxes/v1/               the human labels as delivered (20,054 boxes on all 9,636 frames)
derivation/             stage_full_bench.py (the extraction script) and gt_bench_v1_README.md (the delivered README), byte copies
```

Image name: `<session>_<chunk>_f<frame_idx>.jpg`, `chunk` three digits and `frame_idx`
six digits: the staged `images_full/<session>_<chunk>/f<frame_idx>.jpg` flattened with
the same rule the OSS upload used (`pii/data/gt_bench_full/`, `data/oss_pii_keys.README.md`);
the build asserts the flattening against `data/oss_pii_keys.jsonl`. `frames.csv` is sorted
by image and is the source of truth for which images exist; `chunk` is stored as the
three-digit string, `frame_idx` as an integer, `eye` is `left` on every row. `size`
(bytes) and `md5` are the image file's own; they come from the pii-data repo `frames.csv`
(dataset `gt_bench_full`, hashed at OSS upload) and the build asserts `size` against the
file on disk. `src_key` is the source video object in `we-fpv-sh-ns` and `src_frame` the
frame index decoded from it (equal to `frame_idx`); no frame time in ms is recorded.

`boxes/v1/` holds:

```
frames.csv        image, reviewed   (1 on every frame)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 fisheye frame, one decimal)
README.md         where the labels come from and the rules they follow
job/output/       gt_bundle.json and uuid_map.json (byte copies)
```

Rows are sorted by (image, x1, y1); every image has at least one row (the bundle lists
a frame only when it holds a face, asserted by the build). `ignore` is 0 on every row.
There is no `job/prelabels/`: nothing is known about what the labelers started from.

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 20,054 | 9,636 (all) | 9,636 (all) |

Per chunk: 27 to 2,230 frames; the delivered README says chunks were labelled every 5th
frame across the whole chunk, and PII-516 measured a constant stride of 5 in every chunk.

## Notes and open points

- `gt_bundle.json` holds 10 chunk-uuids; the 10th (`019e93df-...`) has 0 frames and no
  entry in `uuid_map.json`. The 9 mapped uuids cover exactly the 9,636 images (asserted).
- PII-130 (unresolved): on 5 of the 889 frames shared with `gt_bench_sparse`, the sparse
  file carries fewer boxes than this bundle. This set keeps the bundle as its truth; the
  build asserts that the disagreement is exactly those 5 frames (names in PII-130).
- The delivered README says "9,636 frames / 20,054 boxes across 10 sequences" and, in
  another paragraph, that the full labels point at 20,752 frames; the second figure does
  not match the bundle on disk and is left as stated (`derivation/gt_bench_v1_README.md`).
- PII-87 (backlog) states that the gt_bench_v1 images on this box are a re-extraction with
  pixel differences of mean 0.8 to 1.4/255 from the original; not checked here.
- The delivered README states the bench is left eye only and that 16% of faces appear in
  the right eye alone (not re-measured here).

## Provenance

- images: `/data/esteban/pii/datasets/gt_bench_v1/images_full/` (byte copies; md5 equal
  to pii-data `frames.csv` and to the OSS ETags of `pii/data/gt_bench_full/`).
- v1 boxes: `gt_bench_v1/labels/gt_bundle.json` through `labels/uuid_map.json`,
  coordinates already pixel xyxy, written at one decimal; byte-equal to pii-data
  `datasets/gt_bench_full/boxes/v1.csv` (md5 1a479d33eeb9488e790b2346ed2f254f).
- `split.csv`: the 9 sessions are tagged `gt_bench_v1` / `eval` in `data/episode_usage.csv`
  (asserted by the build).
- Not carried over: `gt_bench_v1/.complete` (an empty staging marker) and
  `images_sparse/` (that is `gt_bench_sparse`). Every file of `gt_bench_v1/labels/`
  is under one of the two sets' `boxes/v1/job/output/`.

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out
of this directory.

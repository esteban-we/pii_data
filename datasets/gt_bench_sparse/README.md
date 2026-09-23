# gt_bench_sparse

The sparse slice of the face-PII ground-truth bench: 889 frames from the 9 labelled
chunks (one chunk in each of 9 sessions, recorded 2026-05-15 to 2026-05-28), left eye
only (`vst_left`), raw fisheye 2328x1748, with human face boxes. It is an evaluation
set: no frame of these 9 sessions may be trained on, from this or any other dataset.
Most historical recall figures of the project were scored on this slice.

The same frame names are a subset of `gt_bench_full`, but the bytes are not the same:
this set holds the previous owner's encode of the 889 frames, `gt_bench_full` a later
re-extraction of all 9,636 labelled frames, and 0 of the 889 shared names have an
equal md5. The two are kept as separate datasets for that reason (user decision
2026-09-21, PII-1339).

Delivered by the previous owner 2026-08-27 as `oss://algorithm-datasets/face_pii/gt_bench_v1/`
(`README.md`, `images_sparse/`, `labels/gt_bundle.json`, `labels/gt_eval_sparse.json`) and
staged 2026-08-28 as `/data/esteban/pii/datasets/gt_bench_v1/` (PII-516 checked the label
files md5-equal to the OSS ETags). Rebuilt 2026-09-21 in this layout (PII-1315, PII-1339)
by `data/build_gt_bench_pii2.py --set sparse` in the fpv_face_pii repo; every image is a
byte copy of the staged `images_sparse/` file (md5-checked by `build_gt_bench_pii2.py
--set sparse verify`).

## Layout

```
images/                 889 jpg, 625,250,284 bytes, all 2328x1748
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, src_path
split.csv               session, role; every one of the 9 sessions is eval
boxes/v1/               the human labels as delivered (1,541 boxes on 886 frames)
derivation/             gt_extract.py, the script that produced this slice (byte copy of the repo file)
```

Image name: `<session>_<chunk>_f<frame_idx>.jpg`, `chunk` three digits and `frame_idx`
six digits (the same flat name the OSS upload uses for both gt_bench sets). `frames.csv`
is sorted by image and is the source of truth for which images exist; `chunk` is stored
as the three-digit string, `frame_idx` as an integer, `eye` is `left` on every row.
`size` (bytes) and `md5` are the image file's own; they come from the pii-data repo
`frames.csv` (dataset `gt_bench_sparse`, hashed at OSS upload) and the build asserts
`size` against the file on disk. `src_path` is the path the frame had on the previous
owner's box, as recorded in `gt_eval_sparse.json`; no frame time in ms is recorded
anywhere for this set.

`boxes/v1/` holds:

```
frames.csv        image, reviewed   (1 on every frame)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 fisheye frame, one decimal)
README.md         where the labels come from and the rules they follow
job/output/       gt_eval_sparse.json, the label file as delivered (byte copy)
```

Rows are sorted by (image, x1, y1); an image with no box has no row (3 frames here).
`ignore` is 0 on every row. There is no `job/prelabels/`: nothing is known about what
the labelers started from.

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 1,541 | 886 | 889 (all) |

## How the slice was chosen (PII-515, PII-516)

`derivation/gt_extract.py` (previous owner, repo commit fcbf751, 2026-08-28): per chunk,
the labelled frames sorted by index, `step = len // 120`, the first 120 of `frames[::step]`;
then boxes with long side under 40 px are dropped (frames are kept even if that empties
them). PII-516 re-ran the rule over `gt_bundle.json` and reproduced the 889 names and the
1,541 boxes exactly. Per chunk the slice holds 120 frames, except the two chunks with fewer
labelled frames than that (27 and 22, taken whole).

Two biases stated in the delivered README (`gt_bench_full/derivation/gt_bench_v1_README.md`;
not re-measured here): the slice holds 941 of the 3,044 distinct faces (30.9%), keeping
the faces that linger and dropping the ones that flash past, so recall measured on it
is biased optimistic; and it is the left eye only.

## Known differences from gt_bench_full (PII-130, unresolved)

On 5 of the 889 shared frames `gt_eval_sparse.json` carries fewer boxes than
`gt_bundle.json` gives the same frame (1 vs 2, three cases of 0 vs 1, and 3 vs 4). The
40 px floor explains at most the 0-vs-1 cases. Which file is the truth has not been
decided; this set keeps the sparse file as its own truth, `gt_bench_full` keeps the bundle,
and `build_gt_bench_pii2.py` asserts that the disagreement is exactly those 5 frames.
The names are listed in PII-130 and reproduced by the build script, not here.

PII-87 (backlog) states that the gt_bench_v1 images on this box are a re-extraction, not
byte-identical to the original delivery, with pixel differences of mean 0.8 to 1.4/255;
that claim was not checked here.

## Provenance

- images: `/data/esteban/pii/datasets/gt_bench_v1/images_sparse/` (byte copies; md5 equal
  to pii-data `frames.csv` and to the OSS ETags of `pii/data/gt_bench_sparse/`).
- v1 boxes: `gt_bench_v1/labels/gt_eval_sparse.json`, coordinates already pixel xyxy,
  written at one decimal; byte-equal to pii-data `datasets/gt_bench_sparse/boxes/v1.csv`
  (md5 de1d589fe2e0c81f1181a74ef69ed2d0).
- `split.csv`: the 9 sessions are tagged `gt_bench_v1` / `eval` in `data/episode_usage.csv`
  (asserted by the build).
- Not carried over: `gt_bench_v1/.complete` (an empty staging marker). The delivered
  `README.md` of gt_bench_v1 and `stage_full_bench.py` are under `gt_bench_full/derivation/`.

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out
of this directory.

# faceback_45

Fail-set frames for the face-blur (PII) work: 334 sessions, 369 chunks, both
eyes (left and right), one frame every 60 frames (2 s at 30 fps), recorded
2026-04-19 to 2026-08-23. Extracted on the source box from `/root/failset-frames`
(the `src_path` column of `frames.csv`).

Staged 2026-09-09 from `oss://we-vlm-annotation-data-sh/faceback-45/` into
`/data/esteban/pii/datasets/faceback_45` (verified at staging: 84,954 files, byte
total equal to the manifest sum, 20-image md5 sample plus all non-image objects
equal to the OSS etags). Rebuilt 2026-09-21 in this layout (PII-1315, PII-1316) by
`data/build_faceback45_pii2.py` in the fpv_face_pii repo; every image is a byte
copy of the staged file (md5-checked by `build_faceback45_pii2.py verify`).

## Layout

```
images/                 84,954 jpg, 42,477 per eye, 56,159,427,855 bytes, all 2328x1748
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, t_ms, src_path
split.csv               session, role (train | eval); the split is per session
boxes/v1/               the human labels of the vendor pass (2026-09-04..06)
boxes/v2/               v1 plus the HQ vendor pass over the eval split (2026-09-11)
boxes/v3/               v2 plus the round-2 review of 4,023 eval frames (2026-09-16)
```

Image name: `<session>_c<chunk>_<eye>_f<frame_idx>.jpg` with `frame_idx` zero padded
to six digits; `frames.csv` is sorted by image and `frames.csv` is the source of truth
for which images exist. `size` (bytes) and `md5` are the image file's own; they come
from the pii-data repo `frames.csv` (dataset `faceback_45`, hashed at OSS upload) and
`build_faceback45_pii2.py verify` checks every image against them, so the check
outlives the staging copy.

Each `boxes/vN/` holds:

```
frames.csv        image, reviewed   (1 = the frame was sent in this pass, even if it came back unchanged)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the image's own 2328x1748 space, one decimal)
README.md         why the pass exists and its labeling rules
job/prelabels/    what the vendor started from (byte copies)
job/output/       what the vendor returned (byte copies)
```

A box file is a full replacement over all 84,954 images, never a diff: a consumer
reads one version. Rows are sorted by (image, x1, y1); an image with no box has no
row. Frames not sent in a pass keep the previous version's rows byte for byte.
`ignore=1` would mark a region to load as `gt_bboxes_ignore` rather than a
positive; no version of this dataset has an ignore row.

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 95,006 | 35,311 | 84,954 (all) |
| v2 | 99,015 | 35,951 | 16,940 (the eval split) |
| v3 | 103,453 | 36,498 | 4,023 (eval frames with an armAE34 addition) |

Train rows (75,999 boxes on the 68,014 train frames) are identical in all three
versions; the versions differ only on the eval frames.

## Train/eval split (v1, frozen; WOR-102)

`split.csv`: 267 train sessions (68,014 frames) and 67 eval sessions (16,940 frames,
8,470 per eye), split by session, never by frame; both eyes and every chunk of a
session are on one side. The eval slice is the named slice `eval_faceback_v1`.
Sessions shared with face_mine_v1 keep their face_mine_v1 side (7 eval, 26 train
sessions); the remaining 301 were assigned by the seeded search in
`mining/faceback_split.py` (seed 20260909), see the WOR-102 note in the staging
README for the balance table.

Rules:

- The eval sessions must never be trained on, from this or any later dataset; a
  session listed as eval here or in face_mine_v1 is excluded from training everywhere.
- The split is frozen; later additions land as new named slices.

## Provenance

- images, `frames.csv` (from `upload_manifest.jsonl`, `key` dropped), `split.csv`
  (from `splits/faceback_{train,eval}_sessions_v1.txt`), v1 boxes and v1 job files:
  `/data/esteban/pii/datasets/faceback_45/`
- v2 and v3 boxes: pii-data repo `datasets/faceback_hq/boxes/v1.csv` and `v2.csv`;
  see the per-version README for the job files.
- Not carried over: `_download.log`, `_launch_time.txt` (staging date is above), the
  recognizable-face subsets (`rec`, `rec2`) and the in-tool relabel log of the
  pages site (out of scope, PII-1315).

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out
of this directory.

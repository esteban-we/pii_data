# wider_face

The public WIDER FACE training split (Yang et al., 2016), the general-face floor mixed
roughly 1:1 with our own fisheye data in the early training arms. 12,879 JPEGs across
the 61 official event categories; the official split has 12,880 and the one image
absent here is `58--Hockey/58_Hockey_icehockey_puck_58_909.jpg`, which is missing from
the staged copy, from the pii-data frame table and from every training manifest of the
fpv_face_pii repo (why it was dropped is not recorded). `docs/FACE_DATASETS.md` in the
fpv_face_pii repo gives the staging source as `oss://algorithm-datasets/face_pii/wider_face/`.

Staged locally as `/data/esteban/pii/datasets/wider_face/WIDER_train/images/<scene>/<name>.jpg`.
Rebuilt 2026-09-21 in this layout (PII-1315, PII-1337) by `data/build_wider_pii2.py --set wider_face`
in the fpv_face_pii repo: the scene directories are flattened into one `images/` directory
(basenames are unique across scenes; the build asserts it), every image a byte copy of the
staged file (md5-checked by `build_wider_pii2.py --set wider_face verify`).

Two synthetic derivatives of this set exist as separate datasets, `wider_fisheye_target`
and `wider_fisheye_fill` (the same 12,879 images re-projected through our fisheye camera,
same basenames).

## Layout

```
images/                 12,879 jpg, 1,473,882,349 bytes, width 1024, height 171 to 9108
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, scene, width, height, src_path
split.csv               session, role; every one of the 61 scenes is train
boxes/v1/               the public WIDER FACE annotation, converted (159,390 boxes, 2,399 ignore)
```

WIDER has no sessions, chunks, eyes or frame indices: `session` and `scene` both carry the
event directory name (for example `0--Parade`, 75 to 995 images per scene), and `chunk`,
`eye`, `frame_idx` are empty. `frames.csv` is sorted by image and is the source of truth for
which images exist. `size` (bytes) and `md5` are the image file's own; they come from the
pii-data repo `frames.csv` (dataset `wider_face`) and the build asserts `size` against the
file on disk. `width`, `height` come from the `wider_train_labelv2.txt` header lines and
are asserted equal to pii-data. `src_path` is the staged path the image was copied from.

`boxes/v1/` holds:

```
frames.csv        image, reviewed   (1 on every image)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the image's own space, one decimal)
README.md         where the boxes come from and the conversion rules
```

There is no `job/` directory: the labels are the public annotation, no vendor pass was run.
Rows are sorted by (image, x1, y1); an image with no box has no row (4 images).
`ignore=1` marks a region to load as `gt_bboxes_ignore` rather than a positive.

| version | source | boxes | ignore rows | images with a box | reviewed |
|---|---|---:|---:|---:|---:|
| v1 | official `wider_face_train_bbx_gt.txt` via `training/wider_conv.py` | 159,390 | 2,399 | 12,875 | 12,879 (all) |

## Split

All 61 scenes are `train`; WIDER is never evaluated on. Manifests of the fpv_face_pii
repo that carry the `wider/` blocks: `train_G`, `train_H`, `train_I`, `train_J`, `train_L`,
`train_M`, `train_N`, `train_V`, `train_W`, `train_X` and `train_labelv2.txt`. The `train_Z*`
manifests dropped WIDER after the ablation in PII-31 found the no-WIDER arm ahead on recall
(PII-31, PII-108). The two fisheye derivatives were the other arms of that ablation
(`train_W_fe.txt`, `train_W_fe2.txt` under the mix_ds views tree on this box, not in the repo).

## Licence

WIDER FACE is published for non-commercial research use. The staged copy's README notes that
this was never formally cleared for the commercial detector it trained; raise it rather than
assume it is settled.

## Provenance

- images: `/data/esteban/pii/datasets/wider_face/WIDER_train/images/<scene>/`;
- `size`, `md5`: pii-data `frames.csv`; `width`, `height`, `scene`: the
  `wider_train_labelv2.txt` headers, cross-checked against the directory tree;
- v1 boxes: pii-data `datasets/wider_face/boxes/v1.csv`, byte for byte, itself the
  `wider_train_labelv2.txt` content re-sorted (the build asserts the two equal as per-image
  box sets).
- Not carried over: `wider_face_split/` (the official annotation files, a public download),
  `wider_train_labelv2.txt` (regenerable with `training/wider_conv.py`), `.complete`
  (staging marker). The two source zip archives were never staged.

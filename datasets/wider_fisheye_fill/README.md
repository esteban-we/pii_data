# wider_fisheye_fill

The 12,879 images of `wider_face` (the public WIDER FACE training split) synthetically
re-projected through our head-mounted fisheye camera (rgb-left, Kannala-Brandt, 2328x1748),
with the GT boxes mapped along, zoom policy "fill". Generated 2026-08-30 by
`training/warp_wider_fisheye.py --version 2` in the fpv_face_pii repo (run then from
`/data/esteban/pii/tools/warp_wider_fisheye.py`; CPU, 48 processes, 113 s per the source
README) into `/data/esteban/pii/datasets/wider_fisheye_v2/images/<event>/<name>.jpg`. The sister set
`wider_fisheye_target` is the same generator with the "target" zoom policy
(`--version 1`); same per-image random stream, so hfov,
axis placement and roll per image are identical between the two and only the zoom differs.

Rebuilt 2026-09-21 in this layout (PII-1315, PII-1337) by
`data/build_wider_pii2.py --set wider_fisheye_fill`: the event directories are flattened into
one `images/` directory (basenames unique, asserted), every image a byte copy of the staged
file (md5 computed from the staged file at build time and checked by
`build_wider_pii2.py --set wider_fisheye_fill verify`).

## Layout

```
images/                 12,879 jpg, 5,371,830,585 bytes, all 2328x1748, JPEG quality 92
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, scene, width, height, src_path
split.csv               session, role; every one of the 61 scenes is train
boxes/v1/               the warped WIDER boxes (152,235 boxes, no ignore rows)
derivation/             byte copies of the generator's sidecars (see below)
```

WIDER has no sessions, chunks, eyes or frame indices: `session` and `scene` both carry
the WIDER event directory name (for example `0--Parade`), and `chunk`, `eye`,
`frame_idx` are empty. Image basenames are the WIDER names, so a row here and the row of
the same `image` in `wider_face` are the same source photograph. `frames.csv` is sorted
by image and is the source of truth for which images exist. `size` and `md5` are the
image file's own, computed from the staged file. `width`, `height` are 2328x1748
everywhere (from the labelv2 headers). `src_path` is the staged path the image was copied
from.

`boxes/v1/` holds:

```
frames.csv        image, reviewed   (1 on every image)
boxes.csv         image, x1, y1, x2, y2, ignore   (pixel xyxy in the 2328x1748 frame, one decimal)
README.md         how the boxes were mapped and the drop rules
```

There is no `job/` directory: the boxes are machine-derived from the public annotation, no
vendor pass was run. Rows are sorted by (image, x1, y1); an image with no surviving box has
no row (69 images). No row has `ignore=1`: the generator read the first four fields of
every WIDER box line, so the 2,399 WIDER ignore boxes were warped as ordinary positives
(or dropped by the visibility rules); the flag was not carried.

| version | source | boxes | ignore rows | images with a box | reviewed |
|---|---|---:|---:|---:|---:|
| v1 | `wider_fisheye_v2/wider_fisheye_labelv2.txt` (generator output) | 152,235 | 0 | 12,810 | 12,879 (all) |

`derivation/` (byte copies from `/data/esteban/pii/datasets/wider_fisheye_v2/`, md5-checked by verify):

```
wider_fisheye_params.json    the generator config and the per-image virtual camera (hfov, zoom, axis, roll, f_eff)
wider_fisheye_stats.json     the generator's summary numbers (box counts, size percentiles, zoom binding)
wider_fisheye_labelv2.txt    the generator's label output, prefix `wider_fe2/`; boxes.csv is this file converted
preview/                     12 before/after panels with boxes, two per {tiny, mid, big face} x {axis centre, periphery}
```

## Camera and zoom policy (from the source README and the script header; not re-measured here)

Camera: `vst_fisheye_calib.json["rgb-left"]`, fx = fy = 872.84644, cx = 1151.2087,
cy = 871.10121, k1..k4 = -0.11757304, 0.28808356, -0.23854244, 0.065945865, `cv2.fisheye`
convention; the real image circle is reproduced (full to r = 1100 px, black from 1190 px).
Each WIDER image is treated as a pinhole image with hfov ~ U[50, 90] deg, pointed at a random
canvas point with a roll in [-10, 10] deg, rendered by inverse mapping; everything outside
the source, beyond 65 deg from the virtual axis, or outside the image circle is black.

Zoom policy "fill" (version 2): zoom as large as the max-span cap allows (130 x 124 deg),
reduced so that the median face long side at the fisheye centre stays <= 300 px (face cap),
and never below a 70 deg horizontal min-span; when the face cap and the min-span conflict
the min-span wins (2,015 images, 15.6%, flagged `min_span_conflict` in the params json, with
rendered median faces of 312/449/755 px p10/50/90). Result (stats json): per-image median
face p10/50/90 = 49.6/183.0/388.3 px, effective hfov p10/50/90 = 70/124/130 deg, median
black-pixel fraction 0.39 on a 200-image sample (version 1: 0.79).

## Split

All 61 scenes are `train`. This set was one arm of the WIDER ablation (PII-31): manifest
`train_W_fe2.txt` (= `train_W.txt` with every `wider/` block replaced by the `wider_fe2/` block; under
the mix_ds views tree on this box, not in the repo). The `train_Z*` manifests dropped WIDER
in every form after that ablation (PII-31, PII-108). PII-31 recorded the fill arm as
recall-neutral against train_W and the target arm as worse.

## Licence

A derivative of WIDER FACE, which is published for non-commercial research use; see the
`wider_face` README.

## Provenance

- images: `/data/esteban/pii/datasets/wider_fisheye_v2/images/<event>/`;
- `scene`, `width`, `height`: the `wider_fisheye_labelv2.txt` headers, cross-checked
  against the directory tree; `size`, `md5`: computed from the staged file;
- v1 boxes: `wider_fisheye_labelv2.txt` converted (see `boxes/v1/README.md`);
- `derivation/`: the generator sidecars listed above.
- Not carried over: nothing else was in the staged directory besides its README.

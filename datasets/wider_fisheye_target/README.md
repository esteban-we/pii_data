# wider_fisheye_target

The 12,879 images of `wider_face` (the public WIDER FACE training split) synthetically
re-projected through our head-mounted fisheye camera (rgb-left, Kannala-Brandt, 2328x1748),
with the GT boxes mapped along, zoom policy "target". Generated 2026-08-30 by
`training/warp_wider_fisheye.py --version 1` in the fpv_face_pii repo (run then from
`/data/esteban/pii/tools/warp_wider_fisheye.py`; CPU, 48 processes, 125 s per the source
README) into `/data/esteban/pii/datasets/wider_fisheye/images/<event>/<name>.jpg`. The sister set
`wider_fisheye_fill` is the same generator with the "fill" zoom policy
(`--version 2`); same per-image random stream, so hfov,
axis placement and roll per image are identical between the two and only the zoom differs.

Rebuilt 2026-09-21 in this layout (PII-1315, PII-1337) by
`data/build_wider_pii2.py --set wider_fisheye_target`: the event directories are flattened into
one `images/` directory (basenames unique, asserted), every image a byte copy of the staged
file (md5 computed from the staged file at build time and checked by
`build_wider_pii2.py --set wider_fisheye_target verify`).

## Layout

```
images/                 12,879 jpg, 3,295,707,950 bytes, all 2328x1748, JPEG quality 92
frames.csv              one row per image: image, session, chunk, eye, frame_idx, size, md5, scene, width, height, src_path
split.csv               session, role; every one of the 61 scenes is train
boxes/v1/               the warped WIDER boxes (153,203 boxes, no ignore rows)
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
no row (67 images). No row has `ignore=1`: the generator read the first four fields of
every WIDER box line, so the 2,399 WIDER ignore boxes were warped as ordinary positives
(or dropped by the visibility rules); the flag was not carried.

| version | source | boxes | ignore rows | images with a box | reviewed |
|---|---|---:|---:|---:|---:|
| v1 | `wider_fisheye/wider_fisheye_labelv2.txt` (generator output) | 153,203 | 0 | 12,812 | 12,879 (all) |

`derivation/` (byte copies from `/data/esteban/pii/datasets/wider_fisheye/`, md5-checked by verify):

```
wider_fisheye_params.json    the generator config and the per-image virtual camera (hfov, zoom, axis, roll, f_eff)
wider_fisheye_stats.json     the generator's summary numbers (box counts, size percentiles, zoom binding)
wider_fisheye_labelv2.txt    the generator's label output, prefix `wider_fe/`; boxes.csv is this file converted
preview/                     12 before/after panels with boxes, two per {tiny, mid, big face} x {axis centre, periphery}
```

## Camera and zoom policy (from the source README and the script header; not re-measured here)

Camera: `vst_fisheye_calib.json["rgb-left"]`, fx = fy = 872.84644, cx = 1151.2087,
cy = 871.10121, k1..k4 = -0.11757304, 0.28808356, -0.23854244, 0.065945865, `cv2.fisheye`
convention; the real image circle is reproduced (full to r = 1100 px, black from 1190 px).
Each WIDER image is treated as a pinhole image with hfov ~ U[50, 90] deg, pointed at a random
canvas point with a roll in [-10, 10] deg, rendered by inverse mapping; everything outside
the source, beyond 65 deg from the virtual axis, or outside the image circle is black.

Zoom policy "target" (version 1): a per-image zoom magnifies the image so that its median
face long side lands on a target drawn from a log-normal fitted to the real face-size
distribution (p10/50/90 = 45/72/124 px), clipped to [0.25, 4] and capped so the rendered
image spans at most 130 x 124 deg. Result (stats json): per-image median face p10/50/90 =
39.5/73.9/139.7 px, effective hfov p10/50/90 = 21/56/129 deg, and most images shrink to a
small patch on a mostly black canvas (the source README measured a median 79% black pixels
per image), which motivated the "fill" policy of the sister set.

## Split

All 61 scenes are `train`. This set was one arm of the WIDER ablation (PII-31): manifest
`train_W_fe.txt` (= `train_W.txt` with every `wider/` block replaced by the `wider_fe/` block; under
the mix_ds views tree on this box, not in the repo). The `train_Z*` manifests dropped WIDER
in every form after that ablation (PII-31, PII-108). PII-31 recorded the fill arm as
recall-neutral against train_W and the target arm as worse.

## Licence

A derivative of WIDER FACE, which is published for non-commercial research use; see the
`wider_face` README.

## Provenance

- images: `/data/esteban/pii/datasets/wider_fisheye/images/<event>/`;
- `scene`, `width`, `height`: the `wider_fisheye_labelv2.txt` headers, cross-checked
  against the directory tree; `size`, `md5`: computed from the staged file;
- v1 boxes: `wider_fisheye_labelv2.txt` converted (see `boxes/v1/README.md`);
- `derivation/`: the generator sidecars listed above.
- Not carried over: nothing else was in the staged directory besides its README.

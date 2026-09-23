# wider_face boxes v1: the public WIDER FACE annotation

## Why

The only label pass this set has: the official training annotation
(`wider_face_split/wider_face_train_bbx_gt.txt`) converted by `training/wider_conv.py` in the
fpv_face_pii repo into the labelv2 file every `wider/` training block was built from. No
vendor ever touched these frames, so there is no `job/` directory.

## Conversion rules (`training/wider_conv.py`)

- Official boxes are `x y w h`; written as `x1 y1 x2 y2` with `x2 = x + w`, `y2 = y + h`,
  one decimal, no rounding beyond that (the official values are integers).
- A box with the official `invalid` flag set, or with `w <= 0` or `h <= 0`, becomes
  `ignore=1` instead of being dropped, so it never trains as background. The official
  `blur`, `occlusion`, `pose`, `expression` and `illumination` fields are quality flags and
  are not carried.
- Images the official file lists with zero faces (one all-zero dummy line) have no box row.
- Keypoints are absent (written as the RetinaFace `-1` marker in labelv2; not in this file).

## Counts

`boxes.csv`: 159,390 rows on 12,875 images; 4 images have no row. 2,399 rows have
`ignore=1`. `frames.csv`: `reviewed=1` on all 12,879 images.

The file is byte for byte the pii-data repo file `datasets/wider_face/boxes/v1.csv`
(md5 ae71452bb736650de9ebc81c8bc0cf84). `build_wider_pii2.py --set wider_face` pins that
md5, asserts the rows equal the `wider_train_labelv2.txt` blocks as per-image box sets (five
images list boxes with equal x1,y1 in a different order, nothing else differs), and checks the
per-image counts against pii-data `frames.csv` `n_boxes`.

## Files

- `job/`: none (public annotation, no vendor pass).

# wider_fisheye_target boxes v1: the WIDER boxes mapped through the fisheye warp

## Why

The only label pass this set has: the generator (`training/warp_wider_fisheye.py --version
1`) forward-maps every WIDER FACE training box through the same virtual camera it used to
render the image, and writes the result as `wider_fisheye_labelv2.txt`. `boxes.csv` is that
file converted to the pii2 box table. Machine-derived, no vendor pass, hence no `job/`.

## Mapping rules (from the script header and the source README)

- 17 samples per edge (68 perimeter points) of each source box are unprojected through the
  pinhole model, rotated into the fisheye camera and projected with
  `cv2.fisheye.projectPoints`; the box is the axis-aligned bound of the rendered samples,
  clipped to the 2328x1748 canvas.
- A box is dropped if fewer than 50% of its perimeter samples are visible (off frame,
  outside the image circle r <= 1140 px, or beyond 65 deg from the virtual axis), or if its
  long side ends below 8 px.
- The generator reads the first four fields of every source box line, so the 2,399 WIDER
  boxes marked ignore in `wider_face/boxes/v1` were treated like any other box: those that
  survived are positives here. This file has no `ignore=1` row.
- Coordinates are written at one decimal by the generator; the conversion keeps them as is.

## Counts

`boxes.csv`: 153,203 rows on 12,812 images (96.1% of the 159,390 source boxes; 5,792
dropped as not visible, 395 as too small, per `derivation/wider_fisheye_stats.json`);
67 images have no row (4 of them had no box in WIDER either). `frames.csv`:
`reviewed=1` on all 12,879 images.

`build_wider_pii2.py --set wider_fisheye_target` converts the labelv2 file (19-field lines: four
coordinates plus fifteen -1 keypoint placeholders) and asserts the row count; `verify`
re-derives the box sets from `derivation/wider_fisheye_labelv2.txt` (or the staged file
while it exists) and compares them per image.

## Files

- `job/`: none (machine-derived).
- The generator's sidecars are in the dataset's top-level `derivation/`.

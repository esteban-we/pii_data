# faceback_45 boxes v1: the vendor human pass over every frame

## Why

First human labeling of the whole dataset. All 84,954 frames (42,477 per eye) went to
the vendor's annotation portal as two datasets (`faceback-45-left`,
`faceback-45-right`), seeded with machine prelabels, and came back 2026-09-09 with
one human answer per frame. Validated and converted by `training/faceback_conv.py`
in the fpv_face_pii repo (WOR-99).

## Rules the labelers worked under

- Start from the prelabel boxes (armW face model at score >= 0.75 or corroborated by
  the CrowdHuman head model, long side >= 40 px; `job/prelabels/README.md`); accept,
  move, delete or add boxes so that every visible face is boxed.
- A frame with no face comes back with an empty box list; that is a real negative,
  not a missing label.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted
  here to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the WOR-140 convention). `ignore` is 0 on every row.
- No size floor is applied here; the eval applies its own 40 px long-side floor.

## Counts

Labeled 2026-09-04..06 by 64 (left) / 69 (right) labelers (figures from the WOR-99
validation of the drop):

| | left | right |
|---|---:|---:|
| frames | 42,477 | 42,477 |
| human boxes | 47,998 | 47,008 |
| frames with boxes | 17,743 | 17,568 |
| face-free frames | 24,734 | 24,909 |
| prelabel boxes | 38,436 | 38,202 |
| boxes unchanged from prelabel | 30,631 | 32,003 |
| boxes moved (IoU >= 0.5 with a prelabel) | 2,240 | 2,170 |
| boxes added | 15,127 | 12,835 |
| prelabel boxes deleted | 5,565 | 4,029 |
| frames edited | 9,286 | 9,239 |

`boxes.csv`: 95,006 rows on 35,311 images (75,999 on train frames, 19,007 on eval
frames). `frames.csv`: `reviewed=1` on all 84,954 images. `boxes.csv` is byte for
byte the pii-data file `datasets/faceback_45/boxes/v1.csv`
(md5 5b4b6704b7b782fc455a354acfad0313).

## Files

- `job/prelabels/`: byte copies of `/data/esteban/pii/datasets/faceback_45/prelabels/`
  (`import_left.jsonl`, `import_right.jsonl`: the per-eye import files;
  `prelabels_full.jsonl`: every armW box down to score 0.25 with head-model evidence;
  `README.md`: the prelabel cut and the import procedure).
- `job/output/`: byte copies of `/data/esteban/pii/datasets/faceback_45/labels/`
  (the portal export, `faceback-45-{left,right}.{frames,boxes}.csv`, md5s in its
  `README.md`).
- Source of `boxes.csv`: `/data/esteban/pii/datasets/faceback_45/boxes.jsonl`
  (human boxes, normalised xywh), converted by `data/build_faceback45_pii2.py`.

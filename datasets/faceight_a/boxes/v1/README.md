# faceight_a boxes v1: the vendor human pass over every frame

## Why

First and so far only human labeling of round 1. All 42,408 frames went to the vendor's
annotation portal as the Verdict dataset `faceight`, seeded with armW prelabels, and came
back 2026-09-11 as `face_boxes_faceight.jsonl` (received at
`/data/esteban/tmp/ann_9_11/`, md5 335cad2b33205b88689ec20ae61a19f9). Validated and
converted by `training/ann_9_11_conv.py` in the fpv_face_pii repo (WOR-194).

## Rules the labelers worked under

- Start from the prelabel boxes (the `det_10g_armW` detections at score >= 0.5,
  55,951 boxes on 23,040 frames; a frame with no such box was sent with an empty box
  list, which the portal shows as "the machine looked and found nothing").
- Accept, move, delete or add boxes so that every visible face is boxed. A frame that
  comes back with an empty box list is a real negative, not a missing label.
- Boxes are normalised top-left x, y, w, h at four decimals in the portal; converted here
  to pixel xyxy as x1 = x*W, y1 = y*H, x2 = (x+w)*W, y2 = (y+h)*H at one decimal
  (W x H = 2328 x 1748, the faceback_45 convention). `ignore` is 0 on every row.
- No size floor is applied here; an eval applies its own floor.

The written instruction sheet the labelers were given is not on file; the rules above are
read off the prelabel package and the return.

## Counts

`boxes.csv`: 67,714 boxes on 22,636 images (19,772 frames are face-free); 54,151 boxes on
the train frames and 13,563 on the eval frames. `frames.csv`: `reviewed=1` on all 42,408
images, since the whole set was sent.

Transitions from the prelabels, from the WOR-194 validation of the drop (86 labelers,
2026-09-10 to 2026-09-11): 45,608 boxes unchanged, 5,312 moved (IoU >= 0.5 with a
prelabel), 16,794 added, 5,033 prelabel boxes deleted, 11,002 frames edited (25.9 pct).
That validation reports 55,953 prelabel boxes; the prelabel file and the drop's own
`n_boxes_machine` both count 55,951.

## Files

- `job/prelabels/`: byte copies of `shang:/data/esteban/faceight/verdict/`
  (`import_left.jsonl`: the import file, one record per frame, eye in `view`;
  `prelabels_full.jsonl`: every armW box down to score about 0.1 with the timestamp file
  name in `src_file`; `README.md`: the cut and the import procedure).
- `job/output/`: byte copies of `/data/esteban/pii/datasets/faceight_a/labels/`
  (the drop `face_boxes_faceight.jsonl` and its README).
- `boxes.csv` is derived from `job/output/face_boxes_faceight.jsonl` by
  `data/build_faceight_pii2.py`, and was asserted equal, box for box, to the independent
  conversion `/data/esteban/pii/datasets/faceight_a/boxes.jsonl` (md5
  8d701208b79f10f708eb307c0b1211da) on all 42,408 frames.

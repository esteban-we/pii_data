# faceback_45 labels

Human annotation drop for faceback-45, both eyes, received 2026-09-09 as
`/data/esteban/tmp/faceback_anotation/faceback-45-{left,right}.{frames,boxes}.csv`
and copied here byte for byte (md5 below). Same portal export shape as
`/data/esteban/pii/face-mine_labeled.csv`:

- `*.frames.csv`: one row per frame (42,477 per eye); `boxes_json` holds the human
  boxes; `n_boxes_machine` / `machine_src` describe the prelabel the labeler started
  from (`faceback45/armw_head@1`, see `../prelabels/`).
- `*.boxes.csv`: one row per box; empty `box_i` = face-free frame. `x,y,w,h` are
  normalised top-left+size in the row's `width x height` (all 2328x1748, 4 decimals);
  `px,py,pw,ph` are the SAME box rounded to pixels (verified |x*W - px| <= 0.5 on
  every row), not the prelabel.
- Frame name in the CSV (`image_uri`) omits the eye: `<session>_c<chunk>_f<idx>.jpg`;
  the jpg under `../images/` is `<session>_c<chunk>_<eye>_f<idx>.jpg` where eye is
  the CSV's `ds` suffix (`faceback-45-left` -> `left`).

| file | md5 |
|---|---|
| faceback-45-left.frames.csv | 2fc6b4a62126f0227f134512ee5e5463 |
| faceback-45-left.boxes.csv | 2713a114a2b1cc2e7b53f477985b9d44 |
| faceback-45-right.frames.csv | 3dbd4b360ac57625526fc562ba2838aa |
| faceback-45-right.boxes.csv | 50e32c15b4755bec4d9a2453fd5f637d |

Validated and converted by `training/faceback_conv.py` in the fpv_face_pii repo
(WOR-99): `../boxes.jsonl` (one row per image, human boxes, face_mine_v1 row shape
plus labeler provenance) and the labelv2 manifest
`training/manifests/faceback_45_labelv2.txt` (all 84,954 frames, paths
`faceback/<name>.jpg`). No split here: the split is decided separately.

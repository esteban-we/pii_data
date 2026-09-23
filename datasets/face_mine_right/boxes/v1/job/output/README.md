# face_mine_right labels

Human annotation drop received 2026-09-11 as `/data/esteban/tmp/ann_9_11/face_boxes_face-mine-right.jsonl` and copied
here byte for byte (md5 31f9541be247176ff1612ed835a20b59). One JSON record per frame (62,424) plus a
`{"kind": "end", "frames": N}` trailer; `boxes[{x,y,w,h}]` normalised top-left+size in the record's
`width x height` (all 2328x1748, 4 decimals); `n_boxes_machine` / `machine_src` describe the armW
prelabel (score >= 0.5) the labeler started from. Frame name in `image_uri` is
`<session>_c<chunk>_f<idx>.jpg` = the jpg under `../images/`.

Validated and converted by `training/ann_9_11_conv.py` in the fpv_face_pii repo (WOR-194):
`../boxes.jsonl` and `training/manifests/face_mine_right_labelv2.txt`. No split here.

# faceight_a labels

Human annotation drop received 2026-09-11 as `/data/esteban/tmp/ann_9_11/face_boxes_faceight.jsonl` and copied
here byte for byte (md5 335cad2b33205b88689ec20ae61a19f9). One JSON record per frame (42,408) plus a
`{"kind": "end", "frames": N}` trailer; `boxes[{x,y,w,h}]` normalised top-left+size in the record's
`width x height` (all 2328x1748, 4 decimals); `n_boxes_machine` / `machine_src` describe the armW
prelabel (score >= 0.5) the labeler started from. Frame name in `image_uri` is
`<session>_c<chunk>_f<idx>.jpg`; the jpg under `../images/` is the round-1 file of `data/faceight/frames.csv` for that (session, frame_idx).

Validated and converted by `training/ann_9_11_conv.py` in the fpv_face_pii repo (WOR-194):
`../boxes.jsonl` and `training/manifests/faceight_a_labelv2.txt`. No split here.

# faceight_b labels

Human annotation drop received 2026-09-16 as
`/data/esteban/tmp/pii/faceight_b/face_boxes_faceight_b.jsonl` and copied here byte for byte
(md5 2c1726c751a1499a476222bb60efdbfe). One JSON record per frame (60,000) plus a
`{"kind": "end", "frames": N}` trailer; `boxes[{x,y,w,h}]` normalised top-left+size in the
record's `width x height` (all 2328x1748, 4 decimals); `n_boxes_machine` / `machine_src`
describe the verdict_b prelabel (WOR-195) the labeler started from, `view` is the eye.
Frame name in `image_uri` is `<session>_c<chunk>_f<idx:06d>.jpg` (the Verdict name, no eye
token); the jpg under `../images_left/` or `../images_right/` is the round-2 file of
`data/faceight/frames.csv` for that (session, chunk, view, frame_idx).

Validated and converted by `training/ann_faceight_c_conv.py` in the fpv_face_pii repo
(WOR-1016): `../boxes.jsonl` and `training/manifests/faceight_b_labelv2.txt`. No split here.

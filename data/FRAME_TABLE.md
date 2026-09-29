# data/frames.csv and data/boxes.csv (WOR-125, WOR-131)

One frame table over the eight datasets under `oss://algorithm-datasets/pii/`
(layout: .knuth/docs/oss-pii-layout.md; object list: data/oss_pii_keys.jsonl).
Generator: `python3 data/build_frame_table.py` (stdlib only, about 20 s,
deterministic: two runs are byte-identical). Built 2026-09-09; the `unlabeled`
role was added the same day (WOR-131), boxes.csv unchanged.

## frames.csv (192,701 rows, one per OSS object)

`dataset,image,session_id,chunk,view,frame_idx,width,height,size,md5,role,n_boxes`

- `image` is the OSS key tail (basename). `session_id`, `chunk` (int),
  `view`, `frame_idx` (int) are parsed from the name with the per-dataset rule
  in the generator; `view` is the token as written on the name (`lview`/`rview`
  for pii_frames and face10k_repair, `left`/`right` for faceback_45) and blank
  where the name carries none (face10k_v3, face_mine_v1, gt_bench_*; all are
  vst_left captures) and for wider_face (all four blank).
- `width,height`: 2328x1748 for every fisheye dataset (asserted against every
  source that carries dims); wider_face from the labelv2 header.
- `size,md5`: copied from oss_pii_keys.jsonl (equal to Content-Length / ETag).
- `role`: `train`, `eval` or `unlabeled`. face_mine_v1 and faceback_45 by
  session from data/episode_usage.csv, asserted equal to data/splits/*.txt
  (7,137/1,858 and 267/67 sessions, zero disagreement); gt_bench_sparse and
  gt_bench_full are `eval`; face10k_v3, face10k_repair, pii_frames, wider_face
  are `train`, except: a frame of a dataset whose label source is a labelv2
  manifest (those four; `LABEL_SOURCE == "manifest"` in the generator) that is
  absent from that manifest has never been labeled and gets `role=unlabeled`,
  `n_boxes=0`. Nothing trains on it: the manifest rebuild takes `role=train`
  only. The datasets with a per-frame label file (face_mine_v1, faceback_45,
  gt_bench_*) list every frame by construction, so a frame missing there is a
  build error, never an unlabeled row. Today exactly 16 rows are unlabeled,
  all face10k_repair, all session 20260725_055005_MPCWWA chunk 0 (the vendor
  never returned boxes for them; kept on disk and OSS by user decision,
  2026-09-09); the generator asserts that set and prints the names.
- `n_boxes`: number of boxes.csv rows for the image, ignore rows included.

| dataset | frames | train | eval | unlabeled | boxes | of which ignore |
|---|---|---|---|---|---|---|
| face10k_repair | 3,168 | 3,152 | 0 | 16 | 7,883 | 459 |
| face10k_v3 | 8,339 | 8,339 | 0 | 0 | 25,966 | 505 |
| face_mine_v1 | 62,587 | 49,833 | 12,754 | 0 | 76,859 | 0 |
| faceback_45 | 84,954 | 68,014 | 16,940 | 0 | 95,006 | 0 |
| gt_bench_full | 9,636 | 0 | 9,636 | 0 | 20,054 | 0 |
| gt_bench_sparse | 889 | 0 | 889 | 0 | 1,541 | 0 |
| pii_frames | 10,249 | 10,249 | 0 | 0 | 19,579 | 2,704 |
| wider_face | 12,879 | 12,879 | 0 | 0 | 159,390 | 2,399 |
| total | 192,701 | 152,466 | 40,219 | 16 | 406,278 | 6,067 |

## boxes.csv (406,278 rows, one per box)

`dataset,image,x1,y1,x2,y2,ignore`

Pixel xyxy corners in the image's own space, one decimal, sorted by
(dataset, image, x1, y1). `ignore` is a column the issue did not list: it is 1
for the 3,668 labelv2 lines of the form `x1 y1 x2 y2 1` in the face10k and
pii_frames blocks (plus 2,399 in wider_face), which the trainer loads as
`gt_bboxes_ignore` (training/patches/detection/scrfd/mmdet/datasets/retinaface.py,
`_parse_ann_line`: a 5th value of 1 sets ignore). Without the flag a rebuilt
manifest would turn those regions into positives.

Box sources (priority as fixed by WOR-125, one deliberate deviation):

| dataset | source | why |
|---|---|---|
| face_mine_v1 | face-mine_labeled.csv, not in the live store; it is at /data/esteban/pii_backup/face-mine_labeled.csv (human review, normalized xywh) | datasets/face_mine_v1/boxes.jsonl holds the miner's machine boxes (83,536, box_src face_mine/two_model@1), not the 76,859 human boxes; the manifest was built from the CSV (training/face_mine_conv.py) |
| faceback_45 | datasets/faceback_45/boxes.jsonl (human, normalized xywh) | matches faceback_45_labelv2.txt exactly |
| gt_bench_full | gt_bench_v1/labels/gt_bundle.json via uuid_map.json | the 9 mapped uuids cover exactly the 9,636 images_full names; the 10th uuid (019e93df-7e02-73b3-92b5-1af0dbbab7ae) has 0 frames |
| gt_bench_sparse | gt_bench_v1/labels/gt_eval_sparse.json | 5 of the 889 frames have fewer boxes than the bundle gives the same frame (listed in the issue); the sparse file is kept as its own truth |
| face10k_v3, face10k_repair, pii_frames | labelv2 blocks of training/manifests/train_Z2.txt | no other human-label file on this box reproduces them (face10k_v3/repair_v1/manifest.jsonl does not) |
| wider_face | datasets/wider_face/wider_train_labelv2.txt | |

Conversion from normalized xywh: `x1 = x*W`, `y1 = y*H`, `x2 = (x+w)*W`,
`y2 = (y+h)*H`, formatted `%.1f`, exactly as the conv scripts did.

The 16 face10k_repair images that no manifest ever carried (all
20260725_055005_MPCWWA_chunk_000, 5 lview and 11 rview, listed by the
generator) have `role=unlabeled` and `n_boxes=0`; they are not face-free
negatives (each went to the vendor with 1-2 pre_boxes and came back without
boxes) and must not be read as such.

## Rebuilding a training manifest

`python3 data/build_frame_table.py --check-manifest training/manifests/train_Z2.txt`
rebuilds the manifest (every role=train frame of pii_frames, face10k_v3,
face10k_repair, face_mine_v1, faceback_45; datasets in that order, images and
boxes sorted) and compares: 139,587 frames / 190,520 boxes, identical
(image, box) set at 0.1 px, identical ignore flags, every box line byte for
byte reproducible, no eval-role or unlabeled frame in the manifest, 0 of the
12,754 eval face_mine_v1 frames present, zero surplus (the 16 unlabeled repair
frames stay out of the rebuild; there is no tolerance for surplus any more).
The sorted rebuild is not byte-identical to train_Z2.txt: that file
inherits train_W's unsorted block order (21,396 blocks sit elsewhere) and 34,997
blocks list boxes in non-sorted order; frames.csv carries no order key.
`--rebuild-manifest OUT` writes the sorted rebuild. Dropping one box from a
copy of any source makes the check exit 1 (verified for faceback_45/boxes.jsonl
and face-mine_labeled.csv); flipping one of the 16 unlabeled rows to
`role=train` in a copy of frames.csv makes it exit 1 with a surplus of 1.

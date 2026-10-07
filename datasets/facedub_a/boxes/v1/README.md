# facedub_a boxes/v1

The round-1 human pass, drops of 2026-09-29:

    job/output/face_boxes_facedub_a_left.jsonl    cdda394368dbd8fe6a3152f5e2973434
    job/output/face_boxes_facedub_a_right.jsonl   a2ba42af5d31932d3b4976f34c1e45d5

37568 records (18829 + 18739), one per frame,
plus a {"kind": "end", "frames": N} trailer; 3450 + 5141
frames carry a box, 28418 boxes in total (7415 + 21003).

frames.csv is image,reviewed with reviewed = 1 for every frame: the whole set came back from the
human pass. The drop's own `review_round` field is a per-record vendor counter (values 0, 1 and 2)
and is not used; the frames.csv `round` column is 1, the annotation round of the set.

boxes.csv is image,x1,y1,x2,y2,ignore: normalized top-left xywh turned into pixel xyxy as x*1024,
y*1280, (x+w)*1024, (y+h)*1280 at one decimal, ignore always 0, sorted by (image, x1, y1). A frame's
boxes are a full replacement, never a union with the machine prelabels.

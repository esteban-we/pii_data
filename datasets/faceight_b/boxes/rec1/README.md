# faceight_b boxes/rec1: the recognizable subset

The boxes of `boxes/v2/boxes.csv` that a reviewer called recognizable
in **at least 1 of the 3 passes** of the facereview review (PII-1807,
PII-1833, PII-1915). The source pass is
the v2 armAE34 addition review, over the faceight_b eval sessions;
it put 12,743 boxes on 4,806 frames into the queue and each of them was judged
3 times, by whoever reached it.

A face is recognizable if, were this person a friend, you could tell who it is.
That one line, and nothing else, is what the reviewer's page showed above every batch.

## Counts

| votes | boxes | share of the reviewed set |
|---|---:|---:|
| 0/3 | 11,502 | 90.3% |
| 1/3 | 352 | 2.8% |
| 2/3 | 264 | 2.1% |
| 3/3 | 625 | 4.9% |
| **>= 1 (this set)** | **1,241** | **9.7%** |

## Files

```
boxes.csv    image,x1,y1,x2,y2,ignore
             the 1,241 rows of boxes/v2/boxes.csv with votes >= 1, verbatim,
             sorted by (image, x1, y1). A full file, not a diff, and a strict subset of
             boxes/v2/boxes.csv: coordinates are never re-rounded here
votes.csv    image,x1,y1,x2,y2,votes,pass1,pass2,pass3,user1,user2,user3,ts1,ts2,ts3
             every one of the 12,743 reviewed boxes with each pass's own judgement, its
             reviewer and its timestamp, so another threshold (2/3, 3/3) is rebuilt from
             this file alone and needs neither the review service nor its DB
frames.csv   image,reviewed
             the 4,806 frames those boxes are on, reviewed = 1. Frames outside
             the eval slice, and eval frames with no box, have no row
```

## Reviewers

| reviewer | judgements | marked recognizable |
|---|---:|---:|
| Will | 35,139 | 2,560 |
| changxin-plus | 2,490 | 153 |
| Esteban | 600 | 42 |

Reviewed 2026-10-02 to 2026-10-05 (UTC). A judgement is one pass of one box, so the three columns of
votes.csv come from up to three different people; 38,229 judgements in all.

## Built by

```bash
python3 data/build_facereview_rec.py build  --dataset faceight_b   # in the pii repo
python3 data/build_facereview_rec.py verify --dataset faceight_b
```

from `facereview/data/reviews.faceight_b.db` (read-only) joined to
`boxes/v2/boxes.csv` by (image, box index within the image). `verify`
re-derives all four files and compares them byte for byte.

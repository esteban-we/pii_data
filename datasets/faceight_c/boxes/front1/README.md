# faceight_c boxes/front1: the frontal subset

The boxes of `boxes/v3/boxes.csv` that a reviewer called frontal
in **at least 1 of the 3 passes** of the facereview frontal review
(PII-2617 for the inventory of the review DBs, PII-2688 for this build). Both splits
are here: **8,959 frontal boxes** of the 29,633 judged,
7,231 on train frames and 1,728 on eval
frames. The question the page asked above every batch was

> A face is frontal when an eye or nose, or mouth can be seen. (Only one part also counts).

and a 1 in the review DB's `recognizable` column means frontal for these sets, which
is how facereview carries both questions in one schema (`rule_for`, app.py).

## Where the judgements come from

| served set | split | served on | judged | frontal | mapped to v3 | lost |
|---|---|---|---:|---:|---:|---:|
| `faceight_c_train_front` | train | v3 | 22,916 | 7,231 | 22,916 | 0 |
| `faceight_c_front` | eval | v3 | 5,691 | 1,725 | 5,691 | 0 |
| `faceight_c_front_under30px` | eval | v3 | 1,026 | 3 | 1,026 | 0 |

Each set is its own `reviews.<set>.db` and its own three-pass queue.

Every set was served on v3, the GT in use, so no box version is mapped here; `faceight_c_front_under30px` is the complement of `faceight_c_front` under the 30 px side rule, which is why the eval slice has no unjudged box.

## Counts

Every box of `boxes/v3/boxes.csv` is a row of votes.csv, judged
or not.

| split | boxes | judged | front = 1 | front = 0 | unjudged |
|---|---:|---:|---:|---:|---:|
| train | 27,046 | 22,916 | 7,231 | 15,685 | 4,130 |
| eval | 6,717 | 6,717 | 1,728 | 4,989 | 0 |
| **both** | **33,763** | **29,633** | **8,959** | **20,674** | **4,130** |

4,130 of the 4,130 unjudged boxes have a longer side
under 30 px, which is the queue's size rule (`MIN_SIDE_PX`): facereview
lays no slot for a box that small, so it was never shown to anybody (train 4,130 of 4,130, eval 0 of 0).
They are the `passes = 0` rows of votes.csv, like the small ones.

| votes | boxes | share of the judged set |
|---|---:|---:|
| 0/3 | 20,674 | 69.8% |
| 1/3 | 2,695 | 9.1% |
| 2/3 | 1,898 | 6.4% |
| 3/3 | 4,366 | 14.7% |
| **>= 1 (this set)** | **8,959** | **30.2%** |

## Files

```
boxes.csv    image,x1,y1,x2,y2,ignore
             the 8,959 rows of boxes/v3/boxes.csv with
             votes >= 1, verbatim, sorted by (image, x1, y1). A full file, not
             a diff, and a strict subset of boxes/v3/boxes.csv:
             coordinates are never re-rounded here
votes.csv    image,x1,y1,x2,y2,split,set,src_index,votes,passes,pass1,pass2,pass3,user1,user2,user3,ts1,ts2,ts3
             every one of the 33,763 boxes of boxes/v3/boxes.csv with
             each pass's own judgement, its reviewer and its timestamp, so another
             threshold (2/3, 3/3) is rebuilt from this file alone and needs neither the
             review service nor its DB. `split` is the frame's role in split.csv, `set`
             the served facereview set that judged the box and `src_index` its index in
             THAT set's boxes version, so the row joins back to the review DB; `passes`
             is how many of the 3 passes reached the box, and a pass it never had
             is an empty pass, user and ts
frames.csv   image,reviewed
             the 16,242 frames carrying at least one judged box, reviewed = 1.
             A frame whose every box was too small to queue has no row
```

## Reviewers

| reviewer | judgements | marked frontal |
|---|---:|---:|
| Will | 59,703 | 12,770 |
| Bao | 20,160 | 6,256 |
| Edwin | 8,856 | 526 |
| Esteban | 90 | 31 |
| changxin-plus | 90 | 6 |

Reviewed 2026-10-07 to 2026-10-15 (UTC). A judgement is one pass of one box,
so the three columns of votes.csv come from up to three different people;
88,899 judgements in all.

## Built by

```bash
python3 data/build_facereview_front.py build  --dataset faceight_c   # in the pii repo
python3 data/build_facereview_front.py verify --dataset faceight_c
```

from `/data/facereview/data/reviews.faceight_c_train_front.db`, `/data/facereview/data/reviews.faceight_c_front.db`, `/data/facereview/data/reviews.faceight_c_front_under30px.db`
(read-only, immutable) joined to `boxes/v3/boxes.csv` by (image, box
index within the image) and, where the set was served on an older version, by the box's
coordinates. `verify` re-derives all four files and compares them byte for byte.

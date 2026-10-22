# faceight_b boxes/front1: the frontal subset

The boxes of `boxes/v3/boxes.csv` that a reviewer called frontal
in **at least 1 of the 3 passes** of the facereview frontal review
(PII-2617 for the inventory of the review DBs, PII-2688 for this build). Both splits
are here: **11,765 frontal boxes** of the 49,692 judged,
8,661 on train frames and 3,104 on eval
frames. The question the page asked above every batch was

> A face is frontal when an eye or nose, or mouth can be seen. (Only one part also counts).

and a 1 in the review DB's `recognizable` column means frontal for these sets, which
is how facereview carries both questions in one schema (`rule_for`, app.py).

## Where the judgements come from

| served set | split | served on | judged | frontal | mapped to v3 | lost |
|---|---|---|---:|---:|---:|---:|
| `faceight_b_train_front` | train | v2 | 39,856 | 8,699 | 39,483 | 373 |
| `faceight_b_front` | eval | v3 | 10,209 | 3,104 | 10,209 | 0 |

Each set is its own `reviews.<set>.db` and its own three-pass queue.

The eval slice was re-queued from scratch on v3 as `faceight_b_front` after the v2 recognizable round (boxes/rec1); its 10,209 judged boxes and 3,104 positives are the finished queue, where PII-2617's inventory saw 9,240 and 1,493 mid-round.

**373 judged boxes do not reach v3** and are not in
this version. They were judged on an older boxes version, and the re-labelling that
made v3 moved or removed the box; a review is carried over only
when the target version has a box at the SAME coordinates on the same frame, so a
moved box is dropped rather than guessed at by overlap. By reason:

| served set | reason | boxes |
|---|---|---:|
| `faceight_b_train_front` | no box at these coordinates in v3 | 266 |
| `faceight_b_train_front` | the frame has no box in v3 | 107 |

Dropping them is not a rounding artefact: the nearest v3 box on the same frame is no overlap at all for 272, at IoU below 0.5 for 71, at IoU 0.5 to 0.9 for 30.

Every one of them is on a frame whose box list the re-labelling changed.

## Counts

Every box of `boxes/v3/boxes.csv` is a row of votes.csv, judged
or not.

| split | boxes | judged | front = 1 | front = 0 | unjudged |
|---|---:|---:|---:|---:|---:|
| train | 49,726 | 39,483 | 8,661 | 30,822 | 10,243 |
| eval | 12,781 | 10,209 | 3,104 | 7,105 | 2,572 |
| **both** | **62,507** | **49,692** | **11,765** | **37,927** | **12,815** |

12,341 of the 12,815 unjudged boxes have a longer side
under 30 px, which is the queue's size rule (`MIN_SIDE_PX`): facereview
lays no slot for a box that small, so it was never shown to anybody (train 9,769 of 10,243, eval 2,572 of 2,572).
The other 474 are boxes this version
carries that the version the queue was served on did not, on the frames the
re-labelling changed: big enough to queue, but never in anybody's queue.
They are the `passes = 0` rows of votes.csv, like the small ones.

| votes | boxes | share of the judged set |
|---|---:|---:|
| 0/3 | 37,927 | 76.3% |
| 1/3 | 4,124 | 8.3% |
| 2/3 | 3,452 | 6.9% |
| 3/3 | 4,189 | 8.4% |
| **>= 1 (this set)** | **11,765** | **23.7%** |

## Files

```
boxes.csv    image,x1,y1,x2,y2,ignore
             the 11,765 rows of boxes/v3/boxes.csv with
             votes >= 1, verbatim, sorted by (image, x1, y1). A full file, not
             a diff, and a strict subset of boxes/v3/boxes.csv:
             coordinates are never re-rounded here
votes.csv    image,x1,y1,x2,y2,split,set,src_index,votes,passes,pass1,pass2,pass3,user1,user2,user3,ts1,ts2,ts3
             every one of the 62,507 boxes of boxes/v3/boxes.csv with
             each pass's own judgement, its reviewer and its timestamp, so another
             threshold (2/3, 3/3) is rebuilt from this file alone and needs neither the
             review service nor its DB. `split` is the frame's role in split.csv, `set`
             the served facereview set that judged the box and `src_index` its index in
             THAT set's boxes version, so the row joins back to the review DB; `passes`
             is how many of the 3 passes reached the box, and a pass it never had
             is an empty pass, user and ts
frames.csv   image,reviewed
             the 23,169 frames carrying at least one judged box, reviewed = 1.
             A frame whose every box was too small to queue has no row
```

## Reviewers

| reviewer | judgements | marked frontal |
|---|---:|---:|
| Will | 68,151 | 14,300 |
| huangchuyin-zq | 20,979 | 1,565 |
| linboyu-zq | 19,209 | 1,840 |
| qianjialu-zq | 14,675 | 1,037 |
| Bao | 10,410 | 2,755 |
| yezhirong-zq | 7,912 | 1,466 |
| Edwin | 7,440 | 606 |
| changxin-plus | 120 | 9 |
| Esteban | 90 | 14 |
| Zhiwen | 60 | 0 |
| zouziyang | 30 | 3 |

Reviewed 2026-10-13 to 2026-10-19 (UTC). A judgement is one pass of one box,
so the three columns of votes.csv come from up to three different people;
149,076 judgements in all.

## Built by

```bash
python3 data/build_facereview_front.py build  --dataset faceight_b   # in the pii repo
python3 data/build_facereview_front.py verify --dataset faceight_b
```

from `/data/facereview/data/reviews.faceight_b_train_front.db`, `/data/facereview/data/reviews.faceight_b_front.db`
(read-only, immutable) joined to `boxes/v3/boxes.csv` by (image, box
index within the image) and, where the set was served on an older version, by the box's
coordinates. `verify` re-derives all four files and compares them byte for byte.

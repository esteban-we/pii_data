# faceight_b

Round 2 of the faceight annotation program: 60,000 frames, both eyes (28,503 left,
31,497 right), all 2328x1748, from 25,118 episodes (one session and one chunk each). The
rows are `annotation_round == 2` in `data/faceight/frames.csv` of the fpv_face_pii repo.

Rounds a, b and c are SEPARATE datasets of the same family: their frames are disjoint,
their episodes overlap, and all three follow one frozen episode split (PII-1212). The
letter marks the round, not a subset relation.

Staged 2026-09-22 in this layout (PII-1315, PII-1360) by `data/build_faceight_pii2.py`
in the fpv_face_pii repo. The images have no local source: they were pulled read-only
from `oss://we-vlm-annotation-data-sh/faceight_b/images/` with GetObject (128 parallel
requests, 36,576,699,475 bytes in 472 s, 77.6 MB/s, 0 failures, 0 retries), each object
checked against its Content-Length and against its ETag (single-part, so the ETag is the
object md5) before the file was renamed into place, and the whole set re-checked against
the bucket listing afterwards.

## Layout

```
images/                 60,000 jpg, 28,503 left and 31,497 right, 36,576,699,475 bytes, all 2328x1748
frames.csv              image, session, chunk, eye, frame_idx, t_ms, size, md5, round, local_name, oss_key
split.csv               session, role (train | eval); the split is per session
boxes/v1/               the vendor human pass over every frame (2026-09-16)
boxes/v2/               v1 plus the review of the 9,402 frames with an armAE34 addition (2026-09-18)
```

Image name: `<session>_c<chunk>_<left|right>_f<frame_idx:06d>.jpg`, the faceback_45
convention; `frames.csv` is sorted by image and is the source of truth for which images
exist. `size` and `md5` are the image file's own, computed at build time and equal to the
OSS listing. Two further names of the same frame are kept as columns because every
upstream file uses one of them:

- `local_name`: the faceight frame file `<session>_c<chunk>_<lview|rview>_t<t_ms>.jpg`,
  the `file` column of `data/faceight/frames.csv` and the name under
  `shang:/data/esteban/faceight/frames` and `frames_right`.
- `oss_key`: the eye-less Verdict object
  `faceight_b/images/<session>_c<chunk>_f<frame_idx:06d>.jpg`. The vendor portal accepts
  no eye token in a file name, so the eye lives in the `view` field of every job file.
  Those names do not collide inside this round: the build asserts it, and the 60,000 OSS
  keys match the 60,000 `frames.csv` rows one to one in both directions.

## How the frames were chosen

`.knuth/docs/faceight.md` has the full rule. In short: 34,667 episodes drawn from the
corpus with the scene mix (`mining/faceight_sample.py`, seed 19760703); per episode the
latest chunk of at least 240 s; inside that chunk the frames at `t_ms = 1000 * s + 233`
for the seconds `s` flipped on in `data/faceight/residues.csv`. Rounds 2 and 3 then took
100,000 of the 701,904 unannotated frames of both eyes by cell shares over the 10 cells
of `mining/faceight_cells.py` (score bands of the armW and armAA34 detectors, plus a
faceless cell), breadth first over episodes with one
`numpy.random.default_rng(19760703)` shuffle per cell; the first 60 pct of each cell's
draw order is round 2, the rest round 3 (WOR-191).

## Train/eval split

`split.csv`: 20,090 train and 5,028 eval sessions, 48,073 train and 11,927 eval frames.
The split is per episode, and in faceight an episode is exactly one session, so a session
row is an episode row. It is a projection of the frozen master split over all 34,667
faceight episodes (PII-1212, `mining/faceight_master_split.py`, seed 19760703,
`data/splits/faceight_b_{train,eval}_episodes_v1.txt`). 21,371 of the 25,118 round-2
episodes also carry frames of another round (16,017 of round 1, 15,069 of round 3); an
episode keeps one side in every round, so training on the train slice of any round never
touches an episode held out in another.

Rules:

- The eval episodes must never be trained on, from this or any other round.
- The split is frozen. A new round is a projection of it, never a redraw.

## Provenance

- images: `oss://we-vlm-annotation-data-sh/faceight_b/images/` (the package pushed for
  the vendor in WOR-195); mapping columns from `data/faceight/frames.csv` (md5
  bb56ee8f767a55239700f717f75361ed at the round assignment).
- `split.csv`: `data/splits/faceight_b_{train,eval}_episodes_v1.txt`.
- boxes and job files: see the per-version READMEs.
- The old `/data/esteban/pii/datasets/faceight_b` held no images, only `boxes.jsonl`, a
  README and two symlinks into the shang frame trees.

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out of
this directory.

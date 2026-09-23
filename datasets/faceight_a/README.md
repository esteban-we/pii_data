# faceight_a

Round 1 of the faceight annotation program: 42,408 left-eye frames (`vst_left`), one
frame per line of `frames.csv`, all 2328x1748, from 19,783 episodes (one session and one
chunk each, 1 to 6 frames per episode). The rows are `annotation_round == 1` in
`data/faceight/frames.csv` of the fpv_face_pii repo.

Rounds a, b and c are SEPARATE datasets of the same family: their frames are disjoint,
their episodes overlap, and all three follow one frozen episode split (PII-1212). The
letter marks the round, not a subset relation.

Staged 2026-09-22 in this layout (PII-1315, PII-1360) by `data/build_faceight_pii2.py`
in the fpv_face_pii repo. Every image is a byte copy of
`/data/esteban/pii/datasets/faceight_a/images/<local_name>` renamed to the eye-carrying
name; the build asserted, file by file, that its md5 equals both the source file and the
ETag of `oss://we-vlm-annotation-data-sh/faceight/images/<oss_key>`, and that the byte
total equals the OSS byte total.

## Layout

```
images/                 42,408 jpg, all left eye, 27,185,183,632 bytes, all 2328x1748
frames.csv              image, session, chunk, eye, frame_idx, t_ms, size, md5, round, local_name, oss_key
split.csv               session, role (train | eval); the split is per session
boxes/v1/               the vendor human pass over every frame (2026-09-11)
```

Image name: `<session>_c<chunk>_<left|right>_f<frame_idx:06d>.jpg`, the faceback_45
convention; `frames.csv` is sorted by image and is the source of truth for which images
exist. `size` and `md5` are the image file's own, computed at build time.
Two further names of the same frame are kept as columns because every upstream file uses
one of them:

- `local_name`: the faceight frame file `<session>_c<chunk>_<lview|rview>_t<t_ms>.jpg`,
  the `file` column of `data/faceight/frames.csv` and the name under
  `shang:/data/esteban/faceight/frames/`.
- `oss_key`: the eye-less Verdict object `faceight/images/<session>_c<chunk>_f<frame_idx:06d>.jpg`.
  The vendor portal accepts no eye token in a file name, so the eye lives in the `view`
  field of every job file. Those names do not collide inside a round: the build asserts
  it (checked for all three rounds, 0 collisions).

## How the frames were chosen

`.knuth/docs/faceight.md` has the full rule. In short: 34,667 episodes drawn from the
corpus with the scene mix (`mining/faceight_sample.py`, seed 19760703); per episode the
latest chunk of at least 240 s (`chunk_sel`); inside that chunk the frames at
`t_ms = 1000 * s + 233` for the seconds `s` flipped on in `data/faceight/residues.csv`.
Round 1 was assigned by the WOR-94 rule over the left eye only.

## Train/eval split

`split.csv`: 15,826 train and 3,957 eval sessions, 33,926 train and 8,482 eval frames.
The split is per episode, and in faceight an episode is exactly one session, so a session
row is an episode row. It is a projection of the frozen master split over all 34,667
faceight episodes (PII-1212, `mining/faceight_master_split.py`, seed 19760703,
`data/splits/faceight_a_{train,eval}_episodes_v1.txt`); the round-1 sides were pinned
first, since models were already trained on this train half.

Rules:

- The eval episodes must never be trained on, from this or any later round; the same
  episode keeps the same side in faceight_b, faceight_c and any future round.
- The split is frozen. A new round is a projection of it, never a redraw.

## Provenance

- images and the mapping columns: `/data/esteban/pii/datasets/faceight_a/images` and
  `data/faceight/frames.csv` (md5 bb56ee8f767a55239700f717f75361ed at the round
  assignment).
- `split.csv`: `data/splits/faceight_a_{train,eval}_episodes_v1.txt`.
- v1 boxes and job files: see `boxes/v1/README.md`.
- Not carried over: the `_oss_staging` directory (empty) and the symlink layout of the
  old `faceight_a` (`images` there pointed at the shang frame tree before the OSS staging).

## Handling

The frames are unblurred faces of real people. Same handling rules as any PII batch:
frames stay on this box; usage is materialized as views, never by copying images out of
this directory.

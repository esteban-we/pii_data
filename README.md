# pii_data

The dataset store for the face blur (PII) work, checked out at
`/data/esteban/pii2` on gpu-002 (it becomes `/data/esteban/pii` when PII-1315
finishes). Every dataset, every labeling pass, every training mix and every
training arm is described here. The pixels and the checkpoints are not: they
live on OSS.

The rule: **git is the ledger, OSS is the byte store.** If a file can be
downloaded again (images, checkpoints, ONNX exports) or re-rendered from a
tracked source (the view manifests), it is ignored. Everything that says what
the bytes are, where they came from and how they were labeled is committed.

## Layout

```
datasets/<name>/            one dataset per directory
  README.md                 what it is, where it was staged from, how it was built
  frames.csv                one row per image: image, session, chunk, eye,
                            frame_idx, size, md5, plus provenance (src_path, t_ms)
  split.csv                 session, role (train | eval); the split is per session
  images/                   the JPGs                                    (ignored, OSS)
  boxes/vN/                 one labeling pass; a relabel of a subset is a new vN
    frames.csv              image, reviewed (1 = sent in this pass)
    boxes.csv               image, x1, y1, x2, y2, ignore (pixel xyxy, full replacement)
    README.md               why the pass exists and the labeling rules
    job/prelabels/          what the vendor started from (byte copies)
    job/output/             what the vendor returned (byte copies)
  derivation/               how a derived set was made (scripts, params, previews)

views/<name>/               one declared training mix or eval set
  recipe.yaml               the source of truth: datasets, box versions, splits, filters
  scrfd.txt, d2.json, summary.json                                  (ignored, rendered)

runs/train/
  models.csv                arm, family, epoch, status, pick ONNX with md5 and size
  MANIFEST.tsv              every copied file with size and md5
  <family>/<arm>/
    pick.yaml               which epoch this arm is, the rule that chose it, evidence
    config, launch and export scripts, eval/*.json, small logs
    epochs/, onnx/, train.log                                        (ignored, OSS)
  <family>/stock/           upstream baselines: .sha256, .provenance.txt and .md5
                            tracked, the weights themselves ignored

tables/corpus/              session, episode and chunk tables for the whole corpus
tables/population/          15 GB of parquet over the full frame population   (ignored, OSS)
calib/                      fisheye and annotation calibration JSON
inbox/                      vendor deliveries not yet folded into a dataset
experiments/                one-off probes: scripts and their result JSON
```

## What is tracked

Ignored (see `.gitignore`): `datasets/*/images/`, `views/*/{scrfd.txt,d2.json,summary.json}`,
`runs/train/*/*/{epochs,onnx,logs}/` and `train.log`, every `*.pth` `*.onnx` `*.jit` `*.zip`
anywhere (this is what covers the stock weights), and `tables/population/`.

Everything else is committed. No tracked file may exceed 100 MB; the largest
today is `tables/corpus/chunks_probe.jsonl` at 58 MB. Check before committing:

```
git ls-files -z | xargs -0 stat -c '%s %n' | sort -rn | head
```

## Datasets

| dataset | frames | box versions |
| --- | --- | --- |
| face10k | 11,507 | v1 v2 v3 |
| faceback_45 | 84,954 | v1 v2 v3 |
| faceight_a | 42,408 | v1 |
| faceight_b | 60,000 | v1 v2 |
| faceight_c | 40,000 | v1 v2 |
| face_mine_v1 | 62,587 | v1 v2 v3 |
| face_mine_right | 62,423 | v1 v2 |
| gt_bench_full | 9,636 | v1 |
| gt_bench_sparse | 889 | v1 |
| pii_frames | 10,249 | v1 |
| wider_face | 12,879 | v1 |
| wider_fisheye_fill | 12,879 | v1 |
| wider_fisheye_target | 12,879 | v1 |

423,290 images, 236,516,907,573 bytes, none of them in git. Left and right eyes
are separate datasets (`face_mine_v1` is left, `face_mine_right` is right).
`frames.csv` is the source of truth for which images exist; `size` and `md5` are
mandatory and are what a download is verified against.

## Views

19 views, 11 train mixes and 8 eval sets, each reproducing a legacy manifest
(PII-1373). Frame counts, largest first: train_Z5 303,407; train_Z3 223,214;
train_Z5noFM 203,873; train_Z2 139,587; train_Z4 123,680; train_X 84,452;
train_Z 71,573; train_W, train_W_fe, train_W_fe2 34,619; train_W_nowider 21,740;
eval_faceback_v1, eval_faceback_hq_v1, eval_faceback_hq_v2 16,940;
eval_face_mine_right_v1 12,722; face_mine_v1 12,754; gt_bench_v1 9,636;
eval_faceight_a_v1 8,482; gt_bench_sparse_v1 889.

## Runs

26 SCRFD arms and 4 EgoBlur arms, plus one `stock` directory per family, so 32
rows in `runs/train/models.csv`. Every `pick.yaml` is `rule: last_epoch` for now
(PII-1372 replaces it with a per-epoch sweep); `status` is the W&B tag from
`alex-qiu-worldengineai/pii-face-eval`.

## Getting the bytes

A fresh clone is metadata only: `datasets/*/images/` is empty and the view
manifests are not there.

- Images are on `oss://algorithm-datasets/pii/data/<dataset>/<image>` (8 of the
  13 sets today; PII-1413 finishes the mirror and adds `oss_key` to every
  `frames.csv`). Checkpoints and exports go to `pii/models/<family>/<arm>/`.
- Once PII-1414 lands: `python3 data/oss_sync.py pull [--dataset D | --view V |
  --arm A]` reads the checked-out index and fetches what is missing or
  mismatched, and a tracked `post-merge` hook runs it after `git pull`. Hooks do
  not fire on clone, so the first pull is explicit.
- Views are rendered, not downloaded: `python3 data/build_view.py render --all`,
  verified with `check --all`.

## Build scripts

They live in the `pii` repo (`/home/esteban/repos/pii`), not here:
`data/build_faceback45_pii2.py`, `build_face10k_pii2.py`, `build_faceight_pii2.py`,
`build_facemine_pii2.py`, `build_gt_bench_pii2.py`, `build_pii_frames_pii2.py`,
`build_wider_pii2.py` per dataset; `data/build_view.py` for views;
`data/build_runs_pii2.py` for `runs/train`; `data/oss_pii_upload.py` for OSS.
Each has a `verify` or `check` subcommand that re-checks the tree against sizes
and md5s.

## Superseded

`/home/esteban/repos/pii-data` (one commit, no remote) was the earlier metadata
repo. It is not migrated and not maintained; this repo replaces it.

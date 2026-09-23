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
                            frame_idx, size, md5, oss_key, plus provenance
                            (src_path, t_ms)
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
  models.csv                arm, family, epoch, status, pick ONNX with md5 and size,
                            oss_key of the pick
  MANIFEST.tsv              every copied file with size, md5 and oss_key (empty when
                            the file is git content)
  <family>/<arm>/
    pick.yaml               which epoch this arm is, the rule that chose it, evidence
    config, launch and export scripts, eval/*.json, small logs
    epochs/, onnx/, train.log                                        (ignored, OSS)
  <family>/stock/           upstream baselines: .sha256, .provenance.txt and .md5
                            tracked, the weights themselves ignored

tables/corpus/              session, episode and chunk tables for the whole corpus
tables/population/          15 GB of parquet over the full frame population
                            (ignored, and not on OSS either: PII-1419)
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

A fresh clone is metadata only: `datasets/*/images/` is empty, the checkpoints and
ONNX exports are not there and the view manifests are not rendered.

Every index row that stands for a file on OSS carries its key:
`datasets/<name>/frames.csv` has an `oss_key` column
(`pii/data/<set>/<image>`; the two `face10k` batches keep the prefixes
`face10k_v3` and `face10k_repair` they were uploaded under), and
`runs/train/MANIFEST.tsv` has one that is filled in
(`pii/models/<family>/<arm>/<rel>`) exactly for the files git ignores and empty for
the files git tracks. `runs/train/models.csv` carries the key of each arm's pick.

```
bash data/setup.sh                       # git config core.hooksPath .githooks, then pull
python3 data/oss_sync.py pull            # everything the index names
python3 data/oss_sync.py pull --view train_Z5      # only what one view needs
python3 data/oss_sync.py pull --dataset gt_bench_full --arm armAC
python3 data/oss_sync.py status [--remote]
python3 data/oss_sync.py push --arm armNEW         # after a new training run
python3 data/build_view.py render --all            # views are rendered, not downloaded
```

`pull` skips any file already on disk with the indexed size and md5, verifies every
download against the index md5 before renaming it into place, and prints exactly
`up to date` when nothing was missing. The md5 of a local file is cached in
`.git/oss_sync_md5.json` against its size and mtime, so a rerun does not read the
whole tree again.

`push` uploads single part with `Content-MD5` and asserts the returned ETag equals
the md5. It never overwrites: an object already on OSS whose ETag differs from the
local md5 is reported as a conflict and the run fails. It never deletes. A file
under `runs/train/<family>/<arm>/{epochs,onnx,logs}/` with no MANIFEST row is
uploaded and its row appended (commit it); a new dataset image has to come from the
dataset build script in the `pii` repo, which is what writes `frames.csv`.

The credential is the `default` AK profile of `~/.aliyun/config.json` (RAM user
esteban, region cn-shanghai). A box that has no such file can instead export
`OSS_ACCESS_KEY_ID` and `OSS_ACCESS_KEY_SECRET`, which take precedence, so no secret
has to be written to that box's disk.

The tracked `.githooks/post-merge` hook runs `pull` after every `git pull`. Git does
not run hooks on clone, so the first pull after cloning is the explicit
`bash data/setup.sh` above. Opt out on a box with `git config pii.autopull false`.

## Build scripts

They live in the `pii` repo (`/home/esteban/repos/pii`), not here:
`data/build_faceback45_pii2.py`, `build_face10k_pii2.py`, `build_faceight_pii2.py`,
`build_facemine_pii2.py`, `build_gt_bench_pii2.py`, `build_pii_frames_pii2.py`,
`build_wider_pii2.py` per dataset; `data/build_view.py` for views;
`data/build_runs_pii2.py` for `runs/train`; `data/oss_pii2_mirror.py` for the OSS
mirror. `data/oss_sync.py` and `data/setup.sh` in this repo are the only things a
consumer needs.
Each has a `verify` or `check` subcommand that re-checks the tree against sizes
and md5s.

## Superseded

`/home/esteban/repos/pii-data` (one commit, no remote) was the earlier metadata
repo. It is not migrated and not maintained; this repo replaces it.

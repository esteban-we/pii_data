# pii_data

The dataset store for the face blur (PII) work, checked out at `/data/esteban/pii` on
gpu-002. Every dataset, every labeling pass, every training mix and every training arm is
described here. The pixels and the checkpoints are not: they live on OSS.

The rule: **git is the ledger, OSS is the byte store.** Three commands are the whole
workflow.

```
git clone git@github.com:esteban-we/pii_data.git
bash data/setup.sh
git pull        # and git push
```

1. **Clone** gives the ledger only: no images, no checkpoints, no rendered view manifests.
2. **`bash data/setup.sh`** fast-forwards the checkout, sets `core.hooksPath .githooks`,
   pulls every byte the index names (236 GB of images plus each arm's pick checkpoint,
   about 20 minutes at 250 MB/s) and renders every view. It is idempotent, and anything
   you pass it goes to the pull, so `bash data/setup.sh --view train_Z5` or `--arm armAF`
   takes one slice instead; the render always covers all views, since it reads the
   indexes and not the images.
3. **`git pull` and `git push`** from then on: the hooks setup.sh enabled bring down the
   bytes a pull's new rows name, and send up the bytes this box made before a push leaves.
   A push is refused if an index row names bytes that are on neither this disk nor OSS.

Everything below is detail.

# Details

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

runs/                       one directory per arm (PII-2118), no family level
  models.csv                arm, family, epoch, status, pick ONNX with md5 and size,
                            oss_key of the pick
  MANIFEST.tsv              every copied file with size, md5 and oss_key (empty when
                            the file is git content)
  <arm>/train/
    pick.yaml               which epoch this arm is, the rule that chose it, evidence,
                            `checkpoints:` and the `pick_checkpoint:` this disk keeps
    config, launch and export scripts, eval/*.json, small logs
    epochs/, onnx/, train.log                                        (ignored, OSS)
  stock_<family>/           upstream baselines, one dir per family (PII-2128):
                            .sha256, .provenance.txt and .md5 tracked, the weights
                            themselves ignored. No train/ level, there being no run;
                            the OSS key still says <family>/stock
  train/logs_replay/        PII-1627's replayed mmdet logs, with their own index

tables/corpus/              session, episode and chunk tables for the whole corpus
tables/population/          15 GB of parquet over the full frame population
                            (ignored, and not on OSS either: PII-1419)
calib/                      fisheye and annotation calibration JSON
inbox/                      vendor deliveries not yet folded into a dataset
experiments/                one-off probes: scripts and their result JSON
```

## What is tracked

Ignored (see `.gitignore`): `datasets/*/images/`, `views/*/{scrfd.txt,d2.json,summary.json}`,
`runs/*/train/{epochs,onnx,logs}/` and `train.log`, every `*.pth` `*.onnx` `*.jit` `*.zip`
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
| facedub_a | 37,568 | v1 rec1 |
| faceback_45 | 84,954 | v1 v2 v3 |
| faceight_a | 42,408 | v1 |
| faceight_b | 60,000 | v1 v2 v3 rec1 |
| faceight_c | 40,000 | v1 v2 v3 |
| face_mine_v1 | 62,587 | v1 v2 v3 |
| face_mine_right | 62,423 | v1 v2 |
| gt_bench_full | 9,636 | v1 |
| gt_bench_sparse | 889 | v1 |
| pii_frames | 10,249 | v1 |
| wider_face | 12,879 | v1 |
| wider_fisheye_fill | 12,879 | v1 |
| wider_fisheye_target | 12,879 | v1 |

460,858 images, 244,789,003,925 bytes, none of them in git. Left and right eyes
are separate datasets (`face_mine_v1` is left, `face_mine_right` is right).
`frames.csv` is the source of truth for which images exist; `size` and `md5` are
mandatory and are what a download is verified against. A `rec1` box version is
the recognizable subset of the version it names, out of the three-pass
facereview review (PII-1915): a strict subset of that version's boxes, plus
`votes.csv`, which carries every pass's own judgement, so another threshold is
rebuilt from that file alone.

## Views

20 views, 12 train mixes and 8 eval sets, each reproducing a legacy manifest
(PII-1373). Frame counts, largest first: train_Z5 and train_Z6 303,407 each
(train_Z6 is train_Z5 with the faceight_b and faceight_c blocks on boxes v3, the
mix the DINOv2 arms armAL..armAU trained on, PII-1681); train_Z3 223,214;
train_Z5noFM 203,873; train_Z2 139,587; train_Z4 123,680; train_X 84,452;
train_Z 71,573; train_W, train_W_fe, train_W_fe2 34,619; train_W_nowider 21,740;
eval_faceback_v1, eval_faceback_hq_v1, eval_faceback_hq_v2 16,940;
eval_face_mine_right_v1 12,722; face_mine_v1 12,754; gt_bench_v1 9,636;
eval_faceight_a_v1 8,482; gt_bench_sparse_v1 889.

## Runs

48 SCRFD arms, 5 EgoBlur arms and 1 RF-DETR arm, plus `stock_scrfd` and
`stock_egoblur`, so 56 rows in `runs/models.csv`. Every
`pick.yaml` is `rule: last_epoch` for now (PII-1372 replaces it with a per-epoch
sweep); `status` is the W&B tag from `alex-qiu-worldengineai/pii-face-eval`, and
it is the default `active` for every arm trained after that read (PII-1633).

### Checkpoint retention (PII-1601)

**OSS keeps every epoch. This disk keeps only the epoch `pick.yaml` names.**
DINOv2 checkpoints are multi-GB each, so holding a whole training run on a box
is not feasible, and the rule is uniform over every arm, scrfd and egoblur
included.

An arm under the rule says `checkpoints: pick` in its `pick.yaml` and in the
`checkpoints` column of `models.csv`, and names the one file it keeps in
`pick_checkpoint:` (`epochs/epoch_20.pth` for a scrfd arm,
`epochs/model_final.pth` for an egoblur arm). The older value `local` means
every epoch is also on disk; `upstream release` is `stock_scrfd` and
`stock_egoblur`, which have no epochs. `epochs/latest.pth` is a symlink, not
bytes: it has no OSS object and is never pruned.

**Not every arm's epochs are on OSS.** The fourteen arms PII-2112 registered
(armAU, armAV34, armAW34, armAX, armAY, armAZ34 and armBA34 to armBH34) were
closed out by a chain that pulls ONE checkpoint off the training box, so the
epochs before the pick are still only on fluence1 or on shang and were never
uploaded. Those arms say `pick only (epochs 1 to N are on the training box, not
on OSS)` instead of plain `pick`, and their `MANIFEST.tsv` rows are the pick
epoch, the ONNX exports and the logs: the index claims no object that does not
exist. Their ONNX was exported on this box straight into `<arm>/onnx/`, which is
why those rows have an empty `src` column.

`MANIFEST.tsv` keeps a row for every epoch either way, with size, md5 and
`oss_key`, so the index still proves the OSS copy complete and any epoch can be
fetched back on demand:

```
python3 data/oss_sync.py pull --all-epochs --arm armAF     # bring the run back
python3 data/oss_sync.py status                            # oss_only column
python3 /home/esteban/repos/pii/data/build_runs_pii2.py prune [--arm A] [--apply]
```

`build_runs_pii2.py prune` (in the `pii` repo) is what deletes them: for each
epoch that is not the pick it HEADs the OSS object and requires the ETag to
equal the manifest md5 and the size to match, and only then removes the local
file. It is a dry run unless `--apply`, takes `--arm A` (repeatable), and never
deletes a file that failed the check; it prints those instead. A pruned epoch is
expected absent, not missing: `build_runs_pii2.py verify` reports it as INFO and
`oss_sync.py status` counts it under `oss_only`.

A training run that lands from another box (`remote_wd`, PII-1447) pushes every
epoch to OSS first and only then takes the pick epoch down here.

Applied 2026-09-28: 604 epochs over 28 arms, 69,300,968,439 B, every one of them
HEADed on OSS with ETag equal to the manifest md5 before it was deleted. armW and
armY held only their pick epoch already, so nothing was taken from them. `verify`
after the prune: OK, 1068 files, 81,046,302,640 B.

## Getting the bytes

A fresh clone is metadata only: `datasets/*/images/` is empty, the checkpoints and
ONNX exports are not there and the view manifests are not rendered. Everything else is
on OSS: as of 2026-09-24 the mirror held 423,703 objects and 297,059,234,866 B, which
is every image of the 13 datasets and every file under `runs/` that git ignores.
PII-2112 added 37,603 objects and 36,338,538,553 B on 2026-10-07, the facedub_a images
and the fourteen arms it registered, and `oss_sync.py status --remote` reported
nothing on this disk and not on OSS afterwards.
An object's ETag is its md5, except for the ones over OSS's 5 GB simple-upload cap, which
go up multipart and carry their file md5 in `x-oss-meta-md5` instead (PII-1633).

Every index row that stands for a file on OSS carries its key:
`datasets/<name>/frames.csv` has an `oss_key` column
(`pii/data/<set>/<image>`; the two `face10k` batches keep the prefixes
`face10k_v3` and `face10k_repair` they were uploaded under), and
`runs/MANIFEST.tsv` has one that is filled in
(`pii/models/<family>/<arm>/<rel>`) exactly for the files git ignores and empty for
the files git tracks. `runs/models.csv` carries the key of each arm's pick.
An OSS key still names the family (`pii/models/<family>/<arm>/<rel>`) even though the
family is no longer a directory on disk: PII-2118 was a local rename and changed no key.

`exports/` is the third index (PII-2151). It holds the Verdict annotation CSVs that
`tools/verdict_export.py` in the `pii` repo writes, 90 files and 657,993,513 B: one
`prelabels.csv` and one `round_N.csv` per review round of each Verdict dataset, plus
`all_rounds.csv.gz`. They are the provenance of every `boxes/vN` GT, so the store keeps
them, as OSS content under `pii/exports/<rel>`: git tracks `exports/MANIFEST.tsv`
(`dst_rel`, an empty `src`, `size`, `md5`, `oss_key`, the same columns as
`runs/MANIFEST.tsv`) and `.gitignore` keeps the CSVs themselves out. It is a file of its
own because `data/build_runs_pii2.py` rewrites `runs/MANIFEST.tsv` whole from the run
tree and would drop any row that is not a run's. `pull`, `push`, `status` and
`push --indexed` take it in with the rest; `--exports` restricts `pull` and `status` to
it. `runs/train/logs_replay/` is not store bytes and is ignored where it is: PII-1627's
replayed mmdet logs carry their own index, `logs_replay/MANIFEST.json`.

```
bash data/setup.sh                       # git pull --ff-only, then the hooks, then pull
python3 data/oss_sync.py pull            # everything the index names
python3 data/oss_sync.py pull --view train_Z5      # only what one view needs
python3 data/oss_sync.py pull --dataset gt_bench_full --arm armAC
python3 data/oss_sync.py pull --all-epochs --arm armAF             # also the pruned epochs
python3 data/oss_sync.py pull --exports            # only the Verdict annotation CSVs
python3 data/oss_sync.py status [--remote]
python3 data/oss_sync.py push --arm armNEW         # after a new training run
python3 data/oss_sync.py push --indexed            # what .githooks/pre-push runs
python3 data/build_view.py render --all            # views are rendered, not downloaded
```

`pull` skips any file already on disk with the indexed size and md5, verifies every
download against the index md5 before renaming it into place, and prints exactly
`up to date` when nothing was missing. It also leaves the epochs the retention rule
above keeps on OSS alone unless it is given `--all-epochs`. The md5 of a local file is cached in
`.git/oss_sync_md5.json` against its size and mtime, so a rerun does not read the
whole tree again.

`push` uploads single part with `Content-MD5` and asserts the returned ETag equals
the md5. It never overwrites: an object already on OSS whose md5 differs from the local
one is reported as a conflict and the run fails. It never deletes. A file under
`runs/<arm>/train/{epochs,onnx,logs}/` with no MANIFEST row is uploaded and its
row appended (commit it); a new dataset image has to come from the dataset build script
in the `pii` repo, which is what writes `frames.csv`.

`push --indexed` is the pre-push mode. It covers exactly what the indexes name
(every `frames.csv` `oss_key`, every filled `oss_key` of `runs/MANIFEST.tsv` and of
`exports/MANIFEST.tsv`, and `models.csv`, whose keys it checks are MANIFEST rows),
invents no MANIFEST row, and fails with `NO BYTES <key>` for any indexed object that is
on neither this disk nor OSS.
An epoch the retention rule keeps on OSS only is not such a case.

The credential is the `default` AK profile of `~/.aliyun/config.json` (RAM user
esteban, region cn-shanghai). A box that has no such file can instead export
`OSS_ACCESS_KEY_ID` and `OSS_ACCESS_KEY_SECRET`, which take precedence, so no secret
has to be written to that box's disk.

Two tracked hooks, both live once `core.hooksPath` is `.githooks`, which setup.sh sets:
`.githooks/post-merge` runs `pull` after every `git pull` (opt out with
`git config pii.autopull false`) and `.githooks/pre-push` runs `push --indexed` before
every `git push` (opt out with `git config pii.autopush false`). Git does not run hooks on
clone, so the first pull after cloning is `bash data/setup.sh`, and that run suppresses the
post-merge hook (`PII_SETUP`) so the OSS pull happens once, not twice.

## Build scripts

They live in the `pii` repo (`/home/esteban/repos/pii`), not here:
`data/build_faceback45_pii2.py`, `build_face10k_pii2.py`, `build_faceight_pii2.py`,
`build_facemine_pii2.py`, `build_gt_bench_pii2.py`, `build_pii_frames_pii2.py`,
`build_wider_pii2.py` per dataset; `data/build_view.py` for views;
`data/build_runs_pii2.py` for `runs/`; `data/oss_pii2_mirror.py` for the OSS
mirror. `data/oss_sync.py` and `data/setup.sh` in this repo are the only things a
consumer needs.
Each has a `verify` or `check` subcommand that re-checks the tree against sizes
and md5s.

## Superseded

`/home/esteban/repos/pii-data` (one commit, no remote) was the earlier metadata
repo. It is not migrated and not maintained; this repo replaces it.

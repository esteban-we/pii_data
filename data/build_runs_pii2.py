#!/usr/bin/env python3
"""PII-1379: build <PII_ROOT>/runs/train from the old tree (/data/esteban/pii_backup).

Layout produced (see PII-1379, parent PII-1315):

    /data/esteban/pii/runs/
        models.csv                  one row per arm
        MANIFEST.tsv                one row per copied file (dst, src, size, md5, oss_key)
        <arm>/train/epochs/         epoch_N.pth from runs/train/wd_<arm>*/
        <arm>/train/onnx/           every <arm>*.onnx from runs/train/
        <arm>/train/pick.yaml       rule: last_epoch
        <arm>/train/...             config and logs, as in the work dir
        train/scrfd/stock/          InsightFace and official SCRFD weights
        train/egoblur/stock/        EgoBlur gen1/gen2 releases and the d2 build
    /data/esteban/pii/experiments/verifier_probe/   probe scripts and json

PII-2118 put the arms in runs/<arm>/train/ and moved the two indexes up to runs/.
The family (scrfd, egoblur, rfdetr) is no longer a directory: the OSS key still
carries it, unchanged, and so does the `family` column of models.csv. The two
upstream-release dirs stayed at runs/train/<family>/stock/, since both families
call them `stock` and one runs/stock/ cannot be both.

A MANIFEST src column is empty when the file was produced on this box straight
into the store and was never copied from anywhere (PII-2112: the ONNX exports of
the arms whose close-out chain exported them into <arm>/onnx/). `copy` leaves
those alone and `verify` has no second copy to compare them against.

Rules held by this script:
  * nothing under the source tree is written, renamed or deleted; every copy
    is an rsync out of it.
  * copies are real copies, never hardlinks.
  * no pAUC or any other eval number is written here; every pick.yaml carries
    rule: last_epoch. The per-epoch sweep (PII-1372) rewrites them later.
  * an epoch that no evidence establishes is written `epoch: unknown` with the
    reason in `notes`.

An arm whose work dir is on another box carries `remote_wd: host:/abs/path`
(PII-1447). Those boxes are read-only: the pull is an rsync out of them with
--ignore-existing, and a file already here is asserted byte identical to the
remote copy before anything is pulled. Their MANIFEST rows keep `host:/abs/path`
in the src column, and `verify` reports them instead of re-reading them.

The pull from a remote work dir is not one command (PII-1447): every epoch is
uploaded from that box straight to OSS first, and only then does this box take
what it keeps. The retention rule below is what it keeps, so the order is:
push every epoch to OSS, then `oss_sync.py pull --arm <arm>`, which fetches the
pick epoch and leaves the rest on OSS.

Checkpoint retention (PII-1601)
-------------------------------
OSS holds every epoch of every arm. This disk holds only the epoch that the
arm's pick.yaml names, in `pick_checkpoint:`. An arm under the rule says
`checkpoints: pick` in pick.yaml and in models.csv. MANIFEST.tsv keeps a row
for every epoch either way, so `verify` still proves OSS complete and
`oss_sync.py pull --all-epochs --arm A` can fetch a pruned epoch on demand.
`epochs/latest.pth` is a symlink, not bytes: it has no OSS object and is never
pruned. `verify` and `oss_sync.py status` count a pruned epoch as expected
absent, not as missing.

Commands:
    python3 data/build_runs_pii2.py build     copy, then write the metadata
    python3 data/build_runs_pii2.py copy      rsync only
    python3 data/build_runs_pii2.py meta      pick.yaml, models.csv, MANIFEST
    python3 data/build_runs_pii2.py verify    size and md5 of every copied file
    python3 data/build_runs_pii2.py remote-check   md5 every file that is both
                                              here and in a remote work dir
    python3 data/build_runs_pii2.py prune [--arm A ...] [--apply]
                                              delete the epochs the retention
                                              rule keeps on OSS only; dry run
                                              unless --apply

`verify` exits 1 on the first class of mismatch it finds and prints every bad
file. Both `build` and `verify` are idempotent: a second build copies nothing
and leaves every metadata file byte identical. `copy` re-copies a pruned epoch
out of a local work dir that still holds it, so run `prune` after a build.
"""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

# PII-1449: the live store; data/pii_root.py resolves it (env PII_ROOT or PII2_ROOT,
# default /data/esteban/pii). A clone elsewhere (shang, fluence) sets the env var.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii_root import CODE_ROOT, PII_ROOT  # noqa: E402

# Provenance (PII-1682): the pre-PII-1315 tree this script read. The project retired it, so the
# paths below record where the data came from; they are not a tree to read today.
LEGACY_ROOT = "/data/esteban/pii_backup"


# The manifest/index is the ledger and was never rewritten for the PII-1448 rename: paths in it
# read /data/esteban/pii/... meaning the tree that became LEGACY_ROOT above (PII-1682).
OLD_ROOT_PREFIX = "/data/esteban/pii/"


def resolve_legacy(path: str) -> str:
    """An absolute path recorded before the PII-1448 rename, as it resolved after it."""
    path = str(path)
    if path.startswith(OLD_ROOT_PREFIX) and not path.startswith(LEGACY_ROOT + "/"):
        return LEGACY_ROOT + "/" + path[len(OLD_ROOT_PREFIX):]
    return path

# PII-1448: the source is the PRE-PII-1315 tree, renamed to /data/esteban/pii_backup.
SRC = Path(LEGACY_ROOT)
SRC_TRAIN = SRC / "runs/train"
SRC_WEIGHTS = SRC / "weights"
SRC_EGOBUILD = SRC / "runs/egoblur_build"
SRC_PROBE = SRC / "runs/verifier_probe"

PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
PII2 = Path(PII2_ROOT)
RUNS = PII2 / "runs"
DST_TRAIN = RUNS / "train"    # PII-2118: only <family>/stock/ and logs_replay/ now
DST_PROBE = PII2 / "experiments/verifier_probe"

MANIFEST = RUNS / "MANIFEST.tsv"
MODELS_CSV = RUNS / "models.csv"

STOCK = ("stock",)


def arm_dir(family, arm):
    """Where an arm's files are (PII-2118): runs/<arm>/train/, no family component.
    `stock` is the exception and keeps runs/train/<family>/stock/."""
    return DST_TRAIN / family / arm if arm in STOCK else RUNS / arm / "train"


def runs_rel(family, arm, rel=""):
    """A path under an arm dir, relative to runs/: what models.csv `onnx` holds."""
    d = arm_dir(family, arm).relative_to(RUNS).as_posix()
    return f"{d}/{rel}" if rel else d

# --------------------------------------------------------------------------
# PII-1601: checkpoint retention. `checkpoints: pick` means OSS holds every
# epoch of this arm and this disk holds only the one pick.yaml names. The value
# sits in both pick.yaml and the models.csv `checkpoints` column; the older
# value `local` means every epoch is also here, and `upstream release` is the
# two stock dirs, which have no epochs at all.
# --------------------------------------------------------------------------
RETENTION = "pick"


def checkpoints_value(declared):
    """The models.csv / pick.yaml `checkpoints` value for an arm.

    The rule is uniform over every trained arm, scrfd and egoblur alike, so an
    arm that used to say `local` now says `pick`. `upstream release` is left
    alone: the stock dirs hold released weights, not epochs.
    """
    return RETENTION if declared == "local" else declared


def parse_pick(path):
    """The top-level `key: value` scalars of a pick.yaml. Block bodies, which
    are indented, and comments are skipped."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line or line[0] in " #\t" or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def epoch_rel_parts(dst_rel):
    """(arm, name) for a manifest row under an arm's epochs/, else None.
    PII-2118: runs/<arm>/train/epochs/<name>."""
    parts = dst_rel.split("/")
    if len(parts) == 5 and parts[0] == "runs" and parts[2:4] == ["train", "epochs"]:
        return parts[1], parts[4]
    return None


def retained_names():
    """{arm: kept epoch file name} for every arm under the rule.

    Read off the registry rather than off the tree: runs/ now holds one dir per
    arm, and a dir another agent's run put there is not ours to interpret.
    """
    keep = {}
    for family, arm, _spec in all_arms():
        pick = arm_dir(family, arm) / "pick.yaml"
        if not pick.is_file():
            continue
        spec = parse_pick(pick)
        if spec.get("checkpoints") != RETENTION or not spec.get("pick_checkpoint"):
            continue
        keep[arm] = os.path.basename(spec["pick_checkpoint"])
    return keep


def oss_only_epochs(rows):
    """dst_rel of every manifest row the retention rule keeps on OSS only.

    A symlink row (size -1, epochs/latest.pth) is an alias, not bytes: it has no
    OSS object, so it is never in this set and never pruned.
    """
    keep = retained_names()
    out = set()
    for r in rows:
        got = epoch_rel_parts(r["dst_rel"])
        if not got or int(r["size"]) < 0:
            continue
        arm, name = got
        kept = keep.get(arm)
        if kept and name != kept:
            out.add(r["dst_rel"])
    return out


# --------------------------------------------------------------------------
# PII-1447: work dirs that live on another box. Those boxes are read-only, so
# `copy` rsyncs epoch_*.pth, latest.pth, the config and the logs OUT of them
# with --ignore-existing and writes nothing back. A file already here is never
# overwritten: `remote-check` first asserts it is byte identical to the remote
# copy, and any difference stops the build.
# --------------------------------------------------------------------------
#
# PII-1633: shang's path is NOT /data/esteban/pii/runs/train any more. PII-1450
# renamed shang's old tree to pii_backup exactly as PII-1448 did here, and
# /data/esteban/pii on shang is now a clone of the store, so the work dirs moved
# with the rename. Checked on 2026-09-29: wd_armAA is under pii_backup and absent
# from the clone. Nothing noticed because after the PII-1601 prune every shang
# arm's only local file is its pick epoch, whose manifest src already names the
# remote path, and `remote-check` skips exactly those rows.
REMOTE_TRAIN = {
    "shang": "/data/esteban/pii_backup/runs/train",
    "fluence1": "/mnt/cachefs/esteban/pii/runs/train",
}

# The routes to each box, as (rsync target, extra rsync options, share). The WAN
# between this box and Shanghai polices about 0.55 MB/s per route no matter how
# many rsync flows share it (measured with 1, 2, 4 and 8 flows), but the
# JumpServer bastion (ssh alias `shang`, 10.5.51.210:2222) and shang's own
# address 10.8.23.220 share that one cap, but not evenly: measured in the same
# window, the bastion gets 82 KB/s while the direct route gets 474 KB/s, a stable
# 1 to 6. Arms are assigned in that ratio (the `share`), so both routes finish at
# about the same time instead of the bastion stranding a 3 GB arm for 11 hours.
# Read-only either way: nothing is written on the remote box.
REMOTE_ROUTES = {
    "shang": [("shang", [], 1),
              ("esteban@10.8.23.220",
               ["-e", "ssh -i /home/esteban/.ssh/id_github_work -o BatchMode=yes"], 6)],
    "fluence1": [("fluence1", [], 1)],
}


# PII-1447: shang's own md5sum and stat output for every epoch_*.pth of the 13
# arms it holds, as arm, wd, name, size, md5. The checkpoints do NOT cross the
# 0.56 MB/s WAN (PII-1455): they were uploaded from shang straight to
# pii/models/scrfd/<arm>/epochs/ at 481 MB/s and come down from OSS with
# data/oss_sync.py pull. This file is what lets MANIFEST.tsv name them (size,
# md5, oss_key) before they are on this disk, which is what oss_sync needs. Once
# they are here, `meta` recomputes the same rows from the files themselves.
#
# PII-1633 added a second file in the same format, fluence1's md5 and stat output
# for the eight work dirs that had no arm dir at all (the DINOv2 line armAL,
# armAN, armAP, armAQ, armAR, the SCRFD-34G armAM34, the Sapiens armAS and the
# RF-DETR armAO). fluence1's WAN to this box measured 0.7 MB/s on a 200 MB rsync,
# so those 854.5 GB took the same shang route: fluence1 -> OSS -> here.
REMOTE_EPOCH_TSV = (
    Path(CODE_ROOT + "/.knuth/tmp/pii1447/shang_epochs.tsv"),
    Path(CODE_ROOT + "/.knuth/tmp/pii1633/fluence1_epochs.tsv"),
    Path(CODE_ROOT + "/.knuth/tmp/pii1633/shang_egoblurB_n_epochs.tsv"),
    # PII-1674: armAT's 30 epochs, md5'd on fluence1 the same way.
    Path(CODE_ROOT + "/.knuth/tmp/pii1674/fluence1_armAT_epochs.tsv"),
)


def remote_epoch_index(paths=None):
    """{arm: {name: (size, md5)}} from the remote md5 lists, {} when absent."""
    if paths is None:
        paths = REMOTE_EPOCH_TSV
    elif isinstance(paths, (str, Path)):
        paths = (paths,)
    out = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out.setdefault(r["arm"], {})[r["name"]] = (int(r["size"]), r["md5"])
    return out


def remote_base(spec):
    """The absolute work dir on the other box.

    PII-2112: `wd` is a name under REMOTE_TRAIN[host] for the arms PII-1447 and
    PII-1633 pulled, and an absolute path when the work dir is somewhere else
    (armBB34 trained in shang's own pii_data checkout, not in the pre-rename
    tree REMOTE_TRAIN["shang"] names).
    """
    host, wd = spec["remote_wd"]
    return wd if wd.startswith("/") else f"{REMOTE_TRAIN[host]}/{wd}"


def remote_src(spec, rel=""):
    """`host:/abs/path` of a file in a remote work dir, or of the dir itself."""
    host, _ = spec["remote_wd"]
    return f"{host}:{remote_base(spec)}" + (f"/{rel}" if rel else "")


def is_remote(src):
    """True for a MANIFEST src column that names another box, host:/abs/path."""
    return str(src).split(":", 1)[0] in REMOTE_TRAIN


# PII-1448 renamed the source tree: what MANIFEST.tsv recorded as /data/esteban/pii/... is
# /data/esteban/pii_backup/... now, and /data/esteban/pii is the store this script writes.
# The manifest is the ledger and is not rewritten for a rename; pii_root.resolve_legacy maps
# the src column instead.
local_src = resolve_legacy


def src_for(spec, rel):
    """Where a file in an arm dir came from: the local work dir when that dir
    holds it, else the arm's remote work dir."""
    if spec.get("wd"):
        p = SRC_TRAIN / spec["wd"] / rel
        if p.exists() or p.is_symlink() or not spec.get("remote_wd"):
            return p
    return remote_src(spec, rel)

# --------------------------------------------------------------------------
# W&B status, read read-only from alex-qiu-worldengineai/pii-face-eval on
# 2026-09-22 (run.tags of the 52 runs). A run tagged `deprecated` gives
# status: deprecated; anything else is active. Nothing was written to W&B.
# --------------------------------------------------------------------------
WANDB_TAGS = {
    "armW": [], "armW_repro": ["deprecated"], "armW_nowider": ["deprecated"],
    "armW_fe2": ["deprecated"], "armX": ["deprecated"], "armY": ["deprecated"],
    "armZ": ["deprecated"], "armZ34": ["deprecated"], "armZZ": ["deprecated"],
    "armZZ34": ["deprecated"], "armAA": ["deprecated"], "armAA34": ["deprecated"],
    "armAB": ["deprecated"], "armAC": ["10g-ablations"], "armAC34": ["deprecated"],
    "armAD": ["deprecated"], "armAE": ["deprecated"], "armAE34": [],
    "armAF": ["10g-ablations"], "armAG34": [], "armAH34": [],
    "armAI34": ["in-training"], "armAJ": [], "armAK": [],
    # PII-1412: armAJ34 is W&B run or81ojpk, whose stale `deprecated` tag was
    # removed in PII-1391; armAK34 is run ozb05ivi, created untagged by
    # PII-1395 / PII-1398. Both therefore read as active.
    "armAJ34": [], "armAK34": [],
    "egoblurA": ["deprecated"], "egoblurB": [],
    "egoblurB_n8": ["deprecated"], "egoblurC_n8": [],
    # stock: W&B runs stock10 and stock34 (10G and 34G baselines), both
    # tagged deprecated; egoblur1_* and egoblur2_* likewise.
    "scrfd_stock": ["deprecated"], "egoblur_stock": ["deprecated"],
    # PII-1633: the nine arms it registered (armAL, armAM34, armAN, armAO, armAP,
    # armAQ, armAR, armAS, egoblurB_n), and armAT after them (PII-1674), are NOT
    # in this map. The map is a read of W&B taken on 2026-09-22, before those runs
    # existed, and neither PII-1633 nor PII-1674 read W&B, so guessing a tag here
    # would be inventing one. WANDB_TAGS.get
    # defaults to [], which reads as `status: active`; that is a default, not a
    # finding, and it is what their pick.yaml says today. Whoever next reads the
    # project's run tags should fill these in: armAN's W&B run was deleted
    # (PII-1629) and egoblurB_n's run poxlh4en belongs to a cancelled training
    # (PII-1175), so at least those two are unlikely to stay active.
}

EXPORTER_NOTE = (
    "The recorded exporter hardcodes CKPT = {wd}/epoch_20.pth, so the ONNX is epoch 20."
)

# PII-1447: the arms whose remaining epochs were pulled out of a shang work dir
# into an arm dir that already held the pick epoch.
PULLED = ("epochs 1..{last}, latest.pth, the config and the logs were pulled from "
          "shang:runs/train/{wd}/ by PII-1447 (shang keeps its copy); the epoch_{n}.pth "
          "already here was asserted byte identical to shang's, not overwritten.")

# PII-1633: the same sentence for the fluence1 work dirs. Every epoch went
# fluence1 -> OSS; this disk took the pick, the config and the logs. fluence1
# keeps its copy: nothing there was written, renamed or removed.
PULLED_FLUENCE = ("All {last} epochs were uploaded from "
                  "fluence1:runs/train/{wd}/ straight to OSS by PII-1633 and this disk "
                  "keeps the pick epoch (PII-1601); the config and the logs came by "
                  "rsync. fluence1 keeps its copy, unmodified.")

# PII-1674: armAT finished after PII-1633 had closed its list, so its epochs went
# up on that issue's route but under this one's number.
PULLED_FLUENCE_1674 = PULLED_FLUENCE.replace("PII-1633", "PII-1674")

# PII-2112: the sentence for an arm a close-out chain pulled ONE checkpoint
# from. Those chains pull the final checkpoint, the config, the slurm files and
# the logs and export the ONNX here from that checkpoint, so the epochs before
# the pick are on the training box and in no object store.
PULLED_PICK = (
    "{route} pulled {ckpt} out of {wd}/, with the md5 compared on both hosts, together with "
    "{files}, and made epochs/latest.pth here; the ONNX "
    "under onnx/ was exported on this box from that checkpoint "
    "(training/export/export_armAH34_onnx.py), not copied. Epochs 1 to {prev} stayed on the "
    "training box and went to no object store, so this arm's rows are the pick epoch and the "
    "exports. The training box keeps its copy, unmodified."
)

# The `checkpoints` value for such an arm. PII-1601's plain `pick` would claim
# OSS holds every epoch, which for these arms it does not; PII-1412 already
# wrote a descriptive value where plain one would have been wrong.
CKPT_PICK_ONLY = "pick only (epochs 1 to {prev} are on the training box, not on OSS)"

# What the chain's whitelist brought with the checkpoint. The fluence arms have
# slurm-<job>.{out,err}; armBB34, babysat on shang, has run_arm.sh instead.
FILES_SLURM = "the config, the slurm files and the logs"

# The two close-out chains that pulled these arms here.
ROUTE_1755 = "PII-1755's close-out chain (pull.sh, the PII-1767 route)"
ROUTE_2041 = "PII-2041's chain.sh (the PII-1767 route with the arm substituted)"

# Per arm: work dir under runs/train (None if the checkpoints are not on this
# box), pick epoch, where the epoch comes from, where checkpoints live.
# `here_onnx` (PII-2112): the arm's ONNX exports were written on this box
# straight into <arm>/onnx/, so they are indexed where they are, with no source
# to copy from. `pick_onnx` names the export models.csv should carry when the
# alphabetical first is not the one the arm was scored at.
SCRFD_ARMS = {
    "armW": dict(
        wd=None,
        loose_ckpt=[(SRC_WEIGHTS / "armW_epoch_20.pth", "epochs/epoch_20.pth")],
        loose_onnx=[(SRC_WEIGHTS / "det_10g_armW.onnx", "onnx/armW_det10g.onnx")],
        loose_other=[(SRC_TRAIN / "armW_effective_cfg.py", "armW_effective_cfg.py")],
        epoch=20, checkpoints="local",
        evidence=("highest (only) checkpoint present, weights/armW_epoch_20.pth; "
                  "data/oss_models_upload.py records armW epochs 20, attributed there to Handover.md"),
        notes=("armW predates this box: no export log ties det_10g_armW.onnx to armW_epoch_20.pth. "
               "Handover.md records the ONNX as pulled from oss://we-atlas/weights/face_pii/det_10g_armW.onnx. "
               "The epoch above is the checkpoint's, not a read export record."),
    ),
    "armW_repro": dict(wd="wd_armW_repro", epoch=20, checkpoints="local",
                       evidence="pii_train/export_repro_onnx.py CKPT = wd_armW_repro/epoch_20.pth",
                       notes="Work dir keeps epochs 5, 10, 15, 20 only (checkpoint interval 5)."),
    "armW_nowider": dict(wd="wd_armW_nowider", epoch=20, checkpoints="local",
                         evidence="pii_train/export_nowider_onnx.py CKPT = wd_armW_nowider/epoch_20.pth",
                         notes="Work dir keeps epochs 5, 10, 15, 20 only (checkpoint interval 5)."),
    "armW_fe2": dict(wd="wd_armW_fe2", epoch=20, checkpoints="local",
                     evidence="pii_train/export_fe2_onnx.py CKPT = wd_armW_fe2/epoch_20.pth",
                     notes="Work dir keeps epochs 5, 10, 15, 20 only (checkpoint interval 5)."),
    "armX": dict(wd="wd_armX", epoch=20, checkpoints="local",
                 evidence="pii_train/export_armX_onnx.py CKPT = wd_armX/epoch_20.pth", notes=""),
    "armY": dict(wd="wd_armY_ckpt", epoch=20, checkpoints="local",
                 evidence="wd_armY_ckpt/export_armY_onnx.py CKPT = wd_armY_ckpt/epoch_20.pth",
                 notes=("wd_armY_ckpt was also used as the drop box for the 34G exporters, so this "
                        "dir carries export_arm{AA34,AG34,AH34,Y,Z34,ZZ34}_onnx.py and "
                        "export_stock34_onnx.py, i.e. the recorded exporters of other arms.")),
    "armZ": dict(wd="wd_armZ", epoch=20, checkpoints="local",
                 evidence="pii_train/export_armZ_onnx.py CKPT = wd_armZ/epoch_20.pth", notes=""),
    "armZ34": dict(wd="wd_armZ34", epoch=20, checkpoints="local",
                   evidence="wd_armY_ckpt/export_armZ34_onnx.py CKPT = wd_armZ34/epoch_20.pth", notes=""),
    "armZZ": dict(wd="wd_armZZ", epoch=20, checkpoints="local",
                  evidence="pii_train/export_armZZ_onnx.py CKPT = wd_armZZ/epoch_20.pth", notes=""),
    "armZZ34": dict(wd="wd_armZZ34", epoch=20, checkpoints="local",
                    evidence="wd_armY_ckpt/export_armZZ34_onnx.py CKPT = wd_armZZ34/epoch_20.pth", notes=""),
    "armAA": dict(wd="wd_armAA_ckpt", remote_wd=("shang", "wd_armAA"),
                  epoch=20, checkpoints="local",
                  evidence=("pii_train/export_armAA_onnx.py CKPT = wd_armAA/epoch_20.pth; "
                            "PII-1311 export_epoch.py asserts that same recorded CKPT"),
                  notes=("Trained on shang. epoch_20.pth was scp'd here early into "
                         "wd_armAA_ckpt; " + PULLED.format(last=19, wd="wd_armAA", n=20))),
    "armAA34": dict(wd="wd_armAA34_ckpt", remote_wd=("shang", "wd_armAA34"),
                    epoch=20, checkpoints="local",
                    evidence=("wd_armY_ckpt/export_armAA34_onnx.py CKPT = wd_armAA34/epoch_20.pth; "
                              "pii-data/models.csv row armAA34 epochs 20"),
                    notes=("Trained on shang. epoch_20.pth was scp'd here early into "
                           "wd_armAA34_ckpt; " + PULLED.format(last=19, wd="wd_armAA34", n=20))),
    "armAB": dict(wd=None, remote_wd=("shang", "wd_armAB"), epoch=20, checkpoints="local",
                  evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                            "CKPT == wd_armAB/epoch_20.pth before rebinding it"),
                  notes=("All 20 epochs, latest.pth, the config and the logs were pulled from "
                         "shang:runs/train/wd_armAB/ by PII-1447; shang keeps its copy.")),
    "armAC": dict(wd="wd_armAC_ckpt", remote_wd=("shang", "wd_armAC"),
                  epoch=20, checkpoints="local",
                  evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                            "CKPT == wd_armAC/epoch_20.pth; PII-1349 re-exported the same "
                            "checkpoint here and reproduced the recorded armAC row"),
                  notes=("Trained on shang. epoch_20.pth was scp'd here early into "
                         "wd_armAC_ckpt; " + PULLED.format(last=19, wd="wd_armAC", n=20))),
    "armAC34": dict(wd="wd_armAC34_ckpt", remote_wd=("shang", "wd_armAC34"),
                    epoch=20, checkpoints="local",
                    evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                              "CKPT == wd_armAC34/epoch_20.pth"),
                    notes=("Trained on shang. epoch_20.pth was scp'd here early into "
                           "wd_armAC34_ckpt; " + PULLED.format(last=19, wd="wd_armAC34", n=20))),
    "armAD": dict(wd=None, remote_wd=("shang", "wd_armAD"), epoch=20, checkpoints="local",
                  evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                            "CKPT == wd_armAD/epoch_20.pth"),
                  notes=("All 20 epochs, latest.pth, the config and the logs were pulled from "
                         "shang:runs/train/wd_armAD/ by PII-1447; shang keeps its copy.")),
    "armAE": dict(wd=None, remote_wd=("shang", "wd_armAE"), epoch=20, checkpoints="local",
                  evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                            "CKPT == wd_armAE/epoch_20.pth"),
                  notes=("All 20 epochs, latest.pth, the config and the logs were pulled from "
                         "shang:runs/train/wd_armAE/ by PII-1447; shang keeps its copy. "
                         "EMA arm: the exporter's key audit expects 158 ema_* keys.")),
    "armAE34": dict(wd=None, remote_wd=("shang", "wd_armAE34"), epoch=20, checkpoints="local",
                    evidence=("PII-1311 / PII-1336 export_epoch.py asserts the recorded exporter's "
                              "CKPT == wd_armAE34/epoch_20.pth"),
                    notes=("All 20 epochs, latest.pth, the config and the logs were pulled from "
                           "shang:runs/train/wd_armAE34/ by PII-1447; shang keeps its copy. "
                           "EMA arm: 443 ema_* keys expected at 34G.")),
    "armAF": dict(wd=None, remote_wd=("shang", "wd_armAF"), epoch=80, checkpoints="local",
                  evidence=("PII-995: the shang export logs print ckpt meta exp_name=scrfd_armAF.py "
                            "epoch=20/55/68/80 iter=69760/191840/237184/279040 for the four ONNX; "
                            "armAF_det10g.onnx is the epoch 80 one, the e20/e55/e68 files name theirs"),
                  notes=("80 epoch run, from scratch, one checkpoint per epoch. All 80 epochs, "
                         "latest.pth, the config and the logs were pulled from "
                         "shang:runs/train/wd_armAF/ by PII-1447; shang keeps its copy.")),
    "armAG34": dict(wd="wd_armAG34", epoch=20, checkpoints="local",
                    evidence="wd_armY_ckpt/export_armAG34_onnx.py CKPT = wd_armAG34/epoch_20.pth", notes=""),
    "armAH34": dict(wd="wd_armAH34_fluence", epoch=20, checkpoints="local",
                    evidence=("wd_armY_ckpt/export_armAH34_onnx.py CKPT = "
                              "wd_armAH34_fluence/epoch_20.pth"),
                    notes="Trained on fluence, work dir staged back here."),
    "armAI34": dict(wd="wd_armAI34_fluence", remote_wd=("fluence1", "wd_armAI34_fluence"),
                    epoch=None, checkpoints="local",
                    evidence=("highest epoch_N.pth in the finished work dir; the per-epoch ONNX "
                              "name their own epoch (armAI34_e<N>_det34g.onnx)"),
                    notes=("Trained on fluence, only the 80 epoch_*.pth were staged back here. "
                           "PII-1447 asserted all 80 byte identical to fluence1's and pulled "
                           "latest.pth, the config and the logs from "
                           "fluence1:runs/train/wd_armAI34_fluence/. The run ended at epoch 80 "
                           "(PII-1406) and the unnumbered pick armAI34_det34g.onnx is byte equal "
                           "to armAI34_e80_det34g.onnx (md5 5b3a38db).")),
    "armAJ": dict(wd=None, remote_wd=("shang", "wd_armAJ"), epoch=30, checkpoints="local",
                  evidence=("PII-1349: epoch_30.pth pulled from shang (md5 2c7d810d...), checkpoint "
                            "meta epoch 30 iter 104,640, exported here to armAJ_det10g.onnx"),
                  notes=("30 epoch run. All 30 epochs, latest.pth, the config and the logs were "
                         "pulled from shang:runs/train/wd_armAJ/ by PII-1447; shang keeps its "
                         "copy. The PII-1349 pull left its epoch_30.pth in "
                         "/data/esteban/tmp/pii1348/, which is scratch.")),
    "armAJ34": dict(
        wd=None,
        loose_ckpt=[(Path("/data/esteban/tmp/pii1359/armAJ34_epoch_30.pth"),
                     "epochs/epoch_30.pth")],
        remote_wd=("shang", "wd_armAJ34"),
        epoch=30, checkpoints="local",
        evidence=("PII-1391: wd_armAJ34/epoch_30.pth pulled from shang (md5 3cc7a70f..., "
                  "118,915,853 B), checkpoint meta exp_name scrfd_armAJ34.py epoch 30 iter "
                  "209,280, exported here with the recorded 34G EMA exporter to "
                  "armAJ34_det34g.onnx (md5 ade29673...)"),
        notes=("30 epoch run with EMA: armAE34's recipe carried to 30 epochs, steps [21, 27] "
               "(PII-1355). Trained on shang. epoch_30.pth came here first from the PII-1391 "
               "pull dir /data/esteban/tmp/pii1359/ (scratch); " +
               PULLED.format(last=29, wd="wd_armAJ34", n=30))),
    "armAK": dict(wd=None, remote_wd=("shang", "wd_armAK"), epoch=30, checkpoints="local",
                  evidence=("PII-1349: epoch_30.pth pulled from shang (md5 ab128646...), checkpoint "
                            "meta epoch 30 iter 104,640, exported here to armAK_det10g.onnx"),
                  notes=("30 epoch run, no EMA. All 30 epochs, latest.pth, the config and the "
                         "logs were pulled from shang:runs/train/wd_armAK/ by PII-1447; shang "
                         "keeps its copy. The PII-1349 pull left its epoch_30.pth in "
                         "/data/esteban/tmp/pii1348/, which is scratch.")),
    "armAK34": dict(
        wd=None,
        loose_ckpt=[(Path("/data/esteban/tmp/pii1395/armAK34_epoch_30.pth"),
                     "epochs/epoch_30.pth")],
        remote_wd=("shang", "wd_armAK34"),
        epoch=30, checkpoints="local",
        evidence=("PII-1397 / PII-1395: wd_armAK34/epoch_30.pth pulled from shang (md5 "
                  "9e5fbf36..., 79,387,400 B), checkpoint meta exp_name scrfd_armAK34.py epoch 30 "
                  "iter 209,280 and zero ema_* keys, exported here with the recorded no-EMA 34G "
                  "exporter to armAK34_det34g.onnx (md5 0d6a6e41...)"),
        notes=("30 epoch run, no EMA: armAC34's recipe carried to 30 epochs (PII-1358). "
               "Trained on shang. epoch_30.pth came here first from the PII-1397 pull dir "
               "/data/esteban/tmp/pii1395/ (scratch); " +
               PULLED.format(last=29, wd="wd_armAK34", n=30))),

    # PII-1633: the seven mmdet arms whose work dir is on fluence1 and which had
    # no dir in this tree at all (the gap PII-1603 recorded). Every epoch of each
    # was uploaded fluence1 -> OSS first; this disk keeps the pick (PII-1601).
    # Their ONNX exports are not registered here: armAL, armAM34 and armAN's sit
    # in the old tree under names onnx_arm() cannot parse, and armAP, armAQ,
    # armAR and armAS's are in /data/esteban/tmp/closeout/<arm>/onnx/ (armAR's
    # with ONNX external data, thousands of sibling tensor files). See PII-1634.
    "armAL": dict(wd=None, remote_wd=("fluence1", "wd_armAL_fluence"),
                  epoch=20, checkpoints="local",
                  evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAL_fluence/, "
                            "latest.pth -> epoch_20.pth; babysit.log and slurm-12805.out "
                            "end clean"),
                  notes=(PULLED_FLUENCE.format(last=20, wd="wd_armAL_fluence") +
                         " DINOv2 ViT-L/14 with registers on a SimpleFPN14 neck, crop 672.")),
    "armAM34": dict(wd=None, remote_wd=("fluence1", "wd_armAM34_fluence"),
                    epoch=20, checkpoints="local",
                    evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAM34_fluence/, "
                              "latest.pth -> epoch_20.pth"),
                    notes=(PULLED_FLUENCE.format(last=20, wd="wd_armAM34_fluence") +
                           " SCRFD-34G control run for the armAL cell.")),
    "armAN": dict(wd=None, remote_wd=("fluence1", "wd_armAN_fluence"),
                  epoch=5, checkpoints="local",
                  evidence=("only epoch_1..5.pth exist in "
                            "fluence1:runs/train/wd_armAN_fluence/ and latest.pth -> "
                            "epoch_5.pth, so the run stopped at epoch 5"),
                  notes=(PULLED_FLUENCE.format(last=5, wd="wd_armAN_fluence") +
                         " A 20 epoch run that ended at epoch 5; PII-1629 records that its "
                         "W&B run was deleted, so no curve was replayed for it.")),
    "armAP": dict(wd=None, remote_wd=("fluence1", "wd_armAP_fluence"),
                  epoch=20, checkpoints="local",
                  evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAP_fluence/, "
                            "latest.pth -> epoch_20.pth; slurm job 13493"),
                  notes=(PULLED_FLUENCE.format(last=20, wd="wd_armAP_fluence") +
                         " First of the PII-1520 weekend chain.")),
    "armAQ": dict(wd=None, remote_wd=("fluence1", "wd_armAQ_fluence"),
                  epoch=20, checkpoints="local",
                  evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAQ_fluence/, "
                            "latest.pth -> epoch_20.pth; slurm job 13589"),
                  notes=(PULLED_FLUENCE.format(last=20, wd="wd_armAQ_fluence") +
                         " Third of the PII-1520 weekend chain.")),
    "armAR": dict(wd=None, remote_wd=("fluence1", "wd_armAR_fluence"),
                  epoch=20, checkpoints="local",
                  evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAR_fluence/, "
                            "latest.pth -> epoch_20.pth; slurm job 13561"),
                  notes=(PULLED_FLUENCE.format(last=20, wd="wd_armAR_fluence") +
                         " DINOv2 ViT-g/14 (1.1 B params), so each epoch is 18.2 GB and the "
                         "arm alone is 364.3 GB of the 854.5 GB PII-1633 moved.")),
    "armAS": dict(wd=None, remote_wd=("fluence1", "wd_armAS_fluence"),
                  epoch=20, checkpoints="local",
                  evidence=("20 epoch_*.pth in fluence1:runs/train/wd_armAS_fluence/, "
                            "latest.pth -> epoch_20.pth; slurm job 21956"),
                  notes=("Sapiens backbone. " +
                         PULLED_FLUENCE.format(last=20, wd="wd_armAS_fluence") +
                         " epoch_20.pth, the config and the logs were already on this disk in "
                         "runs/train/scrfd/wd_armAS_fluence/, the shape PII-1610 flagged as "
                         "neither the work dir's nor the store's; PII-1633 moved that dir to "
                         "scrfd/armAS/ (epoch_20.pth into epochs/) and deleted nothing.")),
    # PII-1674: the tenth fluence1 work dir, and the first registered after
    # PII-1633 closed. It finished on 2026-09-30 at 15:58 UTC; armAU, the other
    # PII-1652 arm, is still training and is NOT registered here.
    "armAT": dict(wd=None, remote_wd=("fluence1", "wd_armAT_fluence"),
                  epoch=30, checkpoints="local",
                  evidence=("30 epoch_*.pth in fluence1:runs/train/wd_armAT_fluence/, "
                            "latest.pth -> epoch_30.pth; babysit.log records the trainer "
                            "exiting rc=0 and `epoch_30.pth exists; success`; slurm job "
                            "24121 on fluence2"),
                  notes=(PULLED_FLUENCE_1674.format(last=30, wd="wd_armAT_fluence") +
                         " The PII-1652 experiment: the armAL recipe (DINOv2 ViT-L/14 with "
                         "registers on a SimpleFPN14 neck, crop 672) stretched from 20 to 30 "
                         "epochs with step [21, 27], nothing else changed.")),

    # ----------------------------------------------------------------------
    # PII-2112: fourteen arm dirs that were on this disk with no row in either
    # index. Every one is a finished run on another box (babysit.log records
    # `success` for each) whose close-out chain brought ONE checkpoint here.
    # ----------------------------------------------------------------------
    "armAU": dict(wd=None, remote_wd=("fluence1", "wd_armAU_fluence"),
                  epoch=30, checkpoints=CKPT_PICK_ONLY.format(prev=29),
                  evidence=("epoch_30.pth, the config and the logs pulled from "
                            "fluence1:runs/train/wd_armAU_fluence/; babysit.log records the "
                            "trainer exiting rc=0 and `epoch_30.pth exists; success`; slurm "
                            "job 24122 on fluence4"),
                  notes=("armAR's recipe (DINOv2 ViT-g/14, crop 672) with the backbone layer "
                         "decay at 0.9 instead of 0.8 and 30 epochs with the lr drops at "
                         "[21, 27] (PII-1654). training/closeout/closeout_arm.sh AU pulled "
                         "epoch_30.pth, 18.2 GB, the config and the logs into "
                         "runs/train/scrfd/wd_armAU_fluence/, the shape PII-1610 flagged as "
                         "neither the work dir's nor the store's; PII-2112 moved that dir to "
                         "scrfd/armAU/ (epoch_30.pth into epochs/) as PII-1633 did for armAS "
                         "and PII-1680 for armAT, and deleted nothing. The other 29 epochs "
                         "stayed on fluence1 and are on no object store. There is no "
                         "latest.pth and no ONNX here: the close-out exports live outside "
                         "this tree (PII-1634), and PII-1690 and PII-1693 record them.")),
    "armAV34": dict(wd=None, remote_wd=("fluence1", "wd_armAV34_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armAV34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_5.pth exists; success`; "
                             "slurm job 27805 on fluence6"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_1755,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armAV34_fluence",
                       prev=4)
                       + " The first of PII-1732's four facedub fine-tunes: armAM34's "
                         "epoch_20 (SCRFD-34G, train_Z6) fine-tuned for 5 epochs on "
                         "train_FD, facedub_a only, lr 1e-4 (PII-1748 job chain, close-out "
                         "PII-1757).")),
    "armAW34": dict(wd=None, remote_wd=("fluence1", "wd_armAW34_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armAW34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_5.pth exists; success`; "
                             "slurm job 27807 on fluence6"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_1755,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armAW34_fluence",
                       prev=4)
                       + " PII-1732's second facedub fine-tune: armAM34's epoch_20 "
                         "fine-tuned for 5 epochs on train_Z6 plus facedub_a x4, lr 1e-4 "
                         "(close-out PII-1767). Its epoch_5.pth and ONNX were put on OSS by "
                         "hand before this registration, outside the index (PII-1971).")),
    "armAX": dict(wd=None, remote_wd=("fluence1", "wd_armAX_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armAX_fluence/; babysit.log records the "
                             "trainer exiting rc=0 and `epoch_5.pth exists; success`; slurm "
                             "job 27806 on fluence6"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_1755,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armAX_fluence", prev=4)
                       + " PII-1732's third facedub fine-tune: armAL's epoch_20 (DINOv2 "
                         "ViT-L/14 with registers on SimpleFPN14, crop 672) fine-tuned for "
                         "5 epochs on train_FD, facedub_a only, lr 1e-5 (PII-1763, "
                         "PII-1966).")),
    "armAY": dict(wd=None, remote_wd=("fluence1", "wd_armAY_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   pick_onnx="onnx/armAY_dinov2l_672.onnx",
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armAY_fluence/; babysit.log records the "
                             "trainer exiting rc=0 and `epoch_5.pth exists; success`; slurm "
                             "job 27808 on fluence6"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_1755,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armAY_fluence", prev=4)
                       + " PII-1732's fourth facedub fine-tune: armAL's epoch_20 fine-tuned "
                         "for 5 epochs on train_Z6 plus facedub_a x4, lr 1e-5 (close-out "
                         "PII-1773). Three ONNX were exported here, at 672, 672x504 and "
                         "532x672; `onnx:` names the 672 one, the export the arm was scored "
                         "and published at. epoch_5.pth and the 672 ONNX went on OSS by "
                         "hand (PII-1963), outside the index.")),
    "armAZ34": dict(wd=None, remote_wd=("fluence1", "wd_armAZ34_fluence"),
                   epoch=3, checkpoints=CKPT_PICK_ONLY.format(prev=2),
                   here_onnx=True,
                   evidence=("epoch_3.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armAZ34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_3.pth exists; success`; "
                             "slurm job 35500 on fluence2"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_3.pth", wd="fluence1:runs/train/wd_armAZ34_fluence",
                       prev=2)
                       + " First of PII-2009's four 34G fine-tunes off armAI34's epoch_80: 3 "
                         "epochs of train_Z6, SGD 1e-4, EMA 2e-4, crop 640 (close-out "
                         "PII-2041).")),
    "armBA34": dict(wd=None, remote_wd=("fluence1", "wd_armBA34_fluence"),
                   epoch=6, checkpoints=CKPT_PICK_ONLY.format(prev=5),
                   here_onnx=True,
                   evidence=("epoch_6.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBA34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_6.pth exists; success`; "
                             "slurm job 35520 on fluence2"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_6.pth", wd="fluence1:runs/train/wd_armBA34_fluence",
                       prev=5)
                       + " armAI34's epoch_80 plus 6 epochs of train_Z6, the longer twin of "
                         "armAZ34 (PII-2031 moved it to fluence; close-out PII-2047). Its "
                         "epoch_6.pth and ONNX were put on OSS by hand (PII-2055), outside "
                         "the index.")),
    "armBC34": dict(wd=None, remote_wd=("fluence1", "wd_armBC34_fluence"),
                   epoch=6, checkpoints=CKPT_PICK_ONLY.format(prev=5),
                   here_onnx=True,
                   evidence=("epoch_6.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBC34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_6.pth exists; success`; "
                             "slurm job 35502 on fluence3"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_6.pth", wd="fluence1:runs/train/wd_armBC34_fluence",
                       prev=5)
                       + " armAI34's epoch_80 plus 6 epochs of train_Z6 with facedub_a x4, "
                         "the mix PII-2009 calls train_Z6+FDx4. Its epoch_6.pth and ONNX "
                         "went on OSS by hand (PII-2055), outside the index.")),
    "armBD34": dict(wd=None, remote_wd=("fluence1", "wd_armBD34_fluence"),
                   epoch=3, checkpoints=CKPT_PICK_ONLY.format(prev=2),
                   here_onnx=True,
                   evidence=("epoch_3.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBD34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_3.pth exists; success`; "
                             "slurm job 36594 on fluence4"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_3.pth", wd="fluence1:runs/train/wd_armBD34_fluence",
                       prev=2)
                       + " armAI34's epoch_80 plus 3 epochs of the mix PII-2041's chain "
                         "records as train_Z6+FDx4+AY8.")),
    "armBE34": dict(wd=None, remote_wd=("fluence1", "wd_armBE34_fluence"),
                   epoch=3, checkpoints=CKPT_PICK_ONLY.format(prev=2),
                   here_onnx=True,
                   evidence=("epoch_3.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBE34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_3.pth exists; success`; "
                             "slurm job 36595 on fluence2"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_3.pth", wd="fluence1:runs/train/wd_armBE34_fluence",
                       prev=2)
                       + " armBC34's epoch_6 plus 3 epochs of the set PII-2041's chain "
                         "records as AY8.")),
    "armBF34": dict(wd=None, remote_wd=("fluence1", "wd_armBF34_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBF34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_5.pth exists; success`; "
                             "slurm job 36277 on fluence4"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armBF34_fluence",
                       prev=4)
                       + " armBA34's epoch_6 plus 5 epochs of train_FD with logit KD from "
                         "armAY (PII-2071).")),
    "armBG34": dict(wd=None, remote_wd=("fluence1", "wd_armBG34_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBG34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_5.pth exists; success`; "
                             "slurm job 36278 on fluence4"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armBG34_fluence",
                       prev=4)
                       + " armBC34's epoch_6 plus 5 epochs of train_FD with logit KD from "
                         "armAY, the armBF34 pair on a different parent (PII-2071).")),
    "armBH34": dict(wd=None, remote_wd=("fluence1", "wd_armBH34_fluence"),
                   epoch=5, checkpoints=CKPT_PICK_ONLY.format(prev=4),
                   here_onnx=True,
                   evidence=("epoch_5.pth, the config and the logs pulled from "
                             "fluence1:runs/train/wd_armBH34_fluence/; babysit.log records "
                             "the trainer exiting rc=0 and `epoch_5.pth exists; success`; "
                             "slurm job 37018 on fluence2"),
                   notes=(PULLED_PICK.format(files=FILES_SLURM,
                       route=ROUTE_2041,
                       ckpt="epoch_5.pth", wd="fluence1:runs/train/wd_armBH34_fluence",
                       prev=4)
                       + " armBC34's epoch_6 plus 5 epochs of train_FD, no KD (PII-2081).")),
    "armBB34": dict(
        wd=None,
        remote_wd=("shang", "/data/esteban/pii/runs/train/scrfd/armBB34"),
        epoch=3, checkpoints=CKPT_PICK_ONLY.format(prev=2), here_onnx=True,
        evidence=("epoch_3.pth, run_arm.sh, the config and the logs pulled from "
                  "shang:/data/esteban/pii/runs/train/scrfd/armBB34/; babysit.log records the "
                  "trainer exiting rc=0 and `epoch_3.pth exists; success`; no slurm job, the "
                  "run was babysat on shang GPUs 1,5,6,7"),
        notes=(PULLED_PICK.format(
            files="the config, run_arm.sh and the logs",
            route=ROUTE_2041, ckpt="epoch_3.pth",
            wd="shang:/data/esteban/pii/runs/train/scrfd/armBB34", prev=2) +
            " armAI34's epoch_80 plus 3 epochs of train_Z6 with facedub_a x4 on shang's four "
            "RTX 5090 instead of fluence (PII-2032; PII-2034 built facedub_a there). shang "
            "trained straight into its own pii_data checkout, so the work dir is that "
            "store's arm dir and not a wd_* dir under REMOTE_TRAIN[\"shang\"].")),
}

EGOBLUR_ARMS = {
    "egoblurA": dict(wd="wd_egoblurA", iters=17447),
    "egoblurB": dict(wd="wd_egoblurB", iters=46499),
    "egoblurB_n8": dict(wd="wd_egoblurB_n8_fluence", iters=34879),
    "egoblurC_n8": dict(wd="wd_egoblurC_n8_fluence", iters=47409),
    # PII-1633: the third arm PII-1456 found with no dir in this tree. Launched on
    # shang GPUs 4 and 5 on 2026-09-18 (PII-1084) and cancelled by the user after
    # epoch 1 (PII-1175), which is why there is no model_final.pth: the one
    # checkpoint kept is model_0055803.pth, so it is the pick.
    "egoblurB_n": dict(wd=None, remote_wd=("shang", "wd_egoblurB_n"), iters=55803,
                       pick="epochs/model_0055803.pth",
                       notes=("Cancelled by the user after epoch 1 at iteration 85,139 "
                              "(PII-1175), so there is no model_final.pth and the pick is "
                              "the one checkpoint kept, model_0055803.pth. Detectron2 on "
                              "shang GPUs 4 and 5, W&B run poxlh4en. The checkpoint and "
                              "train.log were uploaded from shang to OSS by PII-1633; the "
                              "config, log.txt, metrics.json and eval/ came by rsync. Its "
                              "wandb/ dir was left behind. shang keeps its copy.")),
}

# --------------------------------------------------------------------------
# PII-1633: a third family. armAO is RF-DETR, not mmdet and not detectron2, so
# its work dir holds neither epoch_N.pth nor model_N.pth: PyTorch-Lightning
# writes checkpoint_<epoch>.ckpt (0..19 for a 20 epoch run) and the rfdetr EMA
# callback writes last_ema.pth plus checkpoint_best_{ema,total}.pth. There is no
# latest.pth symlink and no ONNX: the evaluator loads the .pth directly
# (evaluation/rfdetr/eval_rfdetr_tables.py), and training/closeout/watcher.sh
# names last_ema.pth as the arm's final checkpoint, which is therefore the pick.
# --------------------------------------------------------------------------
RFDETR_ARMS = {
    "armAO": dict(
        wd=None, remote_wd=("fluence1", "wd_armAO_fluence"),
        epoch=20, epoch_unit="epoch", pick="epochs/last_ema.pth",
        checkpoints="local",
        evidence=("checkpoint_0..19.ckpt (one per epoch, checkpoint_interval=1 in "
                  "training/rfdetr/train_armAO.py) in fluence1:runs/train/wd_armAO_fluence/, "
                  "so the run finished its 20 epochs. The pick is last_ema.pth: "
                  "training/closeout/watcher.sh sets FINAL[AO]=last_ema.pth and PII-1551 "
                  "scored that file, md5 672839c5f85bfbe7bee3b0f36b0869db"),
        notes=("RF-DETR 1.11, 8 H100, the fourth run of the PII-1520 weekend chain (slurm "
               "job 13592). 20 checkpoint_*.ckpt of 2.02 GB plus three EMA .pth; "
               "training_config.json and metrics.csv stand in for an mmdet config and log, "
               "and PII-1629 records that there is no mmdet log to replay. " +
               PULLED_FLUENCE.format(last=20, wd="wd_armAO_fluence"))),
}

# Every registered arm, in one place: (family, arm, spec). PII-1633 made this a
# lookup because three of the callers used to say "scrfd" literally.
ARM_REGISTRY = (("scrfd", SCRFD_ARMS), ("egoblur", EGOBLUR_ARMS),
                ("rfdetr", RFDETR_ARMS))


def all_arms():
    for family, arms in ARM_REGISTRY:
        for arm, spec in arms.items():
            yield family, arm, spec


def family_of(arm):
    for family, arms in ARM_REGISTRY:
        if arm in arms:
            return family
    raise KeyError(f"unknown arm {arm!r}")


def spec_of(arm):
    for _, arms in ARM_REGISTRY:
        if arm in arms:
            return arms[arm]
    raise KeyError(f"unknown arm {arm!r}")


def remote_arms():
    """(family, arm, spec) for every arm whose work dir is on another box."""
    return [(f, a, s) for f, a, s in all_arms() if s.get("remote_wd")]


def _mmdet_like_arms():
    """(family, arm, spec) for the families whose arm dir is `epochs/` plus the
    work dir's own root files: scrfd and rfdetr. egoblur keeps its own loop
    because its source layout differs."""
    return [(f, a, s) for f, a, s in all_arms() if f in ("scrfd", "rfdetr")]

SCRFD_STOCK_FILES = [
    "det_10g.onnx",
    "SCRFD_10G_KPS_official.pth",
    "SCRFD_10G_KPS_official.pth.provenance.txt",
    "SCRFD_10G_KPS_official.pth.sha256",
    "SCRFD_34G_official.pth",
    "SCRFD_34G_official.pth.provenance.txt",
    "SCRFD_34G_official.pth.sha256",
    "stock34_det34g.onnx",
    "stock34_det34g.onnx.md5",
    "stock34_det34g.onnx.provenance.txt",
    "stock34_det34g.onnx.sha256",
]

EGOBLUR_STOCK_FILES = [
    "ego_blur_face_gen1.jit",
    "ego_blur_face_gen1.jit.provenance.txt",
    "ego_blur_face_gen1.jit.sha256",
    "ego_blur_face_gen1.zip",
    "ego_blur_face_gen2.jit",
    "ego_blur_face_gen2.zip",
]

EGOBLUR_BUILD_FILES = [
    "egoblur_face_gen2_d2.pth",
    "egoblur_face_gen2_jit_sd.pth",
    "jit_dump.json",
    "jit_dump.py",
    "parity/parity_n50_1200.json",
    "parity/parity_n50_native.json",
]

# Left behind on purpose, reported in PII-1379.
LEFT_BEHIND = """runs/train: venv, insightface, tiny_ds, pii_train, ship_shang, wd_tiny_smoke,
wd_armX_smoke, wd_armZ_dryrun, wd_egoblurA_smoke, wd_egoblurB_smoke, logs_pii1174,
logs_pii1288, logs_pii1340, logs_replay, logs_wor146, install.log, wd_tiny_smoke.log,
ship_shang_rsync.log.
Per work dir: wandb/ and wd_egoblurB/alex-qiu-worldengineai/ (W&B local run state).
weights: dinov2_vitb14_reg4_pretrain.pth, dinov2_vitl14_reg4_pretrain.pth, dinov2_dl.log
(DINOv2 backbones downloaded for the armAL/simple-FPN line, no arm here uses them).
runs/: egoblur_venv, eval_venv, eda_venv, fluence_stage, mirror_logs, cudnn_lib, wandb.
runs/egoblur_build: detectron2/, jit_archive/, build_venv.{sh,log}, pip_freeze.txt,
mem_probe*.json.
runs/verifier_probe: hf/ (71G), venv/ (8.3G), crops/, sheets/, zoom/, logs/."""


# --------------------------------------------------------------------------
# copy
# --------------------------------------------------------------------------

def run(cmd, log):
    log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
    log.flush()
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"FAIL rsync rc={p.returncode}: {' '.join(str(c) for c in cmd)}")


def rsync_dir(src, dst, log, include=None, exclude=None):
    """Copy a directory. --no-H and --copy-unsafe-links are not used: plain -a,
    which copies data (never hardlinks; rsync has no hardlink-to-source mode)."""
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--no-compress"]
    for pat in include or []:
        cmd += ["--include", pat]
    if include:
        cmd += ["--exclude", "*"]
    for pat in exclude or []:
        cmd += ["--exclude", pat]
    cmd += [str(src) + "/", str(dst) + "/"]
    run(cmd, log)


def rsync_file(src, dst, log):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--no-compress", str(src), str(dst)], log)


def remote_sizes(host, wds):
    """Bytes of epoch_*.pth per work dir, in one ssh call (listing only)."""
    script = ("cd %s; for d in %s; do echo $d $(du -cb $d/epoch_*.pth 2>/dev/null "
              "| tail -1 | cut -f1); done" % (REMOTE_TRAIN[host], " ".join(wds)))
    pr = subprocess.run(["ssh", host, script], capture_output=True, text=True)
    out = {}
    for line in pr.stdout.split("\n"):
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def pull_remotes(logpath):
    """Pull every remote work dir, one thread per route (PII-1447).

    Arms are assigned to the routes of their box longest-first, so both routes
    finish at about the same time. Each thread writes its own log next to
    --log. A failed arm is reported and does not stop the others.
    """
    todo = [(fam, arm, spec) for fam, arm, spec in remote_arms()]
    queues, errors = {}, []
    for host in sorted({spec["remote_wd"][0] for _, _, spec in todo}):
        arms = [(f, a, sp) for f, a, sp in todo if sp["remote_wd"][0] == host]
        size = remote_sizes(host, [sp["remote_wd"][1] for _, _, sp in arms])
        # What is left to fetch, not the arm's whole size: a rerun must balance the
        # remainder, and several arms already hold their pick epoch.
        left = {}
        for fam, arm, sp in arms:
            edir = arm_dir(fam, arm) / "epochs"
            here = sum(q.stat().st_size for q in edir.glob("epoch_*.pth")) \
                if edir.is_dir() else 0
            left[arm] = max(0, size.get(sp["remote_wd"][1], 0) - here)
        routes = REMOTE_ROUTES[host]
        load = [0.0] * len(routes)
        for fam, arm, spec in sorted(arms, key=lambda t: -left[t[1]]):
            i = min(range(len(routes)), key=lambda j: load[j] / routes[j][2])
            load[i] += left[arm]
            queues.setdefault((host, i), []).append((fam, arm, spec))
        print(f"{host}: {len(arms)} arms over {len(routes)} routes, left "
              f"{['%.2f GB' % (b / 1e9) for b in load]}", flush=True)

    def work(key):
        host, ridx = key
        route = REMOTE_ROUTES[host][ridx]
        with open(f"{logpath}.{host}{ridx}", "a") as log:
            for fam, arm, spec in queues[key]:
                t0 = time.time()
                log.write(f"\n==== {arm} via {route[0]} start {time.strftime('%F %T')} ====\n")
                try:
                    pull_remote(spec, arm_dir(fam, arm), route, log)
                    msg = "ok"
                except SystemExit as e:
                    errors.append((arm, str(e)))
                    msg = f"FAIL {e}"
                log.write(f"==== {arm} {msg} {time.time() - t0:.1f}s ====\n")
                log.flush()
                print(f"{arm} via {route[0]}: {msg}, {time.time() - t0:.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(queues))) as ex:
        list(ex.map(work, sorted(queues)))
    return errors


def pull_remote(spec, dst, route, log):
    """Pull a work dir that lives on another box (PII-1447).

    Only latest.pth (a symlink), the config and the logs come this way: they are
    23 MB for all 13 arms, while the epoch_*.pth are 20.86 GB and the WAN is
    policed at 0.56 MB/s (PII-1455), so those travel shang -> OSS -> here
    instead. --ignore-existing: a file already here is never rewritten, so a
    killed transfer resumes at whole-file granularity and a checkpoint pulled
    earlier is left alone (remote_check has already asserted it equals the remote
    copy). Deliberately NOT --partial: rsync's default discards an interrupted
    file instead of leaving a short one under the final name, which
    --ignore-existing would then skip for ever. Nothing is written on the remote
    side.
    """
    target, opts = route[0], route[1]
    src = f"{target}:{remote_base(spec)}/"
    (dst / "epochs").mkdir(parents=True, exist_ok=True)
    base = ["rsync", "-a", "--no-compress", "--ignore-existing", "--stats"] + opts
    run(base + ["--include", "latest.pth", "--exclude", "*",
                src, str(dst / "epochs") + "/"], log)
    # PII-1633: *.ckpt as well as *.pth. armAO is RF-DETR and its 20 epochs are
    # checkpoint_<n>.ckpt of 2.02 GB each, which would otherwise come over the
    # 0.7 MB/s WAN instead of from OSS.
    # train.log is excluded too: oss_key_for() gives it an object, so it comes
    # down with oss_sync.py pull instead of over the WAN. egoblurB_n's is 43 MB,
    # which is 90 minutes at the 0.5 MB/s this link gives. Already rsync'd copies
    # are untouched either way (--ignore-existing).
    run(base + ["--exclude", "*.pth", "--exclude", "*.ckpt", "--exclude", "train.log",
                "--exclude", "wandb/***",
                "--exclude", "__pycache__/***", src, str(dst) + "/"], log)


def remote_check(jobs=8):
    """Every file that is both in an arm dir and in that arm's remote work dir
    must be byte identical (PII-1447). A row whose MANIFEST src already names
    the remote path came from there and `verify` covers it, so it is skipped.

    Returns 0 when every pair matches. A difference is a finding: it is printed
    and the exit code is 1, and nothing is copied over it.
    """
    msrc = {}
    if MANIFEST.exists():
        with open(MANIFEST) as fh:
            msrc = {r["dst_rel"]: r["src"] for r in csv.DictReader(fh, delimiter="\t")}
    todo = {}
    for fam, arm, spec in remote_arms():
        dst = arm_dir(fam, arm)
        cands = []
        for sub_dir, prefix in ((dst / "epochs", "epochs/"), (dst, "")):
            if not sub_dir.is_dir():
                continue
            for q in sorted(sub_dir.iterdir()):
                if not q.is_file() or q.is_symlink() or q.name == "pick.yaml":
                    continue
                cands.append((prefix + q.name, q))
        for rel, q in cands:
            wd_rel = rel[len("epochs/"):] if rel.startswith("epochs/") else rel
            remote = remote_src(spec, wd_rel)
            if msrc.get(q.relative_to(PII2).as_posix()) == remote:
                continue
            todo.setdefault(spec["remote_wd"][0], []).append((remote.split(":", 1)[1], q))
    if not todo:
        print("remote-check: nothing is both here and in a remote work dir")
        return 0
    bad = 0
    for host, pairs in sorted(todo.items()):
        paths = "\n".join(rp for rp, _ in pairs)
        pr = subprocess.run(["ssh", host, "xargs -d '\\n' -r md5sum"], input=paths,
                            capture_output=True, text=True)
        got = {}
        for line in pr.stdout.splitlines():
            digest, _, path = line.partition("  ")
            got[path] = digest
        if pr.returncode != 0:
            print(f"remote-check: ssh {host} md5sum rc={pr.returncode}: {pr.stderr[:400]}")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            mine = dict(zip([rp for rp, _ in pairs],
                            ex.map(md5, [str(q) for _, q in pairs])))
        for rp, q in pairs:
            if rp not in got:
                print(f"FAIL remote-check: {host}:{rp} has no md5 (missing on {host}?)")
                bad += 1
            elif got[rp] != mine[rp]:
                print(f"FAIL remote-check: {q} md5 {mine[rp]} != {host}:{rp} md5 {got[rp]}")
                bad += 1
        print(f"remote-check: {len(pairs) - bad} of {len(pairs)} files on {host} "
              f"are byte identical to the copy here")
    return 1 if bad else 0


def onnx_arm(name):
    """armAI34_e12_det34g.onnx -> armAI34 ; armW_fe2_det10g.onnx -> armW_fe2."""
    stem = name[:-len(".onnx")]
    for suf in ("_det10g", "_det34g"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    head, _, tail = stem.rpartition("_")
    if head and tail.startswith("e") and tail[1:].isdigit():
        stem = head
    return stem


def onnx_epoch(name):
    """The epoch a per-epoch export names, or None."""
    stem = name[:-len(".onnx")]
    for suf in ("_det10g", "_det34g"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    _, _, tail = stem.rpartition("_")
    if tail.startswith("e") and tail[1:].isdigit():
        return int(tail[1:])
    return None


# PII-1447: exports whose arm is not in SCRFD_ARMS, so they belong to no dir in
# this tree. They are skipped, loudly: the arm has to be added here first.
# armAL (DINOv2 simple-FPN line) and armAM34 were exported on 2026-09-24 and
# their work dirs are on fluence1, not on this box. PII-1601 added the rest of
# the DINOv2 line: armAN's exports landed in runs/train on 2026-09-24 19:50
# (the PII-1479 pull out of fluence1) and stopped `meta` dead, because this list
# named only armAL and armAM34.
#
# PII-1633 registered all eight arms and every epoch, and this list still stands:
# onnx_arm() parses the InsightFace style name <arm>[_e<N>]_det{10,34}g.onnx, and
# these exports are armAL_dinov2l_1024.onnx / armAN_dinov2l_672.onnx, so the arm
# it returns is armAL_dinov2l_1024, not armAL. Registering the DINOv2 exports
# needs a naming rule, not an arm dir, so it stays out: PII-1634 carries it,
# together with the armAP, armAQ, armAR and armAS exports that sit under
# /data/esteban/tmp/closeout/<arm>/onnx/ (armAR's with ONNX external data).
UNKNOWN_ARM_ONNX = ("armAL", "armAM34", "armAN", "armAO", "armAP", "armAQ",
                    "armAR", "armAS")


def scan_onnx():
    by_arm, skipped = {}, []
    for p in sorted(SRC_TRAIN.glob("*.onnx")):
        arm = onnx_arm(p.name)
        if arm not in SCRFD_ARMS:
            if not arm.startswith(UNKNOWN_ARM_ONNX):
                raise SystemExit(f"unknown arm {arm!r} parsed from {p.name}")
            skipped.append(p.name)
            continue
        by_arm.setdefault(arm, []).append(p)
    if skipped:
        print("INFO no arm dir for these exports, not copied: " + ", ".join(skipped))
    return by_arm


def do_copy(logpath):
    t0 = time.time()
    with open(logpath, "a") as log:
        log.write(f"\n==== copy start {time.strftime('%F %T')} ====\n")

        # scrfd and rfdetr work dirs
        for family, arm, spec in _mmdet_like_arms():
            dst = arm_dir(family, arm)
            if spec["wd"]:
                wd = SRC_TRAIN / spec["wd"]
                rsync_dir(wd, dst / "epochs", log,
                          include=["epoch_*.pth", "latest.pth"])
                rsync_dir(wd, dst, log,
                          exclude=["epoch_*.pth", "latest.pth", "wandb/***",
                                   "__pycache__/***"])
            for src, rel in spec.get("loose_ckpt", []):
                rsync_file(src, dst / rel, log)
            for src, rel in spec.get("loose_other", []):
                rsync_file(src, dst / rel, log)
            for src, rel in spec.get("loose_onnx", []):
                rsync_file(src, dst / rel, log)

        # scrfd onnx
        for arm, paths in scan_onnx().items():
            for p in paths:
                rsync_file(p, arm_dir("scrfd", arm) / "onnx" / p.name, log)

        # scrfd stock
        for name in SCRFD_STOCK_FILES:
            rsync_file(SRC_WEIGHTS / name, DST_TRAIN / "scrfd/stock" / name, log)

        # egoblur work dirs. PII-1633: egoblurB_n's is on shang, and a remote
        # work dir is pulled by pull_remotes(), not rsynced out of the old tree.
        for arm, spec in EGOBLUR_ARMS.items():
            if not spec["wd"]:
                continue
            wd = SRC_TRAIN / spec["wd"]
            dst = arm_dir("egoblur", arm)
            rsync_dir(wd, dst / "epochs", log, include=["model_*.pth"])
            rsync_dir(wd, dst, log,
                      exclude=["model_*.pth", "wandb/***",
                               "alex-qiu-worldengineai/***", "__pycache__/***"])

        # egoblur stock
        for name in EGOBLUR_STOCK_FILES:
            rsync_file(SRC_WEIGHTS / name, DST_TRAIN / "egoblur/stock" / name, log)
        for rel in EGOBLUR_BUILD_FILES:
            rsync_file(SRC_EGOBUILD / rel,
                       DST_TRAIN / "egoblur/stock/egoblur_build" / rel, log)

        # verifier probe: scripts and json only
        rsync_dir(SRC_PROBE, DST_PROBE, log, include=["*.py", "*.json"])

        log.write(f"==== local copy done {time.strftime('%F %T')} "
                  f"{time.time() - t0:.1f}s ====\n")
    errors = pull_remotes(logpath)
    for arm, msg in errors:
        print(f"FAIL remote pull {arm}: {msg}")
    if errors:
        raise SystemExit(f"FAIL {len(errors)} remote work dirs did not pull")
    return time.time() - t0


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_job(args):
    rel, path = args
    p = Path(path)
    if p.is_symlink():
        return rel, -1, "symlink:" + os.readlink(p)
    if not p.exists():
        return rel, -2, "missing:" + str(p)     # PII-1449: report, do not crash the pool
    return rel, p.stat().st_size, md5(p)


def copied_files():
    """Every file this script puts in pii2, as (rel_to_pii2, abs_dst, abs_src)."""
    out = []

    def walk(dst_root, src_of):
        for dirpath, dirnames, filenames in os.walk(dst_root):
            dirnames.sort()
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = p.relative_to(PII2).as_posix()
                if fn in ("MANIFEST.tsv", "models.csv", "pick.yaml", "README.md"):
                    continue
                src = src_of(p)
                if src is None:
                    continue
                out.append((rel, p, src))

    onnx_by_arm = scan_onnx()

    # PII-1633: rfdetr has the same dir shape as a remote scrfd arm (epochs/ plus
    # the work dir's root files), so the two families share this loop. armAO has
    # no onnx/ and no epochs/latest.pth.
    for family, arm, spec in _mmdet_like_arms():
        dst = arm_dir(family, arm)
        if spec["wd"] or spec.get("remote_wd"):
            walk(dst / "epochs", lambda p, sp=spec: src_for(sp, p.name))
            for dirpath, dirnames, filenames in os.walk(dst):
                dirnames[:] = [d for d in sorted(dirnames) if d not in ("epochs", "onnx")]
                for fn in sorted(filenames):
                    if fn in ("pick.yaml",):
                        continue
                    p = Path(dirpath) / fn
                    out.append((p.relative_to(PII2).as_posix(), p,
                                src_for(spec, p.relative_to(dst).as_posix())))
        for src, rel in (spec.get("loose_ckpt", []) + spec.get("loose_other", [])
                         + spec.get("loose_onnx", [])):
            p = dst / rel
            out.append((p.relative_to(PII2).as_posix(), p, src))
        for src in onnx_by_arm.get(arm, []):
            p = dst / "onnx" / src.name
            out.append((p.relative_to(PII2).as_posix(), p, src))
        # PII-2112: an arm whose ONNX was exported on this box straight into
        # the store has no source to copy from, so its row's src column is
        # EMPTY, the way an oss_key is empty for a file git tracks. `copy` has
        # nothing to do for such a file and `verify` has no second copy to
        # compare it against.
        if spec.get("here_onnx"):
            for q in sorted((dst / "onnx").glob("*.onnx")):
                out.append((q.relative_to(PII2).as_posix(), q, ""))

    for name in SCRFD_STOCK_FILES:
        p = DST_TRAIN / "scrfd/stock" / name
        out.append((p.relative_to(PII2).as_posix(), p, SRC_WEIGHTS / name))

    for arm, spec in EGOBLUR_ARMS.items():
        dst = arm_dir("egoblur", arm)
        for dirpath, dirnames, filenames in os.walk(dst):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn == "pick.yaml":
                    continue
                p = Path(dirpath) / fn
                relp = p.relative_to(dst)
                srcrel = Path(*relp.parts[1:]) if relp.parts[0] == "epochs" else relp
                # PII-1633: src_for() picks the local work dir when it holds the
                # file and the arm's remote work dir otherwise, which is what
                # egoblurB_n (on shang, wd None) needs.
                out.append((p.relative_to(PII2).as_posix(), p,
                            src_for(spec, srcrel.as_posix())))

    for name in EGOBLUR_STOCK_FILES:
        p = DST_TRAIN / "egoblur/stock" / name
        out.append((p.relative_to(PII2).as_posix(), p, SRC_WEIGHTS / name))
    for rel in EGOBLUR_BUILD_FILES:
        p = DST_TRAIN / "egoblur/stock/egoblur_build" / rel
        out.append((p.relative_to(PII2).as_posix(), p, SRC_EGOBUILD / rel))

    for p in sorted(DST_PROBE.rglob("*")):
        if p.is_file():
            out.append((p.relative_to(PII2).as_posix(), p,
                        SRC_PROBE / p.relative_to(DST_PROBE)))

    seen, uniq = set(), []
    for rel, dstp, srcp in out:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append((rel, dstp, srcp))
    return sorted(uniq)


def yaml_block(text, indent="  "):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    body = " ".join(lines)
    wrapped, cur = [], ""
    for word in body.split():
        if len(cur) + len(word) + 1 > 92:
            wrapped.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        wrapped.append(cur)
    return "\n".join(indent + l for l in wrapped)


def scrfd_pick(arm, spec, onnx_paths, dst):
    epoch = spec["epoch"]
    loose = [rel.split("/")[-1] for _, rel in spec.get("loose_onnx", [])]
    if spec.get("here_onnx"):
        loose += [q.name for q in sorted((dst / "onnx").glob("*.onnx"))]
    epochs_dir = dst / "epochs"
    local_epochs = sorted(int(p.stem.split("_")[1])
                          for p in epochs_dir.glob("epoch_*.pth")) if epochs_dir.is_dir() else []
    if epoch is None:
        epoch = max(local_epochs) if local_epochs else "unknown"
    per_epoch = sorted((onnx_epoch(p.name), p.name) for p in onnx_paths
                       if onnx_epoch(p.name) is not None)
    plain = sorted([p.name for p in onnx_paths if onnx_epoch(p.name) is None] + loose)
    if plain:
        pick_onnx = "onnx/" + plain[0]
    elif per_epoch:
        pick_onnx = "onnx/" + per_epoch[-1][1]
    else:
        pick_onnx = ""
    # PII-2112: armAY has three exports and the alphabetical first is not the
    # one it was scored at, so the spec may name the pick.
    if spec.get("pick_onnx"):
        pick_onnx = spec["pick_onnx"]
    tags = WANDB_TAGS.get(arm, [])
    status = "deprecated" if "deprecated" in tags else "active"
    notes = spec["notes"]
    checkpoints = checkpoints_value(spec["checkpoints"])
    pick_ckpt = f"epochs/epoch_{epoch}.pth" if isinstance(epoch, int) else ""
    if not local_epochs:
        notes = (notes + " ").strip() + f" checkpoints: {checkpoints}."
    lines = [
        "# PII-1379. rule is last_epoch for every arm until the per-epoch sweep",
        "# (PII-1372) replaces it. No eval number is recorded here.",
        f"arm: {arm}",
        "family: scrfd",
        "rule: last_epoch",
        f"epoch: {epoch}",
        "epoch_unit: epoch",
        f"status: {status}",
        f"wandb_tags: [{', '.join(tags)}]",
        f"onnx: {pick_onnx}" if pick_onnx else "onnx:",
        f"checkpoints: {checkpoints}",
        f"pick_checkpoint: {pick_ckpt}" if pick_ckpt else "pick_checkpoint:",
    ] + ([f"remote_wd: {remote_src(spec)}"] if spec.get("remote_wd") else []) + [
        f"local_epochs: {len(local_epochs)}",
        "epoch_evidence: >-",
        yaml_block(spec["evidence"]),
    ]
    if notes.strip():
        lines += ["notes: >-", yaml_block(notes)]
    return "\n".join(lines) + "\n", epoch, status, pick_onnx, checkpoints


def egoblur_pick(arm, spec, dst):
    epochs_dir = dst / "epochs"
    iters = sorted(int(p.stem.split("_")[1]) for p in epochs_dir.glob("model_0*.pth"))
    last = max(iters) if iters else spec["iters"]
    tags = WANDB_TAGS.get(arm, [])
    status = "deprecated" if "deprecated" in tags else "active"
    # PII-1633: a cancelled run has no model_final.pth, so the pick is explicit
    # in the spec; every completed arm keeps the old default.
    pick_ckpt = spec.get("pick", "epochs/model_final.pth")
    lines = [
        "# PII-1379. rule is last_epoch for every arm until the per-epoch sweep",
        "# (PII-1372) replaces it. No eval number is recorded here.",
        f"arm: {arm}",
        "family: egoblur",
        "rule: last_epoch",
        f"epoch: {last}",
        "epoch_unit: iter",
        f"status: {status}",
        f"wandb_tags: [{', '.join(tags)}]",
        "onnx:",
        f"checkpoints: {RETENTION}",
        f"pick_checkpoint: {pick_ckpt}",
    ] + ([f"remote_wd: {remote_src(spec)}"] if spec.get("remote_wd") else []) + [
        f"local_checkpoints: {len(list(epochs_dir.glob('model_*.pth')))}",
        "epoch_evidence: >-",
        yaml_block(
            spec.get("evidence") or
            ("detectron2 counts iterations, not epochs. The work dir's last_checkpoint file names "
             "model_final.pth and the highest numbered checkpoint is model_%07d.pth, so epoch above "
             "is that iteration. The pick is %s." % (last, pick_ckpt))),
        "notes: >-",
        yaml_block(
            "There is no ONNX for this family: the EgoBlur arms are scored from the detectron2 "
            "checkpoint. " + (spec["notes"] if spec.get("notes") else
                              "Source work dir " + spec["wd"] +
                              "; its wandb/ dir was left behind.")),
    ]
    return "\n".join(lines) + "\n", last, status


def rfdetr_pick(arm, spec, dst):
    """pick.yaml for an RF-DETR arm (PII-1633).

    PyTorch-Lightning writes checkpoint_<epoch>.ckpt, zero based, and the rfdetr
    EMA callback writes last_ema.pth; the pick is the EMA file the evaluator and
    training/closeout/watcher.sh use, not the last .ckpt, so `pick` is explicit
    in the spec instead of derived from the highest epoch number.
    """
    epochs_dir = dst / "epochs"
    local = sorted(p.name for p in epochs_dir.glob("*")
                   if p.is_file()) if epochs_dir.is_dir() else []
    tags = WANDB_TAGS.get(arm, [])
    status = "deprecated" if "deprecated" in tags else "active"
    checkpoints = checkpoints_value(spec["checkpoints"])
    lines = [
        "# PII-1379. rule is last_epoch for every arm until the per-epoch sweep",
        "# (PII-1372) replaces it. No eval number is recorded here.",
        f"arm: {arm}",
        "family: rfdetr",
        "rule: last_epoch",
        f"epoch: {spec['epoch']}",
        f"epoch_unit: {spec.get('epoch_unit', 'epoch')}",
        f"status: {status}",
        f"wandb_tags: [{', '.join(tags)}]",
        "onnx:",
        f"checkpoints: {checkpoints}",
        f"pick_checkpoint: {spec['pick']}",
    ] + ([f"remote_wd: {remote_src(spec)}"] if spec.get("remote_wd") else []) + [
        f"local_checkpoints: {len(local)}",
        "epoch_evidence: >-",
        yaml_block(spec["evidence"]),
        "notes: >-",
        yaml_block(spec["notes"]),
    ]
    return "\n".join(lines) + "\n", spec["epoch"], status


def stock_pick(family, files_note):
    tags = WANDB_TAGS["scrfd_stock" if family == "scrfd" else "egoblur_stock"]
    return "\n".join([
        "# PII-1379. Stock weights, not trained here.",
        f"arm: stock",
        f"family: {family}",
        "rule: last_epoch",
        "epoch: unknown",
        "epoch_unit: epoch",
        "status: deprecated",
        f"wandb_tags: [{', '.join(tags)}]",
        "onnx:",
        "checkpoints: upstream release",
        "pick_checkpoint:",
        "epoch_evidence: >-",
        yaml_block("Upstream released weights: no training run on any box here produced them, so "
                   "no epoch can be established. The .provenance.txt and .sha256 files next to "
                   "each weight record where it came from."),
        "notes: >-",
        yaml_block(files_note),
    ]) + "\n"


def write_meta(logpath):
    t0 = time.time()
    onnx_by_arm = scan_onnx()
    rows = []

    for arm, spec in SCRFD_ARMS.items():
        dst = arm_dir("scrfd", arm)
        text, epoch, status, pick_onnx, checkpoints = scrfd_pick(
            arm, spec, onnx_by_arm.get(arm, []), dst)
        (dst / "pick.yaml").write_text(text)
        pick_abs = dst / pick_onnx if pick_onnx else None
        rows.append(dict(
            arm=arm, family="scrfd", epoch=epoch, status=status,
            onnx=(runs_rel("scrfd", arm, pick_onnx) if pick_onnx else ""),
            onnx_md5=(md5(pick_abs) if pick_abs and pick_abs.exists() else ""),
            onnx_size=(pick_abs.stat().st_size if pick_abs and pick_abs.exists() else ""),
            n_onnx=(len(onnx_by_arm.get(arm, [])) + len(spec.get("loose_onnx", []))
                    + (len(list((dst / "onnx").glob("*.onnx")))
                       if spec.get("here_onnx") else 0)),
            checkpoints=checkpoints,
            source=(str(SRC_TRAIN / spec["wd"]) if spec["wd"] else
                    (remote_src(spec) if spec.get("remote_wd") else
                     (str(spec["loose_onnx"][0][0].parent) if spec.get("loose_onnx")
                      else str(SRC_TRAIN)))),
        ))

    for arm, spec in RFDETR_ARMS.items():
        dst = arm_dir("rfdetr", arm)
        dst.mkdir(parents=True, exist_ok=True)
        text, epoch, status = rfdetr_pick(arm, spec, dst)
        (dst / "pick.yaml").write_text(text)
        rows.append(dict(
            arm=arm, family="rfdetr", epoch=epoch, status=status, onnx="",
            onnx_md5="", onnx_size="", n_onnx=0,
            checkpoints=checkpoints_value(spec["checkpoints"]),
            source=(remote_src(spec) if spec.get("remote_wd")
                    else str(SRC_TRAIN / spec["wd"])),
        ))

    for arm, spec in EGOBLUR_ARMS.items():
        dst = arm_dir("egoblur", arm)
        dst.mkdir(parents=True, exist_ok=True)
        text, last, status = egoblur_pick(arm, spec, dst)
        (dst / "pick.yaml").write_text(text)
        rows.append(dict(
            arm=arm, family="egoblur", epoch=last, status=status, onnx="",
            onnx_md5="", onnx_size="", n_onnx=0, checkpoints=RETENTION,
            source=(str(SRC_TRAIN / spec["wd"]) if spec["wd"] else remote_src(spec)),
        ))

    (DST_TRAIN / "scrfd/stock/pick.yaml").write_text(stock_pick(
        "scrfd",
        "det_10g.onnx is the InsightFace production 10G detector. "
        "SCRFD_10G_KPS_official.pth and SCRFD_34G_official.pth are the official SCRFD "
        "checkpoints the arms fine-tune from. stock34_det34g.onnx is the 34G export made "
        "here by wd_armY_ckpt/export_stock34_onnx.py (copied to scrfd/armY/), so it is the "
        "only file in this dir produced on this box. Source: the old tree's weights/."))
    (DST_TRAIN / "egoblur/stock/pick.yaml").write_text(stock_pick(
        "egoblur",
        "ego_blur_face_gen1 and gen2 are Meta EgoBlur releases (.jit plus the .zip they "
        "came in). egoblur_build/ holds what the gen2 to detectron2 conversion produced: "
        "egoblur_face_gen2_jit_sd.pth (state dict lifted out of the TorchScript archive), "
        "egoblur_face_gen2_d2.pth (the detectron2 checkpoint the egoblur arms start from), "
        "jit_dump.json plus jit_dump.py, and parity/. gen2 has no .sha256 or "
        ".provenance.txt next to it in the source; gen1 does. Sources: "
        "the old tree's weights/ and runs/egoblur_build/."))
    rows.append(dict(arm="stock", family="scrfd", epoch="unknown", status="deprecated",
                     onnx=runs_rel("scrfd", "stock", "det_10g.onnx"),
                     onnx_md5=md5(DST_TRAIN / "scrfd/stock/det_10g.onnx"),
                     onnx_size=(DST_TRAIN / "scrfd/stock/det_10g.onnx").stat().st_size,
                     n_onnx=2, checkpoints="upstream release", source=str(SRC_WEIGHTS)))
    rows.append(dict(arm="stock", family="egoblur", epoch="unknown", status="deprecated",
                     onnx="", onnx_md5="", onnx_size="", n_onnx=0,
                     checkpoints="upstream release", source=str(SRC_WEIGHTS)))

    # oss_key: the object that holds this arm's pick, the ONNX where there is one and the
    # detectron2 checkpoint for egoblur (PII-1413).
    # PII-1633: when there is no ONNX the pick is whatever the arm's pick.yaml
    # names in pick_checkpoint, which covers egoblur's epochs/model_final.pth as
    # before, rfdetr's epochs/last_ema.pth and the DINOv2 arms' epoch file.
    for r in rows:
        pick = r["onnx"]
        if not pick:
            pf = arm_dir(r["family"], r["arm"]) / "pick.yaml"
            ck = parse_pick(pf).get("pick_checkpoint", "") if pf.is_file() else ""
            pick = runs_rel(r["family"], r["arm"], ck) if ck else ""
        r["oss_key"] = oss_key_for(f"runs/{pick}") if pick else ""
    cols = ["arm", "family", "epoch", "status", "onnx", "onnx_md5", "onnx_size",
            "n_onnx", "checkpoints", "source", "oss_key"]
    with open(MODELS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["family"], r["arm"])):
            w.writerow(r)

    # manifest
    files = copied_files()
    # PII-1447: an epoch that is on OSS but not yet on this disk still gets its
    # row, from shang's md5 list, so `oss_sync.py pull --arm` can fetch it.
    pending = []
    for arm, epochs in remote_epoch_index().items():
        dst = arm_dir(family_of(arm), arm)
        for name, (size, digest) in sorted(epochs.items()):
            rel = (dst / "epochs" / name).relative_to(PII2).as_posix()
            if not (dst / "epochs" / name).exists():
                pending.append((rel, size, digest,
                                remote_src(spec_of(arm), name)))
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_md5_job, [(rel, str(p)) for rel, p, _ in files],
                              chunksize=8))
    srcs = {rel: str(s) for rel, _, s in files}
    rows_out = [(rel, srcs[rel], size, digest) for rel, size, digest in results]
    rows_out += [(rel, src, size, digest) for rel, size, digest, src in pending]
    # PII-1601: an epoch this box no longer holds keeps its row, carried over
    # from the manifest that was here before, so MANIFEST.tsv still names every
    # epoch on OSS and `oss_sync.py pull --all-epochs` can fetch it back. An arm
    # whose work dir is on shang gets the same row from the remote md5 list
    # above instead, so the two paths overlap and only the union is counted.
    prev = []
    if MANIFEST.exists():
        with open(MANIFEST) as fh:
            prev = list(csv.DictReader(fh, delimiter="\t"))
    have = {rel for rel, _, _, _ in rows_out}
    for rel in sorted(oss_only_epochs(prev)):
        if rel in have or (PII2 / rel).exists():
            continue
        r = next(p for p in prev if p["dst_rel"] == rel)
        rows_out.append((rel, r["src"], int(r["size"]), r["md5"]))

    # Split what the manifest names but this disk does not hold into the epochs
    # the retention rule deliberately leaves on OSS and anything else, which is
    # a file that has yet to be pulled.
    off_disk = [rel for rel, _, size, _ in rows_out
                if size >= 0 and not (PII2 / rel).exists()]
    pruned_set = oss_only_epochs([{"dst_rel": rel, "size": str(size)}
                                  for rel, _, size, _ in rows_out])
    n_pruned = sum(1 for rel in off_disk if rel in pruned_set)
    n_pending = len(off_disk) - n_pruned
    total = 0
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["dst_rel", "src", "size", "md5", "oss_key"])
        for rel, src, size, digest in sorted(rows_out):
            w.writerow([rel, src, size, digest, "" if size < 0 else oss_key_for(rel)])
            if size > 0:
                total += size
    print(f"meta: {len(rows)} models.csv rows, {len(rows_out)} manifest files "
          f"({n_pruned} pruned epochs, on OSS only; {n_pending} on OSS, not yet "
          f"pulled to this disk), {total:,} B, {time.time() - t0:.1f}s")
    return total


# --------------------------------------------------------------------------
# OSS keys (PII-1413)
# --------------------------------------------------------------------------

# oss://algorithm-datasets/pii/models/<family>/<arm>/<rel> holds every file under
# runs/train that git does not track (checkpoints, ONNX exports, the training logs).
# Everything else in this tree is git content and has an empty oss_key.
MODELS_TOP = "pii/models/"


def oss_key_for(dst_rel: str) -> str:
    """The OSS key of a manifest row, or "" when the file is git content.

    PII-2118 did not change one key. The family the key carries comes from the
    registry for an arm (runs/<arm>/train/<tail>) and from the path itself for the
    two runs/train/<family>/stock/ dirs.
    """
    parts = dst_rel.split("/")
    if parts[0] != "runs" or len(parts) < 4:
        return ""
    if parts[2] == "train":
        arm, tail = parts[1], "/".join(parts[3:])
        try:
            family = family_of(arm)
        except KeyError:
            return ""
    elif parts[1] == "train" and parts[3] in STOCK:
        family, arm, tail = parts[2], parts[3], "/".join(parts[4:])
    else:
        return ""
    base = os.path.basename(tail)
    on_oss = (tail.startswith(("epochs/", "onnx/", "logs/")) or tail == "train.log"
              or base.endswith((".pth", ".onnx", ".jit", ".zip")))
    return f"{MODELS_TOP}{family}/{arm}/{tail}" if on_oss else ""


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

SKIP_NAMES = ("MANIFEST.tsv", "models.csv", "pick.yaml")

# PII-1633: runs/train/logs_replay/ is not built by this script and is not in
# MANIFEST.tsv. It is PII-1627's replayed mmdet logs, one .log and .log.json per
# arm, and it carries its own index, logs_replay/MANIFEST.json, which records
# each entry's source_host, source_path, md5, row count and W&B run id. So it is
# indexed, just not here, and `verify` skips it instead of calling 63 files
# untracked. Registering it in MANIFEST.tsv is PII-1635.
SKIP_DIRS = ("logs_replay",)


def disk_roots():
    """Every tree this script owns: one dir per registered arm, the two stock dirs
    and the verifier probe. PII-2118 replaced the single walk over runs/train/ with
    this list, so a runs/<arm>/train/ another agent's run put there is never
    mistaken for a file of ours that fell out of the manifest."""
    roots = [arm_dir(f, a) for f, a, _ in all_arms()]
    roots += [DST_TRAIN / "scrfd/stock", DST_TRAIN / "egoblur/stock", DST_PROBE]
    return [r for r in roots if r.is_dir()]


def disk_files():
    """Every file that is actually under the pii2 roots this script owns."""
    out = []
    for root in disk_roots():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not (Path(dirpath) == root and d in SKIP_DIRS))
            for fn in sorted(filenames):
                if fn in SKIP_NAMES:
                    continue
                p = Path(dirpath) / fn
                out.append(p.relative_to(PII2).as_posix())
    return sorted(set(out))


def do_verify(jobs, check_source=True):
    if not MANIFEST.exists():
        print("FAIL no MANIFEST.tsv; run build first")
        return 1
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    bad = []

    # 1. the manifest covers exactly the files on disk. PII-1601: an epoch the
    # retention rule keeps on OSS only is expected to be absent here, so it is
    # reported, never failed.
    on_disk = set(disk_files())
    listed = {r["dst_rel"] for r in rows}
    oss_only = oss_only_epochs(rows)
    pruned = sorted(oss_only - on_disk)
    for rel in sorted(on_disk - listed):
        bad.append(f"not in manifest: {rel}")
    for rel in sorted(listed - on_disk):
        if rel in oss_only:
            continue
        bad.append(f"in manifest, missing on disk (oss_sync.py pull fetches it): {rel}")
    if pruned:
        by_arm = {}
        for rel in pruned:
            arm, _ = epoch_rel_parts(rel)
            by_arm[arm] = by_arm.get(arm, 0) + 1
        nbytes = sum(int(r["size"]) for r in rows if r["dst_rel"] in set(pruned))
        print(f"INFO {len(pruned)} epochs over {len(by_arm)} arms are on OSS only "
              f"(checkpoints: {RETENTION}), {nbytes:,} B; "
              f"oss_sync.py pull --all-epochs --arm A fetches them back")

    # The source can grow while a run is still training (armAI34). That is not
    # a failure of what was copied, only a reason to rebuild.
    planned = {rel for rel, _, _ in copied_files()}
    fresh = sorted(planned - listed)
    for rel in fresh:
        print(f"INFO source gained a file since the build, not copied: {rel}")

    work = [(r["dst_rel"], str(PII2 / r["dst_rel"])) for r in rows
            if (PII2 / r["dst_rel"]).exists() or (PII2 / r["dst_rel"]).is_symlink()]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        got = dict((rel, (size, digest)) for rel, size, digest
                   in ex.map(_md5_job, work, chunksize=8))

    src_work, remote_src_rows, made_here_rows = [], 0, 0
    for r in rows:
        rel = r["dst_rel"]
        if rel not in got:
            if rel not in oss_only:
                bad.append(f"missing: {rel}")
            continue
        size, digest = got[rel]
        if str(size) != r["size"] or digest != r["md5"]:
            bad.append(f"copy changed: {rel} size {size} vs {r['size']} "
                       f"md5 {digest} vs {r['md5']}")
        if check_source:
            if not r["src"]:
                # PII-2112: an empty src is a file produced on this box straight
                # into the store (an ONNX export), so there is no second copy.
                made_here_rows += 1
            elif is_remote(r["src"]):
                remote_src_rows += 1
            else:
                src_work.append((rel, local_src(r["src"])))

    if check_source and remote_src_rows:
        print(f"INFO {remote_src_rows} files came from a work dir on another box "
              f"(src is host:/abs/path); their source is not re-read here, use remote-check")
    if check_source and made_here_rows:
        print(f"INFO {made_here_rows} files were made on this box straight into the store "
              f"(src empty); they have no source copy to compare against")
    if check_source:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            srcgot = dict((rel, (size, digest)) for rel, size, digest
                          in ex.map(_md5_job, src_work, chunksize=8))
        by_rel = {r["dst_rel"]: r for r in rows}
        for rel, (size, digest) in srcgot.items():
            r = by_rel[rel]
            if size == -2:
                bad.append(f"source gone: {rel} src {digest[len('missing:'):]}")
                continue
            if str(size) != r["size"] or digest != r["md5"]:
                bad.append(f"source differs: {rel} src {local_src(r['src'])} size {size} "
                           f"vs {r['size']} md5 {digest} vs {r['md5']}")

    # 3. every arm dir has a pick.yaml and it says last_epoch
    for fam, arms in (("scrfd", list(SCRFD_ARMS) + ["stock"]),
                      ("egoblur", list(EGOBLUR_ARMS) + ["stock"]),
                      ("rfdetr", list(RFDETR_ARMS))):
        for arm in arms:
            pick = arm_dir(fam, arm) / "pick.yaml"
            if not pick.exists():
                bad.append(f"no pick.yaml: {fam}/{arm}")
                continue
            text = pick.read_text()
            if "rule: last_epoch" not in text:
                bad.append(f"pick.yaml rule is not last_epoch: {fam}/{arm}")
            if chr(0x2014) in text:
                bad.append(f"pick.yaml has an em-dash: {fam}/{arm}")
            # PII-1601: the one checkpoint an arm under the rule keeps here has
            # to be on this disk, or the arm has no local checkpoint at all.
            spec = parse_pick(pick)
            if spec.get("checkpoints") == RETENTION:
                rel = spec.get("pick_checkpoint", "")
                if not rel:
                    bad.append(f"pick.yaml says checkpoints: {RETENTION} but names no "
                               f"pick_checkpoint: {fam}/{arm}")
                elif not (arm_dir(fam, arm) / rel).exists():
                    bad.append(f"the pick checkpoint is not on this disk: {fam}/{arm}/{rel}")
    if not MODELS_CSV.exists():
        bad.append("no models.csv")

    total = sum(int(r["size"]) for r in rows if int(r["size"]) > 0)
    if bad:
        for b in bad:
            print("FAIL " + b)
        print(f"FAIL {len(bad)} problems over {len(rows)} files")
        return 1
    print(f"OK {len(rows)} files, {total:,} B, size and md5 equal to "
          f"{'source and ' if check_source else ''}manifest")
    return 0


# --------------------------------------------------------------------------
# prune (PII-1601)
# --------------------------------------------------------------------------

def oss_module():
    """data/oss_sync.py out of the pii2 checkout: its OSS client and its
    credential loader are the only ones that touch this bucket (PII-1413), so
    prune borrows them instead of holding a second copy of either."""
    sys.path.insert(0, str(PII2 / "data"))
    # the import must not leave a __pycache__ in the pii2 checkout
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import oss_sync
    finally:
        sys.dont_write_bytecode = saved
    return oss_sync


def do_prune(arms=None, apply_it=False, jobs=8):
    """Delete the epochs the retention rule keeps on OSS only.

    A file is deleted only when a HEAD of its oss_key comes back with the size
    and the ETag the manifest records for it, so every deleted byte is proved
    to be on OSS first. Anything that fails the check is printed and kept.
    """
    if not MANIFEST.exists():
        print("FAIL no MANIFEST.tsv; run build first")
        return 1
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    by_arm, absent = {}, 0
    for rel in sorted(oss_only_epochs(rows)):
        arm, _ = epoch_rel_parts(rel)
        if arms and arm not in arms:
            continue
        p = PII2 / rel
        if not p.exists() or p.is_symlink():
            absent += 1
            continue
        by_arm.setdefault(arm, []).append(
            next(r for r in rows if r["dst_rel"] == rel))
    if arms:
        for arm in arms:
            if arm not in by_arm:
                print(f"INFO nothing to prune for {arm}")
    if not by_arm:
        print(f"nothing to prune ({absent} epochs already on OSS only)")
        return 0

    oss_sync = oss_module()
    ak, sk = oss_sync.load_creds()
    local = threading.local()

    def check(r):
        if not r["oss_key"]:
            return r, "no oss_key in the manifest row"
        if not hasattr(local, "c"):
            local.c = oss_sync.OSS(ak, sk)
        try:
            got = local.c.head(r["oss_key"])
        except Exception as e:                       # network, never a delete
            return r, f"HEAD failed: {e}"
        if got is None:
            return r, "no such object on OSS"
        size, etag = got
        if str(size) != r["size"]:
            return r, f"OSS size {size} != manifest {r['size']}"
        if etag != r["md5"]:
            return r, f"OSS ETag {etag} != manifest md5 {r['md5']}"
        on_disk = (PII2 / r["dst_rel"]).stat().st_size
        if str(on_disk) != r["size"]:
            return r, f"local size {on_disk} != manifest {r['size']}"
        return r, None

    t0 = time.time()
    todo = [r for rs in by_arm.values() for r in rs]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(check, todo))
    okset = {r["dst_rel"] for r, err in results if err is None}
    kept = [(r, err) for r, err in results if err is not None]

    verb = "delete" if apply_it else "would delete"
    n_tot = b_tot = 0
    print(f"{'arm':28} {'files':>7} {'bytes':>18}")
    for arm in sorted(by_arm):
        good = [r for r in by_arm[arm] if r["dst_rel"] in okset]
        nbytes = sum(int(r["size"]) for r in good)
        n_tot += len(good)
        b_tot += nbytes
        print(f"{arm:28} {len(good):>7} {nbytes:>18,}")
    print(f"{'TOTAL':28} {n_tot:>7} {b_tot:>18,}   ({verb}; "
          f"{len(kept)} kept after a failed OSS check; "
          f"{absent} already on OSS only; HEAD pass {time.time() - t0:.1f}s)")
    for r, err in kept:
        print(f"KEPT {r['dst_rel']}: {err}")
    if not apply_it:
        print("dry run, nothing deleted; rerun with --apply")
        return 0

    freed = 0
    for r in todo:
        if r["dst_rel"] not in okset:
            continue
        os.remove(PII2 / r["dst_rel"])
        freed += int(r["size"])
    print(f"deleted {n_tot} files, {freed:,} B freed")
    return 1 if kept else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=["build", "copy", "meta", "verify", "remote-check", "prune"])
    ap.add_argument("--arm", action="append",
                    help="restrict prune to this arm; repeatable")
    ap.add_argument("--apply", action="store_true",
                    help="prune: actually delete; without it prune is a dry run")
    ap.add_argument("--log", default=CODE_ROOT + "/.knuth/tmp/pii1379/rsync.log")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-source", action="store_true",
                    help="verify against the manifest only, do not re-read the source tree")
    ap.add_argument("--no-remote-check", action="store_true",
                    help="skip the md5 assertion against the remote work dirs before a copy")
    a = ap.parse_args()
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)

    if a.command == "remote-check":
        return remote_check(a.jobs)
    if a.command == "prune":
        return do_prune(a.arm, a.apply, a.jobs)
    if a.command in ("build", "copy") and not a.no_remote_check:
        rc = remote_check(a.jobs)
        if rc:
            print("FAIL a file here differs from the remote work dir; nothing copied")
            return rc
    if a.command in ("build", "copy"):
        secs = do_copy(a.log)
        print(f"copy: {secs:.1f}s, log {a.log}")
    if a.command in ("build", "meta"):
        write_meta(a.log)
    if a.command == "verify":
        return do_verify(a.jobs, check_source=not a.no_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The store-local resolver for the build and source scripts in this checkout (PII-1639).

The pii repo has the same file at its root and answers the same question for the train
and eval code. This copy is the one the scripts that live INSIDE the store import, and it
differs in one way: the default live root is the checkout this file belongs to, not a
literal, so a clone anywhere resolves itself.

    PII_ROOT     the live store, the pii_data checkout in the PII-1315 layout.
                 datasets/<name>/{frames.csv,split.csv,boxes/vN/boxes.csv,images/},
                 views/<name>/{recipe.yaml,scrfd.txt,d2.json,summary.json},
                 runs/train/{scrfd,egoblur,rfdetr}/<arm>/{epochs,onnx,pick.yaml},
                 tables/, calib/. Default: the parent of this file's directory.

    LEGACY_ROOT  the pre-PII-1315 tree, read-only as /data/esteban/pii_backup after the
                 PII-1448 rename. It still holds what the store never took:
                 face-mine_labeled.csv, gt_bench_v1/, face_mine_v1_eda/, weights/,
                 closeout/, the corpus CSVs. The build_*_pii2.py scripts read it.

    CODE_ROOT    the pii code repo, the one that keeps training/ and evaluation/. The
                 labelv2 GT manifests and the .knuth/pages media dirs live there, and
                 a few scripts here read them. Default: /home/esteban/repos/pii.

Env vars, highest first: PII_ROOT then PII2_ROOT for the live store; PII_LEGACY_ROOT for
the old tree; PII_CODE_ROOT for the code repo.

Stdlib only, no imports beyond os/pathlib/gzip, so any standalone script can
`from pii_root import PII_ROOT` after adding <store>/data to sys.path.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = str(HERE.parent)
DEFAULT_LEGACY_ROOT = "/data/esteban/pii_backup"
DEFAULT_CODE_ROOT = "/home/esteban/repos/pii"


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.rstrip("/")
    return None


PII_ROOT = _env("PII_ROOT", "PII2_ROOT") or DEFAULT_ROOT
LEGACY_ROOT = _env("PII_LEGACY_ROOT") or DEFAULT_LEGACY_ROOT
CODE_ROOT = _env("PII_CODE_ROOT") or DEFAULT_CODE_ROOT

ROOT = Path(PII_ROOT)
DATASETS = ROOT / "datasets"
VIEWS = ROOT / "views"
RUNS = ROOT / "runs"
TRAIN = RUNS / "train"
TABLES = ROOT / "tables"
CALIB = ROOT / "calib"
DATA = ROOT / "data"

LEGACY = Path(LEGACY_ROOT)
CODE = Path(CODE_ROOT)


def dataset(name: str) -> Path:
    """datasets/<name>/ in the live store."""
    return DATASETS / name


def images(name: str) -> Path:
    return DATASETS / name / "images"


def view(name: str) -> Path:
    """views/<name>/ in the live store (recipe.yaml, scrfd.txt, d2.json, summary.json)."""
    return VIEWS / name


def arm(family: str, name: str) -> Path:
    """runs/train/<family>/<arm>/, family being scrfd, egoblur or rfdetr."""
    return TRAIN / family / name


def onnx(family: str, name: str, filename: str) -> Path:
    return TRAIN / family / name / "onnx" / filename


def legacy(*parts: str) -> Path:
    """A path in the pre-PII-1315 tree (/data/esteban/pii_backup)."""
    return LEGACY.joinpath(*parts)


def code(*parts: str) -> Path:
    """A path in the pii code repo (training/manifests/..., .knuth/pages/media/...)."""
    return CODE.joinpath(*parts)


def manifest(name: str) -> Path:
    """training/manifests/<name> in the code repo: the labelv2 GT manifests."""
    return CODE / "training" / "manifests" / name


def pages_media(*parts: str) -> Path:
    """.knuth/pages/media/... in the code repo: the page dumps and thumbnails."""
    return CODE.joinpath(".knuth", "pages", "media", *parts)


# PII-1448 renamed the old tree out from under every absolute path recorded before the
# move. frames.csv `src_path`, MANIFEST.tsv `src` and data/oss_pii_keys.jsonl `local_path`
# all name /data/esteban/pii/... meaning the OLD tree. Those indexes are the ledger and
# are not rewritten for a rename; a reader resolves them through here instead.
OLD_ROOT_PREFIX = "/data/esteban/pii/"


def resolve_legacy(path: str) -> str:
    """An absolute path recorded before the PII-1448 rename, as it resolves today."""
    path = str(path)
    if path.startswith(OLD_ROOT_PREFIX) and not path.startswith(LEGACY_ROOT.rstrip("/") + "/"):
        return LEGACY_ROOT.rstrip("/") + "/" + path[len(OLD_ROOT_PREFIX):]
    return path


# GitHub refuses a blob over 100 MB, and data/faceight/frames.csv is 107,370,665 B, so
# git tracks data/faceight/frames.csv.gz instead and the plain file is gitignored
# (PII-1639). Readers keep naming the .csv and open it through here: the plain file wins
# when it is on disk (that is what faceight_round.py writes), the .gz is the fallback a
# fresh clone gets. Refresh the .gz after rewriting the CSV:
#     gzip -9 -n -c data/faceight/frames.csv > data/faceight/frames.csv.gz


def open_index(path, mode="rt"):
    """A handle on a store index that is on disk as <path> or as <path>.gz.

    Text modes get newline="" so csv sees the line endings; "rb" gives the
    decompressed bytes, so hashing through here reproduces the plain file's md5.
    """
    path = str(path)
    kw = {} if "b" in mode else {"newline": ""}
    if os.path.exists(path):
        return open(path, mode, **kw)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", mode, **kw)
    raise FileNotFoundError(f"neither {path} nor {path}.gz")


if __name__ == "__main__":
    print(f"PII_ROOT     {PII_ROOT}  (exists={ROOT.is_dir()})")
    print(f"LEGACY_ROOT  {LEGACY_ROOT}  (exists={LEGACY.is_dir()})")
    print(f"CODE_ROOT    {CODE_ROOT}  (exists={CODE.is_dir()})")

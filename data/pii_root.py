#!/usr/bin/env python3
"""The store-local resolver for the build and source scripts in this checkout (PII-1639).

The pii repo has the same file at its root and answers the same question for the train
and eval code. This copy is the one the scripts that live INSIDE the store import, and it
differs in one way: the default live root is the checkout this file belongs to, not a
literal, so a clone anywhere resolves itself.

    PII_ROOT     the live store, the pii_data checkout in the PII-1315 layout.
                 datasets/<name>/{frames.csv,split.csv,boxes/vN/boxes.csv,images/},
                 views/<name>/{recipe.yaml,scrfd.txt,d2.json,summary.json},
                 runs/<arm>/train/{epochs,onnx,pick.yaml} (PII-2118; the two
                 upstream-release dirs stayed at runs/train/<family>/stock/),
                 tables/, calib/. Default: the parent of this file's directory.

    CODE_ROOT    the pii code repo, the one that keeps training/ and evaluation/. The
                 labelv2 GT manifests and the .knuth/pages media dirs live there, and
                 a few scripts here read them. Default: /home/esteban/repos/pii.

Env vars, highest first: PII_ROOT then PII2_ROOT for the live store; PII_CODE_ROOT for the
code repo.

There is one data root. Until PII-1682 there was a second, LEGACY_ROOT, the pre-PII-1315 tree
the build_*_pii2.py scripts read; the project retired it, so each of those scripts now carries
the path as its own provenance constant and nothing resolves it here.

Stdlib only, no imports beyond os/pathlib/gzip, so any standalone script can
`from pii_root import PII_ROOT` after adding <store>/data to sys.path.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = str(HERE.parent)
DEFAULT_CODE_ROOT = "/home/esteban/repos/pii"


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.rstrip("/")
    return None


PII_ROOT = _env("PII_ROOT", "PII2_ROOT") or DEFAULT_ROOT
CODE_ROOT = _env("PII_CODE_ROOT") or DEFAULT_CODE_ROOT

ROOT = Path(PII_ROOT)
DATASETS = ROOT / "datasets"
VIEWS = ROOT / "views"
RUNS = ROOT / "runs"
TRAIN = RUNS / "train"   # PII-2118: <family>/stock/ and logs_replay/ only
TABLES = ROOT / "tables"
CALIB = ROOT / "calib"
DATA = ROOT / "data"

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
    """An arm's dir: runs/<arm>/train/ (PII-2118). The family is kept in the
    signature because it is what the OSS key and models.csv still carry, and
    because `stock` needs it: both families have one, so those two dirs stayed
    at runs/train/<family>/stock/."""
    return TRAIN / family / name if name == "stock" else RUNS / name / "train"


def onnx(family: str, name: str, filename: str) -> Path:
    return arm(family, name) / "onnx" / filename


def code(*parts: str) -> Path:
    """A path in the pii code repo (training/manifests/..., .knuth/pages/media/...)."""
    return CODE.joinpath(*parts)


def manifest(name: str) -> Path:
    """training/manifests/<name> in the code repo: the labelv2 GT manifests."""
    return CODE / "training" / "manifests" / name


def pages_media(*parts: str) -> Path:
    """.knuth/pages/media/... in the code repo: the page dumps and thumbnails."""
    return CODE.joinpath(".knuth", "pages", "media", *parts)


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
    print(f"CODE_ROOT    {CODE_ROOT}  (exists={CODE.is_dir()})")

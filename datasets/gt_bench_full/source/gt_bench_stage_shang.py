#!/usr/bin/env python3
"""Rebuild the gt_bench images on a box that has the capture bucket but not the
dataset bucket (shang). Standalone port of evaluation/stage_full_bench.py step 2
plus the sparse slice; no repo, venv or algorithm-datasets access needed.

PII-1449: the old single gt_bench_v1 dataset became two in the PII-1315 store,
gt_bench_full and gt_bench_sparse, each with a flat images/ dir. --root is the
staging dir this script owns; it still writes the pre-split shapes below, and
the copy into the store is images_full/<session>_<ck>/*.jpg -> gt_bench_full
images/ (flattened to <session>_<ck>_<file>) and images_sparse/*.jpg ->
gt_bench_sparse/images/. The GT that was labels/ is now boxes/v1/boxes.csv of
each dataset, with the raw job files under boxes/v1/job/output/.

Inputs (already staged under ROOT/labels): gt_bundle.json, uuid_map.json,
gt_eval_sparse.json. For each (session, chunk) it downloads
oss://<capture_bucket>/<session>/chunk_<ck>/vst_left/vst_left_video.mp4 with
ossutil (config file, keys never on the command line), extracts the union of
bundle frames and sparse frames with ffmpeg, and writes
  images_full/<session>_<ck>/f<frame>.jpg
  images_sparse/<session>_<ck>_f<frame>.jpg
Idempotent: chunks whose outputs all exist are skipped.

usage: gt_bench_stage_shang.py --root <staging dir for gt_bench_full+sparse> \
         --ossutil-config /path/to/cfg [--workers 9] [--tmp /data/esteban/tmp]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ossutil-config", required=True)
ap.add_argument("--ossutil", default="/usr/local/bin/ossutil")
ap.add_argument("--endpoint", default="oss-cn-shanghai.aliyuncs.com")
ap.add_argument("--region", default="cn-shanghai")
ap.add_argument("--capture-bucket", default="we-fpv-sh-ns")
ap.add_argument("--workers", type=int, default=9)
ap.add_argument("--tmp", default=None)
a = ap.parse_args()

ROOT = Path(a.root)
FULL, SPARSE_DIR = ROOT / "images_full", ROOT / "images_sparse"
BUNDLE = json.load(open(ROOT / "labels/gt_bundle.json"))
UUID_MAP = json.load(open(ROOT / "labels/uuid_map.json"))
SPARSE = json.load(open(ROOT / "labels/gt_eval_sparse.json"))
OSSARGS = ["-c", a.ossutil_config, "-e", a.endpoint, "--region", a.region]

# (session, ck) -> {"full": set(frames), "sparse": set(frames)}
want = defaultdict(lambda: {"full": set(), "sparse": set()})
for uid, (sess, ck, _n) in UUID_MAP.items():
    want[(sess, int(ck))]["full"] |= {int(f) for f in BUNDLE[uid]["frames"]}
for e in SPARSE:
    fr = int(re.search(r"_f(\d+)\.jpg$", e["path"]).group(1))
    want[(e["session"], int(e["chunk"]))]["sparse"].add(fr)
unmapped = [u for u in BUNDLE if u not in UUID_MAP]
print(f"{len(want)} chunks; full frames {sum(len(w['full']) for w in want.values())}, "
      f"sparse frames {sum(len(w['sparse']) for w in want.values())}; unmapped bundle uuids: "
      f"{[(u[-12:], len(BUNDLE[u]['frames'])) for u in unmapped]}", flush=True)


def outputs(sess, ck, w):
    full = {f: FULL / f"{sess}_{ck:03d}" / f"f{f:06d}.jpg" for f in w["full"]}
    sparse = {f: SPARSE_DIR / f"{sess}_{ck:03d}_f{f:06d}.jpg" for f in w["sparse"]}
    return full, sparse


def stage(item):
    (sess, ck), w = item
    full, sparse = outputs(sess, ck, w)
    if all(p.exists() for p in list(full.values()) + list(sparse.values())):
        return f"{sess} c{ck}: already present"
    frames = sorted(w["full"] | w["sparse"])
    key = f"{sess}/chunk_{ck:03d}/vst_left/vst_left_video.mp4"
    print(f"{sess} c{ck}: fetching video, extracting {len(frames)} frames", flush=True)
    with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
        mp4 = Path(tmp) / "v.mp4"
        r = subprocess.run([a.ossutil, "cp", "-u", f"oss://{a.capture_bucket}/{key}", str(mp4), *OSSARGS],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"{sess} c{ck}: ossutil FAILED: {r.stdout[-300:]} {r.stderr[-300:]}"
        sel = "+".join(f"eq(n\\,{f})" for f in frames)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-vf", f"select={sel}",
                        "-vsync", "0", "-q:v", "2", str(Path(tmp) / "out_%06d.jpg")], check=True)
        outs = sorted(Path(tmp).glob("out_*.jpg"))
        if len(outs) != len(frames):
            return f"{sess} c{ck}: FAILED extracted {len(outs)} != {len(frames)}"
        for f, img in zip(frames, outs):
            if f in full:
                full[f].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, full[f])
            if f in sparse:
                sparse[f].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, sparse[f])
    return f"{sess} c{ck}: done, {len(frames)} frames"


with ThreadPoolExecutor(a.workers) as ex:
    for msg in ex.map(stage, sorted(want.items())):
        print(msg, flush=True)
nf = sum(1 for _ in FULL.rglob("*.jpg")) if FULL.exists() else 0
ns = sum(1 for _ in SPARSE_DIR.glob("*.jpg")) if SPARSE_DIR.exists() else 0
print(f"done: images_full {nf} jpg, images_sparse {ns} jpg")

#!/usr/bin/env python3
"""Map gt_bundle chunk-uuids to (session, chunk) via the sparse slice, download the
GT videos, and extract every bundle frame to images_full/. Writes uuid_map.json."""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict

GTB = Path("/data/esteban/pii/datasets/gt_bench_v1")
OUT = GTB / "images_full"
BUNDLE = json.load(open(GTB / "labels/gt_bundle.json"))
SPARSE = json.load(open(GTB / "labels/gt_eval_sparse.json"))

raw = open(os.path.expanduser("~/repos/atlas/sheriff/secrets.jsonc")).read()
blk = re.search(r'"aliyun_oss"\s*:\s*\{(.*?)\}', raw, re.S).group(1)
get = lambda f: re.search(r'"%s"\s*:\s*"([^"]*)"' % f, blk).group(1)
ENV = dict(os.environ, OSS_ACCESS_KEY_ID=get("access_key_id"),
           OSS_ACCESS_KEY_SECRET=get("secret_access_key"))
OSSARGS = ["--endpoint", "https://oss-cn-shanghai.aliyuncs.com", "--region", "cn-shanghai"]

# --- 1. uuid -> (session, chunk): match sparse (session, chunk, frame, first box)
sparse_key = {}
for e in SPARSE:
    if not e["boxes"]:
        continue
    frame = int(re.search(r"_f(\d+)\.jpg$", e["path"]).group(1))
    sparse_key[(round(e["boxes"][0][0], 2), round(e["boxes"][0][1], 2), frame)] = (
        e["session"], int(e["chunk"]))

uuid_map, votes = {}, defaultdict(lambda: defaultdict(int))
for uid, d in BUNDLE.items():
    for fidx, boxes in d["frames"].items():
        if boxes:
            k = (round(boxes[0][0], 2), round(boxes[0][1], 2), int(fidx))
            if k in sparse_key:
                votes[uid][sparse_key[k]] += 1
for uid in BUNDLE:
    if votes[uid]:
        (sess, ck), n = max(votes[uid].items(), key=lambda x: x[1])
        others = sum(v for k, v in votes[uid].items() if k != (sess, ck))
        assert others == 0, (uid, dict(votes[uid]))
        uuid_map[uid] = [sess, ck, n]
print(f"mapped {len(uuid_map)}/{len(BUNDLE)} uuids")
for uid, (s, c, n) in uuid_map.items():
    print(f"  {uid[-12:]} -> {s} chunk {c}  ({n} matching frames)")
unmapped = [u for u in BUNDLE if u not in uuid_map]
print("unmapped (no sparse overlap):", [(u[-12:], len(BUNDLE[u]['frames'])) for u in unmapped])
json.dump(uuid_map, open(GTB / "labels/uuid_map.json", "w"), indent=1)

# --- 2. download videos + extract all bundle frames
for uid, (sess, ck, _n) in sorted(uuid_map.items()):
    frames = sorted(int(f) for f in BUNDLE[uid]["frames"])
    outdir = OUT / f"{sess}_{ck:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    if all((outdir / f"f{f:06d}.jpg").exists() for f in frames):
        print(f"{sess} c{ck}: {len(frames)} frames already present")
        continue
    key = f"{sess}/chunk_{ck:03d}/vst_left/vst_left_video.mp4"
    print(f"{sess} c{ck}: fetching video, extracting {len(frames)} frames", flush=True)
    with tempfile.TemporaryDirectory(dir="/data/esteban/pii") as tmp:
        mp4 = Path(tmp) / "v.mp4"
        subprocess.run(["ossutil", "cp", "-u", f"oss://we-fpv-sh-ns/{key}", str(mp4), *OSSARGS],
                       env=ENV, check=True, capture_output=True)
        sel = "+".join(f"eq(n\\,{f})" for f in frames)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4),
                        "-vf", f"select={sel}", "-vsync", "0", "-q:v", "2",
                        str(Path(tmp) / "out_%06d.jpg")], check=True)
        outs = sorted(Path(tmp).glob("out_*.jpg"))
        assert len(outs) == len(frames), (key, len(outs), len(frames))
        for f, img in zip(frames, outs):
            img.rename(outdir / f"f{f:06d}.jpg")
total = sum(1 for _ in OUT.rglob("*.jpg"))
print(f"done: {total} frames under {OUT}")

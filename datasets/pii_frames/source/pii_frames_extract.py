#!/usr/bin/env python3
"""Standalone port of data/prepare.py:extract_pii_frames (WOR-32) for hosts that
have only stdlib + ffmpeg + ossutil (WOR-82: rebuild pii_frames on shang).

Same frame selection and encoder settings as prepare.py, deliberately verbatim:
  ffmpeg -loglevel error -i v.mp4 -vf select=eq(n\\,F1)+eq(n\\,F2)... -vsync 0 -q:v 2 out_%05d.jpg
Frame selection is index based (decode order n), not time based; outputs are
zipped to the sorted frame list.

Differences from prepare.py: ossutil is driven with a config file (-c) instead
of env credentials; videos are cached in --cache and kept unless --delete-videos;
--jobs videos are processed concurrently; --only restricts to one
<session>/<chunk>/<view>; --offset N adds N to every frame index while keeping
the file names (sabotage test: proves the comparison detects a wrong frame).
"""
import argparse, shutil, subprocess, sys, tempfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VIEW = {"lview": "vst_left", "rview": "vst_right"}


def parse_manifest(manifest):
    videos = defaultdict(set)
    for line in Path(manifest).open():
        if line.startswith("# pii/"):
            rel = line.split()[1]
            _, session, chunk, view, frame = rel.split("/")
            videos[(session, chunk, view)].add(int(frame[1:-4]))
    return {k: sorted(v) for k, v in videos.items()}


def fetch(a, key, mp4):
    if mp4.exists() and mp4.stat().st_size > 0:
        return "cached"
    for attempt in range(3):
        r = subprocess.run(["ossutil", "-c", a.ossutil_config, "cp", "-u", "-f",
                            f"oss://{a.bucket}/{key}", str(mp4)],
                           capture_output=True, text=True)
        if r.returncode == 0 and mp4.exists():
            return "downloaded"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ossutil cp failed for {key}: {r.stderr[-400:]}")


def process(a, session, chunk, view, frames):
    vst = VIEW[view]
    key = f"{session}/{chunk}/{vst}/{vst}_video.mp4"
    outdir = Path(a.out) / session / chunk / view
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) / session / chunk
    cache.mkdir(parents=True, exist_ok=True)
    mp4 = cache / f"{vst}_video.mp4"
    t0 = time.time()
    how = fetch(a, key, mp4)
    t1 = time.time()
    sel_frames = [f + a.offset for f in frames]
    with tempfile.TemporaryDirectory(dir=a.cache) as tmp:
        sel = "+".join(f"eq(n\\,{f})" for f in sel_frames)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4),
                        "-vf", f"select={sel}", "-vsync", "0", "-q:v", "2",
                        str(Path(tmp) / "out_%05d.jpg")], check=True)
        outs = sorted(Path(tmp).glob("out_*.jpg"))
        if len(outs) != len(frames):
            raise RuntimeError(f"{key}: extracted {len(outs)} frames, expected {len(frames)}")
        for f, img in zip(frames, outs):
            shutil.move(str(img), outdir / f"f{f:06d}.jpg")
    if a.delete_videos:
        mp4.unlink()
    return f"{key} {len(frames)} frames {how} dl={t1-t0:.0f}s ffmpeg={time.time()-t1:.0f}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--ossutil-config", required=True)
    p.add_argument("--bucket", default="we-fpv-sh-ns")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--only", help="<session>/<chunk>/<view>")
    p.add_argument("--offset", type=int, default=0, help="SABOTAGE: shift frame index")
    p.add_argument("--delete-videos", action="store_true")
    a = p.parse_args()
    if a.offset:
        print(f"WARNING: --offset {a.offset}: output frames are deliberately WRONG", flush=True)
    videos = parse_manifest(a.manifest)
    if a.only:
        videos = {k: v for k, v in videos.items() if "/".join(k) == a.only}
    out = Path(a.out)
    todo = {k: v for k, v in videos.items()
            if not all((out / k[0] / k[1] / k[2] / f"f{f:06d}.jpg").exists() for f in v)}
    print(f"pii frames: {len(videos)} videos, {len(todo)} to process, jobs={a.jobs}", flush=True)
    failed = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(process, a, *k, v): k for k, v in sorted(todo.items())}
        for fut in as_completed(futs):
            try:
                print("  ok", fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print("  FAIL", "/".join(futs[fut]), e, flush=True)
    total = sum(len(v) for v in videos.values())
    missing = [1 for k, v in videos.items() for f in v
               if not (out / k[0] / k[1] / k[2] / f"f{f:06d}.jpg").exists()]
    print(f"done: {total - len(missing)}/{total} frames present, {failed} videos failed", flush=True)
    sys.exit(1 if (missing or failed) else 0)


if __name__ == "__main__":
    main()

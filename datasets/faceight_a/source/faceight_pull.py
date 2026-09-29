#!/usr/bin/env python
"""faceight: pull one frame per (episode, residue) from OSS by time, then run armW on it.

Selection (data/faceight/README.md): every episode in data/faceight/episodes.txt,
chunk index = column `chunk_sel` of data/episode_usage.csv (latest chunk >= 240 s;
-1 = none, the episode is skipped with reason `no-chunk-240` in failed.jsonl), view
--view (vst_left), one frame per residue s in data/faceight/residues.csv whose row is
`used` with `view == --view` (a missing view column means vst_left) and batch <= --batch,
at t_ms = 1000*s + 233; a --view with no such row is a SystemExit, never an empty fetch.
`--chunk N` overrides chunk_sel for every episode (sabotage tests only).
`--frames-from frames.csv --round N` (WOR-96) replaces the residue rule: the tasks are the
CSV rows with annotation_round == N (their episode, session, chunk, s, batch, t_ms), re-keyed
to --view; each row's chunk must equal chunk_sel in --usage. The OSS snapshot API is
addressed by time (`video/snapshot,t_<ms>,f_jpg`), never by frame number; frame_idx
is derived from the mp4 stts (nearest pts to t_ms) and recorded next to t_ms.

Layout under --out-dir (default /data/esteban/faceight); SFX = --out-suffix, "" for
vst_left and "_right" for vst_right unless given:
  frames<SFX>/<session>_c<chunk:03d>_<lview|rview>_t<ms>.jpg   fetched JPEGs (kept)
  fetch<SFX>.jsonl   one line per fetched frame (file, key, t_ms, frame_idx, fps, bytes)
  timing<SFX>.json   per-video stts cache (moov ranged GETs); the right video is its own mp4,
                     so frame_idx for vst_right comes from the right video's stts
  failed<SFX>.jsonl  every failure, both stages, with reason; retried on rerun
                     (stage "select", reason no-chunk-240: written once per episode, never retried)
  faceight_armW<SFX>.jsonl  final records, one per frame (schema in data/faceight/README.md)

Both stages are resumable: a frame is fetched only if its file is missing or invalid;
a record is written only if its file is not already in the output JSONL (or in --done-jsonl,
WOR-185: a second file whose records also count as done, e.g. the merged file of earlier
shards). `--shard K/N` (WOR-185, detect only) keeps the selected tasks with index % N == K
(the slice is taken over the full selection, before the fetched/done filter, so a restarted
shard gets the same frames whatever is already done); each shard writes its own --out-jsonl
and the shards are concatenated afterwards. The fetch stage ignores --shard.

usage: faceight_pull.py [--stage all|fetch|detect] [--limit N_EPISODES] [--batch B]
                        [--episodes-from FILE] [--chunk N] [--view vst_left|vst_right]
                        [--frames-from CSV --round N] [--out-dir DIR] [--out-suffix SFX]
                        [--workers 48] [--model det_10g_armW.onnx] [--det-size 1024]
                        [--score-thr 0.1] [--face-thr 0.6] [--no-timing] [--head-check]
                        [--out-jsonl FILE] [--done-jsonl FILE] [--shard K/N] [--model-name NAME]

OSS endpoint: shang's internal endpoint / VPC DNS has been dead since 2026-09-08; the
fetch stage runs episode_chunks_probe.setup_endpoint (public endpoint fallback).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # the pii_data checkout (PII-1639)
SHARED = ROOT / "data" / "mining"   # episode_chunks_probe.py, rview_detect.py
sys.path.insert(0, str(ROOT / "data"))
from pii_root import open_index  # noqa: E402  (frames.csv is tracked gzipped, PII-1639)
PHASE_MS = 233
VIEW_SHORT = {"vst_left": "lview", "vst_right": "rview"}
MIN_BYTES = 50_000  # a real 2328x1748 snapshot is ~500-700 KB; anything tiny is an error page


# ----------------------------------------------------------------------------- inputs
def load_residues(path: str, batch_max: int | None, view: str = "vst_left") -> list[tuple[int, int]]:
    """(s, batch) for the `used` rows of residues.csv with `view == view` and batch <= batch_max
    (None = all). The key of the file is (s, view); a file without a `view` column is all vst_left
    (WOR-103). Exits if nothing is selected, so a wrong --view can never mean an empty fetch."""
    out = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        has_view = "view" in (rd.fieldnames or [])
        for row in rd:
            if row["used"].strip().lower() != "true":
                continue
            row_view = row["view"].strip() if has_view else "vst_left"
            if row_view != view:
                continue
            b = int(row["batch"])
            if batch_max is None or b <= batch_max:
                out.append((int(row["s"]), b))
    if not out:
        raise SystemExit(f"{path}: no used residues for view {view} (batch <= {batch_max}); nothing to fetch")
    return sorted(out, key=lambda x: (x[1], x[0]))


def load_sessions(path: str, episodes: list[str]) -> dict[str, tuple[str, int]]:
    """episode_id -> (session_id, chunk_sel). chunk_sel: latest chunk >= 240 s, -1 = none/unprobed."""
    want = set(episodes)
    out = {}
    with open(path) as fh:
        rd = csv.DictReader(fh)
        if "chunk_sel" not in (rd.fieldnames or []):
            raise SystemExit(f"{path}: no `chunk_sel` column (header: {rd.fieldnames}); "
                             "run mining/faceight_chunk_sel (WOR-66) first")
        for row in rd:
            if row["episode_id"] in want and row["session_id"]:
                cs = row["chunk_sel"].strip()
                out[row["episode_id"]] = (row["session_id"], int(cs) if cs else -1)
    missing = [e for e in episodes if e not in out]
    if missing:
        raise SystemExit(f"{len(missing)} episodes without session_id in {path}, e.g. {missing[:3]}")
    return out


def video_key(session: str, chunk: int, view: str) -> str:
    return f"{session}/chunk_{chunk:03d}/{view}/{view}_video.mp4"


def frame_name(session: str, chunk: int, view: str, t_ms: int) -> str:
    return f"{session}_c{chunk:03d}_{VIEW_SHORT.get(view, view)}_t{t_ms}.jpg"


def load_frames_csv(path: str, rnd: int, view: str, limit: int = 0) -> list[dict]:
    """Tasks from a frames.csv (WOR-94): rows with annotation_round == rnd, re-keyed to `view`.
    Row order is kept; --limit keeps the first N distinct episodes in that order."""
    need = {"episode_id", "session_id", "chunk", "s", "batch", "t_ms", "annotation_round"}
    tasks, seen = [], {}  # seen: episode_id -> None, insertion-ordered
    with open_index(path) as fh:
        rd = csv.DictReader(fh)
        miss = need - set(rd.fieldnames or [])
        if miss:
            raise SystemExit(f"{path}: missing columns {sorted(miss)}")
        for row in rd:
            if int(row["annotation_round"]) != rnd:
                continue
            ep, ses = row["episode_id"], row["session_id"]
            chunk, s, t_ms = int(row["chunk"]), int(row["s"]), int(row["t_ms"])
            if t_ms != 1000 * s + PHASE_MS or chunk < 0 or not ses:
                raise SystemExit(f"{path}: bad row {row}")
            if ep not in seen:
                if limit and len(seen) >= limit:
                    continue
                seen[ep] = None
            tasks.append({
                "episode_id": ep, "session_id": ses, "chunk": chunk, "view": view,
                "s": s, "batch": int(row["batch"]), "t_ms": t_ms,
                "key": video_key(ses, chunk, view),
                "file": frame_name(ses, chunk, view, t_ms),
            })
    if not tasks:
        raise SystemExit(f"{path}: no rows with annotation_round == {rnd}")
    return tasks


def build_tasks(args) -> tuple[list[dict], list[tuple[str, str]]]:
    """(tasks, skipped): one task per (episode, residue); skipped = (episode_id, session_id) with chunk_sel -1.
    With --frames-from: one task per CSV row at --round, nothing skipped (chunk is checked against chunk_sel)."""
    if args.frames_from:
        if args.chunk is not None:
            raise SystemExit("--chunk cannot be combined with --frames-from")
        tasks = load_frames_csv(args.frames_from, args.round, args.view, args.limit)
        sessions = load_sessions(args.usage, sorted({t["episode_id"] for t in tasks}))
        bad = [t for t in tasks if sessions[t["episode_id"]] != (t["session_id"], t["chunk"])]
        if bad:
            raise SystemExit(f"{len(bad)} rows of {args.frames_from} disagree with {args.usage} "
                             f"(session_id, chunk_sel), e.g. {bad[0]['file']}")
        return tasks, []
    episodes = [l.strip() for l in open(args.episodes_from or args.episodes) if l.strip()]
    if args.limit:
        episodes = episodes[: args.limit]
    residues = load_residues(args.residues, args.batch, args.view)
    sessions = load_sessions(args.usage, episodes)
    tasks, skipped = [], []
    for ep in episodes:
        ses, chunk = sessions[ep]
        if args.chunk is not None:
            chunk = args.chunk
        if chunk < 0:
            skipped.append((ep, ses))
            continue
        for s, b in residues:
            t_ms = 1000 * s + PHASE_MS
            tasks.append({
                "episode_id": ep, "session_id": ses, "chunk": chunk, "view": args.view,
                "s": s, "batch": b, "t_ms": t_ms,
                "key": video_key(ses, chunk, args.view),
                "file": frame_name(ses, chunk, args.view, t_ms),
            })
    return tasks, skipped


def out_paths(args) -> dict[str, str]:
    """frames dir and the fetch / failed / timing files under --out-dir, with --out-suffix applied."""
    d, sfx = args.out_dir, args.out_suffix
    return {"frames": os.path.join(d, "frames" + sfx), "fetch": os.path.join(d, f"fetch{sfx}.jsonl"),
            "failed": os.path.join(d, f"failed{sfx}.jsonl"), "timing": os.path.join(d, f"timing{sfx}.json")}


def record_skipped(args, skipped: list[tuple[str, str]]) -> int:
    """Append one `no-chunk-240` line per newly skipped episode to failed.jsonl; return the count."""
    path = out_paths(args)["failed"]
    have = {r["episode_id"] for r in read_jsonl(path) if r.get("reason") == "no-chunk-240"}
    new = [(ep, ses) for ep, ses in skipped if ep not in have]
    if new:
        failed = Jsonl(path)
        for ep, ses in new:
            failed.write({"stage": "select", "episode_id": ep, "session_id": ses, "chunk": -1,
                          "reason": "no-chunk-240", "ts": time.strftime("%F %T")})
        failed.close()
    return len(new)


def read_jsonl(path: str) -> list[dict]:
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # torn last line from a kill; the entry is redone
    return out


class Jsonl:
    """Append-only JSONL with a lock; flushes every write (survives kill -9 modulo the last line)."""

    def __init__(self, path: str):
        self.fh = open(path, "a")
        self.lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec) + "\n"
        with self.lock:
            self.fh.write(line)
            self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def valid_jpeg(path: str) -> int:
    """Return byte size if the file looks like a complete JPEG, else 0."""
    try:
        n = os.path.getsize(path)
        if n < MIN_BYTES:
            return 0
        with open(path, "rb") as fh:
            head = fh.read(2)
            fh.seek(-2, os.SEEK_END)
            tail = fh.read(2)
        return n if head == b"\xff\xd8" and tail == b"\xff\xd9" else 0
    except OSError:
        return 0


# ----------------------------------------------------------------------------- fetch
def nearest_frame(pts: list[float], t_ms: int) -> int:
    i = bisect.bisect_left(pts, float(t_ms))
    if i == 0:
        return 0
    if i >= len(pts):
        return len(pts) - 1
    return i if abs(pts[i] - t_ms) < abs(t_ms - pts[i - 1]) else i - 1


def load_osslib(args):
    """Import osslib/mp4meta from --osslib-dir and select a reachable OSS endpoint."""
    sys.path.insert(0, args.osslib_dir)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(SHARED))
    import osslib  # noqa: E402  (credentials + presign live there)
    import mp4meta  # noqa: E402
    from episode_chunks_probe import setup_endpoint  # noqa: E402  (public-endpoint fallback, WOR-62)

    setup_endpoint(osslib)
    return osslib, mp4meta


def head_check(args, tasks: list[dict]) -> None:
    """Print chunk_sel + key + HEAD result for every distinct video; exit 1 if any is missing."""
    osslib, _ = load_osslib(args)
    keys = {}
    for t in tasks:
        keys.setdefault(t["key"], t)
    bad = 0
    for key, t in keys.items():
        n, err = osslib.head(key)
        ok = n is not None and n > 0
        bad += not ok
        print(f"{'OK ' if ok else 'MISSING'} episode {t['episode_id']} chunk_sel {t['chunk']:3d} "
              f"{key} {n if ok else err}", flush=True)
    print(f"head-check: {len(keys)} videos, {len(keys) - bad} present, {bad} missing", flush=True)
    if bad:
        sys.exit(1)


def stage_fetch(args, tasks: list[dict]) -> None:
    osslib, mp4meta = load_osslib(args)

    paths = out_paths(args)
    frames_dir = paths["frames"]
    os.makedirs(frames_dir, exist_ok=True)
    timing_path = paths["timing"]
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    tlock = threading.Lock()
    manifest_path = paths["fetch"]
    have = {r["file"] for r in read_jsonl(manifest_path)}
    manifest = Jsonl(manifest_path)
    failed = Jsonl(paths["failed"])

    by_session: dict[str, list[dict]] = {}
    for t in tasks:
        if t["file"] in have and valid_jpeg(os.path.join(frames_dir, t["file"])):
            continue
        by_session.setdefault(t["key"], []).append(t)
    n_pending = sum(len(v) for v in by_session.values())
    print(f"fetch: {len(tasks)} frames selected; {len(tasks) - n_pending} already fetched; "
          f"{n_pending} pending over {len(by_session)} videos; {args.workers} workers", flush=True)
    if not n_pending:
        return

    stats = {"fetched": 0, "bytes": 0, "failed": 0, "consec_fail": 0}
    slock = threading.Lock()
    stop = threading.Event()

    def fail(t: dict, reason: str) -> None:
        failed.write({"stage": "fetch", "file": t["file"], "key": t["key"], "t_ms": t["t_ms"],
                      "reason": reason, "ts": time.strftime("%F %T")})
        with slock:
            stats["failed"] += 1
            stats["consec_fail"] += 1
            if stats["consec_fail"] >= args.max_consec_fail:
                stop.set()
                print(f"STOP: {stats['consec_fail']} consecutive failures", flush=True)

    def get_timing(key: str):
        """{'ts','stts','nb'} or {'err'}; cached per video in timing.json."""
        with tlock:
            v = timing.get(key)
        if v is not None and "err" not in v:
            return v
        try:
            tm = mp4meta.parse_video_timing(mp4meta.get_moov(key))
            v = {"ts": tm["timescale"], "stts": tm["stts"], "nb": sum(c for c, _ in tm["stts"])}
            if tm["ctts"]:
                v["ctts"] = tm["ctts"]
        except Exception as e:  # noqa: BLE001
            v = {"err": f"{type(e).__name__}:{str(e)[:120]}"}
        with tlock:
            timing[key] = v
        return v

    def process_video(key: str, ts: list[dict]) -> None:
        if stop.is_set():
            return
        pts = fps = nb = None
        if not args.no_timing:
            v = get_timing(key)
            if "err" in v:
                for t in ts:
                    fail(t, "timing:" + v["err"])
                return
            pts = []
            cur = 0
            for cnt, d in v["stts"]:
                for _ in range(cnt):
                    pts.append(cur * 1000.0 / v["ts"])
                    cur += d
            nb = v["nb"]
            dur_ms = cur * 1000.0 / v["ts"]
            fps = round(nb / (dur_ms / 1000.0), 3) if dur_ms else None
        for t in sorted(ts, key=lambda x: x["t_ms"]):
            if stop.is_set():
                return
            path = os.path.join(frames_dir, t["file"])
            frame_idx = nearest_frame(pts, t["t_ms"]) if pts else None
            if pts and t["t_ms"] > dur_ms:
                fail(t, f"short-chunk:t{t['t_ms']}>dur{round(dur_ms)}")
                continue
            n = valid_jpeg(path)
            attempts = 0
            if not n:
                b, attempts, err = osslib.snapshot(key, t["t_ms"])
                if b is None:
                    fail(t, f"snapshot:{err}")
                    continue
                if len(b) < MIN_BYTES or b[:2] != b"\xff\xd8" or b[-2:] != b"\xff\xd9":
                    fail(t, f"bad-jpeg:{len(b)}B")
                    continue
                tmp = path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(b)
                os.replace(tmp, path)
                n = len(b)
            manifest.write({"file": t["file"], "key": key, "t_ms": t["t_ms"], "frame_idx": frame_idx,
                            "fps": fps, "nb": nb, "bytes": n, "attempts": attempts,
                            "ts": time.strftime("%F %T")})
            with slock:
                stats["fetched"] += 1
                stats["bytes"] += n
                stats["consec_fail"] = 0
                k = stats["fetched"]
            if k % 500 == 0:
                el = time.time() - t0
                print(f"fetch {k}/{n_pending}  {k / el:.1f} fps  {stats['bytes'] / k / 1e3:.0f} KB/frame  "
                      f"failed {stats['failed']}  eta {(n_pending - k) / (k / el) / 60:.1f} min", flush=True)

    def save_timing() -> None:
        with tlock:
            snap = dict(timing)
        json.dump(snap, open(timing_path + ".tmp", "w"))
        os.replace(timing_path + ".tmp", timing_path)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_video, k, v) for k, v in by_session.items()]
        last_save = time.time()
        for f in futs:
            f.result()
            if time.time() - last_save > 15:
                save_timing()
                last_save = time.time()
    save_timing()
    manifest.close()
    failed.close()
    el = time.time() - t0
    k = stats["fetched"]
    print(f"fetch done: {k} frames, {stats['failed']} failed, {stats['bytes'] / 1e6:.1f} MB, "
          f"{el:.0f}s ({k / max(el, 1e-9):.2f} fps, {stats['bytes'] / max(k, 1) / 1e3:.0f} KB/frame)"
          + ("  STOPPED EARLY" if stop.is_set() else ""), flush=True)


# ----------------------------------------------------------------------------- detect
def stage_detect(args, tasks: list[dict]) -> None:
    import cv2
    sys.path.insert(0, str(SHARED))
    from rview_detect import SCRFD  # same decode/NMS as the rview run (WOR-53)

    paths = out_paths(args)
    frames_dir = paths["frames"]
    fetched = {r["file"]: r for r in read_jsonl(paths["fetch"])}
    done = {r["file"] for r in read_jsonl(args.out_jsonl)}
    n_done_out = len(done)
    if args.done_jsonl:
        done |= {r["file"] for r in read_jsonl(args.done_jsonl)}
    n_sel = len(tasks)
    if args.shard:
        k, n = args.shard
        tasks = tasks[k::n]
    todo = [t for t in tasks if t["file"] in fetched and t["file"] not in done]
    print(f"detect: {n_sel} selected"
          + (f"; shard {args.shard[0]}/{args.shard[1]}: {len(tasks)} of them" if args.shard else "")
          + f"; {len(fetched)} fetched; {n_done_out} already in {args.out_jsonl}"
          + (f" ({len(done)} with {args.done_jsonl})" if args.done_jsonl else "")
          + f"; {len(todo)} to do; det_size {args.det_size}, score_thr {args.score_thr}",
          flush=True)
    if not todo:
        return

    cv2.setNumThreads(2)
    model = SCRFD(args.model, args.det_size, args.score_thr)
    model_name = args.model_name or Path(args.model).stem
    out = Jsonl(args.out_jsonl)
    failed = Jsonl(paths["failed"])

    def load(t: dict):
        img = cv2.imread(os.path.join(frames_dir, t["file"]), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None
        canvas, ratio = model.preprocess(img)
        return canvas, ratio, img.shape[:2]

    n_ok = n_bad = n_lo = n_hi = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.det_workers) as pool:
        window = args.det_workers * 4
        futs = [pool.submit(load, t) for t in todo[:window]]
        nxt = window
        for i, t in enumerate(todo):
            canvas, ratio, hw = futs[i].result()
            futs[i] = None
            if nxt < len(todo):
                futs.append(pool.submit(load, todo[nxt]))
                nxt += 1
            if canvas is None:
                n_bad += 1
                failed.write({"stage": "detect", "file": t["file"], "reason": "unreadable",
                              "ts": time.strftime("%F %T")})
                continue
            dets = model.detect(canvas, ratio)
            boxes = [[round(float(x), 1) for x in d[:4]] + [round(float(d[4]), 4)] for d in dets]
            n_faces = sum(1 for b in boxes if b[4] >= args.face_thr)
            out.write({
                "episode_id": t["episode_id"], "session_id": t["session_id"],
                "chunk": t["chunk"], "view": t["view"], "s": t["s"], "batch": t["batch"],
                "t_ms": t["t_ms"], "frame_idx": fetched[t["file"]]["frame_idx"],
                "w": int(hw[1]), "h": int(hw[0]), "file": t["file"], "model": model_name,
                "n_faces": n_faces, "boxes": boxes,
            })
            n_ok += 1
            n_lo += len(boxes)
            n_hi += n_faces
            if n_ok % 500 == 0:
                el = time.time() - t0
                print(f"detect {n_ok}/{len(todo)}  {n_ok / el:.1f} fps  boxes/frame {n_lo / n_ok:.2f} "
                      f"(>= {args.face_thr}: {n_hi / n_ok:.2f})  eta {(len(todo) - n_ok) / (n_ok / el) / 60:.1f} min",
                      flush=True)
    out.close()
    failed.close()
    el = time.time() - t0
    print(f"detect done: {n_ok} frames, {n_bad} unreadable, {el:.0f}s ({n_ok / max(el, 1e-9):.1f} fps), "
          f"boxes/frame {n_lo / max(n_ok, 1):.2f} at >= {args.score_thr}, "
          f"{n_hi / max(n_ok, 1):.2f} at >= {args.face_thr}", flush=True)


# ----------------------------------------------------------------------------- main
def parse_shard(text: str) -> tuple[int, int]:
    """'K/N' -> (K, N) with 0 <= K < N."""
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard wants K/N, got {text!r}")
    if n < 1 or not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= K < N")
    return k, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "fetch", "detect"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke)")
    ap.add_argument("--batch", type=int, default=None, help="residues with batch <= B (default: all used)")
    ap.add_argument("--episodes-from", default=None, help="episode list to use instead of --episodes (smoke)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="OVERRIDE: use chunk_<N> for every episode instead of chunk_sel (sabotage tests)")
    ap.add_argument("--view", default="vst_left", choices=sorted(VIEW_SHORT))
    ap.add_argument("--frames-from", default=None, metavar="CSV",
                    help="frames.csv (WOR-94): tasks = its rows at --round, instead of episodes x residues")
    ap.add_argument("--round", type=int, default=1, help="annotation_round to take from --frames-from")
    ap.add_argument("--out-suffix", default=None, metavar="SFX",
                    help='suffix for frames dir / fetch / failed / timing / armW files under --out-dir '
                         '(default: "" for vst_left, "_right" for vst_right)')
    ap.add_argument("--episodes", default=str(ROOT / "data/faceight/episodes.txt"))
    ap.add_argument("--residues", default=str(ROOT / "data/faceight/residues.csv"))
    ap.add_argument("--usage", default=str(ROOT / "data/episode_usage.csv"))
    ap.add_argument("--out-dir", default="/data/esteban/faceight")
    ap.add_argument("--out-jsonl", default=None, help="default <out-dir>/faceight_armW<SFX>.jsonl")
    ap.add_argument("--done-jsonl", default=None, metavar="FILE",
                    help="detect: records whose file is in FILE are also skipped (WOR-185)")
    ap.add_argument("--shard", default=None, metavar="K/N", type=parse_shard,
                    help="detect: only the selected tasks with index %% N == K (WOR-185)")
    ap.add_argument("--osslib-dir", default="/data/esteban/face-mine-rview", help="dir with osslib.py, mp4meta.py")
    ap.add_argument("--workers", type=int, default=48, help="fetch threads (one video per task)")
    ap.add_argument("--max-consec-fail", type=int, default=60)
    ap.add_argument("--no-timing", action="store_true", help="skip moov fetch; frame_idx = null")
    ap.add_argument("--head-check", action="store_true", help="HEAD every selected video key and exit")
    ap.add_argument("--model", default="/data/esteban/armw/weights/det_10g_armW.onnx")
    ap.add_argument("--model-name", default=None, help="value of the record's `model` field (default: onnx stem)")
    ap.add_argument("--det-size", type=int, default=1024)
    ap.add_argument("--score-thr", type=float, default=0.1, help="boxes kept in the record")
    ap.add_argument("--face-thr", type=float, default=0.6, help="threshold for n_faces")
    ap.add_argument("--det-workers", type=int, default=8)
    args = ap.parse_args()
    if args.out_suffix is None:
        args.out_suffix = "" if args.view == "vst_left" else "_" + args.view.split("_", 1)[1]
    if args.out_jsonl is None:
        args.out_jsonl = os.path.join(args.out_dir, f"faceight_armW{args.out_suffix}.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    tasks, skipped = build_tasks(args)
    chunks = sorted({t["chunk"] for t in tasks})
    print(f"{len(tasks)} frames = {len({t['episode_id'] for t in tasks})} episodes x "
          f"{len({t['s'] for t in tasks})} residues {sorted({t['s'] for t in tasks})}; "
          f"chunk {'OVERRIDE ' if args.chunk is not None else 'chunk_sel '}"
          f"{chunks[0] if len(chunks) == 1 else f'{chunks[0]}..{chunks[-1]}' if chunks else 'n/a'}/{args.view}; "
          f"{len(skipped)} episodes skipped (chunk_sel -1); out {args.out_dir} suffix '{args.out_suffix}'"
          + (f"; from {args.frames_from} round {args.round}" if args.frames_from else ""), flush=True)
    if args.head_check:
        head_check(args, tasks)
        return
    n_new = record_skipped(args, skipped)
    if skipped:
        print(f"skipped: {len(skipped)} episodes with no chunk >= 240 s -> failed.jsonl "
              f"(no-chunk-240; {n_new} new, {len(skipped) - n_new} already recorded)", flush=True)
    if args.stage in ("all", "fetch"):
        stage_fetch(args, tasks)
    if args.stage in ("all", "detect"):
        stage_detect(args, tasks)


if __name__ == "__main__":
    main()

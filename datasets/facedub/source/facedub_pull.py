#!/usr/bin/env python
"""facedub: pull one frame per second from both wrist mp4s of the facedub sessions (PII-1490).

Model: mining/faceight_pull.py (PII-61, PII-324). Same osslib loading, public-endpoint
fallback, snapshot retry, stts-derived frame_idx, MIN_BYTES sanity check, resumable stages,
--limit and --workers. This script only pulls frames: the prelabels are egoblurC_n8_fluence
(detectron2, PII-1475 route), produced by mining/facedub_egoblur_detect.py, not by an onnx
SCRFD stage (armAE34 was dropped on 2026-09-25 before any run).

Selection (PII-1489 / PII-1490): every session in data/facedub/sessions.txt, EVERY chunk of it,
both wrists (view uvc_left and uvc_right), s = --residue, +--modulus, ... while
`1000*s + 233 <= duration_ms` of that wrist mp4. The default `--residue 0 --modulus 1` is the
1 fps facedub_a selection; PII-1519 splits the 644 new sessions into ten rounds r = 0..9 with
`--residue r --modulus 10` (0.1 fps each, the union is 1 fps). A session is cut into ~360 s
chunks and the wrist video continues in chunk_001, chunk_002, ..., so the chunks are walked
from 0 upwards until --miss-stop (2)
consecutive chunk indices have no wrist mp4 at all, as episode_chunks_probe.py does for the vst
streams. t_ms is media time from the start of THAT chunk's mp4. duration_ms comes from
`uvc_<hand>_timestamps.json` (`(last_pts_us - first_pts_us) / 1000`), else from the mp4 stts
(moov ranged GETs), else, on chunk_000 only, from the session duration of --durations
(`duration_s` or `csv_duration_min` column; `--durations ''` turns that last fallback off, which
is what PII-1519 does: a wrist mp4 with neither timestamps.json nor a moov is an unfinalized mp4
whose snapshots all 400, so guessing its duration only buys failed snapshots). A wrist chunk with
no duration goes to failed.jsonl (`no-duration`) and is skipped. The OSS snapshot API is
addressed by time (`video/snapshot,t_<ms>,f_jpg`), never by frame number; frame_idx is derived from the mp4 stts
(nearest pts to t_ms) and recorded next to t_ms. A snapshot error never aborts the run: it
lands in failed.jsonl with a reason and is retried on the next run.

Rotation: the snapshot returns the unrotated 1280x1024 frame although the wrist mp4 metadata
says `rotation_deg: 270` (PII-1487), so every frame is rotated 270 degrees clockwise before it
is stored and the stored image is 1024x1280 portrait. Backend --rotate-tool: `jpegtran
-rotate 270 -perfect` (lossless) when jpegtran is on PATH, else Pillow `rotate(90, expand=True)`
saved at --jpeg-quality (90 cw counter-clockwise == 270 cw). The backend actually used is
printed and recorded per frame in fetch.jsonl (`rot`).

Layout under --out-dir (default /data/esteban/facedub):
  frames/<session>_c<chunk:03d>_w<left|right>_t<ms>.jpg   stored (rotated) JPEGs, kept
  wrists.json        plan cache: {"wrists": {"<session>/<chunk>/<hand>": duration_ms, src,
                     rotation_deg, width, height, mp4_bytes}, "sessions": {"<session>":
                     {n_chunks, chunks, probed_to}}}; --wrists moves it out of --out-dir so a
                     series of rounds over one session list probes the chunks once
  fetch.jsonl        one line per stored frame (file, key, t_ms, frame_idx, w, h, bytes, rot)
  timing.json        per-video stts cache (moov ranged GETs); --timing-path moves it out of
                     --out-dir, and it is rewritten only when a new video was parsed.
                     `--timing-path ""` keeps no cache at all: each video's moov is then parsed
                     once per run and its pts stay local to the worker. That is what PII-1519
                     does, because a fragmented stts is ~160 KB per video and 5,282 wrist videos
                     would be an 850 MB JSON file and several GB of live Python lists
  failed.jsonl       every failure, any stage, with a reason; retried on rerun

The fetch stage is resumable: a frame is fetched only if its file is missing or invalid, so a
rerun is a no-op. It needs Pillow (on shang: /usr/bin/python3, PIL 12.3.0; the armw venv has no
Pillow and there is no package mirror from shang).

usage: facedub_pull.py [--stage all|plan|fetch] [--limit N_SESSIONS] [--hand left|right|both]
                       [--sessions FILE] [--durations CSV] [--out-dir DIR] [--workers 48]
                       [--residue R] [--modulus M] [--wrists FILE] [--timing-path FILE]
                       [--rotate-tool auto|jpegtran|pillow] [--jpeg-quality 95]
                       [--no-timing] [--head-check] [--replan] [--frame-list FILE]
                       [--max-chunks 64] [--miss-stop 2]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # the pii_data checkout (PII-1639)
SHARED = ROOT / "data" / "mining"   # episode_chunks_probe.py
PHASE_MS = 233
HANDS = ("left", "right")
SRC_BUCKET = "we-fpv-sh-ns"  # the facedub sessions live in the bucket faceight uses
ROT_CW = 270
MIN_BYTES = 50_000  # a real 1280x1024 wrist snapshot is 130-350 KB; anything tiny is an error page


# ----------------------------------------------------------------------------- keys and names
def view_of(hand: str) -> str:
    return f"uvc_{hand}"


def video_key(session: str, chunk: int, hand: str) -> str:
    return f"{session}/chunk_{chunk:03d}/{view_of(hand)}/{view_of(hand)}_video.mp4"


def timestamps_key(session: str, chunk: int, hand: str) -> str:
    return f"{session}/chunk_{chunk:03d}/{view_of(hand)}/{view_of(hand)}_timestamps.json"


def frame_name(session: str, chunk: int, hand: str, t_ms: int) -> str:
    return f"{session}_c{chunk:03d}_w{hand}_t{t_ms}.jpg"


# ----------------------------------------------------------------------------- inputs
def load_sessions(path: str, limit: int = 0) -> list[str]:
    out = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    if len(set(out)) != len(out):
        raise SystemExit(f"{path}: duplicate session ids")
    if limit:
        out = out[:limit]
    if not out:
        raise SystemExit(f"{path}: no sessions")
    return out


def load_durations(path: str) -> dict[str, int]:
    """session_id -> duration_ms from the episode CSV (the chunk_000 fallback, PII-1489).

    Accepts a `duration_s` column (data/facedub/sessions_duration.csv) or a `csv_duration_min`
    one (data/facedub/sessions_b_duration.csv). An empty --durations means no CSV fallback.
    """
    if not path:
        return {}
    out = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = set(rd.fieldnames or [])
        if "session_id" not in cols:
            raise SystemExit(f"{path}: no session_id column")
        col, mul = ("duration_s", 1000.0) if "duration_s" in cols else ("csv_duration_min", 60000.0)
        if col not in cols:
            raise SystemExit(f"{path}: need duration_s or csv_duration_min, have {sorted(cols)}")
        for row in rd:
            out[row["session_id"]] = int(round(float(row[col]) * mul))
    return out


def out_paths(args) -> dict[str, str]:
    d = args.out_dir
    return {"frames": os.path.join(d, "frames"), "fetch": os.path.join(d, "fetch.jsonl"),
            "failed": os.path.join(d, "failed.jsonl"),
            "timing": os.path.join(d, "timing.json") if args.timing_path is None else args.timing_path,
            "wrists": args.wrists or os.path.join(d, "wrists.json")}


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


def jpeg_size(b: bytes) -> tuple[int, int]:
    """(w, h) from the first SOF marker; (0, 0) if none is found."""
    i = 2
    n = len(b)
    while i + 9 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 2
            continue
        seg = struct.unpack(">H", b[i + 2:i + 4])[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return int(w), int(h)
        i += 2 + seg
    return 0, 0


# ----------------------------------------------------------------------------- osslib
def load_osslib(args):
    """Import osslib/mp4meta from --osslib-dir, check the bucket, select a reachable OSS endpoint."""
    sys.path.insert(0, args.osslib_dir)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(SHARED))
    import osslib  # noqa: E402  (credentials + presign live there)
    import mp4meta  # noqa: E402
    from episode_chunks_probe import setup_endpoint  # noqa: E402  (public-endpoint fallback, WOR-62)

    if osslib.SRC_BUCKET != SRC_BUCKET:
        raise SystemExit(f"{args.osslib_dir}/osslib.py: SRC_BUCKET is {osslib.SRC_BUCKET!r}, "
                         f"the facedub sessions are in {SRC_BUCKET!r}; refusing to run")
    setup_endpoint(osslib)
    return osslib, mp4meta


# ----------------------------------------------------------------------------- plan
def stage_plan(args, sessions: list[str], force: bool = False) -> dict:
    """Walk the chunks of every session and cache each wrist mp4's own duration in wrists.json.

    {"wrists": {"<session>/<chunk>/<hand>": entry}, "sessions": {"<session>": {n_chunks, chunks}}}
    entry = {duration_ms, src: timestamps|stts|csv, rotation_deg, width, height, ...} or {err}.
    A chunk exists when either wrist mp4 HEADs; the walk stops after --miss-stop consecutive
    missing chunks (episode_chunks_probe.py does the same for the vst streams). duration_ms comes
    from `uvc_<hand>_timestamps.json`, else from the mp4 stts (moov ranged GETs), else, on
    chunk_000 only, from the session duration of sessions_duration.csv.
    """
    paths = out_paths(args)
    cache = json.load(open(paths["wrists"])) if os.path.exists(paths["wrists"]) else {}
    if "wrists" not in cache:  # pre-multichunk file: keys were "<session>/<hand>", all chunk 0
        cache = {"wrists": {f"{k.split('/')[0]}/0/{k.split('/')[1]}": v for k, v in cache.items()},
                 "sessions": {}}
    cache.setdefault("sessions", {})
    wr, ses_meta = cache["wrists"], cache["sessions"]
    csv_ms = load_durations(args.durations)
    want = [s for s in sessions if force or s not in ses_meta]
    if want:
        osslib, mp4meta = load_osslib(args)
        lock = threading.Lock()

        def wrist_entry(ses: str, chunk: int, hand: str) -> dict:
            """{duration_ms, src, ...} or {err}; None if the mp4 itself is absent."""
            ent: dict = {}
            b, err = osslib.get(timestamps_key(ses, chunk, hand))
            if b is None:
                ent["ts_err"] = err
            else:
                try:
                    meta = json.loads(b)
                    ent = {"duration_ms": (int(meta["last_pts_us"]) - int(meta["first_pts_us"])) / 1000.0,
                           "src": "timestamps", "rotation_deg": meta.get("rotation_deg"),
                           "width": meta.get("width"), "height": meta.get("height"),
                           "first_pts_us": meta.get("first_pts_us"), "last_pts_us": meta.get("last_pts_us")}
                except Exception as e:  # noqa: BLE001
                    ent["ts_err"] = f"{type(e).__name__}:{str(e)[:120]}"
            key = video_key(ses, chunk, hand)
            n, herr = osslib.head(key)
            ent["mp4_bytes"] = n
            if not n:
                ent["head_err"] = herr
                if "duration_ms" not in ent:
                    return {}  # no object at all: this chunk does not exist for this wrist
            if "duration_ms" not in ent:
                try:  # the mp4 is there but its timestamps.json is not: use the stts duration
                    tm = mp4meta.parse_video_timing(mp4meta.get_moov(key))
                    nb = sum(c * d for c, d in tm["stts"])
                    ent.update({"duration_ms": nb * 1000.0 / tm["timescale"], "src": "stts"})
                except Exception as e:  # noqa: BLE001
                    ent["stts_err"] = f"{type(e).__name__}:{str(e)[:120]}"
            if "duration_ms" not in ent and chunk == 0 and ses in csv_ms:
                ent.update({"duration_ms": float(csv_ms[ses]), "src": "csv"})
            if "duration_ms" not in ent:
                ent["err"] = "no-duration:" + str(ent.get("ts_err") or ent.get("stts_err"))
            return ent

        def probe_session(ses: str) -> None:
            chunks, miss, c = [], 0, 0
            while c < args.max_chunks and miss < args.miss_stop:
                ents = {h: wrist_entry(ses, c, h) for h in args.hands}
                if any(ents.values()):
                    miss = 0
                    chunks.append(c)
                    with lock:
                        for h, e in ents.items():
                            if e:
                                wr[f"{ses}/{c}/{h}"] = e
                else:
                    miss += 1
                c += 1
            with lock:
                ses_meta[ses] = {"n_chunks": len(chunks), "chunks": chunks,
                                 "probed_to": c - 1, "ts": time.strftime("%F %T")}

        print(f"plan: walking the chunks of {len(want)} sessions ({len(ses_meta)} cached)", flush=True)
        with ThreadPoolExecutor(max(1, min(args.workers, len(want)))) as ex:
            for f in [ex.submit(probe_session, s) for s in want]:
                f.result()
        json.dump(cache, open(paths["wrists"] + ".tmp", "w"), indent=1)
        os.replace(paths["wrists"] + ".tmp", paths["wrists"])

    sel = [(s, c, h) for s in sessions for c in ses_meta[s]["chunks"] for h in args.hands
           if f"{s}/{c}/{h}" in wr]
    bad = [k for k in sel if "err" in wr[f"{k[0]}/{k[1]}/{k[2]}"]]
    if bad:
        have = {(r.get("session_id"), r.get("chunk"), r.get("hand")) for r in read_jsonl(paths["failed"])
                if r.get("stage") == "plan"}
        failed = Jsonl(paths["failed"])
        for ses, c, hand in bad:
            if (ses, c, hand) not in have:
                failed.write({"stage": "plan", "session_id": ses, "chunk": c, "hand": hand,
                              "view": view_of(hand), "key": timestamps_key(ses, c, hand),
                              "reason": wr[f"{ses}/{c}/{hand}"]["err"], "ts": time.strftime("%F %T")})
        failed.close()
        print(f"plan: {len(bad)} wrist chunks with no duration -> failed.jsonl: {bad[:4]}", flush=True)
    srcs: dict = {}
    for ses, c, h in sel:
        srcs[wr[f"{ses}/{c}/{h}"].get("src")] = srcs.get(wr[f"{ses}/{c}/{h}"].get("src"), 0) + 1
    rots = sorted({wr[f"{a}/{b}/{c}"].get("rotation_deg") for a, b, c in sel}, key=str)
    nch = {s: ses_meta[s]["n_chunks"] for s in sessions}
    print(f"plan: {len(sel) - len(bad)} wrist chunks with a duration over {len(sessions)} sessions "
          f"({sum(nch.values())} chunks, max {max(nch.values())}); duration source {srcs}; "
          f"rotation_deg in mp4 metadata {rots}", flush=True)
    return cache


def build_tasks(args, sessions: list[str], plan: dict) -> list[dict]:
    """One task per (session, chunk, hand, s) with 1000*s + 233 <= that wrist mp4's duration_ms
    (t_ms is media time from the start of THAT chunk's mp4, which is what the snapshot API takes).

    s runs over --residue, --residue + --modulus, ...: modulus 1 is the 1 fps facedub_a set, and
    the ten residues of modulus 10 partition it (PII-1519).
    """
    wr, ses_meta = plan["wrists"], plan["sessions"]
    tasks = []
    for ses in sessions:
        for chunk in ses_meta[ses]["chunks"]:
            for hand in args.hands:
                ent = wr.get(f"{ses}/{chunk}/{hand}")
                if not ent or "err" in ent:
                    continue
                dur = float(ent["duration_ms"])
                s = args.residue
                while 1000 * s + PHASE_MS <= dur:
                    t_ms = 1000 * s + PHASE_MS
                    tasks.append({"session_id": ses, "hand": hand, "view": view_of(hand),
                                  "chunk": chunk, "s": s, "t_ms": t_ms,
                                  "key": video_key(ses, chunk, hand),
                                  "file": frame_name(ses, chunk, hand, t_ms)})
                    s += args.modulus
    if not tasks:
        raise SystemExit("no frames selected")
    return tasks


# ----------------------------------------------------------------------------- rotation
def pick_rotate_tool(which: str) -> str:
    if which in ("jpegtran", "pillow"):
        return which
    return "jpegtran" if shutil.which("jpegtran") else "pillow"


def rotate_cw270(b: bytes, tool: str, quality: int) -> tuple[bytes, int, int]:
    """Rotate a JPEG 270 degrees clockwise; returns (bytes, w, h) of the result."""
    if tool == "jpegtran":
        r = subprocess.run(["jpegtran", "-rotate", "270", "-perfect", "-copy", "all"],
                           input=b, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError(f"jpegtran rc {r.returncode}: {r.stderr[:120].decode('latin1')}")
        out = r.stdout
    else:
        from PIL import Image
        im = Image.open(io.BytesIO(b))
        im.load()
        buf = io.BytesIO()
        im.rotate(90, expand=True).save(buf, format="JPEG", quality=quality)  # 90 ccw == 270 cw
        out = buf.getvalue()
    w, h = jpeg_size(out)
    if not w:
        raise RuntimeError("rotated jpeg has no SOF marker")
    return out, w, h


# ----------------------------------------------------------------------------- fetch
def nearest_frame(pts: list[float], t_ms: int) -> int:
    i = bisect.bisect_left(pts, float(t_ms))
    if i == 0:
        return 0
    if i >= len(pts):
        return len(pts) - 1
    return i if abs(pts[i] - t_ms) < abs(t_ms - pts[i - 1]) else i - 1


def head_check(args, tasks: list[dict]) -> None:
    """HEAD every distinct wrist video; exit 1 if any is missing."""
    osslib, _ = load_osslib(args)
    keys = {}
    for t in tasks:
        keys.setdefault(t["key"], t)
    bad = 0
    for key, t in keys.items():
        n, err = osslib.head(key)
        ok = n is not None and n > 0
        bad += not ok
        print(f"{'OK ' if ok else 'MISSING'} {key} {n if ok else err}", flush=True)
    print(f"head-check: {len(keys)} videos, {len(keys) - bad} present, {bad} missing", flush=True)
    if bad:
        sys.exit(1)


def stage_fetch(args, tasks: list[dict]) -> None:
    osslib, mp4meta = load_osslib(args)
    tool = pick_rotate_tool(args.rotate_tool)
    print(f"fetch: rotating every frame {ROT_CW} deg clockwise with {tool}"
          + ("" if tool == "jpegtran" else f" (Pillow, quality {args.jpeg_quality})"), flush=True)

    paths = out_paths(args)
    frames_dir = paths["frames"]
    os.makedirs(frames_dir, exist_ok=True)
    timing_path = paths["timing"]  # "" disables the cache: parse each moov once, keep no stts
    timing = json.load(open(timing_path)) if timing_path and os.path.exists(timing_path) else {}
    tlock = threading.Lock()
    tdirty = set()  # keys parsed by THIS run; nothing new means timing.json is not rewritten
    manifest_path = paths["fetch"]
    have = {r["file"] for r in read_jsonl(manifest_path)}
    manifest = Jsonl(manifest_path)
    failed = Jsonl(paths["failed"])

    by_video: dict[str, list[dict]] = {}
    for t in tasks:
        if t["file"] in have and valid_jpeg(os.path.join(frames_dir, t["file"])):
            continue
        by_video.setdefault(t["key"], []).append(t)
    n_pending = sum(len(v) for v in by_video.values())
    print(f"fetch: {len(tasks)} frames selected; {len(tasks) - n_pending} already fetched; "
          f"{n_pending} pending over {len(by_video)} videos; {args.workers} workers", flush=True)
    if not n_pending:
        return

    stats = {"fetched": 0, "bytes": 0, "raw_bytes": 0, "failed": 0, "consec_fail": 0}
    slock = threading.Lock()
    stop = threading.Event()

    def fail(t: dict, reason: str) -> None:
        failed.write({"stage": "fetch", "file": t["file"], "key": t["key"], "t_ms": t["t_ms"],
                      "session_id": t["session_id"], "chunk": t["chunk"], "hand": t["hand"], "s": t["s"],
                      "reason": reason, "ts": time.strftime("%F %T")})
        with slock:
            stats["failed"] += 1
            stats["consec_fail"] += 1
            if stats["consec_fail"] >= args.max_consec_fail:
                stop.set()
                print(f"STOP: {stats['consec_fail']} consecutive failures", flush=True)

    def get_timing(key: str):
        """{'ts','stts','nb'} or {'err'}; cached per video in timing.json."""
        if timing_path:
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
        if timing_path:
            with tlock:
                timing[key] = v
                tdirty.add(key)
        return v

    def process_video(key: str, ts: list[dict]) -> None:
        if stop.is_set():
            return
        pts = fps = nb = dur_ms = None
        if not args.no_timing:
            v = get_timing(key)
            if "err" in v:
                print(f"WARN timing {key}: {v['err']}; frame_idx null for this video", flush=True)
            else:
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
                fail(t, f"past-end:t{t['t_ms']}>stts_dur{round(dur_ms)}")
                continue
            n = valid_jpeg(path)
            attempts = 0
            src_wh = None
            if not n:
                b, attempts, err = osslib.snapshot(key, t["t_ms"])
                if b is None:
                    fail(t, f"snapshot:{err}")
                    continue
                if len(b) < MIN_BYTES or b[:2] != b"\xff\xd8" or b[-2:] != b"\xff\xd9":
                    fail(t, f"bad-jpeg:{len(b)}B")
                    continue
                src_wh = jpeg_size(b)
                try:
                    rb, w, h = rotate_cw270(b, tool, args.jpeg_quality)
                except Exception as e:  # noqa: BLE001
                    fail(t, f"rotate:{type(e).__name__}:{str(e)[:100]}")
                    continue
                tmp = path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(rb)
                os.replace(tmp, path)
                with slock:
                    stats["raw_bytes"] += len(b)
                n = len(rb)
            else:
                with open(path, "rb") as fh:
                    w, h = jpeg_size(fh.read(4096))
            manifest.write({"file": t["file"], "key": key, "session_id": t["session_id"],
                            "chunk": t["chunk"], "hand": t["hand"], "view": t["view"],
                            "s": t["s"], "t_ms": t["t_ms"],
                            "frame_idx": frame_idx, "fps": fps, "nb": nb, "w": w, "h": h,
                            "src_wh": src_wh, "rot": f"{tool}:{ROT_CW}cw", "bytes": n,
                            "attempts": attempts, "ts": time.strftime("%F %T")})
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
        """Rewrite the stts cache only if this run parsed a new video (it is shared by rounds)."""
        if not timing_path:
            return
        with tlock:
            if not tdirty:
                return
            snap = dict(timing)
        json.dump(snap, open(timing_path + ".tmp", "w"))
        os.replace(timing_path + ".tmp", timing_path)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_video, k, v) for k, v in by_video.items()]
        last_save = time.time()
        for f in futs:
            f.result()
            if time.time() - last_save > 120:
                save_timing()
                last_save = time.time()
    save_timing()
    manifest.close()
    failed.close()
    el = time.time() - t0
    k = stats["fetched"]
    print(f"fetch done: {k} frames, {stats['failed']} failed, {stats['bytes'] / 1e6:.1f} MB stored "
          f"({stats['raw_bytes'] / 1e6:.1f} MB raw), {el:.0f}s ({k / max(el, 1e-9):.2f} fps, "
          f"{stats['bytes'] / max(k, 1) / 1e3:.0f} KB/frame)"
          + ("  STOPPED EARLY" if stop.is_set() else ""), flush=True)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "plan", "fetch"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke)")
    ap.add_argument("--hand", default="both", choices=["left", "right", "both"])
    ap.add_argument("--sessions", default=str(ROOT / "data/facedub/sessions.txt"))
    ap.add_argument("--durations", default=str(ROOT / "data/facedub/sessions_duration.csv"))
    ap.add_argument("--out-dir", default="/data/esteban/facedub")
    ap.add_argument("--residue", type=int, default=0, help="first s of the selection (PII-1519 round)")
    ap.add_argument("--modulus", type=int, default=1,
                    help="s step: 1 is 1 fps, 10 gives the ten 0.1 fps rounds")
    ap.add_argument("--wrists", default=None, metavar="FILE",
                    help="plan cache path (default <out-dir>/wrists.json); share it across rounds")
    ap.add_argument("--timing-path", default=None, metavar="FILE",
                    help="stts cache path (default <out-dir>/timing.json); share it across rounds, "
                         "or pass an empty string to keep no cache (one moov parse per video per run)")
    ap.add_argument("--osslib-dir", default="/data/esteban/face-mine-rview", help="dir with osslib.py, mp4meta.py")
    ap.add_argument("--workers", type=int, default=48, help="fetch threads (one video per task)")
    ap.add_argument("--max-consec-fail", type=int, default=60)
    ap.add_argument("--replan", action="store_true", help="re-walk the chunks of every session")
    ap.add_argument("--max-chunks", type=int, default=64, help="highest chunk index probed per session")
    ap.add_argument("--miss-stop", type=int, default=2,
                    help="stop the chunk walk after N consecutive chunks with no wrist mp4")
    ap.add_argument("--rotate-tool", default="auto", choices=["auto", "jpegtran", "pillow"])
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Pillow save quality")
    ap.add_argument("--no-timing", action="store_true", help="skip moov fetch; frame_idx = null")
    ap.add_argument("--head-check", action="store_true", help="HEAD every selected video key and exit")
    ap.add_argument("--frame-list", default=None, metavar="FILE",
                    help="after fetch, write the absolute path of every stored frame to FILE "
                         "(the input of mining/facedub_egoblur_detect.py)")
    args = ap.parse_args()
    if args.modulus < 1 or not 0 <= args.residue < args.modulus:
        raise SystemExit(f"--residue {args.residue} --modulus {args.modulus}: need 0 <= R < M, M >= 1")
    args.hands = HANDS if args.hand == "both" else (args.hand,)
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = load_sessions(args.sessions, args.limit)
    plan = stage_plan(args, sessions, force=args.replan)
    tasks = build_tasks(args, sessions, plan)
    per_hand = {h: sum(1 for t in tasks if t["hand"] == h) for h in args.hands}
    print(f"{len(tasks)} frames = {len({t['session_id'] for t in tasks})} sessions x "
          f"{len({(t['session_id'], t['chunk']) for t in tasks})} session-chunks x "
          f"{'/'.join(view_of(h) for h in args.hands)} at {1.0 / args.modulus:g} fps "
          f"(s = {args.residue} + {args.modulus}k, t_ms = 1000*s + {PHASE_MS} inside each chunk "
          f"mp4); per hand {per_hand}; out {args.out_dir}", flush=True)
    if args.stage == "plan":
        return
    if args.head_check:
        head_check(args, tasks)
        return
    if args.stage in ("all", "fetch"):
        stage_fetch(args, tasks)
    if args.frame_list:
        frames_dir = out_paths(args)["frames"]
        names = sorted(r["file"] for r in read_jsonl(out_paths(args)["fetch"]))
        with open(args.frame_list, "w") as fh:
            for n in names:
                fh.write(os.path.join(frames_dir, n) + "\n")
        print(f"frame list: {len(names)} paths -> {args.frame_list}", flush=True)


if __name__ == "__main__":
    main()

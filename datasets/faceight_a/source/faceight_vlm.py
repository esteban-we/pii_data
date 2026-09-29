#!/usr/bin/env python
"""faceight: ask a VLM how many human heads are in each frame (WOR-70).

Second, detector-independent count next to armW's n_faces. Reads fetch.jsonl in file
order, downscales each frame to --long-side px, sends it with the fixed prompt
PROMPTS[--prompt-id] to a vLLM OpenAI-compatible server, parses one integer.

Output (--out, default /data/esteban/faceight/faceight_vlm.jsonl), one line per frame:
  file, n_heads (int, or null if the answer was not parseable), raw (verbatim answer),
  model, prompt_id, latency_ms
Resumable: files already present in --out are skipped; a frame is written only after a
successful HTTP answer, so a killed run leaves no partial or duplicate records. Transport
errors are retried; after --max-consec-fail consecutive failures the client exits 2 (the
server is presumably down) and a rerun resumes.

Tracking:
  --progress  /data/esteban/faceight/vlm_progress.json  rewritten every ~30 s
              (done, total, fps, eta_min, last_update, errors, started, ...)
  --log       /data/esteban/faceight/vlm.log            one line per 500 frames
  mining/faceight_vlm_status.sh  prints both plus process liveness over one ssh
              (liveness = pid in --pidfile still running; a SIGKILLed client leaves a stale pidfile,
              the status script reports that as "client : NOT running (stale pidfile)").

Server (see mining/faceight_vlm_server.sh): vLLM, Qwen2.5-VL-72B-Instruct-AWQ, TP 2.

usage: faceight_vlm.py [--limit N] [--concurrency 24] [--long-side 1024]
                       [--server http://127.0.0.1:8100/v1] [--out FILE]
                       [--sabotage blank|absurd]   (verification only; use a separate --out)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

PROMPTS = {
    "heads_v1": (
        "Count the human heads visible in this image. Count every real person's head in any "
        "orientation (facing the camera, facing away, seen from above or from the side), "
        "including heads that are only partly visible or cut off by the image border. Do not "
        "count mannequins, dolls, posters, photos, screens, statues or reflections. "
        "Answer with a single integer and nothing else."
    ),
    # verification only: a question the frame cannot answer with a count of heads
    "absurd_v1": (
        "Count the live elephants visible in this image. "
        "Answer with a single integer and nothing else."
    ),
}
WORDS = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_count(raw: str) -> int | None:
    s = raw.strip().strip(".").strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    if s.lower() in WORDS:
        return WORDS[s.lower()]
    m = re.match(r"^[^\d]{0,24}?(\d+)\b", s)  # "There are 2 heads."
    if m and len(s) <= 60:
        return int(m.group(1))
    return None


def encode_frame(path: Path, long_side: int, quality: int, blank: bool) -> tuple[str, int, int]:
    if blank:
        im = Image.new("RGB", (long_side, long_side * 3 // 4), (128, 128, 128))
    else:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        sc = long_side / max(w, h)
        if sc < 1:
            im = im.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode(), im.size[0], im.size[1]


class Progress:
    def __init__(self, path: Path, log: Path, total: int, done0: int):
        self.path, self.log, self.total = path, log, total
        self.done0 = done0
        self.done = done0
        self.nulls = 0
        self.errors = 0
        self.started = time.time()
        self.lock = threading.Lock()
        self.window: list[tuple[float, int]] = [(self.started, done0)]

    def tick(self, null: bool):
        with self.lock:
            self.done += 1
            self.nulls += int(null)
            d = self.done
        if d % 500 == 0:
            self.logline(self.summary())

    def error(self):
        with self.lock:
            self.errors += 1

    def rates(self):
        now = time.time()
        self.window.append((now, self.done))
        while len(self.window) > 2 and now - self.window[0][0] > 300:
            self.window.pop(0)
        t0, d0 = self.window[0]
        fps_recent = (self.done - d0) / max(now - t0, 1e-6) if now - t0 > 5 else 0.0
        fps_all = (self.done - self.done0) / max(now - self.started, 1e-6)
        fps = fps_recent or fps_all
        eta = (self.total - self.done) / fps / 60 if fps > 0 else None
        return fps_recent, fps_all, eta

    def summary(self) -> str:
        fr, fa, eta = self.rates()
        return (f"{self.done}/{self.total}  {fr:.2f} fps (run avg {fa:.2f})  "
                f"eta {eta:.0f} min  null {self.nulls}  errors {self.errors}"
                if eta is not None else f"{self.done}/{self.total}  starting")

    def write(self, alive: bool):
        fr, fa, eta = self.rates()
        rec = {
            "done": self.done, "total": self.total, "fps": round(fr, 3), "fps_run_avg": round(fa, 3),
            "eta_min": None if eta is None else round(eta, 1),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "errors": self.errors, "null_answers_this_run": self.nulls,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "done_at_start": self.done0, "client_alive": alive, "pid": os.getpid(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, self.path)

    def logline(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log, "a") as fh:
            fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", default="/data/esteban/faceight/fetch.jsonl")
    ap.add_argument("--frames-dir", default="/data/esteban/faceight/frames")
    ap.add_argument("--out", default="/data/esteban/faceight/faceight_vlm.jsonl")
    ap.add_argument("--progress", default="/data/esteban/faceight/vlm_progress.json")
    ap.add_argument("--log", default="/data/esteban/faceight/vlm.log")
    ap.add_argument("--pidfile", default="/data/esteban/faceight/vlm_client.pid",
                    help="written at start, removed at exit; faceight_vlm_status.sh reads it")
    ap.add_argument("--server", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="Qwen2.5-VL-72B-Instruct-AWQ", help="served model name")
    ap.add_argument("--prompt-id", default="heads_v1", choices=sorted(PROMPTS))
    ap.add_argument("--limit", type=int, default=0, help="first N frames of fetch.jsonl (smoke)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-consec-fail", type=int, default=40)
    ap.add_argument("--sabotage", choices=["blank", "absurd"], default=None,
                    help="blank: send a flat grey image; absurd: use prompt absurd_v1")
    args = ap.parse_args()
    if args.sabotage == "absurd":
        args.prompt_id = "absurd_v1"
    if args.sabotage and args.out == ap.get_default("out"):
        sys.exit("--sabotage needs its own --out")
    prompt = PROMPTS[args.prompt_id]

    out = Path(args.out)
    done_files: set[str] = set()
    if out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    done_files.add(json.loads(line)["file"])
                except Exception:
                    pass
    todo: list[str] = []
    n_total = 0
    with open(args.fetch) as fh:
        for line in fh:
            if args.limit and n_total >= args.limit:
                break
            f = json.loads(line)["file"]
            n_total += 1
            if f not in done_files:
                todo.append(f)
    prog = Progress(Path(args.progress), Path(args.log), n_total, n_total - len(todo))
    if not args.sabotage:
        Path(args.pidfile).write_text(str(os.getpid()))
    prog.logline(f"start pid {os.getpid()} model {args.model} prompt {args.prompt_id} "
                 f"concurrency {args.concurrency} long_side {args.long_side} "
                 f"todo {len(todo)} already_done {len(done_files & set(todo)) + prog.done0} total {n_total}"
                 + (f" SABOTAGE={args.sabotage}" if args.sabotage else ""))
    if not todo:
        prog.write(alive=False)
        prog.logline("nothing to do")
        return 0

    sess = requests.Session()
    url = args.server.rstrip("/") + "/chat/completions"
    wlock = threading.Lock()
    consec_fail = 0
    stop = threading.Event()
    fout = open(out, "ab", buffering=0)  # one write() syscall per record: no partial lines on abrupt exit

    def one(fname: str):
        nonlocal consec_fail
        if stop.is_set():
            return
        b64, w, h = encode_frame(Path(args.frames_dir) / fname, args.long_side, args.jpeg_quality,
                                 blank=args.sabotage == "blank")
        body = {
            "model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        last_err = None
        for attempt in range(args.retries):
            t0 = time.time()
            try:
                r = sess.post(url, json=body, timeout=args.timeout)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"]
                lat = round((time.time() - t0) * 1000)
                rec = {"file": fname, "n_heads": parse_count(raw), "raw": raw, "model": args.model,
                       "prompt_id": args.prompt_id, "latency_ms": lat}
                with wlock:
                    fout.write((json.dumps(rec) + "\n").encode())
                    consec_fail = 0
                prog.tick(rec["n_heads"] is None)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        prog.error()
        with wlock:
            consec_fail += 1
            cf = consec_fail
        prog.logline(f"ERROR {fname}: {type(last_err).__name__}: {str(last_err)[:160]}")
        if cf >= args.max_consec_fail:
            prog.logline(f"{cf} consecutive failures, stopping (server down?)")
            stop.set()

    def heartbeat():
        while not stop.is_set():
            prog.write(alive=True)
            stop.wait(30)

    def on_term(signum, _frame):
        stop.set()  # queued tasks become no-ops; in-flight requests are dropped (redone on rerun)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    ex = ThreadPoolExecutor(args.concurrency)
    signalled = False
    try:
        for _ in ex.map(one, todo):
            pass
    except SystemExit:
        signalled = True
    finally:
        aborted = stop.is_set()  # consecutive failures or a signal
        stop.set()
        ex.shutdown(wait=not signalled, cancel_futures=True)
        hb.join(timeout=5)
        prog.write(alive=False)
        prog.logline(("aborted " if aborted else "finished ") + prog.summary())
        if not args.sabotage:
            Path(args.pidfile).unlink(missing_ok=True)
        if signalled:
            os._exit(2)  # do not wait for in-flight HTTP requests (up to ~20 s each)
        fout.close()
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())

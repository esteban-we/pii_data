"""Probe OSS for every episode's chunk inventory: which chunk_XXX exist and how long each is.

Metadata only (HEAD + mp4 moov via ranged GET), no frames. One JSONL line per session:
{session_id, episode_id, faceight, n_chunks, total_dur_s, chunks: [{i, bytes, dur_ms, nb, fps}], err}
faceight episodes are probed first. Resumable: sessions already in --out are skipped
(errored ones are redone unless --no-retry-errors; the LAST record per session wins).
Run on shang (next to the bucket), stdlib only:
  cd /root/repos/pii && nohup python3 mining/episode_chunks_probe.py > /data/esteban/faceight/chunks_probe.log 2>&1 &

Chunk walk: HEAD <session>/chunk_<i:03d>/vst_left/vst_left_video.mp4 for i = 0.. and stop after
two consecutive 404s (so a single missing index inside a session is tolerated). A HEAD that fails
for any reason other than HTTP 404 is retried; if it still fails the session is recorded with
err set and n_chunks reflects only what was seen, so "missing" is never inferred from a timeout.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VIDEO = "vst_left/vst_left_video.mp4"
PUBLIC_EP = "oss-cn-shanghai.aliyuncs.com"
FALLBACK_DNS = "223.5.5.5"


def oss_err_code(e) -> str | None:
    """OSS puts the error XML, base64-encoded, in x-oss-err on failed HEADs; return its <Code>."""
    x = e.headers.get("x-oss-err") if e.headers else None
    if not x:
        return None
    try:
        xml = base64.b64decode(x).decode("utf-8", "replace")
        return xml.split("<Code>", 1)[1].split("</Code>", 1)[0]
    except (ValueError, IndexError):
        return None


def setup_endpoint(osslib) -> str:
    """Make sure the bucket host resolves and accepts TCP; otherwise fall back.

    shang's VPC resolvers (100.100.2.x) and the internal OSS addresses went dark on 2026-09-08,
    while the public endpoint stayed reachable. Order: system DNS -> dig @223.5.5.5 (getaddrinfo is
    patched for that one host, so TLS hostname checks still apply) -> public endpoint, same steps.
    """
    import socket
    import subprocess

    def resolve(host: str) -> tuple[list[str], bool]:
        """(ips, via_system_dns)."""
        try:
            return sorted({a[4][0] for a in socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)}), True
        except OSError:
            pass
        try:
            r = subprocess.run(["dig", "+short", "+time=3", f"@{FALLBACK_DNS}", host],
                               capture_output=True, text=True, timeout=15)
            return [x for x in r.stdout.split() if x.replace(".", "").isdigit()], False
        except (OSError, subprocess.SubprocessError):
            return [], False

    def reachable(ip: str) -> bool:
        try:
            socket.create_connection((ip, 443), timeout=5).close()
            return True
        except OSError:
            return False

    for ep in [osslib.EP] + ([PUBLIC_EP] if osslib.EP != PUBLIC_EP else []):
        host = f"{osslib.SRC_BUCKET}.{ep}"
        cand, via_system = resolve(host)
        ips = [ip for ip in cand if reachable(ip)]
        if not ips:
            print(f"endpoint {host}: unresolvable or unreachable ({cand}), trying next", flush=True)
            continue
        osslib.EP = ep
        if via_system:
            print(f"endpoint {host} -> {ips} (system DNS)", flush=True)
            return ep
        orig = socket.getaddrinfo

        def gai(h, port=None, *a, _host=host, _ips=ips, **k):
            if h == _host:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 443)) for ip in _ips]
            return orig(h, port, *a, **k)

        socket.getaddrinfo = gai
        print(f"endpoint {host} -> {ips} (via dig @{FALLBACK_DNS}, getaddrinfo patched)", flush=True)
        return ep
    raise SystemExit("no OSS endpoint reachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "episode_usage.csv"))
    ap.add_argument("--out", default="/data/esteban/faceight/chunks_probe.jsonl")
    ap.add_argument("--osslib-dir", default="/data/esteban/face-mine-rview")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-chunks", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3, help="attempts per HEAD / moov fetch on non-404 errors")
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N to-do sessions")
    ap.add_argument("--sessions", default="", help="comma-separated session ids to probe instead of the CSV")
    ap.add_argument("--no-retry-errors", action="store_true", help="treat errored sessions in --out as done")
    ap.add_argument("--no-endpoint-check", action="store_true", help="use osslib.EP as is (no DNS/reachability fallback)")
    args = ap.parse_args()

    sys.path.insert(0, args.osslib_dir)
    import mp4meta  # noqa: E402
    import osslib  # noqa: E402

    if not args.no_endpoint_check:
        setup_endpoint(osslib)  # mp4meta shares the same osslib module object
    if args.sessions:
        rows = [{"session_id": s, "episode_id": "", "faceight": "0"} for s in args.sessions.split(",") if s]
    else:
        rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
        rows = [r for r in rows if r["session_id"]]
    rows.sort(key=lambda r: (r.get("faceight") != "1", r["session_id"]))
    done: set[str] = set()
    errored: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                rec = json.loads(line)
                s = rec["session_id"]
            except (ValueError, KeyError):
                continue
            if rec.get("err"):
                errored.add(s)
            else:
                done.add(s)
                errored.discard(s)
    if args.no_retry_errors:
        done |= errored
    todo = [r for r in rows if r["session_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} sessions, {len(done)} done, {len(errored - done)} errored, {len(todo)} to do "
          f"({sum(r.get('faceight') == '1' for r in todo)} faceight first)", flush=True)
    if not todo:
        return

    def head(key: str) -> tuple[int | None, bool, str | None]:
        """(bytes, is_404, err). Only an HTTP 404 counts as missing; anything else is retried then err."""
        last = None
        for a in range(args.retries):
            req = urllib.request.Request(osslib.presign(key, method="HEAD"), method="HEAD")
            try:
                r = urllib.request.urlopen(req, timeout=30)
                return int(r.headers.get("Content-Length", 0)), False, None
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    code = oss_err_code(e)
                    if code in (None, "NoSuchKey"):
                        return None, True, None
                    return None, False, f"HTTPError:404:{code}"  # e.g. NoSuchBucket: not a missing chunk
                last = f"HTTPError:{e.code}"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}:{str(e)[:100]}"
            time.sleep(1 + 2 * a)
        return None, False, last

    def moov_timing(key: str) -> dict:
        last = None
        for a in range(args.retries):
            try:
                return mp4meta.parse_video_timing(mp4meta.get_moov(key))
            except ValueError as e:  # malformed file: retrying will not help
                raise e
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1 + 2 * a)
        raise last  # type: ignore[misc]

    lock = threading.Lock()
    out = open(args.out, "a")
    n = [0]
    n_face_todo = sum(r.get("faceight") == "1" for r in todo)
    t0 = time.time()

    def probe(r: dict) -> dict:
        s = r["session_id"]
        rec = {"session_id": s, "episode_id": r.get("episode_id", ""), "faceight": int(r.get("faceight") == "1"),
               "n_chunks": 0, "total_dur_s": 0.0, "chunks": [], "err": None}
        misses = 0
        for i in range(args.max_chunks):
            key = f"{s}/chunk_{i:03d}/{VIDEO}"
            size, missing, err = head(key)
            if missing:
                misses += 1
                if misses >= 2:
                    break
                continue
            if err:
                rec["err"] = f"head:{i}:{err}"
                break
            misses = 0
            c = {"i": i, "bytes": size, "dur_ms": None, "nb": None, "fps": None}
            try:
                tm = moov_timing(key)
                nb = sum(cnt for cnt, _ in tm["stts"])
                dur = sum(cnt * d for cnt, d in tm["stts"]) * 1000.0 / tm["timescale"]
                c.update(dur_ms=round(dur, 1), nb=nb, fps=round(nb / (dur / 1000.0), 3) if dur else None)
            except Exception as e:  # noqa: BLE001
                c["err"] = f"{type(e).__name__}:{str(e)[:80]}"
                rec["err"] = f"moov:{i}:{c['err']}"
            rec["chunks"].append(c)
        else:
            rec["truncated"] = True  # hit --max-chunks without two consecutive 404s
        rec["n_chunks"] = len(rec["chunks"])
        rec["total_dur_s"] = round(sum((c["dur_ms"] or 0) for c in rec["chunks"]) / 1000.0, 1)
        return rec

    def work(r: dict) -> None:
        try:
            rec = probe(r)
        except Exception as e:  # noqa: BLE001
            rec = {"session_id": r["session_id"], "episode_id": r.get("episode_id", ""),
                   "faceight": int(r.get("faceight") == "1"), "err": f"fatal:{type(e).__name__}:{e}"}
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n[0] += 1
            if n[0] == n_face_todo and n_face_todo:
                print(f"faceight subset complete ({n_face_todo} sessions) after {(time.time() - t0) / 60:.1f} min", flush=True)
            if n[0] % 500 == 0:
                el = time.time() - t0
                print(f"{n[0]}/{len(todo)}  {n[0] / el:.1f} sessions/s  eta {(len(todo) - n[0]) / (n[0] / el) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    out.close()
    el = time.time() - t0
    print(f"done: {n[0]} sessions in {el / 60:.1f} min ({n[0] / max(el, 1e-9):.1f} sessions/s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

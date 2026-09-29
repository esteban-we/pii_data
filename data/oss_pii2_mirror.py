#!/usr/bin/env python3
"""PII-1413: mirror every image set and every arm of /data/esteban/pii onto
oss://algorithm-datasets/pii/.

Layout
    pii/data/<oss_dataset>/<image>          images, one prefix per set
    pii/models/<family>/<arm>/<rel>         everything under runs/train that git ignores

The plan is read from the pii2 indexes, never from a directory walk:
    datasets/<name>/frames.csv   image, size, md5 (+ batch for face10k)
    runs/train/MANIFEST.tsv      dst_rel, size, md5

oss_key rules (the same function the build scripts use, see oss_key_for_row):
    face10k          batch v3     -> pii/data/face10k_v3/<image>
                     batch repair -> pii/data/face10k_repair/<image>
    every other set               -> pii/data/<dataset>/<image>
    runs/train/<rel>             -> pii/models/<rel>

Subcommands
    plan            per prefix: local objects, bytes, how many are already on OSS
    upload          single-part PUT with Content-MD5, ETag asserted == md5, 16 parallel,
                    resumable (an object already there with the same size and ETag is
                    skipped). An object whose ETag differs is NEVER overwritten: it is
                    reported and counted as a conflict.
    copy-faceight   cross-bucket server-side CopyObject of the faceight images from
                    we-vlm-annotation-data-sh (eye-less Verdict names) into
                    pii/data/faceight_<set>/ under the pii2 names. Nothing is deleted.
    move-163        the PII-163 objects pii/models/<arm>/... -> pii/models/<family>/<arm>/...
                    CopyObject, verify the destination, then one DeleteObject per verified
                    source key (the only deletes this script can make, --yes required).
    verify          full paginated listing of pii/data/ and pii/models/ against the index
                    union: no missing key, no extra key, size and ETag == md5 everywhere.

Stdlib only. Credential: the `default` AK profile of ~/.aliyun/config.json.
Nothing outside pii/ is ever written; nothing local is written at all.
"""
import argparse
import base64
import csv
import hashlib
import hmac
import http.client
import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import formatdate

PII2 = "/data/esteban/pii"
BUCKET = "algorithm-datasets"
FACEIGHT_BUCKET = "we-vlm-annotation-data-sh"
REGION_HOST = "oss-cn-shanghai.aliyuncs.com"
DATA_TOP = "pii/data/"
MODELS_TOP = "pii/models/"

# pii2 dataset -> the OSS prefix name (equal except face10k, which predates the merge)
SETS = [
    "face10k", "face_mine_right", "face_mine_v1", "faceback_45",
    "faceight_a", "faceight_b", "faceight_c",
    "gt_bench_full", "gt_bench_sparse", "pii_frames",
    "wider_face", "wider_fisheye_fill", "wider_fisheye_target",
]
FACE10K_BATCH_PREFIX = {"v3": "face10k_v3", "repair": "face10k_repair"}


def oss_key_for_row(dataset, row):
    """The pii/data key of one frames.csv row. Deterministic; the build scripts use it."""
    if dataset == "face10k":
        return DATA_TOP + FACE10K_BATCH_PREFIX[row["batch"]] + "/" + row["image"]
    return DATA_TOP + dataset + "/" + row["image"]


def oss_key_for_run(dst_rel):
    """The pii/models key of a runs/train/<rel> manifest row, or None if it is git content."""
    if not dst_rel.startswith("runs/train/"):
        return None
    rel = dst_rel[len("runs/train/"):]
    parts = rel.split("/")
    if len(parts) < 3:
        return None          # models.csv, MANIFEST.tsv
    family, arm, tail = parts[0], parts[1], "/".join(parts[2:])
    base = os.path.basename(tail)
    ignored = (
        tail.startswith("epochs/") or tail.startswith("onnx/") or tail.startswith("logs/")
        or tail == "train.log"
        or base.endswith((".pth", ".onnx", ".jit", ".zip"))
    )
    if not ignored:
        return None
    return MODELS_TOP + family + "/" + arm + "/" + tail


# --------------------------------------------------------------------------- plan

def load_frames(dataset):
    with open(f"{PII2}/datasets/{dataset}/frames.csv", newline="") as f:
        return list(csv.DictReader(f))


def image_plan(names=None):
    """[(local_path, oss_key, size, md5)] for every image of the requested sets."""
    out = []
    for d in (names or SETS):
        for r in load_frames(d):
            key = r.get("oss_key") or oss_key_for_row(d, r)
            if key != oss_key_for_row(d, r):
                raise SystemExit(f"{d}/{r['image']}: frames.csv oss_key {key} != the rule "
                                 f"{oss_key_for_row(d, r)}")
            out.append((f"{PII2}/datasets/{d}/images/{r['image']}", key, int(r["size"]), r["md5"]))
    return out


def model_plan(arms=None):
    out = []
    with open(f"{PII2}/runs/train/MANIFEST.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = oss_key_for_run(r["dst_rel"])
            if int(r["size"]) < 0:
                continue          # symlink (latest.pth): no key, the target is its own row
            if (r.get("oss_key") or None) != k:
                raise SystemExit(f"{r['dst_rel']}: MANIFEST oss_key {r.get('oss_key')!r} != the rule {k!r}")
            if k is None:
                continue
            if arms and k.split("/")[3] not in arms:
                continue
            out.append((f"{PII2}/{r['dst_rel']}", k, int(r["size"]), r["md5"]))
    return out


def faceight_copy_plan(sets):
    """[(src_key_on_annotation_bucket, dst_key, size, md5)] for the faceight images."""
    out = []
    for d in sets:
        for r in load_frames(d):
            src = r.get("src_oss_key") or r["oss_key"]
            out.append((src, DATA_TOP + d + "/" + r["image"], int(r["size"]), r["md5"]))
    return out


# --------------------------------------------------------------------------- client

def load_creds():
    cfg = json.load(open(os.path.expanduser("~/.aliyun/config.json")))
    cur = cfg.get("current", "default")
    for p in cfg["profiles"]:
        if p["name"] == cur:
            if p.get("mode") != "AK":
                raise SystemExit(f"profile {cur} is mode {p.get('mode')}, need AK")
            return p["access_key_id"], p["access_key_secret"]
    raise SystemExit(f"profile {cur} not in ~/.aliyun/config.json")


class OSS:
    """Minimal OSS REST client, V1 signing, one HTTPS connection (one instance per thread)."""

    def __init__(self, ak, sk, bucket=BUCKET, timeout=600):
        self.ak, self.sk, self.bucket, self.timeout = ak, sk, bucket, timeout
        self.host = f"{bucket}.{REGION_HOST}"
        self.conn = None

    def _connect(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = http.client.HTTPSConnection(self.host, timeout=self.timeout,
                                                context=ssl.create_default_context())

    def _sign(self, verb, key, headers, subres=""):
        oss_hdrs = "".join(f"{k.lower()}:{v}\n" for k, v in sorted(headers.items())
                           if k.lower().startswith("x-oss-"))
        s2s = (f"{verb}\n{headers.get('Content-MD5','')}\n{headers.get('Content-Type','')}\n"
               f"{headers['Date']}\n{oss_hdrs}/{self.bucket}/{key}{subres}")
        sig = base64.b64encode(hmac.new(self.sk.encode(), s2s.encode(), hashlib.sha1).digest()).decode()
        return f"OSS {self.ak}:{sig}"

    def request(self, verb, key, body=None, headers=None, query="", retries=6):
        headers = dict(headers or {})
        last = None
        for attempt in range(retries):
            try:
                if self.conn is None:
                    self._connect()
                headers["Date"] = formatdate(usegmt=True)
                headers["Host"] = self.host
                headers["Authorization"] = self._sign(verb, key, headers)
                path = "/" + urllib.parse.quote(key, safe="/") + (("?" + query) if query else "")
                self.conn.request(verb, path, body=body, headers=headers)
                resp = self.conn.getresponse()
                data = resp.read()
                if resp.status >= 500 or resp.status == 429:
                    last = f"HTTP {resp.status}: {data[:200]!r}"
                    self._connect()
                    time.sleep(min(30, 2 ** attempt))
                    continue
                return resp.status, dict(resp.getheaders()), data
            except (http.client.HTTPException, OSError, socket.timeout, ssl.SSLError) as e:
                last = repr(e)
                self._connect()
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"{verb} {key}: gave up after {retries} attempts: {last}")

    def head(self, key):
        st, h, _ = self.request("HEAD", key)
        if st == 404:
            return None
        if st != 200:
            raise RuntimeError(f"HEAD {key}: HTTP {st}")
        return int(h["Content-Length"]), h["ETag"].strip('"').lower(), h

    def put(self, key, data, md5_hex, ctype):
        h = {"Content-MD5": base64.b64encode(bytes.fromhex(md5_hex)).decode(),
             "Content-Type": ctype, "Content-Length": str(len(data))}
        st, rh, body = self.request("PUT", key, body=data, headers=h)
        if st != 200:
            raise RuntimeError(f"PUT {key}: HTTP {st}: {body[:300]!r}")
        etag = rh.get("ETag", "").strip('"').lower()
        if etag != md5_hex:
            raise RuntimeError(f"PUT {key}: ETag {etag} != md5 {md5_hex}")
        return etag

    def copy(self, src_bucket, src_key, dst_key, expect_md5=None, extra=None):
        h = {"x-oss-copy-source": "/" + src_bucket + "/" + urllib.parse.quote(src_key, safe="/"),
             "x-oss-metadata-directive": "COPY"}
        h.update(extra or {})
        st, rh, body = self.request("PUT", dst_key, headers=h)
        if st != 200:
            raise RuntimeError(f"COPY {src_bucket}/{src_key} -> {dst_key}: HTTP {st}: {body[:300]!r}")
        root = ET.fromstring(body)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        et = root.find(f"{ns}ETag")
        etag = (et.text if et is not None and et.text else rh.get("ETag", "")).strip('"').lower()
        if expect_md5 is not None and etag != expect_md5:
            raise RuntimeError(f"COPY {src_key} -> {dst_key}: ETag {etag} != md5 {expect_md5}")
        return etag

    def delete(self, key):
        if not key or key.endswith("/"):
            raise ValueError(f"refusing to delete a prefix-like key {key!r}")
        st, _, body = self.request("DELETE", key)
        if st in (204, 200):
            return "deleted"
        if st == 404:
            return "absent"
        raise RuntimeError(f"DELETE {key}: HTTP {st}: {body[:300]!r}")

    def list(self, prefix, delimiter=""):
        out, marker = {}, ""
        while True:
            q = f"prefix={urllib.parse.quote(prefix, safe='')}&max-keys=1000"
            if marker:
                q += f"&marker={urllib.parse.quote(marker, safe='')}"
            if delimiter:
                q += f"&delimiter={delimiter}"
            st, _, body = self.request("GET", "", query=q)
            if st != 200:
                raise RuntimeError(f"LIST {prefix}: HTTP {st}: {body[:300]!r}")
            root = ET.fromstring(body)
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            for c in root.findall(f"{ns}Contents"):
                out[c.find(f"{ns}Key").text] = (int(c.find(f"{ns}Size").text),
                                                c.find(f"{ns}ETag").text.strip('"').lower())
            for c in root.findall(f"{ns}CommonPrefixes"):
                out[c.find(f"{ns}Prefix").text] = None
            trunc = root.find(f"{ns}IsTruncated")
            if trunc is None or trunc.text != "true":
                return out
            nm = root.find(f"{ns}NextMarker")
            marker = nm.text if nm is not None and nm.text else max(out)


def ctype_of(key):
    return "image/jpeg" if key.endswith(".jpg") else "application/octet-stream"


def listing_for(keys, client):
    """Listings of every prefix (two levels under pii/) that the keys touch."""
    prefixes = sorted({"/".join(k.split("/")[:3]) + "/" for k in keys})
    out = {}
    for p in prefixes:
        out.update(client.list(p))
    return out


# --------------------------------------------------------------------------- upload

def run_parallel(items, work, jobs, label, total_bytes, progress_every=30):
    t0 = time.time()
    done = failed = 0
    nbytes = 0
    errors = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(work, it): it for it in items}
        last = time.time()
        for fut in as_completed(futs):
            try:
                nbytes += fut.result()
                done += 1
            except Exception as e:
                failed += 1
                errors.append((futs[fut], str(e)))
                print(f"[{label}] FAIL {futs[fut][1] if len(futs[fut]) > 1 else futs[fut]}: {e}", flush=True)
            with lock:
                if time.time() - last > progress_every:
                    el = time.time() - t0
                    pct = 100.0 * nbytes / total_bytes if total_bytes else 0
                    print(f"[{label}] {done}/{len(items)} ok, {nbytes/1e9:.2f} GB ({pct:.1f}%), "
                          f"{nbytes/1e6/el:.1f} MB/s, {done/el:.1f} obj/s, failed={failed}", flush=True)
                    last = time.time()
    el = time.time() - t0
    print(f"[{label}] DONE ok={done} failed={failed} bytes={nbytes} wall={el:.0f}s "
          f"{nbytes/1e6/max(el,1e-9):.1f} MB/s {done/max(el,1e-9):.1f} obj/s", flush=True)
    return done, failed, nbytes, el, errors


def cmd_upload(args):
    ak, sk = load_creds()
    plan = []
    if args.group in ("images", "all"):
        plan += image_plan(args.dataset)
    if args.group in ("models", "all"):
        plan += model_plan(args.arm)
    client = OSS(ak, sk)
    remote = listing_for([k for _, k, _, _ in plan], client)
    todo, skipped, conflicts = [], 0, []
    for p, k, size, md5 in plan:
        r = remote.get(k)
        if r is None:
            todo.append((p, k, size, md5))
        elif r == (size, md5):
            skipped += 1
        else:
            conflicts.append((k, r, (size, md5)))
    print(f"plan={len(plan)} already_ok={skipped} to_upload={len(todo)} conflicts={len(conflicts)} "
          f"bytes_to_upload={sum(s for _, _, s, _ in todo)}", flush=True)
    for k, r, want in conflicts[:50]:
        print(f"CONFLICT {k}: remote {r} != local {want} (not overwritten)", flush=True)
    if args.dry_run:
        return 1 if conflicts else 0
    local = threading.local()

    def work(item):
        p, k, size, md5 = item
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk, timeout=args.timeout)
        with open(p, "rb") as f:
            data = f.read()
        if len(data) != size or hashlib.md5(data).hexdigest() != md5:
            raise RuntimeError(f"{p}: local bytes do not match the index (size/md5)")
        local.c.put(k, data, md5, ctype_of(k))
        return len(data)

    done, failed, nbytes, el, _ = run_parallel(todo, work, args.jobs, args.group,
                                               sum(s for _, _, s, _ in todo))
    return 1 if (failed or conflicts) else 0


# --------------------------------------------------------------------------- faceight copy

def cmd_copy_faceight(args):
    ak, sk = load_creds()
    sets = args.set or ["faceight_a", "faceight_b", "faceight_c"]
    plan = faceight_copy_plan(sets)
    client = OSS(ak, sk)
    remote = listing_for([k for _, k, _, _ in plan], client)
    todo, skipped, conflicts = [], 0, []
    for src, dst, size, md5 in plan:
        r = remote.get(dst)
        if r is None:
            todo.append((src, dst, size, md5))
        elif r == (size, md5):
            skipped += 1
        else:
            conflicts.append((dst, r, (size, md5)))
    print(f"plan={len(plan)} already_ok={skipped} to_copy={len(todo)} conflicts={len(conflicts)} "
          f"bytes={sum(s for _, _, s, _ in todo)}", flush=True)
    for k, r, want in conflicts[:50]:
        print(f"CONFLICT {k}: remote {r} != index {want} (not overwritten)", flush=True)
    if args.dry_run:
        return 1 if conflicts else 0
    local = threading.local()

    def work(item):
        src, dst, size, md5 = item
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk)
        local.c.copy(FACEIGHT_BUCKET, src, dst, expect_md5=md5)
        return size

    done, failed, nbytes, el, _ = run_parallel(todo, work, args.jobs, "faceight-copy",
                                               sum(s for _, _, s, _ in todo))
    return 1 if (failed or conflicts) else 0


# --------------------------------------------------------------------------- PII-163 move

def cmd_move_163(args):
    """pii/models/<arm>/<file> -> pii/models/<family>/<arm>/{epochs|onnx}/<file>."""
    ak, sk = load_creds()
    c = OSS(ak, sk)
    plan = {p: (k, s, m) for p, k, s, m in model_plan()}
    by_base = {}
    for p, (k, s, m) in plan.items():
        by_base.setdefault(os.path.basename(k), []).append((k, s, m))
    top = c.list(MODELS_TOP, delimiter="/")
    legacy_prefixes = [k for k, v in top.items() if v is None and k.count("/") == 3
                       and k.split("/")[2] not in ("scrfd", "egoblur")]
    print(f"legacy prefixes under {MODELS_TOP}: {legacy_prefixes}")
    moves = []
    for lp in legacy_prefixes:
        for src, (size, etag) in sorted(c.list(lp).items()):
            base = os.path.basename(src)
            cands = [x for x in by_base.get(base, []) if f"/{lp.split('/')[2]}/" in x[0]]
            if len(cands) != 1:
                raise SystemExit(f"{src}: {len(cands)} destinations in the index ({cands})")
            dst, dsize, dmd5 = cands[0]
            if dsize != size:
                raise SystemExit(f"{src}: remote size {size} != index size {dsize}")
            moves.append((src, etag, dst, dsize, dmd5))
    for src, etag, dst, size, md5 in moves:
        print(f"  {src} (etag {etag}) -> {dst} (md5 {md5}, {size} B)")
    if args.dry_run:
        return 0
    for src, etag, dst, size, md5 in moves:
        r = c.head(dst)
        if r is None:
            new_etag = c.copy(BUCKET, src, dst, expect_md5=None)
            print(f"copied {src} -> {dst}, copy-response ETag {new_etag}")
            r = c.head(dst)
        size_r, etag_r, hdr = r
        # the sources are the multipart objects of PII-163, so their ETag is not the md5;
        # the destination is proved by downloading it and hashing the bytes.
        st, gh, body = c.request("GET", dst)
        if st != 200:
            raise SystemExit(f"GET {dst}: HTTP {st}")
        got = hashlib.md5(body).hexdigest()
        ok = size_r == size and len(body) == size and got == md5
        print(f"verify {dst}: size {size_r} etag {etag_r} downloaded md5 {got} "
              f"(index {md5}) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"{dst}: destination does not match the index, nothing deleted")
    if not args.yes:
        print("destination verified; rerun with --yes to delete the sources")
        return 0
    for src, etag, dst, size, md5 in moves:
        print(f"delete {src}: {c.delete(src)}")
    for lp in sorted({m[0].rsplit('/', 1)[0] + "/" for m in moves}):
        left = c.list(lp)
        print(f"source prefix {lp} now holds {len(left)} objects")
        if left:
            raise SystemExit(f"{lp} is not empty: {sorted(left)[:5]}")
    return 0


# --------------------------------------------------------------------------- verify

def cmd_verify(args):
    ak, sk = load_creds()
    c = OSS(ak, sk)
    want = {}
    for _, k, s, m in image_plan():
        want[k] = (s, m)
    for _, k, s, m in model_plan():
        want[k] = (s, m)
    t0 = time.time()
    have = {}
    for top in (DATA_TOP, MODELS_TOP):
        have.update(c.list(top))
    print(f"listed {len(have)} objects under pii/data/ and pii/models/ in {time.time()-t0:.0f}s")
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    bad = sorted(k for k in set(want) & set(have) if have[k] != want[k])
    per = {}
    for k, (s, _) in have.items():
        p = "/".join(k.split("/")[:4]) if k.startswith(MODELS_TOP) else "/".join(k.split("/")[:3])
        n, b = per.get(p, (0, 0))
        per[p] = (n + 1, b + s)
    for p in sorted(per):
        print(f"{p+'/':44} {per[p][0]:>7} objects {per[p][1]:>14} B")
    print(f"index objects={len(want)} bytes={sum(s for s, _ in want.values())}")
    print(f"remote objects={len(have)} bytes={sum(s for s, _ in have.values())}")
    print(f"missing={len(missing)} extra={len(extra)} size_or_etag_mismatch={len(bad)}")
    for k in missing[:20]:
        print("MISSING", k)
    for k in extra[:20]:
        print("EXTRA", k)
    for k in bad[:20]:
        print("MISMATCH", k, "remote", have[k], "index", want[k])
    ok = not (missing or extra or bad)
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


def cmd_plan(args):
    ak, sk = load_creds()
    c = OSS(ak, sk)
    rows = []
    for d in SETS:
        pl = image_plan([d])
        rows.append((d, len(pl), sum(s for _, _, s, _ in pl)))
    mp = model_plan()
    rows.append(("runs/train (models)", len(mp), sum(s for _, _, s, _ in mp)))
    for name, n, b in rows:
        print(f"{name:24} {n:>7} objects {b:>14} B")
    print(f"{'TOTAL':24} {sum(r[1] for r in rows):>7} objects {sum(r[2] for r in rows):>14} B")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    u = sub.add_parser("upload")
    u.add_argument("--group", choices=["images", "models", "all"], default="all")
    u.add_argument("--dataset", nargs="*")
    u.add_argument("--arm", nargs="*")
    u.add_argument("--jobs", type=int, default=16)
    u.add_argument("--timeout", type=int, default=600,
                   help="per-socket timeout in seconds. A single-part PUT resends the whole "
                        "object on every retry, so this must exceed size/throughput: the 835 MB "
                        "egoblur checkpoints need ~4,200 s at the degraded 200 KB/s per flow "
                        "and all 32 of them timed out at the 600 s default (PII-1413).")
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(fn=cmd_upload)
    f = sub.add_parser("copy-faceight")
    f.add_argument("--set", nargs="*")
    f.add_argument("--jobs", type=int, default=16)
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=cmd_copy_faceight)
    m = sub.add_parser("move-163")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--yes", action="store_true")
    m.set_defaults(fn=cmd_move_163)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()

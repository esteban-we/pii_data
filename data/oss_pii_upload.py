#!/usr/bin/env python3
"""Upload the eight local image sets to oss://algorithm-datasets/pii/data/ (WOR-118, WOR-164).

Layout: one prefix per dataset under pii/data/, JPGs as immediate children, no
sub-levels. Stdlib only (the box has no oss2 SDK). Talks to the OSS REST API
directly with V1 HMAC-SHA1 signing, using the AK profile `default` in
~/.aliyun/config.json.

Subcommands
  upload    idempotent: lists the remote prefix, skips keys that exist with the
            same size, PUTs the rest single-part with Content-MD5, and appends one
            record per uploaded object to <state>/records/<dataset>.jsonl
  count     lists every pii/data/<dataset>/ prefix and compares to the expected table
  assemble  joins the remote listing with the records (computing md5 locally for
            any object without a record) and writes the sorted jsonl
  sample    HEADs N random objects per dataset from a jsonl and checks remote
            size/ETag against the jsonl AND a fresh local size/md5; exits 1 on any
            failure. --key adds explicit keys to the check.

Move subcommands (WOR-164: pii/<dataset>/ -> pii/data/<dataset>/, server-side).
All three read the keys jsonl (dataset, oss_key, size, md5); the destination of a
record is always pii/data/<dataset>/<basename of oss_key>, whether the jsonl still
carries the legacy key or the new one. Nothing local is read.
  copy          server-side CopyObject (x-oss-copy-source) per record, 16 parallel,
                resumable (skips a destination that exists with the same size and
                ETag), asserts the returned ETag == md5, then re-lists the
                destination and asserts count and size/ETag for every record.
                Never deletes.
  verify-moved  lists every destination prefix and asserts: every record present,
                size and ETag == md5, no extra objects; HEADs any missing key to
                print the 404 loudly. --require-source-empty also asserts the
                legacy prefix has 0 objects. Exit 1 on any failure.
  delete-moved  per dataset: re-verifies the destination (same check as
                verify-moved), refuses to proceed on any mismatch, then issues one
                DeleteObject per verified source key (never a prefix delete), then
                lists the source prefix and asserts it is empty. Needs --yes.

Flattening rules (dataset -> oss basename), all proven collision-free on disk:
  flat sets (face10k_v3, face10k_repair, face_mine_v1, faceback_45,
             gt_bench_sparse): basename unchanged
  gt_bench_full: images_full/<session>_<chunk>/f<N>.jpg -> <session>_<chunk>_f<N>.jpg
                 (the naming images_sparse already uses)
  wider_face:    WIDER_train/images/<scene>/<name>.jpg -> <name>.jpg
  pii_frames:    <session>/chunk_<NNN>/<view>/f<N>.jpg -> <session>_c<NNN>_<view>_f<N>.jpg
"""
import argparse
import base64
import hashlib
import hmac
import http.client
import json
import os
import random
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import formatdate

BUCKET = "algorithm-datasets"
HOST = f"{BUCKET}.oss-cn-shanghai.aliyuncs.com"
# PII-1449: the PRE-PII-1315 tree, kept read-only as /data/esteban/pii_backup
# (PII-1448 rename). data/oss_sync.py is the uploader for the new store.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii_root import LEGACY_ROOT  # noqa: E402

ROOT = LEGACY_ROOT + "/datasets"
TOP = "pii/data/"      # images: pii/data/<dataset>/<basename>   (WOR-164)
LEGACY_TOP = "pii/"    # pre-WOR-164 layout: pii/<dataset>/<basename>; only the move commands touch it

# name, local dir (relative to ROOT), expected jpg count, flatten rule
DATASETS = [
    ("face10k_v3", "face10k_v3/images", 8339, "flat"),
    ("face10k_repair", "face10k_v3/repair_v1/images", 3168, "flat"),
    ("face_mine_v1", "face_mine_v1/images", 62587, "flat"),
    ("faceback_45", "faceback_45/images", 84954, "flat"),
    ("gt_bench_sparse", "gt_bench_v1/images_sparse", 889, "flat"),
    ("gt_bench_full", "gt_bench_v1/images_full", 9636, "dir_prefix"),
    ("wider_face", "wider_face/WIDER_train", 12879, "basename"),
    ("pii_frames", "pii_frames", 10249, "pii_frames"),
]
DS = {d[0]: d for d in DATASETS}


def flatten(rule, rel):
    """rel: path of the jpg relative to the dataset dir. Returns the oss basename."""
    parts = rel.split("/")
    if rule == "flat":
        if len(parts) != 1:
            raise ValueError(f"expected flat layout, got {rel}")
        return parts[0]
    if rule == "basename":
        return parts[-1]
    if rule == "dir_prefix":
        if len(parts) != 2:
            raise ValueError(f"expected <dir>/<file>, got {rel}")
        return f"{parts[0]}_{parts[1]}"
    if rule == "pii_frames":
        if len(parts) != 4 or not parts[1].startswith("chunk_"):
            raise ValueError(f"expected <session>/chunk_<NNN>/<view>/<file>, got {rel}")
        session, chunk, view, fname = parts
        return f"{session}_c{chunk[len('chunk_'):]}_{view}_{fname}"
    raise ValueError(rule)


def local_files(name):
    """Yield (local_path, oss_key) for every *.jpg under the dataset; return non-jpg list."""
    _, rel_dir, _, rule = DS[name]
    base = os.path.join(ROOT, rel_dir)
    if not os.path.isdir(base):
        raise SystemExit(f"missing local dir {base}")
    jpgs, others = [], []
    for dp, dns, fns in os.walk(base):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if fn.endswith(".jpg"):
                rel = os.path.relpath(p, base)
                jpgs.append((p, f"{TOP}{name}/{flatten(rule, rel)}"))
            else:
                others.append(p)
    keys = [k for _, k in jpgs]
    if len(set(keys)) != len(keys):
        seen, dups = set(), set()
        for k in keys:
            if k in seen:
                dups.add(k)
            seen.add(k)
        raise SystemExit(f"{name}: {len(dups)} duplicate oss keys after flattening, e.g. {sorted(dups)[:5]}")
    return jpgs, others


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
    """Minimal OSS client, one HTTPS connection per instance (use one per thread)."""

    def __init__(self, ak, sk, timeout=60):
        self.ak, self.sk, self.timeout = ak, sk, timeout
        self.conn = None

    def _connect(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = http.client.HTTPSConnection(HOST, timeout=self.timeout, context=ssl.create_default_context())

    def _sign(self, verb, key, headers, subres=""):
        cmd5 = headers.get("Content-MD5", "")
        ctype = headers.get("Content-Type", "")
        date = headers["Date"]
        oss_hdrs = "".join(f"{k.lower()}:{v}\n" for k, v in sorted(headers.items()) if k.lower().startswith("x-oss-"))
        resource = f"/{BUCKET}/{key}{subres}"
        s2s = f"{verb}\n{cmd5}\n{ctype}\n{date}\n{oss_hdrs}{resource}"
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
                headers["Host"] = HOST
                headers["Authorization"] = self._sign(verb, key, headers)
                path = "/" + key + (("?" + query) if query else "")
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
        """Return (size, etag_lower) or None if 404."""
        st, h, _ = self.request("HEAD", key)
        if st == 404:
            return None
        if st != 200:
            raise RuntimeError(f"HEAD {key}: HTTP {st}")
        return int(h["Content-Length"]), h["ETag"].strip('"').lower()

    def put(self, key, data, md5_hex):
        h = {
            "Content-MD5": base64.b64encode(bytes.fromhex(md5_hex)).decode(),
            "Content-Type": "image/jpeg",
            "Content-Length": str(len(data)),
        }
        st, rh, body = self.request("PUT", key, body=data, headers=h)
        if st != 200:
            raise RuntimeError(f"PUT {key}: HTTP {st}: {body[:300]!r}")
        etag = rh.get("ETag", "").strip('"').lower()
        if etag != md5_hex:
            raise RuntimeError(f"PUT {key}: ETag {etag} != md5 {md5_hex}")
        return etag

    def copy(self, src, dst, md5_hex):
        """Server-side CopyObject src -> dst (same bucket). Returns the new ETag, asserted == md5_hex."""
        h = {
            "x-oss-copy-source": "/" + BUCKET + "/" + urllib.parse.quote(src, safe="/"),
            "x-oss-metadata-directive": "COPY",
        }
        st, rh, body = self.request("PUT", dst, headers=h)
        if st != 200:
            raise RuntimeError(f"COPY {src} -> {dst}: HTTP {st}: {body[:300]!r}")
        root = ET.fromstring(body)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        et = root.find(f"{ns}ETag")
        etag = (et.text if et is not None and et.text else rh.get("ETag", "")).strip('"').lower()
        if etag != md5_hex:
            raise RuntimeError(f"COPY {src} -> {dst}: ETag {etag} != md5 {md5_hex}")
        return etag

    def delete(self, key):
        """DeleteObject for ONE key. Returns 'deleted' (204) or 'absent' (404, e.g. a retried request)."""
        if not key or key.endswith("/"):
            raise ValueError(f"refusing to delete a prefix-like key {key!r}")
        st, _, body = self.request("DELETE", key)
        if st == 204:
            return "deleted"
        if st == 404:
            return "absent"
        raise RuntimeError(f"DELETE {key}: HTTP {st}: {body[:300]!r}")

    def list(self, prefix, delimiter=""):
        """Return {key: (size, etag_lower)} for every object under prefix (GET Bucket v1, paginated).
        With a delimiter, common prefixes are returned as keys mapped to None."""
        out, marker = {}, ""
        while True:
            q = f"prefix={prefix}&max-keys=1000"
            if marker:
                q += f"&marker={marker}"
            if delimiter:
                q += f"&delimiter={delimiter}"
            st, _, body = self.request("GET", "", query=q)
            if st != 200:
                raise RuntimeError(f"LIST {prefix}: HTTP {st}: {body[:300]!r}")
            root = ET.fromstring(body)
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            for c in root.findall(f"{ns}Contents"):
                k = c.find(f"{ns}Key").text
                out[k] = (int(c.find(f"{ns}Size").text), c.find(f"{ns}ETag").text.strip('"').lower())
            for c in root.findall(f"{ns}CommonPrefixes"):
                out[c.find(f"{ns}Prefix").text] = None
            trunc = root.find(f"{ns}IsTruncated")
            if trunc is None or trunc.text != "true":
                return out
            nm = root.find(f"{ns}NextMarker")
            marker = nm.text if nm is not None and nm.text else max(out)


def md5_file(path):
    with open(path, "rb") as f:
        data = f.read()
    return data, hashlib.md5(data).hexdigest()


# ----------------------------------------------------------------------------- upload

def cmd_upload(args):
    ak, sk = load_creds()
    os.makedirs(os.path.join(args.state, "records"), exist_ok=True)
    names = args.dataset or [d[0] for d in DATASETS]
    grand = {"uploaded": 0, "skipped": 0, "bytes": 0, "failed": 0, "nonjpg": 0}
    t_all = time.time()
    for name in names:
        t0 = time.time()
        jpgs, others = local_files(name)
        prefix = f"{TOP}{name}/"
        remote = OSS(ak, sk).list(prefix)
        todo, skipped, resized = [], 0, 0
        for p, k in jpgs:
            r = remote.get(k)
            if r is not None and r[0] == os.path.getsize(p):
                skipped += 1
            else:
                if r is not None:
                    resized += 1
                todo.append((p, k))
        if args.limit:
            todo = todo[: args.limit]
        print(f"[{name}] local jpg={len(jpgs)} nonjpg_skipped={len(others)} remote={len(remote)} "
              f"skip(size match)={skipped} to_upload={len(todo)} (size-mismatch re-uploads={resized})", flush=True)
        if others:
            print(f"[{name}] non-jpg files skipped: {others[:5]}{' ...' if len(others) > 5 else ''}")
        grand["skipped"] += skipped
        grand["nonjpg"] += len(others)
        if args.dry_run or not todo:
            continue
        rec_path = os.path.join(args.state, "records", f"{name}.jsonl")
        rec_lock = threading.Lock()
        rec_f = open(rec_path, "a")
        local = threading.local()

        def work(item):
            p, k = item
            if not hasattr(local, "c"):
                local.c = OSS(ak, sk)
            data, md5 = md5_file(p)
            local.c.put(k, data, md5)
            rec = {"dataset": name, "oss_key": k, "local_path": p, "size": len(data), "md5": md5}
            with rec_lock:
                rec_f.write(json.dumps(rec, sort_keys=True) + "\n")
                rec_f.flush()
            return len(data)

        done, nbytes, failed = 0, 0, []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            last = time.time()
            for fut in as_completed(futs):
                try:
                    nbytes += fut.result()
                    done += 1
                except Exception as e:
                    failed.append((futs[fut][1], str(e)))
                    print(f"[{name}] FAIL {futs[fut][1]}: {e}", flush=True)
                if time.time() - last > args.progress_every:
                    el = time.time() - t0
                    print(f"[{name}] {done}/{len(todo)} uploaded, {nbytes/1e6:.0f} MB, "
                          f"{nbytes/1e6/el:.1f} MB/s, {done/el:.1f} obj/s, failed={len(failed)}", flush=True)
                    last = time.time()
        rec_f.close()
        el = time.time() - t0
        print(f"[{name}] DONE uploaded={done} failed={len(failed)} bytes={nbytes} "
              f"{el:.0f}s {nbytes/1e6/max(el,1e-9):.1f} MB/s", flush=True)
        grand["uploaded"] += done
        grand["bytes"] += nbytes
        grand["failed"] += len(failed)
    el = time.time() - t_all
    print(f"TOTAL uploaded={grand['uploaded']} skipped={grand['skipped']} failed={grand['failed']} "
          f"nonjpg_skipped={grand['nonjpg']} bytes={grand['bytes']} wall={el:.0f}s")
    return 1 if grand["failed"] else 0


# ----------------------------------------------------------------------------- count

def cmd_count(args):
    ak, sk = load_creds()
    c = OSS(ak, sk)
    ok = True
    print(f"{'dataset':16} {'expected':>8} {'remote':>8} {'non_child':>9} {'non_jpg':>7} {'bytes':>14} status")
    for name, _, expected, _ in DATASETS:
        prefix = f"{TOP}{name}/"
        remote = c.list(prefix)
        non_child = sum(1 for k in remote if "/" in k[len(prefix):])
        non_jpg = sum(1 for k in remote if not k.endswith(".jpg"))
        nbytes = sum(s for s, _ in remote.values())
        good = len(remote) == expected and non_child == 0 and non_jpg == 0
        ok &= good
        print(f"{name:16} {expected:>8} {len(remote):>8} {non_child:>9} {non_jpg:>7} {nbytes:>14} {'OK' if good else 'MISMATCH'}")
    top = c.list(TOP, delimiter="/")
    stray = [k for k, v in top.items() if v is not None]  # objects directly under pii/data/
    if stray:
        ok = False
        print(f"stray objects directly under {TOP}: {stray}")
    extra_prefixes = sorted(k for k, v in top.items() if v is None and k[len(TOP):-1] not in DS)
    if extra_prefixes:
        ok = False
        print(f"unexpected prefixes under {TOP}: {extra_prefixes}")
    legacy = c.list(LEGACY_TOP, delimiter="/")
    print(f"top level under {LEGACY_TOP}: {sorted(legacy)}")
    print("ALL OK" if ok else "COUNT CHECK FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- assemble

def cmd_assemble(args):
    ak, sk = load_creds()
    c = OSS(ak, sk)
    records = {}
    rec_dir = os.path.join(args.state, "records")
    for fn in sorted(os.listdir(rec_dir)) if os.path.isdir(rec_dir) else []:
        for line in open(os.path.join(rec_dir, fn)):
            r = json.loads(line)
            records[r["oss_key"]] = r  # last write wins
    out, problems, computed = [], [], 0
    for name, _, expected, _ in DATASETS:
        jpgs, _ = local_files(name)
        local_by_key = dict((k, p) for p, k in jpgs)
        remote = c.list(f"{TOP}{name}/")
        extra = sorted(set(remote) - set(local_by_key))
        missing = sorted(set(local_by_key) - set(remote))
        if extra:
            problems.append(f"{name}: {len(extra)} remote keys with no local file, e.g. {extra[:3]}")
        if missing:
            problems.append(f"{name}: {len(missing)} local files not on remote, e.g. {missing[:3]}")
        for k in sorted(local_by_key):
            if k not in remote:
                continue
            p = local_by_key[k]
            size, etag = remote[k]
            r = records.get(k)
            if r is None or r["local_path"] != p or r["size"] != size:
                _, md5 = md5_file(p)
                r = {"dataset": name, "oss_key": k, "local_path": p, "size": os.path.getsize(p), "md5": md5}
                computed += 1
            if r["size"] != size or r["md5"] != etag:
                problems.append(f"{k}: record size/md5 {r['size']}/{r['md5']} vs remote {size}/{etag}")
            out.append(r)
        print(f"[{name}] remote={len(remote)} expected={expected} written={sum(1 for r in out if r['dataset']==name)}", flush=True)
    out.sort(key=lambda r: (r["dataset"], r["oss_key"]))
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps({k: r[k] for k in ("dataset", "oss_key", "local_path", "size", "md5")}) + "\n")
    print(f"wrote {len(out)} records to {args.out}; md5 computed at assemble time for {computed} objects")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


# ----------------------------------------------------------------------------- sample

def cmd_sample(args):
    ak, sk = load_creds()
    rows = [json.loads(l) for l in open(args.jsonl)]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    rng = random.Random(args.seed)
    picks = []
    for name in sorted(by_ds):
        rs = by_ds[name]
        picks += rng.sample(rs, min(args.n, len(rs)))
    by_key = {r["oss_key"]: r for r in rows}
    for k in args.key or []:
        picks.append(by_key.get(k, {"dataset": "?", "oss_key": k, "local_path": None, "size": None, "md5": None}))
    local = threading.local()

    def check(r):
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk)
        k = r["oss_key"]
        rem = local.c.head(k)
        if rem is None:
            return k, f"FAIL remote 404 NoSuchKey"
        rsize, retag = rem
        if r["local_path"] is None or not os.path.exists(r["local_path"] or ""):
            return k, f"FAIL no local_path in jsonl / local file missing ({r['local_path']})"
        lsize = os.path.getsize(r["local_path"])
        _, lmd5 = md5_file(r["local_path"])
        errs = []
        if rsize != lsize:
            errs.append(f"remote size {rsize} != local {lsize}")
        if rsize != r["size"]:
            errs.append(f"remote size {rsize} != jsonl {r['size']}")
        if retag != lmd5:
            errs.append(f"remote etag {retag} != local md5 {lmd5}")
        if retag != r["md5"]:
            errs.append(f"remote etag {retag} != jsonl md5 {r['md5']}")
        return k, ("FAIL " + "; ".join(errs)) if errs else "ok"

    fails, per_ds = [], {}
    ds_of = {r["oss_key"]: r["dataset"] for r in picks}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for k, res in ex.map(check, picks):
            ds = ds_of.get(k, "?")
            per_ds.setdefault(ds, [0, 0])
            if res == "ok":
                per_ds[ds][0] += 1
            else:
                per_ds[ds][1] += 1
                fails.append((k, res))
    for ds in sorted(per_ds):
        ok, bad = per_ds[ds]
        print(f"{ds:16} checked={ok+bad:>4} ok={ok:>4} failed={bad}")
    for k, res in fails:
        print(f"  {k}: {res}")
    print(f"SAMPLE VERIFY {'PASSED' if not fails else 'FAILED'}: {len(picks)} objects, {len(fails)} failures")
    return 1 if fails else 0


# ----------------------------------------------------------------------------- move (WOR-164)

def dest_key(r):
    """pii/data/<dataset>/<basename>, from a jsonl record carrying either the legacy or the new key."""
    return f"{TOP}{r['dataset']}/{r['oss_key'].rsplit('/', 1)[1]}"


def legacy_key(r):
    return f"{LEGACY_TOP}{r['dataset']}/{r['oss_key'].rsplit('/', 1)[1]}"


def load_rows(path, datasets=None):
    """Rows of the keys jsonl grouped by dataset (insertion order of DATASETS, then unknown names)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        for k in ("dataset", "oss_key", "size", "md5"):
            if k not in r:
                raise SystemExit(f"{path}: record without {k}: {r}")
        if r["oss_key"].count("/") not in (2, 3) or not r["oss_key"].startswith(LEGACY_TOP):
            raise SystemExit(f"{path}: unexpected key shape {r['oss_key']}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    order = [d[0] for d in DATASETS] + sorted(set(by_ds) - set(DS))
    names = [n for n in (datasets or order) if n in by_ds]
    return {n: by_ds[n] for n in names}


def check_destination(c, name, rows):
    """List pii/data/<name>/ and compare to rows. Returns (listing, failures:list[str])."""
    prefix = f"{TOP}{name}/"
    remote = c.list(prefix)
    fails = []
    want = {}
    for r in rows:
        d = dest_key(r)
        if d in want:
            fails.append(f"duplicate destination in keys file: {d}")
        want[d] = r
    for d, r in want.items():
        got = remote.get(d)
        if got is None:
            hd = c.head(d)
            fails.append(f"MISSING {d} (HEAD -> {'404 NoSuchKey' if hd is None else hd})")
            continue
        size, etag = got
        if size != r["size"]:
            fails.append(f"SIZE {d}: remote {size} != keys file {r['size']}")
        if etag != r["md5"]:
            fails.append(f"ETAG {d}: remote {etag} != keys file md5 {r['md5']}")
    extra = sorted(set(remote) - set(want))
    if extra:
        fails.append(f"EXTRA {len(extra)} objects under {prefix} not in keys file, e.g. {extra[:5]}")
    if len(remote) != len(want):
        fails.append(f"COUNT {prefix}: remote {len(remote)} != keys file {len(want)}")
    return remote, fails


def _pool(items, fn, jobs, label, progress_every):
    """Run fn over items with bounded threads; return (n_ok, results, failures). fn gets a thread-local OSS."""
    ak, sk = load_creds()
    local = threading.local()

    def wrapped(it):
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk)
        return fn(local.c, it)

    t0, last, done, results, failures = time.time(), time.time(), 0, [], []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(wrapped, it): it for it in items}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
                done += 1
            except Exception as e:
                failures.append((futs[fut], str(e)))
                print(f"{label} FAIL {futs[fut]}: {e}", flush=True)
            if time.time() - last > progress_every:
                el = time.time() - t0
                print(f"{label} {done}/{len(items)} done, {done/el:.1f} obj/s, failed={len(failures)}", flush=True)
                last = time.time()
    return done, results, failures


def cmd_copy(args):
    ak, sk = load_creds()
    groups = load_rows(args.jsonl, args.dataset)
    grand_fail = 0
    for name, rows in groups.items():
        t0 = time.time()
        c = OSS(ak, sk)
        dst_prefix = f"{TOP}{name}/"
        existing = c.list(dst_prefix)
        todo, skipped = [], 0
        for r in rows:
            d = dest_key(r)
            got = existing.get(d)
            if got is not None and got == (r["size"], r["md5"]):
                skipped += 1
            else:
                todo.append(r)
        if args.limit:
            todo = todo[: args.limit]
        print(f"[{name}] records={len(rows)} dest_existing={len(existing)} skip(size+etag match)={skipped} "
              f"to_copy={len(todo)}", flush=True)
        if args.dry_run or not todo:
            continue

        def do_copy(client, r):
            return client.copy(legacy_key(r), dest_key(r), r["md5"])

        done, _, failures = _pool(todo, do_copy, args.jobs, f"[{name}]", args.progress_every)
        el_copy = time.time() - t0
        print(f"[{name}] COPIED {done}/{len(todo)} failed={len(failures)} {el_copy:.0f}s "
              f"{done/max(el_copy,1e-9):.1f} obj/s", flush=True)
        if args.limit:
            print(f"[{name}] --limit given, skipping the full-destination assertion")
            grand_fail += len(failures)
            continue
        t1 = time.time()
        _, fails = check_destination(OSS(ak, sk), name, rows)
        for f in fails[:50]:
            print(f"[{name}] PROBLEM {f}")
        print(f"[{name}] DEST CHECK {'OK' if not fails else 'FAILED'}: {len(rows)} records, "
              f"{len(fails)} problems, {time.time()-t1:.0f}s; dataset wall {time.time()-t0:.0f}s", flush=True)
        grand_fail += len(failures) + len(fails)
        if fails or failures:
            print(f"[{name}] stopping at first dataset with a mismatch")
            break
    print(f"COPY {'OK' if not grand_fail else 'FAILED'} ({grand_fail} problems)")
    return 1 if grand_fail else 0


def cmd_verify_moved(args):
    ak, sk = load_creds()
    c = OSS(ak, sk)
    groups = load_rows(args.jsonl, args.dataset)
    total_fail = 0
    print(f"{'dataset':16} {'records':>8} {'dest':>8} {'source':>8} status")
    for name, rows in groups.items():
        remote, fails = check_destination(c, name, rows)
        src = c.list(f"{LEGACY_TOP}{name}/")
        if args.require_source_empty and src:
            fails.append(f"SOURCE {LEGACY_TOP}{name}/ still holds {len(src)} objects, e.g. {sorted(src)[:3]}")
        print(f"{name:16} {len(rows):>8} {len(remote):>8} {len(src):>8} {'OK' if not fails else 'FAILED'}", flush=True)
        for f in fails[:50]:
            print(f"  {f}")
        total_fail += len(fails)
    print(f"VERIFY-MOVED {'PASSED' if not total_fail else 'FAILED'}: {total_fail} problems")
    return 1 if total_fail else 0


def cmd_delete_moved(args):
    if not args.yes:
        raise SystemExit("delete-moved deletes source objects; pass --yes")
    ak, sk = load_creds()
    groups = load_rows(args.jsonl, args.dataset)
    for name, rows in groups.items():
        t0 = time.time()
        c = OSS(ak, sk)
        src_prefix = f"{LEGACY_TOP}{name}/"
        _, fails = check_destination(c, name, rows)
        if fails:
            for f in fails[:50]:
                print(f"[{name}] PROBLEM {f}")
            print(f"[{name}] destination check FAILED ({len(fails)} problems); NOTHING deleted, stopping")
            return 1
        src = c.list(src_prefix)
        verified_src = {legacy_key(r) for r in rows}
        unknown = sorted(set(src) - verified_src)
        if unknown:
            print(f"[{name}] {len(unknown)} objects under {src_prefix} are not in the keys file, e.g. {unknown[:5]}; "
                  f"NOTHING deleted, stopping")
            return 1
        todo = sorted(set(src) & verified_src)
        print(f"[{name}] destination verified for {len(rows)} records; source holds {len(src)}; deleting {len(todo)} "
              f"one key at a time", flush=True)
        if not todo:
            continue

        def do_delete(client, key):
            return client.delete(key)

        done, results, failures = _pool(todo, do_delete, args.jobs, f"[{name}]", args.progress_every)
        left = c.list(src_prefix)
        el = time.time() - t0
        print(f"[{name}] DELETED {results.count('deleted')} absent-on-retry={results.count('absent')} "
              f"failed={len(failures)}; source listing after: {len(left)} objects; {el:.0f}s", flush=True)
        if failures or left:
            print(f"[{name}] SOURCE NOT EMPTY or delete failures; stopping")
            return 1
    print("DELETE-MOVED OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="/tmp/wor118", help="dir for per-dataset upload records")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--dataset", action="append", choices=[d[0] for d in DATASETS])
    u.add_argument("--jobs", type=int, default=12)
    u.add_argument("--limit", type=int, default=0)
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--progress-every", type=float, default=60)
    sub.add_parser("count")
    a = sub.add_parser("assemble")
    a.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "oss_pii_keys.jsonl"))
    s = sub.add_parser("sample")
    s.add_argument("--jsonl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "oss_pii_keys.jsonl"))
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=118)
    s.add_argument("--jobs", type=int, default=8)
    s.add_argument("--key", action="append", help="extra oss key(s) to check explicitly")
    default_jsonl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oss_pii_keys.jsonl")
    for cmd in ("copy", "verify-moved", "delete-moved"):
        m = sub.add_parser(cmd)
        m.add_argument("--jsonl", default=default_jsonl)
        m.add_argument("--dataset", action="append")
        m.add_argument("--jobs", type=int, default=16)
        m.add_argument("--progress-every", type=float, default=30)
        if cmd == "copy":
            m.add_argument("--limit", type=int, default=0)
            m.add_argument("--dry-run", action="store_true")
        if cmd == "verify-moved":
            m.add_argument("--require-source-empty", action="store_true")
        if cmd == "delete-moved":
            m.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    return {"upload": cmd_upload, "count": cmd_count, "assemble": cmd_assemble, "sample": cmd_sample,
            "copy": cmd_copy, "verify-moved": cmd_verify_moved, "delete-moved": cmd_delete_moved}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

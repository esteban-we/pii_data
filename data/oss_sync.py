#!/usr/bin/env python3
"""pii_data <-> oss://algorithm-datasets/pii/: git holds the ledger, OSS holds the bytes.

Git tracks every index in this repo. The bytes it does not track (dataset images,
checkpoints, ONNX exports, training logs) live on OSS under the key each index row
carries in its `oss_key` column, and this script is the only thing that moves them.

    datasets/<name>/frames.csv    image, size, md5, oss_key -> datasets/<name>/images/<image>
    runs/train/MANIFEST.tsv       dst_rel, size, md5, oss_key -> <dst_rel> (oss_key empty
                                  when the file is git content)

Subcommands
    pull [--dataset D ...] [--view V ...] [--arm A ...]
        Download every indexed object that is missing locally or whose bytes differ from
        the index. A local file that already matches is never rewritten, so a rerun
        resumes. Every download is verified against the index md5 before it is renamed
        into place. Prints one line per prefix, and exactly `up to date` when nothing
        was missing.
    push [--dataset D ...] [--arm A ...]
        Upload local files that are in the index but not on OSS, and files under
        runs/train/ that are on disk in a known layout with no index row yet (the row is
        appended to MANIFEST.tsv; commit it afterwards). ETag is asserted equal to the
        md5. An object already on OSS whose ETag differs is an error, never an
        overwrite. Nothing is ever deleted.
    status [--remote]
        Three sets per prefix: missing locally, local but not pushed, and (with
        --remote, which lists OSS) objects under pii/ that no index row claims.

Hook: .githooks/post-merge runs `pull` after every `git pull` once
`git config core.hooksPath .githooks` is set (data/setup.sh does that). Opt out on a
box with `git config pii.autopull false`. Git does not run hooks on clone, so the
first pull after a clone has to be explicit: `bash data/setup.sh && python3 data/oss_sync.py pull`.

Stdlib only. Credential: the `default` AK profile of ~/.aliyun/config.json.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import http.client
import json
import os
import re
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import formatdate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "algorithm-datasets"
HOST = f"{BUCKET}.oss-cn-shanghai.aliyuncs.com"
TOP = "pii/"
DATA_TOP = "pii/data/"
MODELS_TOP = "pii/models/"
JOBS = 16


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- index

class Item:
    __slots__ = ("path", "key", "size", "md5", "group")

    def __init__(self, path, key, size, md5, group):
        self.path, self.key, self.size, self.md5, self.group = path, key, size, md5, group

    @property
    def prefix(self):
        return self.key.rsplit("/", 1)[0] + "/"


def datasets_on_disk():
    d = os.path.join(ROOT, "datasets")
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "frames.csv")))


def read_frames(dataset):
    with open(os.path.join(ROOT, "datasets", dataset, "frames.csv"), newline="") as f:
        return list(csv.DictReader(f))


def dataset_items(datasets=None, want_images=None):
    """want_images: {dataset: set(image)} to restrict a view pull."""
    out = []
    for d in datasets or datasets_on_disk():
        rows = read_frames(d)
        if not rows or "oss_key" not in rows[0]:
            die(f"datasets/{d}/frames.csv has no oss_key column; rebuild it with the pii repo build script")
        keep = want_images.get(d) if want_images else None
        for r in rows:
            if keep is not None and r["image"] not in keep:
                continue
            out.append(Item(os.path.join(ROOT, "datasets", d, "images", r["image"]),
                            r["oss_key"], int(r["size"]), r["md5"], d))
    return out


def manifest_rows():
    p = os.path.join(ROOT, "runs/train/MANIFEST.tsv")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and "oss_key" not in rows[0]:
        die("runs/train/MANIFEST.tsv has no oss_key column; rebuild it with the pii repo build script")
    return rows


def run_items(arms=None):
    out = []
    for r in manifest_rows():
        if not r["oss_key"]:
            continue
        arm = r["oss_key"].split("/")[3]
        if arms and arm not in arms:
            continue
        out.append(Item(os.path.join(ROOT, r["dst_rel"]), r["oss_key"], int(r["size"]), r["md5"], arm))
    return out


# --------------------------------------------------------------------------- view recipes

def view_images(names):
    """{dataset: set(image)} for every image the named views need."""
    want = {}
    for v in names:
        path = os.path.join(ROOT, "views", v, "recipe.yaml")
        if not os.path.isfile(path):
            die(f"no such view: {path}")
        for src in parse_sources(path):
            d = src["dataset"]
            rows = read_frames(d)
            split = src.get("split")
            if split:
                role = {}
                with open(os.path.join(ROOT, "datasets", d, "split.csv"), newline="") as f:
                    for r in csv.DictReader(f):
                        role[r["session"]] = r["role"]
                rows = [r for r in rows if role.get(r["session"]) == split]
            for col, val in (src.get("where") or {}).items():
                rows = [r for r in rows if r.get(col) == val]
            want.setdefault(d, set()).update(r["image"] for r in rows)
    return want


def parse_sources(path):
    """The `sources:` list of a recipe: dataset, split and the flow-mapping `where`."""
    out, in_sources = [], False
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_sources = line.strip() == "sources:"
            continue
        if not in_sources:
            continue
        body = line.strip()
        if body.startswith("- "):
            out.append({})
            body = body[2:]
        if not out:
            continue
        key, _, rest = body.partition(":")
        key, rest = key.strip(), rest.strip()
        if key == "where" and rest.startswith("{"):
            out[-1]["where"] = {k.strip(): v.strip().strip("'\"")
                                for k, _, v in (p.partition(":") for p in rest[1:-1].split(",")) if k.strip()}
        elif rest:
            out[-1][key] = rest.strip("'\"")
    return [s for s in out if s.get("dataset")]


# --------------------------------------------------------------------------- OSS client

def load_creds():
    """OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET when set, else the aliyun CLI profile.

    The environment wins so that a box which has no ~/.aliyun/config.json (shang,
    fluence) can pull without a secret being written to its disk.
    """
    ak, sk = os.environ.get("OSS_ACCESS_KEY_ID"), os.environ.get("OSS_ACCESS_KEY_SECRET")
    if ak and sk:
        return ak, sk
    p = os.path.expanduser("~/.aliyun/config.json")
    if not os.path.isfile(p):
        die(f"no OSS credential: set OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET, or put the "
            f"aliyun CLI profile `default` (RAM user esteban) at {p}")
    cfg = json.load(open(p))
    cur = cfg.get("current", "default")
    for prof in cfg["profiles"]:
        if prof["name"] == cur:
            if prof.get("mode") != "AK":
                die(f"profile {cur} is mode {prof.get('mode')}, need AK")
            return prof["access_key_id"], prof["access_key_secret"]
    die(f"profile {cur} not in {p}")


class OSS:
    """Minimal OSS REST client (V1 signing), one HTTPS connection per instance."""

    def __init__(self, ak, sk, timeout=600):
        self.ak, self.sk, self.timeout = ak, sk, timeout
        self.conn = None

    def _connect(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = http.client.HTTPSConnection(HOST, timeout=self.timeout,
                                                context=ssl.create_default_context())

    def _sign(self, verb, key, headers):
        oss_hdrs = "".join(f"{k.lower()}:{v}\n" for k, v in sorted(headers.items())
                           if k.lower().startswith("x-oss-"))
        s2s = (f"{verb}\n{headers.get('Content-MD5','')}\n{headers.get('Content-Type','')}\n"
               f"{headers['Date']}\n{oss_hdrs}/{BUCKET}/{key}")
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

    def get(self, key):
        st, h, data = self.request("GET", key)
        if st != 200:
            raise RuntimeError(f"GET {key}: HTTP {st}: {data[:200]!r}")
        if int(h["Content-Length"]) != len(data):
            raise RuntimeError(f"GET {key}: {len(data)} B != Content-Length {h['Content-Length']}")
        return data, h["ETag"].strip('"').lower()

    def head(self, key):
        st, h, _ = self.request("HEAD", key)
        if st == 404:
            return None
        if st != 200:
            raise RuntimeError(f"HEAD {key}: HTTP {st}")
        return int(h["Content-Length"]), h["ETag"].strip('"').lower()

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


# --------------------------------------------------------------------------- local state

# Hashing every image on every `git pull` would read 236 GB, so the md5 of a file is
# cached under .git/ against its (size, mtime_ns). The cache is never trusted for a file
# whose size or mtime changed, and it is not tracked by git.
CACHE_PATH = os.path.join(ROOT, ".git", "oss_sync_md5.json")
_cache = None
_cache_lock = threading.Lock()
_cache_dirty = False


def cache_load():
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(CACHE_PATH))
        except Exception:
            _cache = {}
    return _cache


def cache_save():
    if _cache_dirty and os.path.isdir(os.path.dirname(CACHE_PATH)):
        tmp = CACHE_PATH + ".part"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, CACHE_PATH)


def md5_cached(path, size):
    global _cache_dirty
    c = cache_load()
    st = os.stat(path)
    hit = c.get(path)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    digest = md5_file(path)
    with _cache_lock:
        c[path] = [st.st_size, st.st_mtime_ns, digest]
        _cache_dirty = True
    return digest


def local_ok(it):
    """True when the file on disk already is the indexed object."""
    try:
        if os.path.getsize(it.path) != it.size:
            return False
    except OSError:
        return False
    return md5_cached(it.path, it.size) == it.md5


def select(args):
    items = []
    if getattr(args, "view", None):
        items += dataset_items(None, view_images(args.view))
    elif getattr(args, "dataset", None):
        items += dataset_items(args.dataset)
    elif not getattr(args, "arm", None):
        items += dataset_items()
    if getattr(args, "arm", None) or not (getattr(args, "dataset", None) or getattr(args, "view", None)):
        items += run_items(getattr(args, "arm", None))
    return items


def by_prefix(items):
    out = {}
    for it in items:
        out.setdefault(it.prefix, []).append(it)
    return out


def parallel(items, work, jobs, total_bytes, label):
    t0 = time.time()
    done = failed = 0
    nbytes = 0
    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(work, it): it for it in items}
        last = time.time()
        for fut in as_completed(futs):
            try:
                nbytes += fut.result()
                done += 1
            except Exception as e:
                failed += 1
                errors.append((futs[fut].key, str(e)))
                print(f"FAIL {futs[fut].key}: {e}", flush=True)
            if time.time() - last > 30:
                el = time.time() - t0
                print(f"{label} {done}/{len(items)}, {nbytes/1e9:.2f} of {total_bytes/1e9:.2f} GB, "
                      f"{nbytes/1e6/el:.1f} MB/s, {done/el:.1f} obj/s, failed={failed}", flush=True)
                last = time.time()
    el = max(time.time() - t0, 1e-9)
    print(f"{label} {done} objects, {nbytes} B, {el:.0f}s, {nbytes/1e6/el:.1f} MB/s, failed={failed}")
    return done, failed, errors


# --------------------------------------------------------------------------- pull

def restore_symlinks(arms=None):
    """MANIFEST rows with size -1 are symlinks (epochs/latest.pth); the target is in md5."""
    made = []
    for r in manifest_rows():
        if int(r["size"]) >= 0 or not r["md5"].startswith("symlink:"):
            continue
        target = r["md5"][len("symlink:"):]
        path = os.path.join(ROOT, r["dst_rel"])
        if arms and r["dst_rel"].split("/")[3] not in arms:
            continue
        if os.path.islink(path) and os.readlink(path) == target:
            continue
        if os.path.exists(path) and not os.path.islink(path):
            print(f"not a symlink, left alone: {path}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.islink(path):
            os.remove(path)
        os.symlink(target, path)
        made.append(r["dst_rel"])
    return made


def cmd_pull(args):
    items = select(args)
    todo = [it for it in items if not local_ok(it)]
    links = restore_symlinks(getattr(args, "arm", None)) if not getattr(args, "dataset", None) \
        and not getattr(args, "view", None) else []
    for rel in links:
        print(f"symlink restored: {rel}")
    if not todo and not links:
        print("up to date")
        return 0
    if not todo:
        return 0
    for p, its in sorted(by_prefix(todo).items()):
        print(f"{p:44} {len(its):>7} objects {sum(i.size for i in its):>14} B to fetch")
        if len(its) <= 10:
            for it in its:
                print(f"    {it.key}")
    ak, sk = load_creds()
    local = threading.local()

    def work(it):
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk)
        data, etag = local.c.get(it.key)
        got = hashlib.md5(data).hexdigest()
        if got != it.md5 or len(data) != it.size:
            raise RuntimeError(f"{it.key}: got {len(data)} B md5 {got}, index says {it.size} B md5 {it.md5}")
        os.makedirs(os.path.dirname(it.path), exist_ok=True)
        tmp = it.path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, it.path)
        return len(data)

    done, failed, _ = parallel(todo, work, args.jobs, sum(i.size for i in todo), "pull")
    return 1 if failed else 0


# --------------------------------------------------------------------------- push

def unindexed_runs():
    """Files under runs/train/<family>/<arm>/{epochs,onnx,logs}/ and train.log with no MANIFEST row."""
    known = {r["dst_rel"] for r in manifest_rows()}
    out = []
    base = os.path.join(ROOT, "runs/train")
    for family in sorted(os.listdir(base)):
        fdir = os.path.join(base, family)
        if not os.path.isdir(fdir):
            continue
        for arm in sorted(os.listdir(fdir)):
            adir = os.path.join(fdir, arm)
            if not os.path.isdir(adir):
                continue
            for dp, _, fns in os.walk(adir):
                for fn in sorted(fns):
                    p = os.path.join(dp, fn)
                    if os.path.islink(p):
                        continue
                    rel = os.path.relpath(p, ROOT)
                    tail = os.path.relpath(p, adir)
                    on_oss = (tail.startswith(("epochs/", "onnx/", "logs/")) or tail == "train.log"
                              or fn.endswith((".pth", ".onnx", ".jit", ".zip")))
                    if on_oss and rel not in known:
                        out.append((rel, f"{MODELS_TOP}{family}/{arm}/{tail}"))
    return out


def append_manifest(new_rows):
    p = os.path.join(ROOT, "runs/train/MANIFEST.tsv")
    with open(p, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for r in new_rows:
            w.writerow(r)
    print(f"appended {len(new_rows)} rows to runs/train/MANIFEST.tsv (commit it)")


def cmd_push(args):
    items = select(args)
    new_rows = []
    if not args.dataset:
        for rel, key in unindexed_runs():
            path = os.path.join(ROOT, rel)
            size, md5 = os.path.getsize(path), md5_file(path)
            arm = key.split("/")[3]
            if args.arm and arm not in args.arm:
                continue
            items.append(Item(path, key, size, md5, arm))
            new_rows.append([rel, path, size, md5, key])
    missing_local = [it for it in items if not os.path.isfile(it.path)]
    items = [it for it in items if os.path.isfile(it.path)]
    ak, sk = load_creds()
    c = OSS(ak, sk)
    remote = {}
    for p in sorted({it.prefix for it in items}):
        remote.update(c.list(p))
    todo, ok, conflicts = [], 0, []
    for it in items:
        r = remote.get(it.key)
        if r is None:
            todo.append(it)
        elif r == (it.size, it.md5):
            ok += 1
        else:
            conflicts.append((it, r))
    print(f"local={len(items)} on_oss_and_equal={ok} to_push={len(todo)} conflicts={len(conflicts)} "
          f"not_on_disk={len(missing_local)}")
    for it, r in conflicts:
        print(f"CONFLICT {it.key}: OSS has {r}, local is ({it.size}, {it.md5}); refusing to overwrite")
    if conflicts:
        return 1
    if not todo:
        print("up to date")
        return 0
    for p, its in sorted(by_prefix(todo).items()):
        print(f"{p:44} {len(its):>7} objects {sum(i.size for i in its):>14} B to push")
    if args.dry_run:
        return 0
    local = threading.local()

    def work(it):
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk)
        with open(it.path, "rb") as f:
            data = f.read()
        if len(data) != it.size or hashlib.md5(data).hexdigest() != it.md5:
            raise RuntimeError(f"{it.path}: bytes changed since the index was read")
        local.c.put(it.key, data, it.md5, ctype_of(it.key))
        return len(data)

    done, failed, _ = parallel(todo, work, args.jobs, sum(i.size for i in todo), "push")
    pushed = {it.key for it in todo}
    if new_rows and not failed:
        append_manifest([r for r in new_rows if r[4] in pushed])
    return 1 if failed else 0


# --------------------------------------------------------------------------- status

def cmd_status(args):
    items = select(args)
    missing, present = [], []
    for it in items:
        (present if local_ok(it) else missing).append(it)
    remote = {}
    if args.remote:
        ak, sk = load_creds()
        c = OSS(ak, sk)
        for top in (DATA_TOP, MODELS_TOP):
            remote.update(c.list(top))
    # --remote lists everything under pii/data/ and pii/models/, so the "unindexed" set is
    # compared against the WHOLE index, not against the scope the other columns use.
    indexed = {it.key for it in (dataset_items() + run_items())} if args.remote \
        else {it.key for it in items}
    unpushed = [it for it in present if args.remote and it.key not in remote]
    if not args.remote:
        ak, sk = load_creds()
        c = OSS(ak, sk)
        for p in sorted({it.prefix for it in items}):
            remote.update(c.list(p))
        unpushed = [it for it in present if it.key not in remote]
    unindexed = sorted(set(remote) - indexed)
    prefixes = sorted({it.prefix for it in items} | {k.rsplit("/", 1)[0] + "/" for k in unindexed})
    print(f"{'prefix':44} {'indexed':>8} {'missing':>8} {'unpushed':>9} {'unindexed':>10}")
    for p in prefixes:
        n = sum(1 for it in items if it.prefix == p)
        m = sum(1 for it in missing if it.prefix == p)
        u = sum(1 for it in unpushed if it.prefix == p)
        x = sum(1 for k in unindexed if k.startswith(p))
        print(f"{p:44} {n:>8} {m:>8} {u:>9} {x:>10}")
    print(f"{'TOTAL':44} {len(items):>8} {len(missing):>8} {len(unpushed):>9} {len(unindexed):>10}")
    for it in missing[:10]:
        print("missing locally:", it.path)
    for it in unpushed[:10]:
        print("local but not on OSS:", it.key)
    for k in unindexed[:10]:
        print("on OSS but not in the index:", k)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--dataset", nargs="*")
    p.add_argument("--view", nargs="*")
    p.add_argument("--arm", nargs="*")
    p.add_argument("--jobs", type=int, default=JOBS)
    p.set_defaults(fn=cmd_pull)
    q = sub.add_parser("push")
    q.add_argument("--dataset", nargs="*")
    q.add_argument("--arm", nargs="*")
    q.add_argument("--jobs", type=int, default=JOBS)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(fn=cmd_push)
    s = sub.add_parser("status")
    s.add_argument("--dataset", nargs="*")
    s.add_argument("--view", nargs="*")
    s.add_argument("--arm", nargs="*")
    s.add_argument("--remote", action="store_true",
                   help="list all of pii/data/ and pii/models/ to find objects no index row claims")
    s.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    rc = args.fn(args)
    cache_save()
    sys.exit(rc)


if __name__ == "__main__":
    main()

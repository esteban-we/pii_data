#!/usr/bin/env python3
"""Upload the trained arm checkpoints and ONNX exports to oss://algorithm-datasets/pii/models/ (WOR-163).

Layout: pii/models/<arm>/epoch_20.pth and pii/models/<arm>/<arm>_det{10g|34g}.onnx, plus
pii/models/stock/ for the InsightFace baseline ONNX and the two official SCRFD checkpoints the
arms fine-tune from. Additive only: nothing outside pii/models/ is read or written, and an
existing key whose ETag differs from the local md5 is reported, never overwritten.

Reuses the stdlib OSS client of data/oss_pii_upload.py (V1 signing, ~/.aliyun default profile,
oss-cn-shanghai) without modifying it. Every object goes up as ONE single-part PutObject with a
Content-MD5 header and the response ETag is asserted equal to the local md5, so ETag == md5 for
every object under pii/models/. Single-part PutObject is limited to 5 GB by OSS; the largest file
here is 76 MB and the script refuses anything above the limit rather than falling back to multipart
(which would break the ETag rule).

Subcommands
  inventory   print arm, file, size, md5 for every entry of the table (missing files flagged), no network
  upload      idempotent: skip a key that already exists with the same size and ETag == md5, PUT
              the rest, assert ETag == md5 on the response; --only <arm> restricts to one arm
  write-csv   write the index (default /home/esteban/repos/pii-data/models.csv): one row per
              existing local file, provenance from the PROVENANCE table below, results column
              from the `models` keys of evaluation/results/results_*.json
  verify      list pii/models/, HEAD every object (no sampling) and assert: the remote set per
              arm equals the local file set, size and ETag == local md5, and every models.csv row
              (--csv) matches an object by key, size and md5 with no extra rows or objects.
              Exit 1 on any failure.
"""
import argparse
import base64
import csv
import glob
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oss_pii_upload import OSS as _OSS, load_creds  # noqa: E402


class OSS(_OSS):
    """oss_pii_upload.OSS plus V1 sub-resource signing for the multipart calls (?uploads,
    ?partNumber=N&uploadId=...), which the shared client never needs. Everything else inherited."""

    SUBRES = ("uploads", "partNumber", "uploadId")

    def request(self, verb, key, body=None, headers=None, query="", retries=6):
        parts = [q for q in query.split("&") if q and q.split("=")[0] in self.SUBRES]
        self._subres = ("?" + "&".join(sorted(parts))) if parts else ""
        return super().request(verb, key, body=body, headers=headers, query=query, retries=retries)

    def _sign(self, verb, key, headers, subres=""):
        return super()._sign(verb, key, headers, subres=getattr(self, "_subres", ""))

TOP = "pii/models/"
SINGLE_PUT_LIMIT = 5 * 1024 ** 3  # OSS PutObject maximum; above it only multipart works
# PII-1639: the results files stayed in the pii code repo.
RESULTS_DIR = os.path.join(CODE_ROOT, "evaluation", "results")
DEFAULT_CSV = "/home/esteban/repos/pii-data/models.csv"
# PII-1449: this uploader reads the PRE-PII-1315 tree, kept read-only as
# /data/esteban/pii_backup by the PII-1448 rename. data/oss_sync.py replaces it for the
# new store; this stays only to re-read what it once pushed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii_root import CODE_ROOT  # noqa: E402

# Provenance (PII-1682): the pre-PII-1315 tree this script read. The project retired it, so the
# paths below record where the data came from; they are not a tree to read today.
LEGACY_ROOT = "/data/esteban/pii_backup"

RT = LEGACY_ROOT + "/runs/train"
WT = LEGACY_ROOT + "/weights"

# (arm, oss basename, local path). Order is the order of models.csv.
FILES = [
    ("stock", "det_10g.onnx", f"{WT}/det_10g.onnx"),
    ("stock", "SCRFD_10G_KPS_official.pth", f"{WT}/SCRFD_10G_KPS_official.pth"),
    ("stock", "SCRFD_34G_official.pth", f"{WT}/SCRFD_34G_official.pth"),
    ("armW", "epoch_20.pth", f"{WT}/armW_epoch_20.pth"),
    ("armW", "armW_det10g.onnx", f"{WT}/det_10g_armW.onnx"),
    ("armW_repro", "epoch_20.pth", f"{RT}/wd_armW_repro/epoch_20.pth"),
    ("armW_repro", "armW_repro_det10g.onnx", f"{RT}/armW_repro_det10g.onnx"),
    ("armW_nowider", "epoch_20.pth", f"{RT}/wd_armW_nowider/epoch_20.pth"),
    ("armW_nowider", "armW_nowider_det10g.onnx", f"{RT}/armW_nowider_det10g.onnx"),
    ("armW_fe2", "epoch_20.pth", f"{RT}/wd_armW_fe2/epoch_20.pth"),
    ("armW_fe2", "armW_fe2_det10g.onnx", f"{RT}/armW_fe2_det10g.onnx"),
    ("armX", "epoch_20.pth", f"{RT}/wd_armX/epoch_20.pth"),
    ("armX", "armX_det10g.onnx", f"{RT}/armX_det10g.onnx"),
    ("armY", "epoch_20.pth", f"{RT}/wd_armY_ckpt/epoch_20.pth"),
    ("armY", "armY_det34g.onnx", f"{RT}/armY_det34g.onnx"),
    ("armZ", "epoch_20.pth", f"{RT}/wd_armZ/epoch_20.pth"),
    ("armZ", "armZ_det10g.onnx", f"{RT}/armZ_det10g.onnx"),
    ("armZ34", "epoch_20.pth", f"{RT}/wd_armZ34/epoch_20.pth"),
    ("armZ34", "armZ34_det34g.onnx", f"{RT}/armZ34_det34g.onnx"),
    ("armZZ", "epoch_20.pth", f"{RT}/wd_armZZ/epoch_20.pth"),
    ("armZZ", "armZZ_det10g.onnx", f"{RT}/armZZ_det10g.onnx"),
    ("armZZ34", "epoch_20.pth", f"{RT}/wd_armZZ34/epoch_20.pth"),
    ("armZZ34", "armZZ34_det34g.onnx", f"{RT}/armZZ34_det34g.onnx"),
    # armAA / armAA34 trained on shang; epoch_20.pth staged here by scp (WOR-163), md5 checked both sides
    ("armAA", "epoch_20.pth", f"{RT}/wd_armAA_ckpt/epoch_20.pth"),
    ("armAA", "armAA_det10g.onnx", f"{RT}/armAA_det10g.onnx"),
    ("armAA34", "epoch_20.pth", f"{RT}/wd_armAA34_ckpt/epoch_20.pth"),
    ("armAA34", "armAA34_det34g.onnx", f"{RT}/armAA34_det34g.onnx"),
]

# Per-arm provenance for models.csv. Every value is read from a file named in the comment;
# unknown stays "" (never guessed). base = load_from of the config that trained the run;
# manifest = data.train.ann_file basename; gpus_box = <box>/<n gpus>.
PROVENANCE = {
    # stock: not trained here. Sources: weights/*.provenance.txt, *.sha256 (WOR-47).
    "stock": dict(base="", manifest="", epochs="", gpus_box=""),
    # armW: shipped model. base from the config chain scrfd_armW -> armO -> armN ->
    # scrfd_10g_bnkps_ft2.py (load_from SCRFD_10G_KPS_official.pth); manifest and epochs from
    # scrfd_armW.py / scrfd_armN.py (total_epochs=20); box+gpus per Handover.md section 10
    # ("samples_per_gpu 32 x 2 GPUs", run dir gpu1:/data/liangzhenghao/train/wd_armW), not
    # independently verifiable from this box.
    "armW": dict(base="SCRFD_10G_KPS_official.pth", manifest="train_W.txt", epochs="20", gpus_box="gpu1/2 (per Handover.md)"),
    # armW_repro / _nowider / _fe2: wd_<arm>/scrfd_*.py (effective config copy): load_from
    # SCRFD_10G_KPS_official.pth, total_epochs 20, ann_file; log line 6 "GPU 0,1: NVIDIA RTX 6000 Ada".
    "armW_repro": dict(base="SCRFD_10G_KPS_official.pth", manifest="train_W.txt", epochs="20", gpus_box="here/2"),
    "armW_nowider": dict(base="SCRFD_10G_KPS_official.pth", manifest="train_W_nowider.txt", epochs="20", gpus_box="here/2"),
    "armW_fe2": dict(base="SCRFD_10G_KPS_official.pth", manifest="train_W_fe2.txt", epochs="20", gpus_box="here/2"),
    # armX: wd_armX/scrfd_armX.py, log "GPU 0" (one visible GPU; WOR-44 pinned CUDA_VISIBLE_DEVICES=5).
    "armX": dict(base="SCRFD_10G_KPS_official.pth", manifest="train_X.txt", epochs="20", gpus_box="here/1"),
    # armY: wd_armY_ckpt/embedded_config.py (load_from /home/esteban/pii/weights/SCRFD_34G_official.pth,
    # train_X.txt, total_epochs 20); box we-gpu-3, one RTX 5090 (WOR-51, scrfd_armY.py header).
    "armY": dict(base="SCRFD_34G_official.pth", manifest="train_X.txt", epochs="20", gpus_box="we-gpu-3/1"),
    # armZ/armZ34/armZZ/armZZ34: wd_<arm>/scrfd_<arm>.py + training/arms.yaml (box here, gpus list).
    "armZ": dict(base="armW_epoch_20.pth", manifest="train_Z.txt", epochs="20", gpus_box="here/1"),
    "armZ34": dict(base="SCRFD_34G_official.pth", manifest="train_Z.txt", epochs="20", gpus_box="here/1"),
    "armZZ": dict(base="armW_epoch_20.pth", manifest="train_Z.txt", epochs="20", gpus_box="here/2"),
    "armZZ34": dict(base="SCRFD_34G_official.pth", manifest="train_Z.txt", epochs="20", gpus_box="here/2"),
    # armAA/armAA34: training/configs/scrfd_armAA*.py (_base_ armZ / armZ34, ann_file train_Z2.txt)
    # + training/arms.yaml (box shang, 2 gpus each) + WOR-109/110/111.
    "armAA": dict(base="armW_epoch_20.pth", manifest="train_Z2.txt", epochs="20", gpus_box="shang/2"),
    "armAA34": dict(base="SCRFD_34G_official.pth", manifest="train_Z2.txt", epochs="20", gpus_box="shang/2"),
}

CSV_COLUMNS = ["arm", "file", "oss_key", "size", "md5", "format", "etag_rule", "base", "manifest", "epochs",
               "gpus_box", "results", "local_path"]

# Objects uploaded with `upload-multipart` (user decision, WOR-163 2026-09-10: the single-flow path
# was crippled). Their ETag is md5(concat part md5s)-N, not the file md5; the file md5 is in
# x-oss-meta-md5. Value = part size used. Every other object is single-part (ETag == md5).
MULTIPART = {
    "pii/models/armAA34/epoch_20.pth": 2 * 1024 * 1024,
    "pii/models/armAA34/armAA34_det34g.onnx": 2 * 1024 * 1024,
}


def expected_etag(key, data, md5_hex):
    """What the object's ETag must be under the rule recorded for this key."""
    ps = MULTIPART.get(key)
    if ps is None:
        return md5_hex
    return expected_multipart_etag([hashlib.md5(data[o:o + ps]).hexdigest() for o in range(0, len(data), ps)])


def key_of(arm, name):
    return f"{TOP}{arm}/{name}"


def md5_file(path):
    with open(path, "rb") as f:
        data = f.read()
    return data, hashlib.md5(data).hexdigest()


def inventory():
    """[(arm, name, key, path, size|None, md5|None)] in table order; size/md5 None when missing."""
    rows = []
    for arm, name, path in FILES:
        if os.path.isfile(path):
            _, md5 = md5_file(path)
            rows.append((arm, name, key_of(arm, name), path, os.path.getsize(path), md5))
        else:
            rows.append((arm, name, key_of(arm, name), path, None, None))
    return rows


def print_inventory(rows):
    print(f"{'arm':<13} {'file':<28} {'size':>10}  {'md5':<32}  local_path")
    missing = []
    for arm, name, key, path, size, md5 in rows:
        if size is None:
            missing.append(key)
            print(f"{arm:<13} {name:<28} {'MISSING':>10}  {'':<32}  {path}")
        else:
            print(f"{arm:<13} {name:<28} {size:>10}  {md5}  {path}")
    present = [r for r in rows if r[4] is not None]
    print(f"{len(present)} present, {len(missing)} missing, {sum(r[4] for r in present):,} bytes")
    for k in missing:
        print(f"GAP: {k}")
    return missing


def put_object(c, key, data, md5_hex):
    """Single-part PutObject with Content-MD5; returns the response ETag, asserted == md5_hex."""
    if len(data) > SINGLE_PUT_LIMIT:
        raise RuntimeError(f"{key}: {len(data)} bytes exceeds the 5 GB single PutObject limit; "
                           "multipart is not implemented (it would break ETag == md5)")
    h = {
        "Content-MD5": base64.b64encode(bytes.fromhex(md5_hex)).decode(),
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(data)),
    }
    st, rh, body = c.request("PUT", key, body=data, headers=h)
    if st != 200:
        raise RuntimeError(f"PUT {key}: HTTP {st}: {body[:300]!r}")
    etag = rh.get("ETag", "").strip('"').lower()
    if etag != md5_hex:
        raise RuntimeError(f"PUT {key}: ETag {etag} != md5 {md5_hex}")
    return etag


def expected_multipart_etag(part_md5s):
    """OSS multipart ETag rule, established empirically on 2026-09-10 (WOR-163) against the
    completed pii/models/armAA34/epoch_20.pth: MD5 of the concatenated UPPERCASE hex part ETag
    strings, then '-<n parts>'. This is NOT the S3 rule (MD5 of the concatenated *binary* part
    MD5s); that form was asserted by an earlier revision of this file and does not match OSS."""
    return hashlib.md5("".join(m.upper() for m in part_md5s).encode()).hexdigest() + f"-{len(part_md5s)}"


def put_multipart(ak, sk, key, data, md5_hex, part_size, jobs, timeout):
    """Multipart upload with parallel part PUTs (one connection per thread), each part carrying
    Content-MD5 and its ETag asserted == the part md5. The object's ETag is NOT the file md5;
    the file md5 is stored as x-oss-meta-md5 and the final ETag is asserted equal to
    expected_multipart_etag(). Returns (etag, part_md5s)."""
    import xml.etree.ElementTree as ET
    c = OSS(ak, sk, timeout=timeout)
    h = {"Content-Type": "application/octet-stream", "x-oss-meta-md5": md5_hex}
    st, rh, body = c.request("POST", key, headers=h, query="uploads")
    if st != 200:
        raise RuntimeError(f"InitiateMultipartUpload {key}: HTTP {st}: {body[:300]!r}")
    root = ET.fromstring(body)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    upload_id = root.find(f"{ns}UploadId").text
    parts = [(i + 1, data[o:o + part_size]) for i, o in enumerate(range(0, len(data), part_size))]
    local = threading.local()

    def work(item):
        n, chunk = item
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk, timeout=timeout)
        pm = hashlib.md5(chunk).hexdigest()
        ph = {"Content-MD5": base64.b64encode(bytes.fromhex(pm)).decode(),
              "Content-Length": str(len(chunk))}
        st, rh, body = local.c.request("PUT", key, body=chunk, headers=ph,
                                       query=f"partNumber={n}&uploadId={upload_id}")
        if st != 200:
            raise RuntimeError(f"UploadPart {key} #{n}: HTTP {st}: {body[:300]!r}")
        et = rh.get("ETag", "").strip('"').lower()
        if et != pm:
            raise RuntimeError(f"UploadPart {key} #{n}: ETag {et} != part md5 {pm}")
        return n, pm

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        done = dict(ex.map(work, parts))
    part_md5s = [done[n] for n, _ in parts]
    # OSS compares these ETags case-sensitively and returns them uppercase; sending the
    # lowercased hex is rejected with InvalidPart (WOR-163, 2026-09-10).
    xml = "<CompleteMultipartUpload>" + "".join(
        f"<Part><PartNumber>{n}</PartNumber><ETag>\"{done[n].upper()}\"</ETag></Part>" for n, _ in parts
    ) + "</CompleteMultipartUpload>"
    st, rh, body = c.request("POST", key, body=xml.encode(), headers={"Content-Type": "application/xml"},
                             query=f"uploadId={upload_id}")
    if st != 200:
        raise RuntimeError(f"CompleteMultipartUpload {key}: HTTP {st}: {body[:300]!r}")
    root = ET.fromstring(body)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    et = root.find(f"{ns}ETag")
    etag = (et.text if et is not None and et.text else rh.get("ETag", "")).strip('"').lower()
    exp = expected_multipart_etag(part_md5s)
    if etag != exp:
        raise RuntimeError(f"Complete {key}: ETag {etag} != expected multipart ETag {exp}")
    return etag, part_md5s


def cmd_upload_multipart(args):
    """NOT the default path. Only for a slow lossy link when the user accepts ETag != md5 for the
    object; proves the multipart ETag rule and stores the file md5 as x-oss-meta-md5."""
    t0 = time.time()
    rows = [r for r in inventory() if r[4] is not None and r[0] in args.only]
    ak, sk = load_creds()
    remote = OSS(ak, sk).list(TOP)
    for arm, name, key, path, size, md5 in rows:
        if key in remote:
            print(f"skip {key}: exists {remote[key]}")
            continue
        data, md5b = md5_file(path)
        assert md5b == md5
        t = time.time()
        etag, pm = put_multipart(ak, sk, key, data, md5, args.part_size, args.jobs, args.timeout)
        hs = OSS(ak, sk).head(key)
        print(f"MPUT   {key}: {size} bytes, {len(pm)} parts of {args.part_size}, ETag {etag} == "
              f"expected multipart ETag, file md5 {md5} in x-oss-meta-md5, HEAD {hs}, {time.time() - t:.0f}s")
    print(f"multipart run {time.time() - t0:.0f}s")


def cmd_download_check(args):
    """GET each --only arm's objects back and md5 the bytes (content proof for multipart objects).
    Aborts a download whose rate is below --min-rate MB/s after --grace seconds, as the path may be
    as crippled downstream as upstream; says so and exits 2 (not a verification failure)."""
    rows = [r for r in inventory() if r[4] is not None and r[0] in args.only]
    ak, sk = load_creds()
    rc = 0
    for arm, name, key, path, size, md5 in rows:
        c = OSS(ak, sk, timeout=args.grace + 5)
        c._connect()
        h = {"Date": __import__("email.utils", fromlist=["formatdate"]).formatdate(usegmt=True), "Host": c.conn.host}
        h["Authorization"] = c._sign("GET", key, h)
        c.conn.request("GET", "/" + key, headers=h)
        resp = c.conn.getresponse()
        if resp.status != 200:
            print(f"GET {key}: HTTP {resp.status}"); rc = 1; continue
        m, got, t0, aborted = hashlib.md5(), 0, time.time(), False
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            m.update(chunk); got += len(chunk)
            el = time.time() - t0
            if el > args.grace and got / el < args.min_rate * 1e6:
                aborted = True
                break
        el = time.time() - t0
        if aborted:
            print(f"ABORTED {key}: {got / 1e6:.1f} MB in {el:.0f}s = {got / el / 1e6:.2f} MB/s < {args.min_rate} MB/s; download path crippled too")
            rc = max(rc, 2); c.conn.close(); continue
        ok = got == size and m.hexdigest() == md5
        print(f"{'OK ' if ok else 'BAD'} {key}: downloaded {got} bytes in {el:.0f}s ({got / el / 1e6:.2f} MB/s), md5 {m.hexdigest()} {'==' if ok else '!='} local {md5}")
        if not ok:
            rc = 1
    sys.exit(rc)


def cmd_inventory(args):
    print_inventory(inventory())


def cmd_upload(args):
    t0 = time.time()
    rows = inventory()
    print_inventory(rows)
    if args.only:
        rows = [r for r in rows if r[0] in args.only]
    ak, sk = load_creds()
    remote = OSS(ak, sk).list(TOP)
    print(f"remote {TOP}: {len(remote)} objects before")
    todo, skip, conflict = [], 0, []
    for arm, name, key, path, size, md5 in rows:
        if size is None:
            continue
        r = remote.get(key)
        if r is not None:
            if r == (size, md5):
                print(f"skip   {key}: exists, size {size}, ETag == md5")
                skip += 1
            else:
                print(f"CONFLICT {key}: remote size/ETag {r} != local ({size}, {md5}); not overwritten")
                conflict.append(key)
            continue
        todo.append((key, path, size, md5))
    if args.limit:
        todo = todo[: args.limit]
    local = threading.local()

    def work(item):
        key, path, size, md5 = item
        if not hasattr(local, "c"):
            local.c = OSS(ak, sk, timeout=args.timeout)  # one connection per thread
        data, md5b = md5_file(path)
        assert md5b == md5 and len(data) == size, path
        t = time.time()
        etag = put_object(local.c, key, data, md5)
        hs = local.c.head(key)  # independent re-read after the PUT
        if hs != (size, md5):
            raise RuntimeError(f"HEAD after PUT {key}: {hs} != ({size}, {md5})")
        return f"PUT    {key}: {size} bytes, ETag {etag} == md5, HEAD ok, {time.time() - t:.0f}s"

    up, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
                up += 1
            except Exception as e:
                print(f"FAILED {futs[f][0]}: {e!r}", flush=True)
                failed.append(futs[f][0])
    print(f"uploaded {up}, skipped {skip}, failed {len(failed)}, conflicts {len(conflict)}, {time.time() - t0:.0f}s")
    if conflict or failed:
        sys.exit(1)


def results_for(arm):
    """Basenames of evaluation/results/results_*.json whose models dict has a key == arm."""
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("models") if isinstance(d, dict) else None
        if isinstance(m, dict) and arm in m:
            out.append(os.path.basename(f))
    return out


def csv_rows(only=None):
    rows = []
    for arm, name, key, path, size, md5 in inventory():
        if size is None or (only and arm not in only):
            continue
        p = PROVENANCE[arm]
        rows.append({
            "arm": arm, "file": name, "oss_key": key, "size": str(size), "md5": md5,
            "format": name.rsplit(".", 1)[-1],
            "etag_rule": "multipart" if key in MULTIPART else "single",
            "base": p["base"], "manifest": p["manifest"], "epochs": p["epochs"], "gpus_box": p["gpus_box"],
            # the "stock" model in the results files is det_10g.onnx; the two official .pth are
            # fine-tune bases that were never scored
            "results": "" if (arm == "stock" and name != "det_10g.onnx") else ";".join(results_for(arm)),
            "local_path": path,
        })
    return rows


def cmd_write_csv(args):
    rows = csv_rows(args.only)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}: {len(rows)} rows")


def cmd_verify(args):
    t0 = time.time()
    fails = []
    local = {r[2]: r for r in inventory() if r[4] is not None}
    c = OSS(*load_creds())
    remote = c.list(TOP)
    print(f"remote {TOP}: {len(remote)} objects; local present: {len(local)}")
    # 1. per-arm set equality
    arms = sorted({k.split("/")[2] for k in list(local) + list(remote)})
    for arm in arms:
        ls = {k for k in local if k.split("/")[2] == arm}
        rs = {k for k in remote if k.split("/")[2] == arm}
        if ls != rs:
            fails.append(f"{arm}: local-only {sorted(ls - rs)} remote-only {sorted(rs - ls)}")
        print(f"{arm:<13} local {len(ls)} remote {len(rs)} {'OK' if ls == rs else 'MISMATCH'}")
    # 2. HEAD every object: size, ETag under the key's rule (single: == md5; multipart: ==
    #    md5(concat part md5s)-N computed from the local bytes, plus x-oss-meta-md5 == file md5)
    for key in sorted(remote):
        st, h, _ = c.request("HEAD", key)
        hs = (int(h["Content-Length"]), h["ETag"].strip('"').lower()) if st == 200 else None
        meta = h.get("x-oss-meta-md5", "") if st == 200 else ""
        if key in local:
            data, md5 = md5_file(local[key][3])
            exp = (local[key][4], expected_etag(key, data, md5))
            meta_ok = (meta == md5) if key in MULTIPART else True
        else:
            exp, meta_ok = None, False
        ok = hs is not None and hs == exp and remote[key] == exp and meta_ok
        rule = "multipart" if key in MULTIPART else "single"
        print(f"{'OK ' if ok else 'BAD'} {key}: rule {rule} HEAD {hs} list {remote[key]} expected {exp}"
              f"{' meta-md5 ' + meta if key in MULTIPART else ''}")
        if not ok:
            fails.append(f"{key}: HEAD {hs} list {remote[key]} expected {exp} meta {meta}")
    # 3. models.csv rows == objects
    if args.csv:
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
        seen = set()
        for r in rows:
            k = r["oss_key"]
            seen.add(k)
            if k not in remote:
                fails.append(f"csv row {k}: no such object")
            elif k in MULTIPART:
                if remote[k][0] != int(r["size"]) or r.get("etag_rule") != "multipart":
                    fails.append(f"csv row {k}: size/etag_rule ({r['size']}, {r.get('etag_rule')}) != remote size {remote[k][0]} / multipart")
                if k in local and r["md5"] != local[k][5]:
                    fails.append(f"csv row {k}: csv md5 {r['md5']} != local md5 {local[k][5]}")
                st, h, _ = c.request("HEAD", k)
                if h.get("x-oss-meta-md5", "") != r["md5"]:
                    fails.append(f"csv row {k}: csv md5 {r['md5']} != x-oss-meta-md5 {h.get('x-oss-meta-md5')}")
            elif remote[k] != (int(r["size"]), r["md5"]) or r.get("etag_rule", "single") != "single":
                fails.append(f"csv row {k}: csv ({r['size']}, {r['md5']}, {r.get('etag_rule')}) != remote {remote[k]}")
            if k in local and (local[k][4], local[k][5]) != (int(r["size"]), r["md5"]):
                fails.append(f"csv row {k}: csv ({r['size']}, {r['md5']}) != local ({local[k][4]}, {local[k][5]})")
        if len(seen) != len(rows):
            fails.append(f"csv has duplicate oss_key rows ({len(rows)} rows, {len(seen)} keys)")
        extra = sorted(set(remote) - seen)
        if extra:
            fails.append(f"objects without a csv row: {extra}")
        print(f"csv {args.csv}: {len(rows)} rows, {len(seen)} keys, objects {len(remote)}")
    print(f"verify: {len(remote)} objects HEADed, {len(fails)} failures, {time.time() - t0:.1f}s")
    for x in fails:
        print("FAIL", x)
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory").set_defaults(fn=cmd_inventory)
    p = sub.add_parser("upload")
    p.add_argument("--only", nargs="*", help="restrict to these arms")
    p.add_argument("--limit", type=int, default=0, help="upload at most N objects")
    p.add_argument("--jobs", type=int, default=1, help="parallel PUTs (one connection each)")
    p.add_argument("--timeout", type=int, default=300, help="socket timeout per request, seconds")
    p.set_defaults(fn=cmd_upload)
    p = sub.add_parser("upload-multipart", help="NOT default: parallel-part multipart (ETag != md5), needs --only")
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--part-size", type=int, default=4 * 1024 * 1024)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(fn=cmd_upload_multipart)
    p = sub.add_parser("download-check", help="GET objects back and md5 them; abort if the path is crippled")
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--grace", type=float, default=30.0, help="seconds before the rate test applies")
    p.add_argument("--min-rate", type=float, default=1.0, help="MB/s below which the download is aborted")
    p.set_defaults(fn=cmd_download_check)
    p = sub.add_parser("write-csv")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--only", nargs="*", help="restrict rows to these arms (e.g. while the rest is deferred)")
    p.set_defaults(fn=cmd_write_csv)
    p = sub.add_parser("verify")
    p.add_argument("--csv", default=DEFAULT_CSV, help="models.csv to check; '' to skip")
    p.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

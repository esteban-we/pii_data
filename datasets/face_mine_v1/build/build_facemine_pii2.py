#!/usr/bin/env python3
"""Build the face_mine datasets in the pii2 layout (PII-1315; left eye PII-1318, right eye PII-1319).
Stdlib only, idempotent; every source is read-only.

    build  --eye left|right     copy images (skipped with --no-images, e.g. after an rsync),
                                write frames.csv, split.csv and boxes/vN
    verify --eye left|right     md5 every image against the frames.csv md5 column (and against
                                the source while it exists), check row counts, box files,
                                reviewed sets, pinned md5s and job copies; exit 1 on any mismatch

Left eye -> /data/esteban/pii/datasets/face_mine_v1, right eye -> .../face_mine_right
(separate datasets, PII-1315). Layout written (README.md files are written by hand):

    images/                     byte copy of the source images (no links)
    frames.csv                  image,session,chunk,eye,frame_idx,size,md5,src_key,src_frame,oss_key
                                (size = bytes of the image, md5 of its bytes; src_key is the
                                source video key and src_frame the frame index in that video,
                                both from the source boxes.jsonl). Left: size and md5 asserted
                                equal to pii-data frames.csv (dataset=face_mine_v1).
    split.csv                   session,role (train|eval) from data/splits/{train,eval}_sessions_v1.txt,
                                restricted to the sessions that have a frame in this eye
    boxes/vN/frames.csv         image,reviewed
    boxes/vN/boxes.csv          image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal, sorted by
                                image,x1,y1; images with no box have no row)
    boxes/vN/job/prelabels/     what the vendor started from (byte copies)
    boxes/vN/job/output/        what the vendor returned (byte copies)
    A pending version (still at the vendor) has job/prelabels and a README only.

Left (face_mine_v1):
  v1  human pass over all 62,587 frames (2026-09-01). boxes.csv from
      /data/esteban/pii_backup/face-mine_labeled.csv (normalized xywh -> pixel xyxy, WOR-140
      convention); asserted byte-equal to pii-data face_mine_v1/boxes/v1.csv. reviewed=1 all.
      job/prelabels: the miner's boxes.jsonl (face_mine/two_model@1); job/output: the csv.
  v2  armY false-positive eval frames relabeled (PII-142): 3,600 frames of the eval split.
      boxes.csv = vendor jsonl on those frames, v1 rows elsewhere; asserted byte-equal to
      pii-data face_mine_v1/boxes/v2.csv, reviewed set = v2.frames.txt.
  v3  pending: armAG34-over-human relabel of every frame (PII-1273); prelabels only.
Right (face_mine_right):
  v1  human pass over all 62,423 frames (2026-09-04..09, WOR-194). boxes.csv derived from the
      vendor drop (labels/face_boxes_face-mine-right.jsonl, byte-equal to /data/esteban/tmp/
      ann_9_11/); asserted equal to face_mine_right/boxes.jsonl (the WOR-194 conversion) while
      that file exists. job/output also carries the vendor's csv exports of the same labels
      (/data/esteban/tmp/faceback_anotation/face-mine-right.{boxes,frames}.csv). No prelabel
      file exists: the armW >= 0.5 prelabels lived in the portal only and the drop carries
      just their count per frame (n_boxes_machine, machine_src); job/prelabels/ is absent.
      pinned_md5 is the md5 of boxes.csv as first built (2026-09-21), no pii-data file exists.
  v2  pending: armAG34-over-human relabel (PII-1273); prelabels only.
      size and md5 in frames.csv are computed from the source bytes (no table has them).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool

# PII-1449: the live store; data/pii_root.py resolves it (env PII_ROOT or PII2_ROOT,
# default /data/esteban/pii). A clone elsewhere (shang, fluence) sets the env var.
STORE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(STORE, "data"))
from pii_root import LEGACY_ROOT, PII_ROOT  # noqa: E402

PII = LEGACY_ROOT    # PII-1448: the old tree, now /data/esteban/pii_backup
PII_DATA = "/home/esteban/repos/pii-data"
SPLITS = os.path.join(STORE, "data", "splits")
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
WRITE_ROOT = PII2_ROOT + "/"
RELABEL_V2 = f"{PII}/datasets/face_mine_relabel_v2"

W, H = 2328, 1748
NAME = re.compile(r"^(\d{8}_\d{6}_[A-Z0-9]{6})_c(\d{3})_f(\d{6})\.jpg$")
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
FRAME_COLS = ["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "src_key", "src_frame",
              "oss_key"]
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]

EYES = {
    "left": {
        "dataset": "face_mine_v1",
        "src_images": f"{PII}/datasets/face_mine_v1/images",
        "src_boxes_jsonl": f"{PII}/datasets/face_mine_v1/boxes.jsonl",   # miner output; provenance only
        "view": "lview", "video": "vst_left",
        "n_images": 62587, "n_sessions": 8995, "n_eval_sessions": 1858, "n_eval_frames": 12754,
        "pii_data_dataset": "face_mine_v1",     # rows of pii-data frames.csv carrying size/md5
        "versions": {
            "v1": {
                "kind": "labeled_csv",
                "output_file": "face-mine_labeled.csv",
                "expect_boxes": 76859,
                "pinned_md5": "a6a35c2e37ad70b3fd525f91b4a61298",   # pii-data face_mine_v1/boxes/v1.csv
                "job": {"prelabels": [f"{PII}/datasets/face_mine_v1/boxes.jsonl"],
                        "output": [f"{PII}/face-mine_labeled.csv"]},
            },
            "v2": {
                "kind": "drop_subset",
                "output_file": "face_boxes_face-mine-v2.jsonl",
                "prelabel_file": "armY_fp_thr0.5.jsonl",
                "n_reviewed": 3600,
                "expect_boxes": 81563,
                "pinned_md5": "7561137eb5c0c506310f2dc839de08d2",   # pii-data face_mine_v1/boxes/v2.csv
                "pinned_reviewed_md5": "b30c60df2820fae8429efdfa1d058fda",   # v2.frames.txt
                "job": {"prelabels": [f"{PII}/datasets/face_mine_v2/armY_fp_thr0.5.jsonl",
                                      f"{PII}/datasets/face_mine_v2/README.md"],
                        "output": ["/data/esteban/tmp/face_boxes_face-mine-v2.jsonl"]},
            },
            "v3": {
                "kind": "pending",
                "prelabel_file": "import_left.jsonl",
                "changed_file": "changed_left.txt",
                "job": {"prelabels": [f"{RELABEL_V2}/{n}" for n in
                                      ("import_left.jsonl", "changed_left.txt", "README.md", "stats.json")]},
            },
        },
    },
    "right": {
        "dataset": "face_mine_right",
        "src_images": f"{PII}/datasets/face_mine_right/images",
        "src_boxes_jsonl": f"{PII}/datasets/face_mine_right/boxes.jsonl",   # human boxes, WOR-194
        "view": "rview", "video": "vst_right",
        "n_images": 62423, "n_sessions": 8994, "n_eval_sessions": 1857, "n_eval_frames": 12722,
        "pii_data_dataset": None,               # not in pii-data: md5 computed from the source
        "versions": {
            "v1": {
                "kind": "human_jsonl",
                "output_file": "face_boxes_face-mine-right.jsonl",
                "drop_records": 62424,          # one record has no image (README of the source)
                "expect_boxes": 90902,
                "pinned_md5": "5188364d192dc67faf52f91017cfb7ed",   # boxes.csv as first built 2026-09-21 (no pii-data file)
                "job": {"output": [f"{PII}/datasets/face_mine_right/labels/face_boxes_face-mine-right.jsonl",
                                   f"{PII}/datasets/face_mine_right/labels/README.md",
                                   "/data/esteban/tmp/faceback_anotation/face-mine-right.boxes.csv",
                                   "/data/esteban/tmp/faceback_anotation/face-mine-right.frames.csv"]},
            },
            "v2": {
                "kind": "pending",
                "prelabel_file": "import_right.jsonl",
                "changed_file": "changed_right.txt",
                "job": {"prelabels": [f"{RELABEL_V2}/{n}" for n in
                                      ("import_right.jsonl", "changed_right.txt", "README.md", "stats.json")]},
            },
        },
    },
}

FAILS: list[str] = []
CFG: dict = {}
DST = ""


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check(ok: bool, msg: str) -> None:
    if not ok:
        die(msg)


def fail(msg: str) -> None:
    """verify: record and keep going."""
    FAILS.append(msg)
    print(f"FAIL: {msg}")


def fmt(v: float) -> str:
    return f"{v:.1f}"


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def dst(*parts: str) -> str:
    p = os.path.join(DST, *parts)
    check(os.path.abspath(p).startswith(WRITE_ROOT), f"refusing to write outside {WRITE_ROOT}: {p}")
    return p


def write_csv(path: str, header: list[str], rows) -> None:
    check(path.startswith(WRITE_ROOT), f"refusing to write outside {WRITE_ROOT}: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str, header: list[str]) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        check(hdr == header, f"{path}: header {hdr} != {header}")
        return [row for row in r]


def box_key(row):
    return (row[0], float(row[1]), float(row[2]))


def group_boxes(rows) -> dict[str, list[tuple]]:
    d: dict[str, list[tuple]] = {}
    for r in rows:
        d.setdefault(r[0], []).append(tuple(r[1:]))
    return d


def flatten(by: dict[str, list[tuple]]) -> list[list[str]]:
    rows = [[im, *b] for im, bl in by.items() for b in bl]
    rows.sort(key=box_key)
    return rows


def csv_bytes(header: list[str], rows) -> bytes:
    lines = [",".join(header)] + [",".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def pix(b: dict) -> tuple:
    """normalized top-left xywh -> pixel xyxy strings (WOR-140 convention), ignore=0."""
    return (fmt(b["x"] * W), fmt(b["y"] * H), fmt((b["x"] + b["w"]) * W), fmt((b["y"] + b["h"]) * H), "0")


def sort_boxes(bl: list[tuple]) -> list[tuple]:
    return sorted(bl, key=lambda t: (float(t[0]), float(t[1])))


def _md5_one(path: str) -> tuple[str, str]:
    return os.path.basename(path), md5_of(path)


def md5_dir(root: str, names: list[str], jobs: int, label: str) -> dict[str, str]:
    t0 = time.time()
    out = {}
    with Pool(jobs) as pool:
        for i, (n, m) in enumerate(pool.imap_unordered(_md5_one, [os.path.join(root, n) for n in names], chunksize=64), 1):
            out[n] = m
            if i % 20000 == 0:
                print(f"  md5 {label}: {i}/{len(names)} ({time.time() - t0:.0f}s)")
    print(f"md5 {label}: {len(names)} files ({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------- sources

def parse_name(name: str) -> tuple[str, str, str]:
    m = NAME.match(name)
    check(m is not None, f"bad image name {name}")
    return m.group(1), m.group(2), str(int(m.group(3)))


def load_source_provenance() -> dict[str, tuple[str, str]]:
    """{image: (src_key, src_frame)} from the source boxes.jsonl; one row per image, fields
    consistent with the name."""
    out = {}
    for ln in open(CFG["src_boxes_jsonl"], encoding="utf-8"):
        d = json.loads(ln)
        fn = d["image"].rsplit("/", 1)[-1]
        s, c, f = parse_name(fn)
        check((d["session"], d["chunk"], str(d["frame"])) == (s, c, f), f"boxes.jsonl: fields of {fn}")
        check(d["view"] == CFG["view"] and (d["width"], d["height"]) == (W, H), f"boxes.jsonl: view/dims {fn}")
        check(d["src_key"] == f"{s}/chunk_{c}/{CFG['video']}/{CFG['video']}_video.mp4", f"boxes.jsonl: src_key {fn}")
        check(d["src_frame"] == d["frame"] and fn not in out, f"boxes.jsonl: src_frame/dup {fn}")
        out[fn] = (d["src_key"], str(d["src_frame"]))
    check(len(out) == CFG["n_images"], f"boxes.jsonl has {len(out)} rows, expected {CFG['n_images']}")
    return out


def load_pii_data_rows() -> dict[str, tuple[str, str]] | None:
    """{image: (size, md5)} from pii-data frames.csv for this eye's dataset, or None."""
    ds = CFG["pii_data_dataset"]
    p = os.path.join(PII_DATA, "frames.csv")
    if ds is None or not os.path.isfile(p):
        return None
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["dataset"] == ds:
                check(row["image_ref"] == "" and row["view"] == "" and row["md5"], f"pii-data row {row['image']}")
                out[row["image"]] = (row["size"], row["md5"])
    return out


def load_splits(sessions: set[str]) -> dict[str, str]:
    roles = {}
    for role in ("train", "eval"):
        for s in open(os.path.join(SPLITS, f"{role}_sessions_v1.txt"), encoding="utf-8").read().split():
            check(s not in roles, f"session {s} in both split files")
            roles[s] = role
    check(sessions <= set(roles), f"{len(sessions - set(roles))} sessions with frames not in the split files")
    roles = {s: r for s, r in roles.items() if s in sessions}
    check(len(roles) == CFG["n_sessions"], f"{len(roles)} sessions, expected {CFG['n_sessions']}")
    check(sum(1 for r in roles.values() if r == "eval") == CFG["n_eval_sessions"], "eval session count")
    return roles


def load_labeled_csv(path: str, images: set[str]) -> dict[str, list[tuple]]:
    """face-mine_labeled.csv (one row per box; empty box_i = face-free) -> {image: sorted boxes}."""
    by: dict[str, list[tuple]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = row["image_uri"].rsplit("/", 1)[-1]
            check(fn in images and row["view"] == CFG["view"], f"{path}: {fn}")
            check((row["width"], row["height"]) == (str(W), str(H)), f"{path}: dims {fn}")
            bl = by.setdefault(fn, [])
            if row["box_i"] != "":
                bl.append(pix({k: float(row[k]) for k in "xywh"}))
    check(set(by) == images, f"{path}: frame set != images")
    return {k: sort_boxes(v) for k, v in by.items()}


def load_drop(path: str, images: set[str], null_records: int = 0) -> dict[str, list[tuple]]:
    """Vendor drop jsonl -> {image: sorted boxes}; trailer checked; records without an image
    (image_uri null) are counted and must number null_records."""
    out: dict[str, list[tuple]] = {}
    n = nulls = 0
    trailer = None
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        if "session" not in d:
            check(trailer is None and d.get("kind") == "end", f"{path}: unexpected non-record line")
            trailer = d
            continue
        n += 1
        if d.get("image_uri") is None:
            nulls += 1
            continue
        fn = d["image_uri"].rsplit("/", 1)[-1]
        check(fn in images and fn not in out and d["view"] == CFG["view"], f"{path}: {fn}")
        check((d["width"], d["height"]) == (W, H) and d["n_boxes"] == len(d["boxes"]), f"{path}: dims/n_boxes {fn}")
        out[fn] = sort_boxes([pix(b) for b in d["boxes"]])
    check(trailer is not None and trailer["frames"] == n, f"{path}: trailer/record count mismatch")
    check(nulls == null_records, f"{path}: {nulls} records without an image, expected {null_records}")
    return out


def load_human_jsonl(path: str, images: set[str]) -> dict[str, list[tuple]]:
    """face_mine_right/boxes.jsonl (box_src human) -> {image: sorted boxes}."""
    out: dict[str, list[tuple]] = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        fn = d["image"].rsplit("/", 1)[-1]
        check(fn in images and fn not in out and d["box_src"] == "human", f"{path}: {fn}")
        check(d["n_boxes"] == len(d["boxes"]), f"{path}: n_boxes {fn}")
        out[fn] = sort_boxes([pix(b) for b in d["boxes"]])
    check(set(out) == images, f"{path}: frame set != images")
    return out


def load_prelabel_frames(path: str, key: str) -> set[str]:
    return {json.loads(ln)[key] for ln in open(path, encoding="utf-8")}


# ---------------------------------------------------------------- versions

def derive_versions(images: list[str], eval_frames: set[str], job_root, args) -> dict[str, tuple[list, set]]:
    """{version: (box rows, reviewed set)} for the non-pending versions, re-derived from the job
    files (job_root(version, sub, basename) -> path). Pending versions are absent."""
    image_set = set(images)
    out: dict[str, tuple[list, set]] = {}
    prev_by: dict[str, list[tuple]] = {}
    for version, spec in CFG["versions"].items():
        kind = spec["kind"]
        if kind == "pending":
            frames = load_prelabel_frames(job_root(version, "prelabels", spec["prelabel_file"]), "image")
            check(frames == image_set, f"{version}: prelabel frame set != images")
            changed = set(open(job_root(version, "prelabels", spec["changed_file"]), encoding="utf-8").read().split())
            check(changed <= image_set, f"{version}: changed list has unknown images")
            continue
        if kind == "labeled_csv":
            by = load_labeled_csv(job_root(version, "output", spec["output_file"]), image_set)
            reviewed = image_set
        elif kind == "human_jsonl":
            drop = load_drop(job_root(version, "output", spec["output_file"]), image_set,
                             null_records=spec["drop_records"] - CFG["n_images"])
            check(set(drop) == image_set, f"{version}: drop frame set != images")
            if os.path.isfile(CFG["src_boxes_jsonl"]):   # the WOR-194 conversion, while pii exists
                check(load_human_jsonl(CFG["src_boxes_jsonl"], image_set) == drop,
                      f"{version}: source boxes.jsonl != vendor drop")
            by = drop
            reviewed = image_set
        elif kind == "drop_subset":
            drop = load_drop(job_root(version, "output", spec["output_file"]), image_set)
            reviewed = set(drop)
            check(len(reviewed) == spec["n_reviewed"] and reviewed <= eval_frames,
                  f"{version}: reviewed set is not {spec['n_reviewed']} eval frames")
            pre = load_prelabel_frames(job_root(version, "prelabels", spec["prelabel_file"]), "file")
            check(pre == reviewed, f"{version}: prelabel frame set != vendor output frame set")
            if spec.get("pinned_reviewed_md5"):
                check(md5_bytes(("\n".join(sorted(reviewed)) + "\n").encode()) == spec["pinned_reviewed_md5"],
                      f"{version}: reviewed set != pinned frames list")
            by = {k: v for k, v in prev_by.items() if k not in reviewed}
            by.update(drop)
        else:
            die(f"unknown kind {kind}")
        rows = flatten(by)
        check(len(rows) == spec["expect_boxes"], f"{version}: {len(rows)} boxes, expected {spec['expect_boxes']}")
        if spec.get("pinned_md5"):
            check(md5_bytes(csv_bytes(BOX_COLS, rows)) == spec["pinned_md5"], f"{version}: boxes != pinned md5")
        out[version] = (rows, reviewed)
        prev_by = {k: v for k, v in by.items() if v}
    return out


# ---------------------------------------------------------------- build

def copy_images(names: list[str]) -> None:
    src_dir, dst_dir = CFG["src_images"], dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    t0 = time.time()
    for n in names:
        s, d = os.path.join(src_dir, n), os.path.join(dst_dir, n)
        check(not os.path.islink(s), f"source image is a symlink: {s}")
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        shutil.copy2(s, d)   # full byte copy, mtime kept; never a link
        copied += 1
        if copied % 5000 == 0:
            print(f"  images: {copied} copied ({time.time() - t0:.0f}s)")
    print(f"images: {copied} copied, {len(names) - copied} already present ({time.time() - t0:.0f}s)")


def copy_job_files(version: str) -> None:
    for sub, files in CFG["versions"][version]["job"].items():
        out_dir = dst("boxes", version, "job", sub)
        os.makedirs(out_dir, exist_ok=True)
        for s in files:
            check(os.path.isfile(s), f"job source missing: {s}")
            d = os.path.join(out_dir, os.path.basename(s))
            if os.path.exists(d) and md5_of(d) == md5_of(s):
                continue
            shutil.copyfile(s, d)
        print(f"  {version}/job/{sub}: {len(files)} files")


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    names = sorted(os.listdir(CFG["src_images"]))
    check(len(names) == CFG["n_images"], f"source images: {len(names)} files, expected {CFG['n_images']}")
    check(len(set(names)) == len(names) and all(NAME.match(n) for n in names), "source image names")
    if not args.no_images:
        copy_images(names)
    prov = load_source_provenance()
    check(set(prov) == set(names), "source boxes.jsonl frame set != source images")

    # frames.csv: size and md5 of the source bytes; left: asserted equal to pii-data
    size = {n: str(os.path.getsize(os.path.join(CFG["src_images"], n))) for n in names}
    md5 = md5_dir(CFG["src_images"], names, args.jobs, "source")
    ref = load_pii_data_rows()
    if ref is not None:
        check(set(ref) == set(names), "pii-data frames.csv rows != source images")
        bad = [n for n in names if ref[n] != (size[n], md5[n])]
        check(not bad, f"{len(bad)} images differ from pii-data size/md5, e.g. {bad[:3]}")
        print(f"frames: size and md5 equal to pii-data for all {len(names)} images")
    frames = []
    for n in names:
        s, c, f = parse_name(n)
        frames.append([n, s, c, args.eye, f, size[n], md5[n], prov[n][0], prov[n][1],
                       f"pii/data/{CFG['dataset']}/{n}"])
    sessions = {r[1] for r in frames}
    roles = load_splits(sessions)
    eval_frames = {r[0] for r in frames if roles[r[1]] == "eval"}
    check(len(eval_frames) == CFG["n_eval_frames"], f"{len(eval_frames)} eval frames, expected {CFG['n_eval_frames']}")
    write_csv(dst("frames.csv"), FRAME_COLS, frames)
    write_csv(dst("split.csv"), ["session", "role"], [[s, roles[s]] for s in sorted(roles)])
    print(f"frames.csv {len(frames)} rows ({sum(int(size[n]) for n in names)} bytes), split.csv {len(roles)} "
          f"sessions ({CFG['n_eval_sessions']} eval, {len(eval_frames)} eval frames)")

    for version in CFG["versions"]:
        copy_job_files(version)
    versions = derive_versions(names, eval_frames, lambda v, sub, b: dst("boxes", v, "job", sub, b), args)
    for version, spec in CFG["versions"].items():
        if spec["kind"] == "pending":
            for p in ("frames.csv", "boxes.csv"):
                check(not os.path.exists(dst("boxes", version, p)), f"{version} is pending but has {p}")
            print(f"boxes/{version}: pending (prelabels only)")
            continue
        rows, reviewed = versions[version]
        write_csv(dst("boxes", version, "boxes.csv"), BOX_COLS, rows)
        write_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"],
                  [[im, "1" if im in reviewed else "0"] for im in names])
        by = group_boxes(rows)
        ev = [r for r in rows if r[0] in eval_frames]
        rv = [r for r in rows if r[0] in reviewed]
        print(f"boxes/{version}: {len(rows)} boxes on {len(by)} images ({len(names) - len(by)} face-free); "
              f"train {len(rows) - len(ev)} boxes, eval {len(ev)} boxes; {len(reviewed)} reviewed carrying "
              f"{len(rv)} boxes on {len({r[0] for r in rv})} images")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def verify(args) -> None:
    t0 = time.time()
    frames = read_csv(dst("frames.csv"), FRAME_COLS)
    images = [r[0] for r in frames]
    image_set = set(images)
    if len(frames) != CFG["n_images"]:
        fail(f"frames.csv has {len(frames)} rows, expected {CFG['n_images']}")
    if images != sorted(images) or len(image_set) != len(images):
        fail("frames.csv not sorted by image or has duplicates")
    for r in frames:
        m = NAME.match(r[0])
        if (m is None or (m.group(1), m.group(2), str(int(m.group(3)))) != (r[1], r[2], r[4]) or r[3] != args.eye
                or not MD5_HEX.match(r[6]) or not r[5].isdigit()
                or r[7] != f"{r[1]}/chunk_{r[2]}/{CFG['video']}/{CFG['video']}_video.mp4" or r[8] != r[4]):
            fail(f"frames.csv: bad row for {r[0]}")
            break
    size_of = {r[0]: int(r[5]) for r in frames}
    md5_col = {r[0]: r[6] for r in frames}
    ref = load_pii_data_rows()
    if ref is not None:
        if set(ref) != image_set:
            fail("pii-data frames.csv rows != frames.csv images")
        elif any(ref[n] != (r[5], r[6]) for n, r in zip(images, frames)):
            fail("frames.csv size/md5 differ from pii-data frames.csv")
        else:
            print(f"frames.csv: size and md5 equal to pii-data for all {len(images)} images")
    if os.path.isfile(CFG["src_boxes_jsonl"]):
        prov = load_source_provenance()
        if any((r[7], r[8]) != prov.get(r[0]) for r in frames):
            fail("frames.csv src_key/src_frame differ from the source boxes.jsonl")

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(roles) != CFG["n_sessions"] or len(split) != CFG["n_sessions"]:
        fail(f"split.csv has {len(split)} rows, expected {CFG['n_sessions']}")
    if {r[1] for r in frames} != set(roles) or set(roles.values()) - {"train", "eval"}:
        fail("split.csv sessions do not match frames.csv, or bad role")
    if os.path.isdir(SPLITS) and load_splits({r[1] for r in frames}) != roles:
        fail("split.csv differs from data/splits/{train,eval}_sessions_v1.txt")
    eval_frames = {r[0] for r in frames if roles.get(r[1]) == "eval"}
    if len(eval_frames) != CFG["n_eval_frames"]:
        fail(f"{len(eval_frames)} eval frames, expected {CFG['n_eval_frames']}")

    # images: every file present, no extras, no links, size and md5 equal to frames.csv (and the source)
    listed = sorted(os.listdir(dst("images")))
    if listed != images:
        fail(f"images/ lists {len(listed)} files; frames.csv has {len(images)}; "
             f"extra {sorted(set(listed) - image_set)[:3]}, missing {sorted(image_set - set(listed))[:3]}")
    src_dir = CFG["src_images"]
    have_src = os.path.isdir(src_dir)
    for n in listed:
        p = os.path.join(DST, "images", n)
        if os.path.islink(p):
            fail(f"images/{n} is a symlink")
        elif n in size_of and os.path.getsize(p) != size_of[n]:
            fail(f"images/{n}: size {os.path.getsize(p)} != frames.csv size {size_of[n]}")
        elif have_src and os.path.isfile(os.path.join(src_dir, n)):
            ss, ds_ = os.stat(os.path.join(src_dir, n)), os.stat(p)
            if ss.st_ino == ds_.st_ino and ss.st_dev == ds_.st_dev:
                fail(f"images/{n} is a hardlink of the source")
    if args.no_md5:
        print("images: md5 check skipped (--no-md5)")
    else:
        listed_set = set(listed)
        present = [n for n in images if n in listed_set]
        got = md5_dir(dst("images"), present, args.jobs, "images")
        bad = [n for n in present if got[n] != md5_col[n]]
        if bad:
            for n in bad[:20]:
                fail(f"images/{n}: md5 {got[n]} != frames.csv {md5_col[n]}")
            fail(f"{len(bad)} images differ from frames.csv md5")
        else:
            print(f"images: {len(present)} md5 equal to frames.csv ({time.time() - t0:.0f}s)")
        if have_src:
            src = md5_dir(src_dir, present, args.jobs, "source")
            bad = [n for n in present if src[n] != got[n]]
            if bad:
                fail(f"{len(bad)} images differ from the source, e.g. {bad[:3]}")
            else:
                print(f"images: {len(present)} md5 equal to the source ({time.time() - t0:.0f}s)")
        else:
            print(f"  note: source {src_dir} gone, images compared to frames.csv md5 only")

    # job copies (compared to the source while it exists)
    for version, spec in CFG["versions"].items():
        for sub, files in spec["job"].items():
            for s in files:
                d = dst("boxes", version, "job", sub, os.path.basename(s))
                if not os.path.isfile(d):
                    fail(f"missing job copy {d}")
                elif os.path.isfile(s) and md5_of(s) != md5_of(d):
                    fail(f"job copy differs from source: {d}")
                elif not os.path.isfile(s):
                    print(f"  note: source gone, job copy not compared: {d}")
    if FAILS and any("missing job copy" in f for f in FAILS):
        print(f"VERIFY FAILED: {len(FAILS)} problem(s), job copies missing so box files not re-derived")
        sys.exit(1)

    # box versions, re-derived from the job copies
    expected = derive_versions(images, eval_frames, lambda v, sub, b: dst("boxes", v, "job", sub, b), args)
    for version, spec in CFG["versions"].items():
        if spec["kind"] == "pending":
            for p in ("frames.csv", "boxes.csv"):
                if os.path.exists(dst("boxes", version, p)):
                    fail(f"boxes/{version} is pending but has {p}")
            print(f"boxes/{version}: pending, prelabels present")
            continue
        exp_rows, exp_reviewed = expected[version]
        vf = read_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"])
        if [r[0] for r in vf] != images:
            fail(f"boxes/{version}/frames.csv image column != frames.csv")
        if set(r[1] for r in vf) - {"0", "1"}:
            fail(f"boxes/{version}/frames.csv: reviewed not 0/1")
        reviewed = {r[0] for r in vf if r[1] == "1"}
        if reviewed != exp_reviewed:
            fail(f"boxes/{version}/frames.csv reviewed set: {len(reviewed)} frames, expected {len(exp_reviewed)}")
        rows = read_csv(dst("boxes", version, "boxes.csv"), BOX_COLS)
        if len(rows) != spec["expect_boxes"]:
            fail(f"boxes/{version}/boxes.csv has {len(rows)} rows, expected {spec['expect_boxes']}")
        if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
            fail(f"boxes/{version}/boxes.csv not sorted by (image, x1, y1)")
        if {r[0] for r in rows} - image_set:
            fail(f"boxes/{version}/boxes.csv: box on an image not in frames.csv")
        if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
            fail(f"boxes/{version}/boxes.csv: coordinate not one decimal or bad ignore")
        if rows != exp_rows:
            fail(f"boxes/{version}/boxes.csv differs from the rows re-derived from job/")
        if spec.get("pinned_md5") and md5_of(dst("boxes", version, "boxes.csv")) != spec["pinned_md5"]:
            fail(f"boxes/{version}/boxes.csv md5 != pinned md5")
        print(f"boxes/{version}: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
              f"{len(reviewed)} reviewed ({time.time() - t0:.0f}s)")

    for p in ["README.md"] + [f"boxes/{v}/README.md" for v in CFG["versions"]]:
        if not os.path.isfile(dst(p)):
            fail(f"missing {p}")
    for p in ("_download.log", "_launch_time.txt"):
        if os.path.exists(dst(p)):
            fail(f"{p} must not be in the dataset")

    if FAILS:
        print(f"VERIFY FAILED: {len(FAILS)} problem(s) ({time.time() - t0:.0f}s)")
        sys.exit(1)
    print(f"VERIFY OK ({time.time() - t0:.0f}s)")


def main() -> None:
    global CFG, DST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eye", choices=("left", "right"), required=True)
    b.add_argument("--no-images", action="store_true", help="skip the image copy (e.g. rsync ran already)")
    b.add_argument("--jobs", type=int, default=16, help="md5 workers")
    v = sub.add_parser("verify")
    v.add_argument("--eye", choices=("left", "right"), required=True)
    v.add_argument("--no-md5", action="store_true", help="skip the per-image md5 pass")
    v.add_argument("--jobs", type=int, default=16, help="md5 workers")
    args = ap.parse_args()
    CFG = EYES[args.eye]
    DST = os.path.join(WRITE_ROOT, "datasets", CFG["dataset"])
    {"build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()

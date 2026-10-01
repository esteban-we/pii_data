#!/usr/bin/env python3
"""Build /data/esteban/pii/datasets/face10k in the pii2 layout (PII-1315, PII-1320).
Stdlib only, idempotent; every source is read-only.

    build            copy images (skipped with --no-images), write frames.csv, split.csv,
                     boxes/v1..v3 (frames.csv, boxes.csv, job/prelabels, job/output)
    verify           md5 every image against the frames.csv md5 column (and against the
                     source while it exists), check row counts, box files, reviewed sets,
                     the unlabeled flag and job copies; exit 1 on any mismatch

One image set: pii/datasets/face10k_v3/images (8,339, batch v3) plus
face10k_v3/repair_v1/images (3,168, batch repair) = 11,507 files in one flat images/
directory (names asserted unique before any copy).

Layout written (README.md files are written by hand, not here):

    images/                     byte copies (no links)
    frames.csv                  image,session,chunk,eye,frame_idx,size,md5,batch,unlabeled,oss_key
                                (size and md5 from pii-data frames.csv, datasets face10k_v3
                                and face10k_repair; size asserted equal to the file on disk;
                                batch v3|repair; unlabeled=1 on the 16 repair frames the
                                original vendor never returned, PII-131)
    split.csv                   session,role: every session is train
    boxes/vN/frames.csv         image,reviewed
    boxes/vN/boxes.csv          image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal,
                                sorted by image,x1,y1; images with no box have no row)
    boxes/vN/job/prelabels/     what the vendor started from (byte copies)
    boxes/vN/job/output/        what the vendor returned (byte copies)

v1: pii-data face10k_v3/boxes/v1.csv (25,966) + face10k_repair/boxes/v1.csv (7,883),
    merged and re-sorted; reviewed=1 on every frame except the 16 unlabeled ones, which
    have no box row. Cross-checked against pii-data frames.csv n_boxes and, when the
    file is present, against the face10k labelv2 blocks of training/manifests/train_Z2.txt
    (the file the boxes were recovered from, PII-330). job/prelabels: the two source
    manifests plus the README, ANNOTATION_SPEC and stats the vendor was given.
    job/output: none on disk.
v2: pii-data facecheck1/boxes/v1.csv (49,379), reviewed=1 on all 11,507; re-derived from
    /data/esteban/tmp/face_boxes_facecheck1.jsonl (normalized xywh -> pixel xyxy at one
    decimal, the PII-140 convention) and asserted equal. job/output: that jsonl.
v3: same with facecheck2 (55,946 boxes).
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
# datasets/<name>/build/<this file> is three levels under the store root, so four
# dirname() calls (PII-1681: the PII-1649 move left three, which resolved to
# <store>/datasets and made every builder fail on `import pii_root`).
STORE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(STORE, "data"))
from pii_root import CODE_ROOT, PII_ROOT  # noqa: E402

# Provenance (PII-1682): the pre-PII-1315 tree this script read. The project retired it, so the
# paths below record where the data came from; they are not a tree to read today.
LEGACY_ROOT = "/data/esteban/pii_backup"

SRC_V3 = LEGACY_ROOT + "/datasets/face10k_v3"    # PII-1448: the old tree
SRC_REPAIR = os.path.join(SRC_V3, "repair_v1")
SRC_IMAGES = {"v3": os.path.join(SRC_V3, "images"), "repair": os.path.join(SRC_REPAIR, "images")}
SRC_FACECHECK = {"v2": "/data/esteban/tmp/face_boxes_facecheck1.jsonl",
                 "v3": "/data/esteban/tmp/face_boxes_facecheck2.jsonl"}
PII_DATA = "/home/esteban/repos/pii-data/datasets"
PII_DATA_FRAMES = "/home/esteban/repos/pii-data/frames.csv"
TRAIN_Z2 = os.path.join(CODE_ROOT, "training", "manifests", "train_Z2.txt")
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DST = PII2_ROOT + "/datasets/face10k"
WRITE_ROOT = PII2_ROOT + "/"

W, H = 2328, 1748
N_BATCH = {"v3": 8339, "repair": 3168}
N_IMAGES = sum(N_BATCH.values())
N_SESSIONS = 361
N_UNLABELED = 16
UNLABELED_SESSION = "20260725_055005_MPCWWA"
EXPECT_BOXES = {"v1": 25966 + 7883, "v2": 49379, "v3": 55946}
# pii-data box files the versions must reproduce
MD5_PII_DATA = {
    "face10k_v3/boxes/v1.csv": "eab68f114ece62cf9deef487e1b60c34",
    "face10k_repair/boxes/v1.csv": "c204316dda8b7c3c4c3f803c27a13b6d",
    "facecheck1/boxes/v1.csv": "93339899800b5b14d29abf13afde01dd",
    "facecheck2/boxes/v1.csv": "7352c56b6a2909d8acd6035d641a73f2",
}
PII_DATA_DATASET = {"v3": "face10k_v3", "repair": "face10k_repair"}
Z2_PREFIX = {"face10k/images/": "v3", "face10k/repair_v1/": "repair"}

FRAME_COLS = ["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "batch", "unlabeled",
              "oss_key"]

# PII-1413: the object that holds this image on oss://algorithm-datasets. The two batches
# predate the merge into one dataset and keep their own prefixes.
OSS_PREFIX = {"v3": "pii/data/face10k_v3/", "repair": "pii/data/face10k_repair/"}
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
EYE_OF_VIEW = {"vst_left": "left", "lview": "left", "rview": "right"}
KPS_PLACEHOLDER = ["-1.0"] * 15

# (source path, destination relative to boxes/vN/job/)
JOB_COPIES = {
    "v1": [(os.path.join(SRC_V3, n), f"prelabels/face10k_v3/{n}")
           for n in ("manifest.jsonl", "README.md", "ANNOTATION_SPEC.md", "stats.json")]
          + [(os.path.join(SRC_REPAIR, n), f"prelabels/repair_v1/{n}") for n in ("manifest.jsonl", "README.md")],
    "v2": [(SRC_FACECHECK["v2"], "output/face_boxes_facecheck1.jsonl")],
    "v3": [(SRC_FACECHECK["v3"], "output/face_boxes_facecheck2.jsonl")],
}

FAILS: list[str] = []


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


def csv_bytes(header: list[str], rows) -> bytes:
    lines = [",".join(header)] + [",".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def expected_name(session: str, chunk: str, eye: str, frame_idx: str, batch: str) -> str:
    if batch == "v3":
        return f"{session}_{chunk}_f{int(frame_idx):06d}.jpg"
    view = {"left": "lview", "right": "rview"}[eye]
    return f"{session}_chunk_{chunk}_{view}_f{int(frame_idx):06d}.jpg"


# ---------------------------------------------------------------- sources

def load_manifests() -> dict[str, dict]:
    """{image: {session, chunk, eye, frame_idx, batch}} from the two source manifests.
    Asserts the two batches share no image name (the flat images/ dir depends on it)."""
    out: dict[str, dict] = {}
    for batch, root in (("v3", SRC_V3), ("repair", SRC_REPAIR)):
        n = 0
        for ln in open(os.path.join(root, "manifest.jsonl"), encoding="utf-8"):
            d = json.loads(ln)
            n += 1
            image = d["image"].rsplit("/", 1)[-1]
            check(d["image"] == "images/" + image, f"{batch} manifest: unexpected image path {d['image']}")
            check((d["width"], d["height"]) == (W, H), f"{batch} manifest: dims of {image}")
            chunk = f"{d['chunk_index']:03d}" if batch == "v3" else d["chunk"]
            rec = {"session": d["session"], "chunk": chunk, "eye": EYE_OF_VIEW[d["view"]],
                   "frame_idx": str(d["frame"]), "batch": batch}
            check(image == expected_name(rec["session"], chunk, rec["eye"], rec["frame_idx"], batch),
                  f"{batch} manifest: {image} does not match its fields")
            check(image not in out, f"image name {image} appears in both batches (or twice)")
            out[image] = rec
        check(n == N_BATCH[batch], f"{batch} manifest has {n} rows, expected {N_BATCH[batch]}")
    for batch, d in SRC_IMAGES.items():
        listed = set(os.listdir(d))
        check(listed == {k for k, v in out.items() if v["batch"] == batch},
              f"{d} does not list exactly the {batch} manifest images")
    return out


def load_pii_data_frames() -> dict[str, tuple[str, str, str, int]]:
    """{image: (size, md5, role, n_boxes)} from pii-data frames.csv, datasets face10k_v3 + face10k_repair."""
    out: dict[str, tuple[str, str, str, int]] = {}
    with open(PII_DATA_FRAMES, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["dataset"] not in PII_DATA_DATASET.values():
                continue
            check(d["image"] not in out, f"pii-data frames.csv: duplicate {d['image']}")
            check(MD5_HEX.match(d["md5"]) is not None and d["size"].isdigit(),
                  f"pii-data frames.csv: bad size/md5 for {d['image']}")
            check(d["image_ref"] == "", f"pii-data frames.csv: {d['image']} has an image_ref")
            out[d["image"]] = (d["size"], d["md5"], d["role"], int(d["n_boxes"]))
    check(len(out) == N_IMAGES, f"pii-data frames.csv has {len(out)} face10k rows, expected {N_IMAGES}")
    return out


def build_frames() -> list[list[str]]:
    """FRAME_COLS rows sorted by image."""
    man = load_manifests()
    pd = load_pii_data_frames()
    check(set(man) == set(pd), "source manifests and pii-data frames.csv list different images")
    rows = []
    unlabeled = []
    for image in sorted(man):
        m = man[image]
        size, md5, role, _ = pd[image]
        check(role in ("train", "unlabeled"), f"pii-data role {role} for {image}")
        if role == "unlabeled":
            check(m["batch"] == "repair" and m["session"] == UNLABELED_SESSION,
                  f"unexpected unlabeled frame {image}")
            unlabeled.append(image)
        rows.append([image, m["session"], m["chunk"], m["eye"], m["frame_idx"], size, md5,
                     m["batch"], "1" if role == "unlabeled" else "0",
                     OSS_PREFIX[m["batch"]] + image])
    check(len(unlabeled) == N_UNLABELED, f"{len(unlabeled)} unlabeled frames, expected {N_UNLABELED}")
    return rows


def load_pii_data_boxes(rel: str) -> list[list[str]]:
    p = os.path.join(PII_DATA, rel)
    check(md5_of(p) == MD5_PII_DATA[rel], f"pii-data {rel} is not the pinned file")
    rows = read_csv(p, BOX_COLS)
    check([box_key(r) for r in rows] == sorted(box_key(r) for r in rows), f"{rel} not sorted")
    return rows


def load_vendor_jsonl(path: str) -> dict[str, list[tuple]]:
    """facecheck drop -> {image: sorted pixel box tuples} (empty list = face-free frame)."""
    out: dict[str, list[tuple]] = {}
    n = 0
    trailer = None
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        if "session" not in d:
            check(trailer is None and d.get("kind") == "end", f"{path}: unexpected non-record line")
            trailer = d
            continue
        n += 1
        name = d["image_uri"].rsplit("/", 1)[-1]
        check(name not in out, f"{path}: {name} appears twice")
        check((d["width"], d["height"]) == (W, H), f"{path}: dims {name}")
        check(d["n_boxes"] == len(d["boxes"]), f"{path}: n_boxes mismatch {name}")
        bl = [(fmt(b["x"] * W), fmt(b["y"] * H), fmt((b["x"] + b["w"]) * W), fmt((b["y"] + b["h"]) * H), "0")
              for b in d["boxes"]]
        bl.sort(key=lambda t: (float(t[0]), float(t[1])))
        out[name] = bl
    check(trailer is not None and trailer["frames"] == n, f"{path}: trailer/record count mismatch")
    return out


def load_z2_face10k() -> dict[str, list[tuple]]:
    """{image: sorted box tuples} from the face10k labelv2 blocks of train_Z2.txt."""
    out: dict[str, list[tuple]] = {}
    cur = None
    with open(TRAIN_Z2, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if ln.startswith("#"):
                parts = ln.split()
                check(len(parts) == 4, f"train_Z2: bad header {ln!r}")
                cur = None
                for prefix in Z2_PREFIX:
                    if parts[1].startswith(prefix):
                        check((int(parts[2]), int(parts[3])) == (W, H), f"train_Z2: dims {parts[1]}")
                        name = parts[1][len(prefix):]
                        check(name not in out, f"train_Z2: duplicate {parts[1]}")
                        cur = out.setdefault(name, [])
            elif cur is not None:
                v = ln.split()
                if len(v) == 19:
                    check(v[4:] == KPS_PLACEHOLDER, f"train_Z2: non-placeholder keypoints: {ln!r}")
                    ignore = "0"
                elif len(v) == 5 and v[4] == "1":
                    ignore = "1"
                else:
                    die(f"train_Z2: box line with {len(v)} fields: {ln!r}")
                cur.append(tuple(fmt(float(x)) for x in v[:4]) + (ignore,))
    for bl in out.values():
        bl.sort(key=lambda t: (float(t[0]), float(t[1])))
    return out


# ---------------------------------------------------------------- build

def copy_images(frames: list[list[str]]) -> None:
    dst_dir = dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    t0 = time.time()
    for i, r in enumerate(frames, 1):
        name, size, batch = r[0], int(r[5]), r[7]
        s = os.path.join(SRC_IMAGES[batch], name)
        d = os.path.join(dst_dir, name)
        check(not os.path.islink(s), f"source image is a symlink: {s}")
        check(os.path.getsize(s) == size, f"{s}: size {os.path.getsize(s)} != pii-data size {size}")
        if os.path.exists(d) and not os.path.islink(d) and os.path.getsize(d) == size:
            continue
        shutil.copy2(s, d)   # full byte copy, mtime kept; never a link
        copied += 1
        if copied % 2000 == 0:
            print(f"  images: {copied} copied, {i}/{len(frames)} seen, {time.time() - t0:.0f}s")
    print(f"images: {copied} copied, {len(frames) - copied} already present ({time.time() - t0:.0f}s)")


def copy_job_files(version: str) -> None:
    for s, rel in JOB_COPIES[version]:
        check(os.path.isfile(s), f"job source missing: {s}")
        d = dst("boxes", version, "job", rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d) and md5_of(d) == md5_of(s):
            continue
        shutil.copyfile(s, d)
    print(f"  {version}/job: {len(JOB_COPIES[version])} files")


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    frames = build_frames()
    all_images = [r[0] for r in frames]
    image_set = set(all_images)
    batch_of = {r[0]: r[7] for r in frames}
    unlabeled = {r[0] for r in frames if r[8] == "1"}
    if not args.no_images:
        copy_images(frames)

    sessions = sorted({r[1] for r in frames})
    check(len(sessions) == N_SESSIONS, f"{len(sessions)} sessions, expected {N_SESSIONS}")
    write_csv(dst("frames.csv"), FRAME_COLS, frames)
    write_csv(dst("split.csv"), ["session", "role"], [[s, "train"] for s in sessions])
    print(f"frames.csv {len(frames)} rows ({N_BATCH['v3']} v3 + {N_BATCH['repair']} repair, "
          f"{len(unlabeled)} unlabeled), split.csv {len(sessions)} sessions, all train")

    # v1: the two pii-data files merged; no box on an unlabeled frame
    v1_v3 = load_pii_data_boxes("face10k_v3/boxes/v1.csv")
    v1_rep = load_pii_data_boxes("face10k_repair/boxes/v1.csv")
    check(all(batch_of.get(r[0]) == "v3" for r in v1_v3), "face10k_v3 v1.csv: box on a non-v3 image")
    check(all(batch_of.get(r[0]) == "repair" for r in v1_rep), "face10k_repair v1.csv: box on a non-repair image")
    v1 = sorted(v1_v3 + v1_rep, key=box_key)
    check(len(v1) == EXPECT_BOXES["v1"], f"v1 rows {len(v1)}")
    check(not ({r[0] for r in v1} & unlabeled), "v1: box on an unlabeled frame")
    n_boxes_pd = {img: v[3] for img, v in load_pii_data_frames().items()}
    v1_by = group_boxes(v1)
    check(all(len(v1_by.get(img, [])) == n for img, n in n_boxes_pd.items()),
          "v1 per-image counts != pii-data frames.csv n_boxes")
    if os.path.isfile(TRAIN_Z2):
        z2 = load_z2_face10k()
        check(set(z2) == image_set - unlabeled, f"train_Z2 face10k blocks cover {len(z2)} frames, "
                                                 f"expected the {len(image_set) - len(unlabeled)} labeled ones")
        check({k: v for k, v in z2.items() if v} == v1_by, "v1 boxes != train_Z2.txt face10k blocks")
        print(f"  v1: equal to the {len(z2)} face10k blocks of train_Z2.txt")
    else:
        print(f"  note: {TRAIN_Z2} not present, v1 not cross-checked against it")

    versions = {"v1": (v1, image_set - unlabeled)}
    # v2, v3: the facecheck passes, re-derived from the vendor jsonl and asserted equal to pii-data
    for version, rel in (("v2", "facecheck1/boxes/v1.csv"), ("v3", "facecheck2/boxes/v1.csv")):
        rows = load_pii_data_boxes(rel)
        check(len(rows) == EXPECT_BOXES[version], f"{version} rows {len(rows)}")
        vendor = load_vendor_jsonl(SRC_FACECHECK[version])
        check(set(vendor) == image_set, f"{version}: vendor jsonl frames != the 11,507 images")
        check({k: v for k, v in vendor.items() if v} == group_boxes(rows),
              f"pii-data {rel} differs from the vendor jsonl {SRC_FACECHECK[version]}")
        versions[version] = (rows, image_set)

    for version, (rows, reviewed) in versions.items():
        write_csv(dst("boxes", version, "boxes.csv"), BOX_COLS, rows)
        write_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"],
                  [[im, "1" if im in reviewed else "0"] for im in all_images])
        copy_job_files(version)
        n_img = len({r[0] for r in rows})
        print(f"boxes/{version}: {len(rows)} boxes on {n_img} images, {len(reviewed)} reviewed")
    for version, rel in (("v2", "facecheck1/boxes/v1.csv"), ("v3", "facecheck2/boxes/v1.csv")):
        check(md5_of(dst("boxes", version, "boxes.csv")) == MD5_PII_DATA[rel], f"written {version} boxes.csv md5")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def _md5_check(job: tuple[str, str, str, bool]):
    """(name, batch, expected md5 from frames.csv, compare_src) -> (name, error or None)."""
    name, batch, expected, compare_src = job
    s = os.path.join(SRC_IMAGES[batch], name)
    d = os.path.join(DST, "images", name)
    if os.path.islink(d):
        return name, "dst is a symlink"
    if not os.path.isfile(d):
        return name, "missing in dst"
    md = md5_of(d)
    if md != expected:
        return name, f"md5 {md} != frames.csv {expected}"
    if compare_src:
        if not os.path.isfile(s):
            return name, "missing in src"
        if os.stat(s).st_ino == os.stat(d).st_ino and os.stat(s).st_dev == os.stat(d).st_dev:
            return name, "dst is a hardlink of src"
        ms = md5_of(s)
        if ms != md:
            return name, f"md5 {md} != source {ms}"
    return name, None


def verify(args) -> None:
    t0 = time.time()
    frames = read_csv(dst("frames.csv"), FRAME_COLS)
    images = [r[0] for r in frames]
    image_set = set(images)
    if len(frames) != N_IMAGES:
        fail(f"frames.csv has {len(frames)} rows, expected {N_IMAGES}")
    if images != sorted(images) or len(image_set) != len(images):
        fail("frames.csv not sorted by image or has duplicates")
    size_of = {r[0]: int(r[5]) for r in frames if r[5].isdigit()}
    md5_of_image = {r[0]: r[6] for r in frames}
    batch_of = {r[0]: r[7] for r in frames}
    unlabeled = {r[0] for r in frames if r[8] == "1"}
    for r in frames:
        if r[7] not in N_BATCH or r[8] not in ("0", "1"):
            fail(f"frames.csv: bad batch/unlabeled on {r[0]}")
            break
        if r[0] != expected_name(r[1], r[2], r[3], r[4], r[7]):
            fail(f"frames.csv: {r[0]} does not match its fields")
            break
    for batch, n in N_BATCH.items():
        if sum(1 for r in frames if r[7] == batch) != n:
            fail(f"frames.csv: batch {batch} count != {n}")
    if len(unlabeled) != N_UNLABELED or any(batch_of[u] != "repair" or not u.startswith(UNLABELED_SESSION) for u in unlabeled):
        fail(f"frames.csv: unlabeled set is {len(unlabeled)} frames, expected the {N_UNLABELED} "
             f"repair frames of {UNLABELED_SESSION}")
    if any(not MD5_HEX.match(r[6]) or not r[5].isdigit() for r in frames):
        fail("frames.csv: md5 not 32 hex chars or size not an integer")
    src_present = all(os.path.isfile(os.path.join(root, "manifest.jsonl")) for root in (SRC_V3, SRC_REPAIR))
    if src_present:
        man = load_manifests()
        if {k: [v["session"], v["chunk"], v["eye"], v["frame_idx"], v["batch"]] for k, v in man.items()} \
                != {r[0]: [r[1], r[2], r[3], r[4], r[7]] for r in frames}:
            fail("frames.csv fields differ from the source manifests")
    else:
        print("  note: source manifests not present, frames.csv fields not cross-checked")
    if os.path.isfile(PII_DATA_FRAMES):
        pd = load_pii_data_frames()
        if {k: (v[0], v[1]) for k, v in pd.items()} != {r[0]: (r[5], r[6]) for r in frames}:
            fail("frames.csv size/md5 differ from pii-data frames.csv")
        if {k for k, v in pd.items() if v[2] == "unlabeled"} != unlabeled:
            fail("frames.csv unlabeled flag differs from pii-data role=unlabeled")
    else:
        pd = None
        print(f"  note: {PII_DATA_FRAMES} not present, size/md5/unlabeled not cross-checked")

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(roles) != N_SESSIONS or len(split) != N_SESSIONS:
        fail(f"split.csv has {len(split)} rows, expected {N_SESSIONS}")
    if {r[1] for r in frames} != set(roles) or set(roles.values()) != {"train"}:
        fail("split.csv sessions do not match frames.csv, or a role is not train")

    # images: every file present, no extras, size and md5 as in frames.csv (and as the source)
    listed = sorted(os.listdir(dst("images")))
    if listed != images:
        fail(f"images/ lists {len(listed)} files; frames.csv has {len(images)}; "
             f"extra {sorted(set(listed) - image_set)[:3]}, missing {sorted(image_set - set(listed))[:3]}")
    for n in listed:
        p = os.path.join(DST, "images", n)
        if n in size_of and os.path.getsize(p) != size_of[n]:
            fail(f"images/{n}: size {os.path.getsize(p)} != frames.csv size {size_of[n]}")
    if args.no_md5:
        print("images: md5 check skipped (--no-md5)")
    else:
        compare_src = all(os.path.isdir(d) for d in SRC_IMAGES.values())
        if not compare_src:
            print("  note: source images not present; md5 checked against frames.csv only")
        jobs = [(n, batch_of[n], md5_of_image[n], compare_src) for n in images]
        bad = 0
        with Pool(args.jobs) as pool:
            for i, (name, err) in enumerate(pool.imap_unordered(_md5_check, jobs, chunksize=64), 1):
                if err:
                    bad += 1
                    if bad <= 20:
                        fail(f"images/{name}: {err}")
                if i % 4000 == 0:
                    print(f"  md5: {i}/{len(images)} ({time.time() - t0:.0f}s)")
        if bad:
            fail(f"{bad} images differ")
        else:
            print(f"images: {len(images)} md5 equal to frames.csv"
                  f"{' and to the source' if compare_src else ''} ({time.time() - t0:.0f}s)")

    # box versions
    pinned = {"v2": "facecheck1/boxes/v1.csv", "v3": "facecheck2/boxes/v1.csv"}
    for version in ("v1", "v2", "v3"):
        vf = read_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"])
        if [r[0] for r in vf] != images:
            fail(f"boxes/{version}/frames.csv image column != frames.csv")
        if set(r[1] for r in vf) - {"0", "1"}:
            fail(f"boxes/{version}/frames.csv: reviewed not 0/1")
        reviewed = {r[0] for r in vf if r[1] == "1"}
        rows = read_csv(dst("boxes", version, "boxes.csv"), BOX_COLS)
        if len(rows) != EXPECT_BOXES[version]:
            fail(f"boxes/{version}/boxes.csv has {len(rows)} rows, expected {EXPECT_BOXES[version]}")
        if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
            fail(f"boxes/{version}/boxes.csv not sorted by (image, x1, y1)")
        if {r[0] for r in rows} - image_set:
            fail(f"boxes/{version}/boxes.csv: box on an image not in frames.csv")
        if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
            fail(f"boxes/{version}/boxes.csv: coordinate not one decimal or bad ignore")
        if version == "v1":
            if {r[0] for r in rows} & unlabeled:
                fail("boxes/v1/boxes.csv: box on an unlabeled frame")
            for batch, rel in (("v3", "face10k_v3/boxes/v1.csv"), ("repair", "face10k_repair/boxes/v1.csv")):
                part = [r for r in rows if batch_of.get(r[0]) == batch]
                if md5_bytes(csv_bytes(BOX_COLS, part)) != MD5_PII_DATA[rel]:
                    fail(f"boxes/v1/boxes.csv rows of batch {batch} != pii-data {rel}")
            if pd is not None:
                by = group_boxes(rows)
                if any(len(by.get(img, [])) != v[3] for img, v in pd.items()):
                    fail("boxes/v1/boxes.csv per-image counts != pii-data frames.csv n_boxes")
            expected_reviewed = image_set - unlabeled
        else:
            if md5_of(dst("boxes", version, "boxes.csv")) != MD5_PII_DATA[pinned[version]]:
                fail(f"boxes/{version}/boxes.csv md5 != pii-data {pinned[version]}")
            out = dst("boxes", version, "job", JOB_COPIES[version][0][1])
            if os.path.isfile(out):
                vendor = load_vendor_jsonl(out)
                if set(vendor) != image_set:
                    fail(f"boxes/{version}/job/output covers {len(vendor)} frames, expected all {N_IMAGES}")
                if {k: v for k, v in vendor.items() if v} != group_boxes(rows):
                    fail(f"boxes/{version}/boxes.csv != its job/output jsonl")
            expected_reviewed = image_set
        if reviewed != expected_reviewed:
            fail(f"boxes/{version}/frames.csv reviewed set: {len(reviewed)} frames, "
                 f"expected {len(expected_reviewed)}")
        for s, rel in JOB_COPIES[version]:
            d = dst("boxes", version, "job", rel)
            if not os.path.isfile(d):
                fail(f"missing job copy {d}")
            elif os.path.isfile(s) and md5_of(s) != md5_of(d):
                fail(f"job copy differs from source: {d}")
            elif not os.path.isfile(s):
                print(f"  note: source gone, job copy not compared: {d}")
        print(f"boxes/{version}: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
              f"{len(reviewed)} reviewed ({time.time() - t0:.0f}s)")

    for p in ("README.md", "boxes/v1/README.md", "boxes/v2/README.md", "boxes/v3/README.md"):
        if not os.path.isfile(dst(p)):
            fail(f"missing {p}")
    for p in ("_download.log", "_launch_time.txt", "_probe.txt", ".complete"):
        if os.path.exists(dst(p)):
            fail(f"{p} must not be in the dataset")

    if FAILS:
        print(f"VERIFY FAILED: {len(FAILS)} problem(s) ({time.time() - t0:.0f}s)")
        sys.exit(1)
    print(f"VERIFY OK ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-images", action="store_true", help="skip the image copy")
    v = sub.add_parser("verify")
    v.add_argument("--no-md5", action="store_true", help="skip the per-image md5 comparison")
    v.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    {"build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()

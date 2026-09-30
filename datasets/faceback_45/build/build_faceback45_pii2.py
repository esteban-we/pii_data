#!/usr/bin/env python3
"""Build /data/esteban/pii/datasets/faceback_45 in the pii2 layout (PII-1315, PII-1316).
Stdlib only, idempotent; every source is read-only.

    build            copy images (skipped with --no-images), write frames.csv, split.csv,
                     boxes/v1..v3 (frames.csv, boxes.csv, job/prelabels, job/output)
    verify           md5 every image against the frames.csv md5 column (and against the
                     source while it exists), check row counts, box files, reviewed sets
                     and job copies; exit 1 on any mismatch

Layout written (README.md files are written by hand, not here):

    images/                     byte copy of pii/datasets/faceback_45/images (no links)
    frames.csv                  image,session,chunk,eye,frame_idx,size,md5,t_ms,src_path,oss_key
                                (size and md5 from pii-data frames.csv, dataset=faceback_45;
                                size asserted equal to the manifest bytes; PII-1315)
    split.csv                   session,role (train|eval), by session
    boxes/vN/frames.csv         image,reviewed
    boxes/vN/boxes.csv          image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal,
                                sorted by image,x1,y1; images with no box have no row)
    boxes/vN/job/prelabels/     what the vendor started from (byte copies)
    boxes/vN/job/output/        what the vendor returned (byte copies)

v1: faceback_45/boxes.jsonl (human, normalized xywh) -> pixel xyxy with the WOR-140
    convention (x*W, y*H, (x+w)*W, (y+h)*H at one decimal); asserted byte-equal to
    pii-data datasets/faceback_45/boxes/v1.csv. reviewed=1 everywhere.
v2: pii-data faceback_hq/boxes/v1.csv rows on the 16,940 eval frames, v1 rows on the
    train frames. reviewed=1 on the eval frames. Cross-checked against the vendor jsonl.
v3: pii-data faceback_hq/boxes/v2.csv rows on the eval frames, v2 rows on the train
    frames. reviewed=1 on the 4,023 frames of faceback_hq/boxes/v2.frames.txt (the
    frames the round-2 vendor output covers). Cross-checked against the vendor jsonl.
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
from pii_root import LEGACY_ROOT, PII_ROOT  # noqa: E402

SRC = LEGACY_ROOT + "/datasets/faceback_45"    # PII-1448: the old tree
SRC_RELABEL = LEGACY_ROOT + "/datasets/faceback_hq_relabel_v1"
SRC_HQ_EVAL = "/data/esteban/tmp/hq_eval"
SRC_HQ_2 = "/data/esteban/tmp/pii/fb_hq_2"
PII_DATA = "/home/esteban/repos/pii-data/datasets"
PII_DATA_FRAMES = "/home/esteban/repos/pii-data/frames.csv"
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DST = PII2_ROOT + "/datasets/faceback_45"
WRITE_ROOT = PII2_ROOT + "/"

W, H = 2328, 1748
N_IMAGES = 84954
N_SESSIONS = 334
N_EVAL_FRAMES = 16940
N_V3_REVIEWED = 4023
EXPECT_BOXES = {"v1": 95006, "v2": 23016 + 75999, "v3": 27454 + 75999}
# pii-data box files the versions must reproduce (v1 whole; v2/v3 on the eval frames)
MD5_FB45_V1 = "5b4b6704b7b782fc455a354acfad0313"
MD5_HQ_V1 = "1f50860639a223f11e90c754422add0c"
MD5_HQ_V2 = "1c44ee062e7d80703c31f1c9f2ce896f"
MD5_HQ_V2_FRAMES = "df5bbb42ae66d7f5d9a45b08166a847d"

FRAME_COLS = ["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "t_ms", "src_path",
              "oss_key"]
OSS_PREFIX = "pii/data/faceback_45/"   # PII-1413: where the image bytes live on OSS
MANIFEST_COLS = ["image", "session", "chunk", "eye", "frame_idx", "t_ms", "bytes", "src_path"]
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
EYE_OF_VIEW = {"lview": "left", "rview": "right"}
NO_EYE_NAME = re.compile(r"^(.+)_f(\d+)\.jpg$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")

JOB_COPIES = {
    "v1": {"prelabels": [os.path.join(SRC, "prelabels", n) for n in
                         ("import_left.jsonl", "import_right.jsonl", "prelabels_full.jsonl", "README.md")],
           "output": [os.path.join(SRC, "labels", n) for n in
                      ("faceback-45-left.frames.csv", "faceback-45-left.boxes.csv",
                       "faceback-45-right.frames.csv", "faceback-45-right.boxes.csv", "README.md")]},
    "v2": {"output": [os.path.join(SRC_HQ_EVAL, n) for n in
                      ("face_boxes_faceback-45-HQ-eval-left.jsonl", "face_boxes_faceback-45-HQ-eval-right.jsonl")]},
    "v3": {"prelabels": [os.path.join(SRC_RELABEL, n) for n in
                         ("additions.jsonl", "import.jsonl", "README.md", "stats.json")],
           "output": [os.path.join(SRC_HQ_2, n) for n in
                      ("face_boxes_faceback_hq_2-left.jsonl", "face_boxes_faceback_hq_2-right.jsonl")]},
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


# ---------------------------------------------------------------- sources

def load_manifest() -> list[list[str]]:
    """MANIFEST_COLS rows from upload_manifest.jsonl, sorted by image; key dropped."""
    rows = []
    seen = set()
    for ln in open(os.path.join(SRC, "upload_manifest.jsonl"), encoding="utf-8"):
        d = json.loads(ln)
        image = d["key"].rsplit("/", 1)[-1]
        check(image == f"{d['session']}_c{d['chunk']}_{d['eye']}_f{d['frame_idx']:06d}.jpg",
              f"manifest: key {d['key']} does not match its fields")
        check(image not in seen, f"manifest: duplicate {image}")
        seen.add(image)
        rows.append([image, d["session"], d["chunk"], d["eye"], str(d["frame_idx"]),
                     str(d["t_ms"]), str(d["bytes"]), d["src_path"]])
    rows.sort(key=lambda r: r[0])
    check(len(rows) == N_IMAGES, f"manifest has {len(rows)} rows, expected {N_IMAGES}")
    return rows


def load_pii_data_md5() -> dict[str, tuple[str, str]]:
    """{image: (size, md5)} from pii-data frames.csv rows with dataset=faceback_45."""
    out: dict[str, tuple[str, str]] = {}
    with open(PII_DATA_FRAMES, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            if d["dataset"] != "faceback_45":
                continue
            check(d["image"] not in out, f"pii-data frames.csv: duplicate {d['image']}")
            check(MD5_HEX.match(d["md5"]) is not None and d["size"].isdigit(),
                  f"pii-data frames.csv: bad size/md5 for {d['image']}")
            out[d["image"]] = (d["size"], d["md5"])
    check(len(out) == N_IMAGES, f"pii-data frames.csv has {len(out)} faceback_45 rows, expected {N_IMAGES}")
    return out


def build_frames() -> list[list[str]]:
    """FRAME_COLS rows: manifest fields plus size and md5 from pii-data (no re-hashing)."""
    hashes = load_pii_data_md5()
    rows = []
    for image, session, chunk, eye, frame_idx, t_ms, nbytes, src_path in load_manifest():
        check(image in hashes, f"pii-data frames.csv has no md5 for {image}")
        size, md5 = hashes[image]
        check(size == nbytes, f"{image}: pii-data size {size} != manifest bytes {nbytes}")
        rows.append([image, session, chunk, eye, frame_idx, size, md5, t_ms, src_path,
                     OSS_PREFIX + image])
    return rows


def load_splits() -> dict[str, str]:
    roles = {}
    for role in ("train", "eval"):
        p = os.path.join(SRC, "splits", f"faceback_{role}_sessions_v1.txt")
        for s in open(p, encoding="utf-8").read().split():
            check(s not in roles, f"session {s} in both split files")
            roles[s] = role
    check(len(roles) == N_SESSIONS, f"splits list {len(roles)} sessions, expected {N_SESSIONS}")
    return roles


def load_faceback_jsonl() -> list[list[str]]:
    """v1 rows (pixel xyxy strings) from boxes.jsonl, sorted by (image, x1, y1)."""
    rows = []
    seen = set()
    for ln in open(os.path.join(SRC, "boxes.jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        fn = r["image"].rsplit("/", 1)[-1]
        check((r["width"], r["height"]) == (W, H), f"boxes.jsonl: dims {fn}")
        check(fn not in seen, f"boxes.jsonl: duplicate {fn}")
        check(r["n_boxes"] == len(r["boxes"]), f"boxes.jsonl: n_boxes mismatch {fn}")
        seen.add(fn)
        for b in r["boxes"]:
            rows.append([fn, fmt(b["x"] * W), fmt(b["y"] * H),
                         fmt((b["x"] + b["w"]) * W), fmt((b["y"] + b["h"]) * H), "0"])
    check(len(seen) == N_IMAGES, f"boxes.jsonl has {len(seen)} frames, expected {N_IMAGES}")
    rows.sort(key=box_key)
    return rows


def load_vendor_jsonl(paths: list[str]) -> dict[str, list[tuple]]:
    """Vendor drop -> {owner image: sorted pixel box tuples} (empty list = face-free).
    Owner name = basename with the eye inserted from the record's view (WOR-176)."""
    out: dict[str, list[tuple]] = {}
    for p in paths:
        n = 0
        trailer = None
        for ln in open(p, encoding="utf-8"):
            d = json.loads(ln)
            if "session" not in d:
                check(trailer is None and d.get("kind") == "end", f"{p}: unexpected non-record line")
                trailer = d
                continue
            n += 1
            base = d["image_uri"].rsplit("/", 1)[-1]
            eye = EYE_OF_VIEW[d["view"]]
            m = NO_EYE_NAME.match(base)
            check(m is not None, f"{p}: bad name {base}")
            name = f"{m.group(1)}_{eye}_f{m.group(2)}.jpg"
            check(name not in out, f"{p}: {name} mapped twice")
            check((d["width"], d["height"]) == (W, H), f"{p}: dims {name}")
            check(d["n_boxes"] == len(d["boxes"]), f"{p}: n_boxes mismatch {name}")
            bl = [(fmt(b["x"] * W), fmt(b["y"] * H), fmt((b["x"] + b["w"]) * W),
                   fmt((b["y"] + b["h"]) * H), "0") for b in d["boxes"]]
            bl.sort(key=lambda t: (float(t[0]), float(t[1])))
            out[name] = bl
        check(trailer is not None and trailer["frames"] == n, f"{p}: trailer/record count mismatch")
    return out


def load_pii_data_boxes(rel: str) -> list[list[str]]:
    return read_csv(os.path.join(PII_DATA, rel), BOX_COLS)


# ---------------------------------------------------------------- build

def copy_images() -> None:
    src_dir = os.path.join(SRC, "images")
    dst_dir = dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(os.listdir(src_dir))
    check(len(names) == N_IMAGES, f"source images: {len(names)} files, expected {N_IMAGES}")
    copied = 0
    t0 = time.time()
    for i, n in enumerate(names, 1):
        s = os.path.join(src_dir, n)
        d = os.path.join(dst_dir, n)
        check(not os.path.islink(s), f"source image is a symlink: {s}")
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        shutil.copy2(s, d)   # full byte copy, mtime kept; never a link
        copied += 1
        if copied % 5000 == 0:
            print(f"  images: {copied} copied, {i}/{len(names)} seen, {time.time() - t0:.0f}s")
    print(f"images: {copied} copied, {len(names) - copied} already present ({time.time() - t0:.0f}s)")


def copy_job_files(version: str) -> None:
    for sub, files in JOB_COPIES[version].items():
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
    if not (args.no_images or args.frames_only):
        copy_images()

    frames = build_frames()
    roles = load_splits()
    all_images = [r[0] for r in frames]
    sessions = {r[1] for r in frames}
    check(sessions == set(roles), "manifest sessions != split sessions")
    eval_frames = {r[0] for r in frames if roles[r[1]] == "eval"}
    check(len(eval_frames) == N_EVAL_FRAMES, f"{len(eval_frames)} eval frames, expected {N_EVAL_FRAMES}")

    write_csv(dst("frames.csv"), FRAME_COLS, frames)
    if args.frames_only:
        print(f"frames.csv {len(frames)} rows (--frames-only; split.csv, boxes and images untouched)")
        return
    write_csv(dst("split.csv"), ["session", "role"], [[s, roles[s]] for s in sorted(roles)])
    print(f"frames.csv {len(frames)} rows, split.csv {len(roles)} sessions "
          f"({sum(1 for s in roles if roles[s] == 'eval')} eval)")

    # v1: boxes.jsonl -> pixel xyxy; must equal pii-data faceback_45/boxes/v1.csv byte for byte
    v1 = load_faceback_jsonl()
    check(set(r[0] for r in v1) <= set(all_images), "v1: box on an unknown image")
    v1_bytes = csv_bytes(BOX_COLS, v1)
    ref = open(os.path.join(PII_DATA, "faceback_45", "boxes", "v1.csv"), "rb").read()
    check(md5_bytes(ref) == MD5_FB45_V1, "pii-data faceback_45/boxes/v1.csv is not the pinned file")
    check(v1_bytes == ref, "v1 boxes differ from pii-data faceback_45/boxes/v1.csv")
    check(len(v1) == EXPECT_BOXES["v1"], f"v1 rows {len(v1)}")
    v1_by = group_boxes(v1)
    train_rows = [r for r in v1 if r[0] not in eval_frames]

    # v2: hq v1 on the eval frames, v1 rows on the rest
    hq1 = load_pii_data_boxes("faceback_hq/boxes/v1.csv")
    check(md5_of(os.path.join(PII_DATA, "faceback_hq/boxes/v1.csv")) == MD5_HQ_V1, "hq v1.csv not pinned")
    check(set(r[0] for r in hq1) <= eval_frames, "hq v1: box on a non-eval frame")
    vendor = load_vendor_jsonl(JOB_COPIES["v2"]["output"])
    check(set(vendor) == eval_frames, "hq eval vendor output does not cover exactly the eval frames")
    check({k: v for k, v in vendor.items() if v} == group_boxes(hq1),
          "hq v1.csv differs from the vendor jsonl in /data/esteban/tmp/hq_eval")
    v2 = sorted(train_rows + hq1, key=box_key)
    check(len(v2) == EXPECT_BOXES["v2"], f"v2 rows {len(v2)}")

    # v3: hq v2 on the eval frames, v2 rows on the rest; reviewed = v2.frames.txt
    hq2 = load_pii_data_boxes("faceback_hq/boxes/v2.csv")
    check(md5_of(os.path.join(PII_DATA, "faceback_hq/boxes/v2.csv")) == MD5_HQ_V2, "hq v2.csv not pinned")
    check(set(r[0] for r in hq2) <= eval_frames, "hq v2: box on a non-eval frame")
    rev_path = os.path.join(PII_DATA, "faceback_hq/boxes/v2.frames.txt")
    check(md5_of(rev_path) == MD5_HQ_V2_FRAMES, "hq v2.frames.txt not pinned")
    v3_reviewed = set(open(rev_path, encoding="utf-8").read().split())
    check(len(v3_reviewed) == N_V3_REVIEWED and v3_reviewed <= eval_frames, "v2.frames.txt set")
    vendor2 = load_vendor_jsonl(JOB_COPIES["v3"]["output"])
    check(set(vendor2) == v3_reviewed, "round-2 vendor output frames != v2.frames.txt")
    hq1_by, hq2_by = group_boxes(hq1), group_boxes(hq2)
    check({k: v for k, v in vendor2.items() if v} == {k: v for k, v in hq2_by.items() if k in v3_reviewed},
          "hq v2.csv on the reviewed frames differs from the vendor jsonl in /data/esteban/tmp/pii/fb_hq_2")
    check({k: v for k, v in hq1_by.items() if k not in v3_reviewed}
          == {k: v for k, v in hq2_by.items() if k not in v3_reviewed},
          "hq v2.csv changes a frame outside v2.frames.txt")
    v3 = sorted(train_rows + hq2, key=box_key)
    check(len(v3) == EXPECT_BOXES["v3"], f"v3 rows {len(v3)}")

    for version, rows, reviewed in (("v1", v1, set(all_images)), ("v2", v2, eval_frames), ("v3", v3, v3_reviewed)):
        write_csv(dst("boxes", version, "boxes.csv"), BOX_COLS, rows)
        write_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"],
                  [[im, "1" if im in reviewed else "0"] for im in all_images])
        copy_job_files(version)
        n_img = len({r[0] for r in rows})
        print(f"boxes/{version}: {len(rows)} boxes on {n_img} images, {len(reviewed)} reviewed")
    check(md5_of(dst("boxes", "v1", "boxes.csv")) == MD5_FB45_V1, "written v1 boxes.csv md5")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def _md5_check(job: tuple[str, str, bool]):
    """(name, expected md5 from frames.csv, compare_src) -> (name, error or None)."""
    name, expected, compare_src = job
    s = os.path.join(SRC, "images", name)
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
    size_of = {r[0]: int(r[5]) for r in frames}
    md5_of_image = {r[0]: r[6] for r in frames}
    for r in frames:
        if r[0] != f"{r[1]}_c{r[2]}_{r[3]}_f{int(r[4]):06d}.jpg":
            fail(f"frames.csv: {r[0]} does not match its fields")
            break
    if any(not MD5_HEX.match(r[6]) or not r[5].isdigit() for r in frames):
        fail("frames.csv: md5 not 32 hex chars or size not an integer")
    if os.path.isfile(os.path.join(SRC, "upload_manifest.jsonl")):
        if load_manifest() != [[r[0], r[1], r[2], r[3], r[4], r[7], r[5], r[8]] for r in frames]:
            fail("frames.csv differs from upload_manifest.jsonl")
    if os.path.isfile(PII_DATA_FRAMES):
        if load_pii_data_md5() != {r[0]: (r[5], r[6]) for r in frames}:
            fail("frames.csv size/md5 differ from pii-data frames.csv (dataset=faceback_45)")
    else:
        print(f"  note: {PII_DATA_FRAMES} not present, size/md5 columns not cross-checked")

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(roles) != N_SESSIONS or len(split) != N_SESSIONS:
        fail(f"split.csv has {len(split)} rows, expected {N_SESSIONS}")
    if {r[1] for r in frames} != set(roles) or set(roles.values()) - {"train", "eval"}:
        fail("split.csv sessions do not match frames.csv, or bad role")
    if os.path.isdir(os.path.join(SRC, "splits")) and load_splits() != roles:
        fail("split.csv differs from the source split files")
    eval_frames = {r[0] for r in frames if roles.get(r[1]) == "eval"}
    if len(eval_frames) != N_EVAL_FRAMES:
        fail(f"{len(eval_frames)} eval frames, expected {N_EVAL_FRAMES}")

    # images: every file present, no extras, md5 equal to the source
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
        compare_src = os.path.isdir(os.path.join(SRC, "images"))
        if not compare_src:
            print(f"  note: source images {SRC}/images not present; md5 checked against frames.csv only")
        jobs = [(n, md5_of_image[n], compare_src) for n in images]
        bad = 0
        with Pool(args.jobs) as pool:
            for i, (name, err) in enumerate(pool.imap_unordered(_md5_check, jobs, chunksize=64), 1):
                if err:
                    bad += 1
                    if bad <= 20:
                        fail(f"images/{name}: {err}")
                if i % 10000 == 0:
                    print(f"  md5: {i}/{len(images)} ({time.time() - t0:.0f}s)")
        if bad:
            fail(f"{bad} images differ from frames.csv md5" + (" or the source" if compare_src else ""))
        else:
            print(f"images: {len(images)} md5 equal to frames.csv"
                  + (" and the source" if compare_src else "") + f" ({time.time() - t0:.0f}s)")

    # box versions
    boxes_by_version = {}
    for version in ("v1", "v2", "v3"):
        vf = read_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"])
        if [r[0] for r in vf] != images:
            fail(f"boxes/{version}/frames.csv image column != frames.csv")
        if set(r[1] for r in vf) - {"0", "1"}:
            fail(f"boxes/{version}/frames.csv: reviewed not 0/1")
        reviewed = {r[0] for r in vf if r[1] == "1"}
        rows = read_csv(dst("boxes", version, "boxes.csv"), BOX_COLS)
        boxes_by_version[version] = rows
        if len(rows) != EXPECT_BOXES[version]:
            fail(f"boxes/{version}/boxes.csv has {len(rows)} rows, expected {EXPECT_BOXES[version]}")
        if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
            fail(f"boxes/{version}/boxes.csv not sorted by (image, x1, y1)")
        if {r[0] for r in rows} - image_set:
            fail(f"boxes/{version}/boxes.csv: box on an image not in frames.csv")
        if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
            fail(f"boxes/{version}/boxes.csv: coordinate not one decimal or bad ignore")
        eval_rows = [r for r in rows if r[0] in eval_frames]
        train_rows = [r for r in rows if r[0] not in eval_frames]
        if version == "v1":
            if md5_of(dst("boxes", "v1", "boxes.csv")) != MD5_FB45_V1:
                fail("boxes/v1/boxes.csv md5 != pii-data faceback_45/boxes/v1.csv")
            expected_reviewed = image_set
            v1_train = train_rows
        else:
            ref_md5, ref_name = (MD5_HQ_V1, "v1.csv") if version == "v2" else (MD5_HQ_V2, "v2.csv")
            if md5_bytes(csv_bytes(BOX_COLS, eval_rows)) != ref_md5:
                fail(f"boxes/{version}/boxes.csv eval-frame rows != pii-data faceback_hq/boxes/{ref_name}")
            if train_rows != v1_train:
                fail(f"boxes/{version}/boxes.csv train rows != v1 train rows")
            if version == "v2":
                expected_reviewed = eval_frames
            else:
                out = [os.path.join(DST, "boxes", "v3", "job", "output", os.path.basename(p))
                       for p in JOB_COPIES["v3"]["output"]]
                expected_reviewed = set(load_vendor_jsonl(out))
                if len(expected_reviewed) != N_V3_REVIEWED:
                    fail(f"v3 job/output covers {len(expected_reviewed)} frames, expected {N_V3_REVIEWED}")
                if md5_bytes(("\n".join(sorted(expected_reviewed)) + "\n").encode()) != MD5_HQ_V2_FRAMES:
                    fail("v3 job/output frame set != pii-data faceback_hq/boxes/v2.frames.txt")
        if reviewed != expected_reviewed:
            fail(f"boxes/{version}/frames.csv reviewed set: {len(reviewed)} frames, "
                 f"expected {len(expected_reviewed)}")
        # job copies
        for sub, files in JOB_COPIES[version].items():
            for s in files:
                d = dst("boxes", version, "job", sub, os.path.basename(s))
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
    for p in ("_download.log", "_launch_time.txt"):
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
    b.add_argument("--no-images", action="store_true", help="skip the image copy (e.g. rsync runs elsewhere)")
    b.add_argument("--frames-only", action="store_true",
                   help="rewrite the top-level frames.csv only (implies --no-images; nothing else touched)")
    v = sub.add_parser("verify")
    v.add_argument("--no-md5", action="store_true",
                   help="skip the per-image md5 pass (frames.csv column, and the source while it exists)")
    v.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    {"build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build /data/esteban/pii/datasets/pii_frames in the pii2 layout (PII-1315, PII-1338).
Stdlib only, idempotent; every source is read-only.

    build            copy images (skipped with --no-images), write frames.csv, split.csv,
                     boxes/v1 (frames.csv, boxes.csv)
    verify           md5 every image against the frames.csv md5 column (and against the
                     source while it exists), check row counts, box file, reviewed set;
                     exit 1 on any mismatch

Source tree (nested): pii/datasets/pii_frames/<session>/chunk_<NNN>/<view>/f<NNNNNN>.jpg,
10,249 files, 25 sessions, view in {lview, rview}. Flattened into one images/ directory as
<session>_c<NNN>_<view>_f<NNNNNN>.jpg, the same rule as the OSS keys in
data/oss_pii_keys.jsonl (the build asserts every flattened name equals its oss_key
basename and every local_path equals the source file). The view token is kept exactly
as on disk (lview/rview), not normalized to left/right.

Layout written (README.md files are written by hand, not here):

    images/                     byte copies (no links)
    frames.csv                  image,session,chunk,view,frame_idx,size,md5,oss_key
                                (size and md5 from pii-data frames.csv dataset=pii_frames;
                                size asserted equal to the file on disk)
    split.csv                   session,role: every session is train
    boxes/v1/frames.csv         image,reviewed (1 on all 10,249)
    boxes/v1/boxes.csv          image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal,
                                sorted by image,x1,y1; images with no box have no row)

v1: pii-data datasets/pii_frames/boxes/v1.csv (19,579 rows on 8,761 images, 2,704 ignore),
    copied byte for byte; cross-checked against pii-data frames.csv n_boxes and, when the
    files are present, against the pii/ labelv2 blocks of training/manifests/train_Z2.txt
    and train_W.txt (the manifests the boxes were recovered from). job/prelabels and
    job/output: none on disk (the vendor drop is lost).
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
from pii_root import CODE_ROOT, LEGACY_ROOT, PII_ROOT  # noqa: E402

SRC = LEGACY_ROOT + "/datasets/pii_frames"    # PII-1448: the old tree
PII_DATA_BOXES = "/home/esteban/repos/pii-data/datasets/pii_frames/boxes/v1.csv"
PII_DATA_FRAMES = "/home/esteban/repos/pii-data/frames.csv"
OSS_KEYS = os.path.join(STORE, "data", "oss_pii_keys.jsonl")
MANIFESTS = [os.path.join(CODE_ROOT, "training", "manifests", n)
             for n in ("train_Z2.txt", "train_W.txt")]
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DST = PII2_ROOT + "/datasets/pii_frames"
WRITE_ROOT = PII2_ROOT + "/"

W, H = 2328, 1748
N_IMAGES = 10249
N_SESSIONS = 25
N_BOXES = 19579
N_IMAGES_WITH_BOX = 8761
N_IGNORE = 2704
MD5_PII_DATA_BOXES = "3b8cb6d1a325aeefb59987e47fac588c"

FRAME_COLS = ["image", "session", "chunk", "view", "frame_idx", "size", "md5", "oss_key"]
OSS_PREFIX = "pii/data/pii_frames/"   # PII-1413: where the image bytes live on OSS
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
SESSION_RE = re.compile(r"^\d{8}_\d{6}_[A-Z]{6}$")
CHUNK_RE = re.compile(r"^chunk_(\d{3})$")
FRAME_RE = re.compile(r"^f(\d{6})\.jpg$")
VIEWS = ("lview", "rview")
KPS_PLACEHOLDER = ["-1.0"] * 15

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


def flat_name(session: str, chunk: str, view: str, frame_idx: str) -> str:
    return f"{session}_c{int(chunk):03d}_{view}_f{int(frame_idx):06d}.jpg"


def src_path(session: str, chunk: str, view: str, frame_idx: str) -> str:
    return os.path.join(SRC, session, f"chunk_{int(chunk):03d}", view, f"f{int(frame_idx):06d}.jpg")


def manifest_name(session: str, chunk: str, view: str, frame_idx: str) -> str:
    return f"pii/{session}/chunk_{int(chunk):03d}/{view}/f{int(frame_idx):06d}.jpg"


# ---------------------------------------------------------------- sources

def walk_source() -> dict[str, dict]:
    """{flat image name: {session, chunk, view, frame_idx}} from the nested source tree.
    Every regular file under SRC must be a frame jpg at depth 4; names asserted unique."""
    out: dict[str, dict] = {}
    n_files = 0
    for root, dirs, files in os.walk(SRC):
        dirs.sort()
        rel = os.path.relpath(root, SRC)
        parts = [] if rel == "." else rel.split(os.sep)
        for f in sorted(files):
            n_files += 1
            check(len(parts) == 3, f"unexpected file depth: {os.path.join(root, f)}")
            session, chunk_dir, view = parts
            m_c, m_f = CHUNK_RE.match(chunk_dir), FRAME_RE.match(f)
            check(SESSION_RE.match(session) is not None and m_c is not None and view in VIEWS and m_f is not None,
                  f"unexpected source path: {os.path.join(root, f)}")
            check(not os.path.islink(os.path.join(root, f)), f"source image is a symlink: {os.path.join(root, f)}")
            name = flat_name(session, m_c.group(1), view, m_f.group(1))
            check(name not in out, f"flattened name collision: {name}")
            out[name] = {"session": session, "chunk": m_c.group(1), "view": view, "frame_idx": str(int(m_f.group(1)))}
    check(n_files == N_IMAGES, f"source tree has {n_files} files, expected {N_IMAGES}")
    return out


def load_pii_data_frames() -> dict[str, tuple[str, str, str, int, dict]]:
    """{image: (size, md5, role, n_boxes, fields)} from pii-data frames.csv, dataset pii_frames."""
    out: dict[str, tuple] = {}
    with open(PII_DATA_FRAMES, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["dataset"] != "pii_frames":
                continue
            check(d["image"] not in out, f"pii-data frames.csv: duplicate {d['image']}")
            check(MD5_HEX.match(d["md5"]) is not None and d["size"].isdigit(),
                  f"pii-data frames.csv: bad size/md5 for {d['image']}")
            check(d["image_ref"] == "", f"pii-data frames.csv: {d['image']} has an image_ref")
            check((d["width"], d["height"]) == (str(W), str(H)), f"pii-data frames.csv: dims of {d['image']}")
            fields = {"session": d["session_id"], "chunk": f"{int(d['chunk']):03d}", "view": d["view"],
                      "frame_idx": str(int(d["frame_idx"]))}
            check(d["image"] == flat_name(**fields), f"pii-data frames.csv: {d['image']} does not match its fields")
            out[d["image"]] = (d["size"], d["md5"], d["role"], int(d["n_boxes"]), fields)
    check(len(out) == N_IMAGES, f"pii-data frames.csv has {len(out)} pii_frames rows, expected {N_IMAGES}")
    return out


def load_oss_keys() -> dict[str, tuple[str, int, str]]:
    """{oss basename: (local_path, size, md5)} for dataset pii_frames."""
    out: dict[str, tuple[str, int, str]] = {}
    for ln in open(OSS_KEYS, encoding="utf-8"):
        d = json.loads(ln)
        if d["dataset"] != "pii_frames":
            continue
        name = d["oss_key"].rsplit("/", 1)[-1]
        check(name not in out, f"oss keys: duplicate basename {name}")
        out[name] = (d["local_path"], d["size"], d["md5"])
    check(len(out) == N_IMAGES, f"oss keys list {len(out)} pii_frames objects, expected {N_IMAGES}")
    return out


def build_frames() -> list[list[str]]:
    """FRAME_COLS rows sorted by image. Asserts source tree, pii-data frames.csv and the
    OSS key map agree on names, fields, paths, size and md5."""
    src = walk_source()
    pd = load_pii_data_frames()
    check(set(src) == set(pd), "source tree and pii-data frames.csv list different images")
    if os.path.isfile(OSS_KEYS):
        oss = load_oss_keys()
        check(set(oss) == set(src), "oss_pii_keys.jsonl basenames != flattened source names")
        for name, (local_path, size, md5) in oss.items():
            m = src[name]
            check(local_path == src_path(m["session"], m["chunk"], m["view"], m["frame_idx"]),
                  f"oss keys: local_path of {name} is not the source file")
            check((str(size), md5) == (pd[name][0], pd[name][1]), f"oss keys: size/md5 of {name} != pii-data")
        print(f"  oss_pii_keys.jsonl: {len(oss)} basenames equal to the flattened names, local_path and size/md5 agree")
    else:
        print(f"  note: {OSS_KEYS} not present, flattened names not checked against the OSS keys")
    rows = []
    for image in sorted(src):
        m = src[image]
        size, md5, role, _, fields = pd[image]
        check(fields == m, f"pii-data fields differ from the source path for {image}")
        check(role == "train", f"pii-data role {role} for {image}")
        rows.append([image, m["session"], m["chunk"], m["view"], m["frame_idx"], size, md5,
                     OSS_PREFIX + image])
    return rows


def load_pii_data_boxes() -> list[list[str]]:
    check(md5_of(PII_DATA_BOXES) == MD5_PII_DATA_BOXES, f"pii-data {PII_DATA_BOXES} is not the pinned file")
    rows = read_csv(PII_DATA_BOXES, BOX_COLS)
    check([box_key(r) for r in rows] == sorted(box_key(r) for r in rows), "pii-data v1.csv not sorted")
    return rows


def load_manifest_pii(path: str) -> dict[str, list[tuple]]:
    """{flat image: sorted box tuples} from the pii/ labelv2 blocks of a training manifest."""
    out: dict[str, list[tuple]] = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if ln.startswith("#"):
                parts = ln.split()
                check(len(parts) == 4, f"{path}: bad header {ln!r}")
                cur = None
                if parts[1].startswith("pii/"):
                    check((int(parts[2]), int(parts[3])) == (W, H), f"{path}: dims {parts[1]}")
                    p = parts[1].split("/")
                    m_c, m_f = CHUNK_RE.match(p[2]), FRAME_RE.match(p[4])
                    check(len(p) == 5 and m_c is not None and p[3] in VIEWS and m_f is not None,
                          f"{path}: unexpected pii block name {parts[1]}")
                    name = flat_name(p[1], m_c.group(1), p[3], m_f.group(1))
                    check(name not in out, f"{path}: duplicate {parts[1]}")
                    cur = out.setdefault(name, [])
            elif cur is not None:
                v = ln.split()
                if len(v) == 19:
                    check(v[4:] == KPS_PLACEHOLDER, f"{path}: non-placeholder keypoints: {ln!r}")
                    ignore = "0"
                elif len(v) == 5 and v[4] == "1":
                    ignore = "1"
                else:
                    die(f"{path}: box line with {len(v)} fields: {ln!r}")
                cur.append(tuple(fmt(float(x)) for x in v[:4]) + (ignore,))
    for bl in out.values():
        bl.sort(key=lambda t: (float(t[0]), float(t[1])))
    return out


def check_v1_against_sources(rows: list[list[str]], image_set: set[str], pd) -> None:
    by = group_boxes(rows)
    check(all(len(by.get(img, [])) == v[3] for img, v in pd.items()),
          "v1 per-image counts != pii-data frames.csv n_boxes")
    for path in MANIFESTS:
        if not os.path.isfile(path):
            print(f"  note: {path} not present, v1 not cross-checked against it")
            continue
        man = load_manifest_pii(path)
        check(set(man) == image_set, f"{path}: pii blocks cover {len(man)} frames, expected {len(image_set)}")
        check({k: v for k, v in man.items() if v} == by, f"v1 boxes != pii blocks of {path}")
        print(f"  v1: equal to the {len(man)} pii blocks of {os.path.basename(path)}")


# ---------------------------------------------------------------- build

def copy_images(frames: list[list[str]]) -> None:
    dst_dir = dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    t0 = time.time()
    for i, r in enumerate(frames, 1):
        name, size = r[0], int(r[5])
        s = src_path(r[1], r[2], r[3], r[4])
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


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    frames = build_frames()
    all_images = [r[0] for r in frames]
    image_set = set(all_images)
    if not args.no_images:
        copy_images(frames)

    sessions = sorted({r[1] for r in frames})
    check(len(sessions) == N_SESSIONS, f"{len(sessions)} sessions, expected {N_SESSIONS}")
    write_csv(dst("frames.csv"), FRAME_COLS, frames)
    write_csv(dst("split.csv"), ["session", "role"], [[s, "train"] for s in sessions])
    n_view = {v: sum(1 for r in frames if r[3] == v) for v in VIEWS}
    print(f"frames.csv {len(frames)} rows ({n_view['lview']} lview, {n_view['rview']} rview), "
          f"split.csv {len(sessions)} sessions, all train")

    v1 = load_pii_data_boxes()
    check(len(v1) == N_BOXES, f"v1 rows {len(v1)}")
    check({r[0] for r in v1} <= image_set, "v1: box on an image not in frames.csv")
    check(len({r[0] for r in v1}) == N_IMAGES_WITH_BOX, "v1: images with a box != expected")
    check(sum(1 for r in v1 if r[5] == "1") == N_IGNORE, "v1: ignore rows != expected")
    check_v1_against_sources(v1, image_set, load_pii_data_frames())

    write_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS, v1)
    check(md5_of(dst("boxes", "v1", "boxes.csv")) == MD5_PII_DATA_BOXES, "written v1 boxes.csv md5")
    write_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"], [[im, "1"] for im in all_images])
    n_zero = len(image_set - {r[0] for r in v1})
    print(f"boxes/v1: {len(v1)} boxes on {len({r[0] for r in v1})} images, {n_zero} images with no box, "
          f"{len(all_images)} reviewed; byte-equal to pii-data v1.csv")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def _md5_check(job: tuple[str, str, str, bool]):
    """(name, source path, expected md5 from frames.csv, compare_src) -> (name, error or None)."""
    name, s, expected, compare_src = job
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
    for r in frames:
        if r[3] not in VIEWS or r[0] != flat_name(r[1], r[2], r[3], r[4]) or len(r[2]) != 3:
            fail(f"frames.csv: {r[0]} does not match its fields")
            break
    if any(not MD5_HEX.match(r[6]) or not r[5].isdigit() for r in frames):
        fail("frames.csv: md5 not 32 hex chars or size not an integer")
    src_present = os.path.isdir(SRC)
    if src_present:
        src = walk_source()
        if src != {r[0]: {"session": r[1], "chunk": r[2], "view": r[3], "frame_idx": r[4]} for r in frames}:
            fail("frames.csv rows differ from the source tree")
    else:
        print("  note: source tree not present, frames.csv fields not cross-checked")
    if os.path.isfile(PII_DATA_FRAMES):
        pd = load_pii_data_frames()
        if {k: (v[0], v[1]) for k, v in pd.items()} != {r[0]: (r[5], r[6]) for r in frames}:
            fail("frames.csv size/md5 differ from pii-data frames.csv")
    else:
        pd = None
        print(f"  note: {PII_DATA_FRAMES} not present, size/md5 not cross-checked")
    if os.path.isfile(OSS_KEYS):
        oss = load_oss_keys()
        if {k: (str(v[1]), v[2]) for k, v in oss.items()} != {r[0]: (r[5], r[6]) for r in frames}:
            fail("frames.csv names/size/md5 differ from oss_pii_keys.jsonl")
    else:
        print(f"  note: {OSS_KEYS} not present, names not cross-checked against the OSS keys")

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
        if not src_present:
            print("  note: source images not present; md5 checked against frames.csv only")
        jobs = [(r[0], src_path(r[1], r[2], r[3], r[4]), md5_of_image[r[0]], src_present) for r in frames]
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
                  f"{' and to the source' if src_present else ''} ({time.time() - t0:.0f}s)")

    # boxes/v1
    vf = read_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"])
    if [r[0] for r in vf] != images:
        fail("boxes/v1/frames.csv image column != frames.csv")
    if {r[1] for r in vf} != {"1"}:
        fail("boxes/v1/frames.csv: reviewed must be 1 on every frame")
    rows = read_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS)
    if len(rows) != N_BOXES:
        fail(f"boxes/v1/boxes.csv has {len(rows)} rows, expected {N_BOXES}")
    if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
        fail("boxes/v1/boxes.csv not sorted by (image, x1, y1)")
    if {r[0] for r in rows} - image_set:
        fail("boxes/v1/boxes.csv: box on an image not in frames.csv")
    if len({r[0] for r in rows}) != N_IMAGES_WITH_BOX:
        fail(f"boxes/v1/boxes.csv: {len({r[0] for r in rows})} images with a box, expected {N_IMAGES_WITH_BOX}")
    if sum(1 for r in rows if r[5] == "1") != N_IGNORE:
        fail(f"boxes/v1/boxes.csv: ignore rows != {N_IGNORE}")
    if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
        fail("boxes/v1/boxes.csv: coordinate not one decimal or bad ignore")
    if md5_of(dst("boxes", "v1", "boxes.csv")) != MD5_PII_DATA_BOXES:
        fail("boxes/v1/boxes.csv md5 != pii-data datasets/pii_frames/boxes/v1.csv")
    if pd is not None:
        by = group_boxes(rows)
        if any(len(by.get(img, [])) != v[3] for img, v in pd.items()):
            fail("boxes/v1/boxes.csv per-image counts != pii-data frames.csv n_boxes")
    for path in MANIFESTS:
        if not os.path.isfile(path):
            print(f"  note: {path} not present, v1 not cross-checked against it")
            continue
        man = load_manifest_pii(path)
        if set(man) != image_set or {k: v for k, v in man.items() if v} != group_boxes(rows):
            fail(f"boxes/v1/boxes.csv != pii blocks of {path}")
    print(f"boxes/v1: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
          f"{len(vf)} reviewed ({time.time() - t0:.0f}s)")

    for p in ("README.md", "boxes/v1/README.md"):
        if not os.path.isfile(dst(p)):
            fail(f"missing {p}")
    for p in ("_download.log", "_launch_time.txt", "_probe.txt", ".complete"):
        if os.path.exists(dst(p)):
            fail(f"{p} must not be in the dataset")
    for p in ("README.md", "boxes/v1/README.md"):
        if os.path.isfile(dst(p)) and "—" in open(dst(p), encoding="utf-8").read():
            fail(f"{p} contains an em-dash")

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

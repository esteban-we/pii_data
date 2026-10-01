#!/usr/bin/env python3
"""Build the three WIDER-derived sets under /data/esteban/pii/datasets in the pii2 layout
(PII-1315, PII-1337). Stdlib only, idempotent; every source is read-only.

    --set wider_face | wider_fisheye_target | wider_fisheye_fill   (required)
    build            copy images (skipped with --no-images), write frames.csv, split.csv,
                     boxes/v1 (frames.csv, boxes.csv) and, for the fisheye sets, derivation/
    verify           md5 every image against the frames.csv md5 column (and against the
                     source while it exists), check row counts, box files, reviewed set and
                     derivation copies; exit 1 on any mismatch

Three separate image sets (user decision 2026-09-21: separate sets, not versions):

    wider_face             pii/datasets/wider_face/WIDER_train/images/<scene>/<name>.jpg
                           flattened to images/<name>.jpg (12,879; basenames asserted unique).
                           size, md5 from pii-data frames.csv (dataset wider_face; size asserted
                           equal to the file on disk). boxes/v1 = pii-data
                           datasets/wider_face/boxes/v1.csv byte for byte (159,390 rows, 2,399
                           ignore), cross-checked against wider_train_labelv2.txt (the file
                           training/wider_conv.py writes) as per-image sorted box sets.
    wider_fisheye_target   pii/datasets/wider_fisheye/images/<event>/<name>.jpg (the
                           warp_wider_fisheye.py --version 1 build, zoom policy "target"),
                           flattened the same way. md5 computed from the source file.
                           boxes/v1 from wider_fisheye_labelv2.txt (153,203 boxes, no ignore).
    wider_fisheye_fill     same for pii/datasets/wider_fisheye_v2 (--version 2, zoom policy
                           "fill"), 152,235 boxes.

Layout written (README.md files are written by hand, not here):

    images/                  byte copies (no links), flat
    frames.csv               image,session,chunk,eye,frame_idx,size,md5,scene,width,height,src_path,
                             oss_key (PII-1413: pii/data/<set>/<image> on oss://algorithm-datasets)
                             session = scene = the WIDER event directory name (WIDER has no
                             sessions); chunk, eye, frame_idx empty; width,height from the
                             labelv2 header (wider_face: also asserted equal to pii-data)
    split.csv                session,role: every one of the 61 scenes is train
    boxes/v1/frames.csv      image,reviewed (1 on every image)
    boxes/v1/boxes.csv       image,x1,y1,x2,y2,ignore (pixel xyxy, 1 decimal, sorted by image,x1,y1)
    boxes/v1/job/            none (public WIDER annotation / machine-derived)
    derivation/              fisheye sets only: byte copies of wider_fisheye_params.json,
                             wider_fisheye_stats.json, wider_fisheye_labelv2.txt and preview/
"""
import argparse
import csv
import hashlib
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
from pii_root import PII_ROOT  # noqa: E402

# Provenance (PII-1682): the pre-PII-1315 tree this script read. The project retired it, so the
# paths below record where the data came from; they are not a tree to read today.
LEGACY_ROOT = "/data/esteban/pii_backup"


# The manifest/index is the ledger and was never rewritten for the PII-1448 rename: paths in it
# read /data/esteban/pii/... meaning the tree that became LEGACY_ROOT above (PII-1682).
OLD_ROOT_PREFIX = "/data/esteban/pii/"


def resolve_legacy(path: str) -> str:
    """An absolute path recorded before the PII-1448 rename, as it resolved after it."""
    path = str(path)
    if path.startswith(OLD_ROOT_PREFIX) and not path.startswith(LEGACY_ROOT + "/"):
        return LEGACY_ROOT + "/" + path[len(OLD_ROOT_PREFIX):]
    return path

PII = LEGACY_ROOT + "/datasets"    # PII-1448: the old tree, now /data/esteban/pii_backup
PII_DATA = "/home/esteban/repos/pii-data/datasets"
PII_DATA_FRAMES = "/home/esteban/repos/pii-data/frames.csv"
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DST_ROOT = PII2_ROOT + "/datasets"
WRITE_ROOT = PII2_ROOT + "/"

N_IMAGES = 12879
N_SCENES = 61
FE_W, FE_H = 2328, 1748
PINNED_V1 = ("wider_face/boxes/v1.csv", "ae71452bb736650de9ebc81c8bc0cf84")

SETS = {
    "wider_face": dict(
        src_root=os.path.join(PII, "wider_face"),
        src_images=os.path.join(PII, "wider_face", "WIDER_train", "images"),
        labelv2=os.path.join(PII, "wider_face", "wider_train_labelv2.txt"),
        prefix="wider/", n_boxes=159390, n_ignore=2399, md5_from="pii-data", dims=None,
        derivation=[]),
    "wider_fisheye_target": dict(
        src_root=os.path.join(PII, "wider_fisheye"),
        src_images=os.path.join(PII, "wider_fisheye", "images"),
        labelv2=os.path.join(PII, "wider_fisheye", "wider_fisheye_labelv2.txt"),
        prefix="wider_fe/", n_boxes=153203, n_ignore=0, md5_from="source", dims=(FE_W, FE_H),
        derivation=["wider_fisheye_params.json", "wider_fisheye_stats.json", "wider_fisheye_labelv2.txt"]
                   + [f"preview/{i:02d}_{n}.jpg" for i, n in enumerate(
                       ["tinyface_center"] * 2 + ["tinyface_periph"] * 2 + ["midface_center"] * 2
                       + ["midface_periph"] * 2 + ["bigface_center"] * 2 + ["bigface_periph"] * 2)]),
}
SETS["wider_fisheye_fill"] = dict(
    SETS["wider_fisheye_target"],
    src_root=os.path.join(PII, "wider_fisheye_v2"),
    src_images=os.path.join(PII, "wider_fisheye_v2", "images"),
    labelv2=os.path.join(PII, "wider_fisheye_v2", "wider_fisheye_labelv2.txt"),
    prefix="wider_fe2/", n_boxes=152235)

FRAME_COLS = ["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "scene", "width", "height",
              "src_path", "oss_key"]
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
KPS_PLACEHOLDER = ["-1.0"] * 15

SET = None      # the selected SETS entry
DST = None      # /data/esteban/pii/datasets/<set>
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


def full_key(t):
    return tuple(float(x) for x in t[:4]) + (t[4],)


def group_boxes(rows) -> dict[str, list[tuple]]:
    """{image: box tuples sorted by full key}; used for order-insensitive comparisons."""
    d: dict[str, list[tuple]] = {}
    for r in rows:
        d.setdefault(r[0], []).append(tuple(r[1:]))
    for v in d.values():
        v.sort(key=full_key)
    return d


# ---------------------------------------------------------------- sources

def load_labelv2(path: str, prefix: str):
    """-> ({name: scene}, {name: (w, h)}, {name: [box tuples]}) from a labelv2 file whose
    headers are '# <prefix><scene>/<name>.jpg W H'. Box lines: 19 fields (4 coords + 15
    placeholder keypoints, ignore=0) or 5 fields ending in 1 (ignore=1)."""
    scene: dict[str, str] = {}
    dims: dict[str, tuple[int, int]] = {}
    boxes: dict[str, list[tuple]] = {}
    cur = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if ln.startswith("#"):
                parts = ln.split()
                check(len(parts) == 4 and parts[1].startswith(prefix), f"{path}: bad header {ln!r}")
                rel = parts[1][len(prefix):]
                check(rel.count("/") == 1 and rel.endswith(".jpg"), f"{path}: unexpected path {rel}")
                sc, name = rel.split("/")
                check(name not in scene, f"{path}: {name} appears twice (flat images/ needs unique names)")
                scene[name] = sc
                dims[name] = (int(parts[2]), int(parts[3]))
                cur = boxes.setdefault(name, [])
            else:
                check(cur is not None, f"{path}: box line before any header")
                v = ln.split()
                if len(v) == 19:
                    check(v[4:] == KPS_PLACEHOLDER, f"{path}: non-placeholder keypoints: {ln!r}")
                    ignore = "0"
                elif len(v) == 5 and v[4] == "1":
                    ignore = "1"
                else:
                    die(f"{path}: box line with {len(v)} fields: {ln!r}")
                cur.append(tuple(fmt(float(x)) for x in v[:4]) + (ignore,))
    for bl in boxes.values():
        bl.sort(key=full_key)
    check(len(scene) == N_IMAGES, f"{path}: {len(scene)} images, expected {N_IMAGES}")
    return scene, dims, boxes


def list_source_images() -> dict[str, str]:
    """{name: scene} from the source images tree; asserts one level of scene dirs, jpg only,
    unique basenames, and exactly N_SCENES scenes."""
    root = SET["src_images"]
    out: dict[str, str] = {}
    scenes = sorted(os.listdir(root))
    check(len(scenes) == N_SCENES, f"{root}: {len(scenes)} entries, expected {N_SCENES} scene dirs")
    for sc in scenes:
        d = os.path.join(root, sc)
        check(os.path.isdir(d) and not os.path.islink(d), f"{d}: not a plain directory")
        for name in os.listdir(d):
            p = os.path.join(d, name)
            check(os.path.isfile(p) and not os.path.islink(p) and name.endswith(".jpg"), f"{p}: not a plain jpg")
            check(name not in out, f"basename {name} appears in scenes {out.get(name)} and {sc}")
            out[name] = sc
    check(len(out) == N_IMAGES, f"{root}: {len(out)} images, expected {N_IMAGES}")
    return out


def load_pii_data_frames() -> dict[str, tuple[str, str, int, int, int]]:
    """{image: (size, md5, width, height, n_boxes)} from pii-data frames.csv, dataset wider_face."""
    out: dict[str, tuple[str, str, int, int, int]] = {}
    with open(PII_DATA_FRAMES, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["dataset"] != "wider_face":
                continue
            check(d["image"] not in out, f"pii-data frames.csv: duplicate {d['image']}")
            check(MD5_HEX.match(d["md5"]) is not None and d["size"].isdigit(),
                  f"pii-data frames.csv: bad size/md5 for {d['image']}")
            check(d["role"] == "train" and d["image_ref"] == "", f"pii-data frames.csv: {d['image']} role/image_ref")
            out[d["image"]] = (d["size"], d["md5"], int(d["width"]), int(d["height"]), int(d["n_boxes"]))
    check(len(out) == N_IMAGES, f"pii-data frames.csv has {len(out)} wider_face rows, expected {N_IMAGES}")
    return out


def _src_md5(job: tuple[str, str]):
    name, path = job
    return name, os.path.getsize(path), md5_of(path)


def build_frames(jobs: int) -> list[list[str]]:
    """FRAME_COLS rows sorted by image."""
    scene_fs = list_source_images()
    scene_lb, dims, _ = load_labelv2(SET["labelv2"], SET["prefix"])
    check(scene_fs == scene_lb, "source images tree and labelv2 headers list different scene/name pairs")
    if SET["dims"] is not None:
        check(set(dims.values()) == {SET["dims"]}, f"labelv2 dims are not all {SET['dims']}")
    names = sorted(scene_fs)
    src_of = {n: os.path.join(SET["src_images"], scene_fs[n], n) for n in names}
    if SET["md5_from"] == "pii-data":
        pd = load_pii_data_frames()
        check(set(pd) == set(names), "pii-data frames.csv and the source tree list different images")
        meta = {}
        for n in names:
            size, md5, w, h, _ = pd[n]
            check(os.path.getsize(src_of[n]) == int(size), f"{src_of[n]}: size != pii-data size {size}")
            check((w, h) == dims[n], f"{n}: pii-data dims {(w, h)} != labelv2 {dims[n]}")
            meta[n] = (size, md5)
        print(f"  size/md5 from pii-data frames.csv for {len(meta)} images")
    else:
        t0 = time.time()
        meta = {}
        with Pool(jobs) as pool:
            for n, size, md5 in pool.imap_unordered(_src_md5, [(n, src_of[n]) for n in names], chunksize=64):
                meta[n] = (str(size), md5)
        print(f"  size/md5 computed from {len(meta)} source images ({time.time() - t0:.0f}s)")
    rows = []
    for n in names:
        size, md5 = meta[n]
        w, h = dims[n]
        rows.append([n, scene_fs[n], "", "", "", size, md5, scene_fs[n], str(w), str(h), src_of[n],
                     f"pii/data/{SET['name']}/{n}"])
    return rows


def build_boxes(scene_names: set[str]) -> list[list[str]]:
    """boxes/v1 rows (BOX_COLS, sorted by image, x1, y1)."""
    _, _, lb = load_labelv2(SET["labelv2"], SET["prefix"])
    lb_nonempty = {k: v for k, v in lb.items() if v}
    if SET["md5_from"] == "pii-data":
        rel, md5 = PINNED_V1
        p = os.path.join(PII_DATA, rel)
        check(md5_of(p) == md5, f"pii-data {rel} is not the pinned file")
        rows = read_csv(p, BOX_COLS)
        check([box_key(r) for r in rows] == sorted(box_key(r) for r in rows), f"{rel} not sorted")
        check(group_boxes(rows) == lb_nonempty, f"pii-data {rel} != {SET['labelv2']} as per-image box sets")
        n_boxes = {img: v[4] for img, v in load_pii_data_frames().items()}
        by = group_boxes(rows)
        check(all(len(by.get(img, [])) == n for img, n in n_boxes.items()),
              "per-image counts != pii-data frames.csv n_boxes")
        print(f"  v1: pii-data {rel} (md5 {md5}), equal to {os.path.basename(SET['labelv2'])} as box sets")
    else:
        rows = [[img, *b] for img in sorted(lb_nonempty) for b in lb_nonempty[img]]
        print(f"  v1: converted from {SET['labelv2']}")
    check(len(rows) == SET["n_boxes"], f"v1 rows {len(rows)}, expected {SET['n_boxes']}")
    check(sum(1 for r in rows if r[5] == "1") == SET["n_ignore"], "v1 ignore row count")
    check({r[0] for r in rows} <= scene_names, "v1: box on an image not in the set")
    return rows


# ---------------------------------------------------------------- build

def copy_images(frames: list[list[str]]) -> None:
    dst_dir = dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    t0 = time.time()
    for i, r in enumerate(frames, 1):
        name, size, s = r[0], int(r[5]), r[10]
        d = os.path.join(dst_dir, name)
        check(not os.path.islink(s), f"source image is a symlink: {s}")
        check(os.path.getsize(s) == size, f"{s}: size {os.path.getsize(s)} != frames.csv size {size}")
        if os.path.exists(d) and not os.path.islink(d) and os.path.getsize(d) == size:
            continue
        shutil.copy2(s, d)   # full byte copy, mtime kept; never a link
        copied += 1
        if copied % 2000 == 0:
            print(f"  images: {copied} copied, {i}/{len(frames)} seen, {time.time() - t0:.0f}s")
    print(f"images: {copied} copied, {len(frames) - copied} already present ({time.time() - t0:.0f}s)")


def copy_derivation() -> None:
    for rel in SET["derivation"]:
        s = os.path.join(SET["src_root"], rel)
        check(os.path.isfile(s), f"derivation source missing: {s}")
        d = dst("derivation", rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d) and md5_of(d) == md5_of(s):
            continue
        shutil.copyfile(s, d)
    if SET["derivation"]:
        print(f"  derivation/: {len(SET['derivation'])} files")


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    frames = build_frames(args.jobs)
    names = [r[0] for r in frames]
    if not args.no_images:
        copy_images(frames)
    scenes = sorted({r[1] for r in frames})
    check(len(scenes) == N_SCENES, f"{len(scenes)} scenes, expected {N_SCENES}")
    write_csv(dst("frames.csv"), FRAME_COLS, frames)
    write_csv(dst("split.csv"), ["session", "role"], [[s, "train"] for s in scenes])
    print(f"frames.csv {len(frames)} rows, split.csv {len(scenes)} scenes, all train")
    rows = build_boxes(set(names))
    write_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS, rows)
    write_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"], [[n, "1"] for n in names])
    if SET["md5_from"] == "pii-data":
        check(md5_of(dst("boxes", "v1", "boxes.csv")) == PINNED_V1[1], "written boxes.csv md5 != pinned pii-data file")
    print(f"boxes/v1: {len(rows)} boxes on {len({r[0] for r in rows})} images, "
          f"{sum(1 for r in rows if r[5] == '1')} ignore, {len(names)} reviewed")
    copy_derivation()
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def _md5_check(job: tuple[str, str, str, str, bool]):
    """(name, dst image path, expected md5 from frames.csv, src path, compare_src) -> (name, error or None).
    Workers get everything through the tuple: Python 3.14 starts them with forkserver, so
    module globals set in main() are not inherited."""
    name, d, expected, s, compare_src = job
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
    if any(r[1] != r[7] or r[2:5] != ["", "", ""] for r in frames):
        fail("frames.csv: session != scene, or chunk/eye/frame_idx not empty")
    if any(not MD5_HEX.match(r[6]) or not r[5].isdigit() or not (r[8].isdigit() and r[9].isdigit()) for r in frames):
        fail("frames.csv: md5 not 32 hex chars, or size/width/height not integers")
    if SET["dims"] is not None and any((int(r[8]), int(r[9])) != SET["dims"] for r in frames if r[8].isdigit()):
        fail(f"frames.csv: width/height not {SET['dims']}")
    # PII-1448: src_path was recorded when the source tree was /data/esteban/pii; resolve_legacy
    # maps it to where that tree is now, so the column is still checked, not skipped.
    if any(resolve_legacy(r[10]) != os.path.join(SET["src_images"], r[1], r[0]) for r in frames):
        fail("frames.csv: src_path is not <src_images>/<scene>/<image>")
    size_of = {r[0]: int(r[5]) for r in frames if r[5].isdigit()}
    md5_col = {r[0]: r[6] for r in frames}
    src_of = {r[0]: resolve_legacy(r[10]) for r in frames}

    src_present = os.path.isfile(SET["labelv2"]) and os.path.isdir(SET["src_images"])
    if src_present:
        scene_fs = list_source_images()
        scene_lb, dims, _ = load_labelv2(SET["labelv2"], SET["prefix"])
        if {r[0]: r[1] for r in frames} != scene_fs or scene_fs != scene_lb:
            fail("frames.csv scenes differ from the source tree / labelv2 headers")
        if {r[0]: (int(r[8]), int(r[9])) for r in frames if r[8].isdigit()} != dims:
            fail("frames.csv width/height differ from the labelv2 headers")
    else:
        print("  note: source tree or labelv2 not present, scenes/dims not cross-checked")
    pd = None
    if SET["md5_from"] == "pii-data":
        if os.path.isfile(PII_DATA_FRAMES):
            pd = load_pii_data_frames()
            if {k: (v[0], v[1]) for k, v in pd.items()} != {r[0]: (r[5], r[6]) for r in frames}:
                fail("frames.csv size/md5 differ from pii-data frames.csv")
        else:
            print(f"  note: {PII_DATA_FRAMES} not present, size/md5 not cross-checked")

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(roles) != N_SCENES or len(split) != N_SCENES:
        fail(f"split.csv has {len(split)} rows, expected {N_SCENES}")
    if {r[1] for r in frames} != set(roles) or set(roles.values()) != {"train"}:
        fail("split.csv sessions do not match frames.csv scenes, or a role is not train")

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
        compare_src = os.path.isdir(SET["src_images"])
        if not compare_src:
            print("  note: source images not present; md5 checked against frames.csv only")
        jobs = [(n, os.path.join(DST, "images", n), md5_col[n], src_of[n], compare_src) for n in images]
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

    # boxes/v1
    vf = read_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"])
    if [r[0] for r in vf] != images:
        fail("boxes/v1/frames.csv image column != frames.csv")
    if {r[1] for r in vf} != {"1"}:
        fail("boxes/v1/frames.csv: reviewed must be 1 on every image")
    rows = read_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS)
    if len(rows) != SET["n_boxes"]:
        fail(f"boxes/v1/boxes.csv has {len(rows)} rows, expected {SET['n_boxes']}")
    if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
        fail("boxes/v1/boxes.csv not sorted by (image, x1, y1)")
    if {r[0] for r in rows} - image_set:
        fail("boxes/v1/boxes.csv: box on an image not in frames.csv")
    if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
        fail("boxes/v1/boxes.csv: coordinate not one decimal or bad ignore")
    if sum(1 for r in rows if r[5] == "1") != SET["n_ignore"]:
        fail(f"boxes/v1/boxes.csv: ignore rows != {SET['n_ignore']}")
    if SET["md5_from"] == "pii-data":
        if md5_of(dst("boxes", "v1", "boxes.csv")) != PINNED_V1[1]:
            fail(f"boxes/v1/boxes.csv md5 != pii-data {PINNED_V1[0]} ({PINNED_V1[1]})")
        if pd is not None:
            by = group_boxes(rows)
            if any(len(by.get(img, [])) != v[4] for img, v in pd.items()):
                fail("boxes/v1/boxes.csv per-image counts != pii-data frames.csv n_boxes")
    labelv2 = SET["labelv2"] if src_present else dst("derivation", "wider_fisheye_labelv2.txt")
    if os.path.isfile(labelv2):
        _, _, lb = load_labelv2(labelv2, SET["prefix"])
        if {k: v for k, v in lb.items() if v} != group_boxes(rows):
            fail(f"boxes/v1/boxes.csv != {labelv2} as per-image box sets")
    else:
        print("  note: no labelv2 file to compare boxes.csv against")
    if os.path.exists(dst("boxes", "v1", "job")):
        fail("boxes/v1/job exists; this pass has no vendor job (PII-1337)")
    print(f"boxes/v1: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
          f"{len(vf)} reviewed ({time.time() - t0:.0f}s)")

    for rel in SET["derivation"]:
        s = os.path.join(SET["src_root"], rel)
        d = dst("derivation", rel)
        if not os.path.isfile(d):
            fail(f"missing derivation copy {d}")
        elif os.path.isfile(s) and md5_of(s) != md5_of(d):
            fail(f"derivation copy differs from source: {d}")
        elif not os.path.isfile(s):
            print(f"  note: source gone, derivation copy not compared: {d}")
    if SET["derivation"]:
        extra = sorted(set(os.path.relpath(os.path.join(r, f), dst("derivation"))
                           for r, _, fs in os.walk(dst("derivation")) for f in fs) - set(SET["derivation"]))
        if extra:
            fail(f"derivation/ has unexpected files: {extra[:5]}")

    for p in ("README.md", "boxes/v1/README.md"):
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
    global SET, DST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-images", action="store_true", help="skip the image copy")
    b.add_argument("--jobs", type=int, default=16, help="md5 workers (fisheye sets)")
    v = sub.add_parser("verify")
    v.add_argument("--no-md5", action="store_true", help="skip the per-image md5 comparison")
    v.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    SET = dict(SETS[args.set], name=args.set)
    DST = os.path.join(DST_ROOT, args.set)
    {"build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()

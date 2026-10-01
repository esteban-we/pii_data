#!/usr/bin/env python3
"""Build /data/esteban/pii/datasets/gt_bench_sparse and gt_bench_full in the pii2 layout
(PII-1315, PII-1339). Stdlib only, idempotent; every source is read-only.

    --set sparse|full build     copy images (skipped with --no-images), write frames.csv,
                                split.csv, boxes/v1 (frames.csv, boxes.csv, job/output),
                                derivation/ copies
    --set sparse|full verify    md5 every image against the frames.csv md5 column (and
                                against the source while it exists), check row counts, the
                                box file against its pinned md5 and against a re-derivation
                                from the job/output copy, the reviewed set, job and
                                derivation copies; exit 1 on any mismatch

Both sets hold the same 9 sessions (all eval) of the face-PII ground-truth bench
(/data/esteban/pii_backup/datasets/gt_bench_v1). They are two datasets because the bytes
differ: the sparse images are the previous owner's encode of 889 frames, the full
images are a re-extraction of all 9,636 labelled frames (evaluation/stage_full_bench.py,
2026-08-29); 0 of the 889 shared names have equal md5 (user decision 2026-09-21).

Layout written (README.md files are written by hand, not here):

    images/                     byte copies (no links)
    frames.csv                  sparse: image,session,chunk,eye,frame_idx,size,md5,src_path
                                full:   image,session,chunk,eye,frame_idx,size,md5,src_key,src_frame
                                both: plus oss_key (PII-1413)
                                (size and md5 from pii-data frames.csv, dataset gt_bench_sparse /
                                gt_bench_full, hashed at OSS upload; size asserted equal to the
                                file on disk. src_path: the path inside gt_eval_sparse.json;
                                src_key/src_frame: the video key and frame index
                                stage_full_bench.py decoded)
    split.csv                   session,role: the 9 sessions, all eval
    boxes/v1/frames.csv         image,reviewed (1 everywhere)
    boxes/v1/boxes.csv          image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal, sorted by
                                image,x1,y1; images with no box have no row)
    boxes/v1/job/output/        the label files as delivered (byte copies); no prelabels exist
    derivation/                 the scripts and README the set was derived with (byte copies)

sparse images: gt_bench_v1/images_sparse/<name>.jpg (already flat).
full images:   gt_bench_v1/images_full/<session>_<chunk>/f<NNNNNN>.jpg flattened to
               <session>_<chunk>_f<NNNNNN>.jpg, the rule of the OSS upload (asserted against
               data/oss_pii_keys.jsonl, dataset gt_bench_full).
sparse boxes:  labels/gt_eval_sparse.json (list of {path, boxes, session, chunk}; pixel xyxy
               floats) -> one decimal; 1,541 rows on 886 of the 889 frames; asserted byte-equal
               to pii-data datasets/gt_bench_sparse/boxes/v1.csv.
full boxes:    labels/gt_bundle.json ({uuid: {image_size, frames: {idx: [[x1,y1,x2,y2]..]}}})
               through labels/uuid_map.json (uuid -> [session, chunk, n]); 20,054 rows on all
               9,636 frames; asserted byte-equal to pii-data datasets/gt_bench_full/boxes/v1.csv.
               The 10th uuid of the bundle is not in uuid_map and has 0 frames (asserted).
Both builds assert that the sparse and full boxes disagree on exactly the 5 frames of PII-130.
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


# The manifest/index is the ledger and was never rewritten for the PII-1448 rename: paths in it
# read /data/esteban/pii/... meaning the tree that became LEGACY_ROOT above (PII-1682).
OLD_ROOT_PREFIX = "/data/esteban/pii/"


def resolve_legacy(path: str) -> str:
    """An absolute path recorded before the PII-1448 rename, as it resolved after it."""
    path = str(path)
    if path.startswith(OLD_ROOT_PREFIX) and not path.startswith(LEGACY_ROOT + "/"):
        return LEGACY_ROOT + "/" + path[len(OLD_ROOT_PREFIX):]
    return path

SRC = LEGACY_ROOT + "/datasets/gt_bench_v1"    # PII-1448: the old tree
SRC_LABELS = os.path.join(SRC, "labels")
PII_DATA = "/home/esteban/repos/pii-data/datasets"
PII_DATA_FRAMES = "/home/esteban/repos/pii-data/frames.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
OSS_KEYS = os.path.join(STORE, "data", "oss_pii_keys.jsonl")
EPISODE_USAGE = os.path.join(STORE, "data", "episode_usage.csv")
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DST_ROOT = PII2_ROOT + "/datasets"
WRITE_ROOT = PII2_ROOT + "/"

W, H = 2328, 1748
N_SESSIONS = 9
VIDEO = "vst_left"
MD5_LABEL = {  # the label files as delivered (PII-516: equal to the OSS ETags)
    "gt_bundle.json": "47b67814ee9aee4480f0784fef170bcf",
    "gt_eval_sparse.json": "cfd17fff00f32a86e70285469c5f31dc",
    "uuid_map.json": "62f45bbff51d91095eeadf008c12ad91",
}
UNMAPPED_UUID = "019e93df-7e02-73b3-92b5-1af0dbbab7ae"   # in gt_bundle.json, 0 frames, not in uuid_map
# PII-130: gt_eval_sparse.json has fewer boxes than gt_bundle.json on these 5 frames (unresolved)
PII_130_FRAMES = {
    "20260515_081443_BUXVDW_011_f001100.jpg", "20260515_081443_BUXVDW_011_f004035.jpg",
    "20260515_081443_BUXVDW_011_f005670.jpg", "20260515_081443_BUXVDW_011_f007790.jpg",
    "20260527_053115_CAHMSV_006_f002880.jpg",
}

SETS = {
    "sparse": dict(
        name="gt_bench_sparse",
        n_images=889, n_boxes=1541, n_box_images=886,
        md5_v1="de1d589fe2e0c81f1181a74ef69ed2d0",     # pii-data gt_bench_sparse/boxes/v1.csv
        frame_cols=["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "src_path", "oss_key"],
        job_output=["gt_eval_sparse.json"],
        derivation={"gt_extract.py": os.path.join(HERE, os.pardir, "source", "gt_extract.py")},
    ),
    "full": dict(
        name="gt_bench_full",
        n_images=9636, n_boxes=20054, n_box_images=9636,
        md5_v1="1a479d33eeb9488e790b2346ed2f254f",     # pii-data gt_bench_full/boxes/v1.csv
        frame_cols=["image", "session", "chunk", "eye", "frame_idx", "size", "md5", "src_key", "src_frame",
                    "oss_key"],
        job_output=["gt_bundle.json", "uuid_map.json"],
        derivation={"stage_full_bench.py": os.path.join(CODE_ROOT, "evaluation", "stage_full_bench.py"),
                    "gt_bench_v1_README.md": os.path.join(SRC, "README.md")},
    ),
}

NAME_RE = re.compile(r"^(\d{8}_\d{6}_[A-Z]{6})_(\d{3})_f(\d{6})\.jpg$")
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]

CFG: dict = {}
DST = ""
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


def csv_bytes(header: list[str], rows) -> bytes:
    lines = [",".join(header)] + [",".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def box_sort_key(row):
    return (row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), row[5])


def parse_name(image: str) -> tuple[str, str, int]:
    m = NAME_RE.match(image)
    check(m is not None, f"image name does not match <session>_<chunk>_f<idx>.jpg: {image}")
    return m.group(1), m.group(2), int(m.group(3))


# ---------------------------------------------------------------- sources

def source_images() -> dict[str, str]:
    """{flat image name: absolute source path}. full: images_full/<session>_<chunk>/f<idx>.jpg
    -> <session>_<chunk>_f<idx>.jpg (the OSS rule, asserted against oss_pii_keys.jsonl)."""
    out: dict[str, str] = {}
    if CFG["name"] == "gt_bench_sparse":
        d = os.path.join(SRC, "images_sparse")
        for n in sorted(os.listdir(d)):
            check(n.endswith(".jpg"), f"non-jpg in images_sparse: {n}")
            out[n] = os.path.join(d, n)
    else:
        root = os.path.join(SRC, "images_full")
        for sub in sorted(os.listdir(root)):
            check(re.match(r"^\d{8}_\d{6}_[A-Z]{6}_\d{3}$", sub) is not None, f"images_full: odd dir {sub}")
            for n in sorted(os.listdir(os.path.join(root, sub))):
                check(re.match(r"^f\d{6}\.jpg$", n) is not None, f"images_full/{sub}: odd file {n}")
                flat = f"{sub}_{n}"
                check(flat not in out, f"flattening collision: {flat}")
                out[flat] = os.path.join(root, sub, n)
    for n, p in out.items():
        check(not os.path.islink(p), f"source image is a symlink: {p}")
        parse_name(n)
    check(len(out) == CFG["n_images"], f"source has {len(out)} images, expected {CFG['n_images']}")
    if os.path.isfile(OSS_KEYS):
        # PII-1448 renamed the tree: every local_path in the key map was written when the old
        # store was /data/esteban/pii, and it is /data/esteban/pii_backup now. The map itself is
        # retired (PII-1449) and is not rewritten; resolve_legacy maps its paths.
        keys = {}
        for ln in open(OSS_KEYS, encoding="utf-8"):
            d = json.loads(ln)
            if d["dataset"] == CFG["name"]:
                keys[d["oss_key"].rsplit("/", 1)[-1]] = resolve_legacy(d["local_path"])
        check(keys == out, f"flattened names/paths differ from data/oss_pii_keys.jsonl (dataset={CFG['name']})")
    else:
        print(f"  note: {OSS_KEYS} not present, flattening not cross-checked")
    return out


def load_pii_data_md5() -> dict[str, tuple[str, str]]:
    """{image: (size, md5)} from pii-data frames.csv rows with dataset=<name>."""
    out: dict[str, tuple[str, str]] = {}
    with open(PII_DATA_FRAMES, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["dataset"] != CFG["name"]:
                continue
            check(d["image"] not in out, f"pii-data frames.csv: duplicate {d['image']}")
            check(MD5_HEX.match(d["md5"]) is not None and d["size"].isdigit(),
                  f"pii-data frames.csv: bad size/md5 for {d['image']}")
            check(d["role"] == "eval", f"pii-data frames.csv: {d['image']} role {d['role']} != eval")
            out[d["image"]] = (d["size"], d["md5"])
    check(len(out) == CFG["n_images"],
          f"pii-data frames.csv has {len(out)} {CFG['name']} rows, expected {CFG['n_images']}")
    return out


def load_label(name: str, root: str = SRC_LABELS):
    p = os.path.join(root, name)
    check(os.path.isfile(p), f"label file missing: {p}")
    check(md5_of(p) == MD5_LABEL[name], f"{p} is not the delivered file (md5 != {MD5_LABEL[name]})")
    return json.load(open(p, encoding="utf-8"))


def derive_sparse(root: str = SRC_LABELS) -> tuple[dict[str, list[tuple]], dict[str, str]]:
    """gt_eval_sparse.json -> ({image: [box tuples]}, {image: src_path})."""
    boxes: dict[str, list[tuple]] = {}
    src_path: dict[str, str] = {}
    for e in load_label("gt_eval_sparse.json", root):
        fn = e["path"].rsplit("/", 1)[-1]
        session, chunk, _ = parse_name(fn)
        check(fn not in boxes, f"gt_eval_sparse: duplicate {fn}")
        check(e["session"] == session and int(e["chunk"]) == int(chunk), f"gt_eval_sparse: fields of {fn}")
        boxes[fn] = [tuple(fmt(c) for c in b) + ("0",) for b in e["boxes"]]
        src_path[fn] = e["path"]
    return boxes, src_path


def derive_full(root: str = SRC_LABELS) -> dict[str, list[tuple]]:
    """gt_bundle.json + uuid_map.json -> {image: [box tuples]}."""
    bundle = load_label("gt_bundle.json", root)
    umap = load_label("uuid_map.json", root)
    check(set(bundle) - set(umap) == {UNMAPPED_UUID} and set(umap) <= set(bundle),
          "gt_bundle/uuid_map uuid sets: expected exactly one unmapped uuid")
    check(bundle[UNMAPPED_UUID]["frames"] == {}, f"unmapped uuid {UNMAPPED_UUID} has frames")
    boxes: dict[str, list[tuple]] = {}
    for uuid, v in bundle.items():
        check(tuple(v["image_size"]) == (H, W), f"gt_bundle {uuid}: image_size {v['image_size']}")
        if uuid not in umap:
            continue
        session, chunk, _n = umap[uuid]
        for fidx, bl in v["frames"].items():
            fn = f"{session}_{int(chunk):03d}_f{int(fidx):06d}.jpg"
            check(fn not in boxes, f"gt_bundle: duplicate frame {fn}")
            check(len(bl) > 0, f"gt_bundle: frame {fn} listed with 0 boxes")
            boxes[fn] = [tuple(fmt(c) for c in b) + ("0",) for b in bl]
    return boxes


def box_rows(by_image: dict[str, list[tuple]]) -> list[list[str]]:
    rows = [[fn, *b] for fn, bl in by_image.items() for b in bl]
    rows.sort(key=box_sort_key)
    return rows


def check_pii_130(sparse: dict[str, list[tuple]], full: dict[str, list[tuple]]) -> None:
    check(set(sparse) <= set(full), "sparse frame names are not a subset of the full names")
    differ = {fn for fn in sparse if sorted(sparse[fn]) != sorted(full[fn])}
    check(differ == PII_130_FRAMES,
          f"sparse vs full boxes differ on {sorted(differ)}, expected the 5 PII-130 frames")
    for fn in differ:
        check(len(sparse[fn]) < len(full[fn]), f"PII-130: sparse has more boxes than full on {fn}")


def derive_v1(root: str = SRC_LABELS) -> tuple[list[list[str]], dict[str, str]]:
    """This set's v1 rows (sorted) and the sparse src_path map; asserts the PII-130 invariant."""
    sparse, src_path = derive_sparse(root)
    full = derive_full(root)
    check_pii_130(sparse, full)
    check(len(sparse) == SETS["sparse"]["n_images"] and len(full) == SETS["full"]["n_images"],
          f"label files cover {len(sparse)} sparse / {len(full)} full frames")
    rows = box_rows(sparse if CFG["name"] == "gt_bench_sparse" else full)
    return rows, src_path


def load_split_check(sessions: set[str]) -> None:
    """While data/episode_usage.csv exists: every session is tagged gt_bench_v1 / eval there."""
    if not os.path.isfile(EPISODE_USAGE):
        print(f"  note: {EPISODE_USAGE} not present, session tags not cross-checked")
        return
    tagged = set()
    with open(EPISODE_USAGE, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["session_id"] in sessions:
                check(d["dataset"] == "gt_bench_v1" and d["role"] == "eval",
                      f"episode_usage.csv: {d['session_id']} is {d['dataset']}/{d['role']}, expected gt_bench_v1/eval")
                tagged.add(d["session_id"])
    check(tagged == sessions, f"episode_usage.csv lacks {sorted(sessions - tagged)}")


def build_frames(images: dict[str, str], src_path: dict[str, str]) -> list[list[str]]:
    hashes = load_pii_data_md5()
    check(set(hashes) == set(images), "pii-data frames.csv image set != source images")
    rows = []
    for image in sorted(images):
        session, chunk, frame_idx = parse_name(image)
        size, md5 = hashes[image]
        check(int(size) == os.path.getsize(images[image]), f"{image}: pii-data size {size} != file on disk")
        base = [image, session, chunk, "left", str(frame_idx), size, md5]
        key = f"pii/data/{CFG['name']}/{image}"       # PII-1413
        if CFG["name"] == "gt_bench_sparse":
            check(image in src_path, f"{image} not in gt_eval_sparse.json")
            rows.append(base + [src_path[image], key])
        else:
            rows.append(base + [f"{session}/chunk_{chunk}/{VIDEO}/{VIDEO}_video.mp4", str(frame_idx), key])
    return rows


# ---------------------------------------------------------------- build

def copy_images(images: dict[str, str]) -> None:
    dst_dir = dst("images")
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    t0 = time.time()
    for i, n in enumerate(sorted(images), 1):
        s = images[n]
        d = os.path.join(dst_dir, n)
        if os.path.exists(d) and not os.path.islink(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        shutil.copy2(s, d)   # full byte copy, mtime kept; never a link
        copied += 1
        if copied % 2000 == 0:
            print(f"  images: {copied} copied, {i}/{len(images)} seen, {time.time() - t0:.0f}s")
    extra = sorted(set(os.listdir(dst_dir)) - set(images))
    check(not extra, f"images/ has files not in the source: {extra[:5]}")
    print(f"images: {copied} copied, {len(images) - copied} already present ({time.time() - t0:.0f}s)")


def copy_file(s: str, d: str) -> None:
    check(os.path.isfile(s), f"copy source missing: {s}")
    check(d.startswith(WRITE_ROOT), f"refusing to write outside {WRITE_ROOT}: {d}")
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.exists(d) and md5_of(d) == md5_of(s):
        return
    shutil.copyfile(s, d)


def copy_extras() -> None:
    for n in CFG["job_output"]:
        copy_file(os.path.join(SRC_LABELS, n), dst("boxes", "v1", "job", "output", n))
    print(f"  boxes/v1/job/output: {len(CFG['job_output'])} files (no prelabels exist for this set)")
    for n, s in CFG["derivation"].items():
        copy_file(s, dst("derivation", n))
    print(f"  derivation: {len(CFG['derivation'])} files")


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    images = source_images()
    v1, src_path = derive_v1()
    frames = build_frames(images, src_path)
    sessions = {r[1] for r in frames}
    check(len(sessions) == N_SESSIONS, f"{len(sessions)} sessions, expected {N_SESSIONS}")
    load_split_check(sessions)
    check({r[0] for r in v1} <= set(images), "v1: box on an unknown image")
    check(len(v1) == CFG["n_boxes"], f"v1 rows {len(v1)}, expected {CFG['n_boxes']}")
    check(len({r[0] for r in v1}) == CFG["n_box_images"], "v1: images with a box")
    ref_path = os.path.join(PII_DATA, CFG["name"], "boxes", "v1.csv")
    if os.path.isfile(ref_path):
        ref = open(ref_path, "rb").read()
        check(md5_bytes(ref) == CFG["md5_v1"], f"{ref_path} is not the pinned file")
        check(csv_bytes(BOX_COLS, v1) == ref, f"derived v1 boxes differ from {ref_path}")
    else:
        print(f"  note: {ref_path} not present; v1 checked against the pinned md5 only")
    check(md5_bytes(csv_bytes(BOX_COLS, v1)) == CFG["md5_v1"], "derived v1 boxes md5 != pinned")

    if not args.no_images:
        copy_images(images)
    write_csv(dst("frames.csv"), CFG["frame_cols"], frames)
    write_csv(dst("split.csv"), ["session", "role"], [[s, "eval"] for s in sorted(sessions)])
    print(f"frames.csv {len(frames)} rows, split.csv {len(sessions)} sessions (all eval)")
    write_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS, v1)
    write_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"], [[r[0], "1"] for r in frames])
    copy_extras()
    check(md5_of(dst("boxes", "v1", "boxes.csv")) == CFG["md5_v1"], "written boxes.csv md5")
    print(f"boxes/v1: {len(v1)} boxes on {CFG['n_box_images']} images, {len(frames)} reviewed")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def _md5_check(job: tuple[str, str, str, str]):
    """(dst image path, name, expected md5 from frames.csv, source path or '') -> (name, error or None).
    The dst path travels in the job: Pool workers do not inherit the DST global under forkserver/spawn."""
    d, name, expected, s = job
    if os.path.islink(d):
        return name, "dst is a symlink"
    if not os.path.isfile(d):
        return name, "missing in dst"
    md = md5_of(d)
    if md != expected:
        return name, f"md5 {md} != frames.csv {expected}"
    if s:
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
    cols = CFG["frame_cols"]
    frames = read_csv(dst("frames.csv"), cols)
    images = [r[0] for r in frames]
    image_set = set(images)
    if len(frames) != CFG["n_images"]:
        fail(f"frames.csv has {len(frames)} rows, expected {CFG['n_images']}")
    if images != sorted(images) or len(image_set) != len(images):
        fail("frames.csv not sorted by image or has duplicates")
    for r in frames:
        m = NAME_RE.match(r[0])
        if m is None or [r[1], r[2], r[3], r[4]] != [m.group(1), m.group(2), "left", str(int(m.group(3)))]:
            fail(f"frames.csv: {r[0]} does not match its fields")
            break
    if any(not MD5_HEX.match(r[6]) or not r[5].isdigit() for r in frames):
        fail("frames.csv: md5 not 32 hex chars or size not an integer")
    if CFG["name"] == "gt_bench_full":
        for r in frames:
            if r[7] != f"{r[1]}/chunk_{r[2]}/{VIDEO}/{VIDEO}_video.mp4" or r[8] != r[4]:
                fail(f"frames.csv: src_key/src_frame of {r[0]}")
                break
    size_of = {r[0]: int(r[5]) for r in frames}
    md5_col = {r[0]: r[6] for r in frames}

    src_ok = os.path.isdir(SRC)
    src_images: dict[str, str] = {}
    if src_ok:
        src_images = source_images()
        if set(src_images) != image_set:
            fail("frames.csv image set != source images")
    else:
        print(f"  note: source {SRC} not present; images checked against frames.csv only")
    if os.path.isfile(PII_DATA_FRAMES):
        if load_pii_data_md5() != {r[0]: (r[5], r[6]) for r in frames}:
            fail(f"frames.csv size/md5 differ from pii-data frames.csv (dataset={CFG['name']})")
    else:
        print(f"  note: {PII_DATA_FRAMES} not present, size/md5 columns not cross-checked")

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(split) != N_SESSIONS or len(roles) != N_SESSIONS or set(roles.values()) != {"eval"}:
        fail(f"split.csv: {len(split)} rows, roles {set(roles.values())}; expected {N_SESSIONS} eval sessions")
    if {r[1] for r in frames} != set(roles):
        fail("split.csv sessions do not match frames.csv")

    # images: every file present, no extras, sizes, md5 vs frames.csv (and the source)
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
        jobs = [(os.path.join(DST, "images", n), n, md5_col[n], src_images.get(n, "")) for n in images]
        bad = 0
        with Pool(args.jobs) as pool:
            for i, (name, err) in enumerate(pool.imap_unordered(_md5_check, jobs, chunksize=16), 1):
                if err:
                    bad += 1
                    if bad <= 20:
                        fail(f"images/{name}: {err}")
                if i % 2000 == 0:
                    print(f"  md5: {i}/{len(images)} ({time.time() - t0:.0f}s)")
        if bad:
            fail(f"{bad} images differ from frames.csv md5" + (" or the source" if src_ok else ""))
        else:
            print(f"images: {len(images)} md5 equal to frames.csv"
                  + (" and the source" if src_ok else "") + f" ({time.time() - t0:.0f}s)")

    # boxes/v1
    vf = read_csv(dst("boxes", "v1", "frames.csv"), ["image", "reviewed"])
    if [r[0] for r in vf] != images:
        fail("boxes/v1/frames.csv image column != frames.csv")
    if {r[1] for r in vf} != {"1"}:
        fail(f"boxes/v1/frames.csv: reviewed must be 1 on every frame ({sum(r[1] != '1' for r in vf)} are not)")
    rows = read_csv(dst("boxes", "v1", "boxes.csv"), BOX_COLS)
    if len(rows) != CFG["n_boxes"]:
        fail(f"boxes/v1/boxes.csv has {len(rows)} rows, expected {CFG['n_boxes']}")
    if len({r[0] for r in rows}) != CFG["n_box_images"]:
        fail(f"boxes/v1/boxes.csv covers {len({r[0] for r in rows})} images, expected {CFG['n_box_images']}")
    if [box_sort_key(r) for r in rows] != sorted(box_sort_key(r) for r in rows):
        fail("boxes/v1/boxes.csv not sorted by (image, x1, y1)")
    if {r[0] for r in rows} - image_set:
        fail("boxes/v1/boxes.csv: box on an image not in frames.csv")
    if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] != "0" for r in rows):
        fail("boxes/v1/boxes.csv: coordinate not one decimal or ignore != 0")
    if md5_of(dst("boxes", "v1", "boxes.csv")) != CFG["md5_v1"]:
        fail(f"boxes/v1/boxes.csv md5 != pinned {CFG['md5_v1']} (pii-data {CFG['name']}/boxes/v1.csv)")

    # job/output copies: pinned md5 always, equal to the source while it exists; boxes re-derivable
    out_dir = dst("boxes", "v1", "job", "output")
    for n in CFG["job_output"]:
        d = os.path.join(out_dir, n)
        if not os.path.isfile(d):
            fail(f"missing job copy {d}")
        elif md5_of(d) != MD5_LABEL[n]:
            fail(f"job copy md5 != delivered file: {d}")
        s = os.path.join(SRC_LABELS, n)
        if os.path.isfile(d) and os.path.isfile(s) and md5_of(s) != md5_of(d):
            fail(f"job copy differs from source: {d}")
    if os.path.isdir(dst("boxes", "v1", "job", "prelabels")):
        fail("boxes/v1/job/prelabels exists; no prelabels exist for this set")
    if not FAILS or all("job copy" not in f for f in FAILS):
        rederived = None
        if CFG["name"] == "gt_bench_sparse":
            rederived, src_path = derive_sparse(out_dir)
            if [r[7] for r in frames] != [src_path.get(r[0]) for r in frames]:
                fail("frames.csv src_path differs from job/output/gt_eval_sparse.json")
        else:
            rederived = derive_full(out_dir)
        if box_rows(rederived) != rows:
            fail("boxes/v1/boxes.csv differs from a re-derivation of job/output")
    if os.path.isdir(SRC_LABELS):
        sp, _ = derive_sparse()
        check_pii_130(sp, derive_full())

    for n, s in CFG["derivation"].items():
        d = dst("derivation", n)
        if not os.path.isfile(d):
            fail(f"missing derivation copy {d}")
        elif os.path.isfile(s) and md5_of(s) != md5_of(d):
            fail(f"derivation copy differs from source: {d}")
        elif not os.path.isfile(s):
            print(f"  note: source gone, derivation copy not compared: {d}")
    for p in ("README.md", "boxes/v1/README.md"):
        if not os.path.isfile(dst(p)):
            fail(f"missing {p}")
    for p in ("_download.log", "_launch_time.txt"):
        if os.path.exists(dst(p)):
            fail(f"{p} must not be in the dataset")
    print(f"boxes/v1: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
          f"{sum(r[1] == '1' for r in vf)} reviewed ({time.time() - t0:.0f}s)")

    if FAILS:
        print(f"VERIFY FAILED: {len(FAILS)} problem(s) ({time.time() - t0:.0f}s)")
        sys.exit(1)
    print(f"VERIFY OK ({time.time() - t0:.0f}s)")


def main() -> None:
    global CFG, DST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=sorted(SETS), help="which dataset: sparse or full")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-images", action="store_true", help="skip the image copy")
    v = sub.add_parser("verify")
    v.add_argument("--no-md5", action="store_true",
                   help="skip the per-image md5 pass (frames.csv column, and the source while it exists)")
    v.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    CFG = SETS[args.set]
    DST = os.path.join(DST_ROOT, CFG["name"])
    {"build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()

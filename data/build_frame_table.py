#!/usr/bin/env python3
"""Build data/frames.csv and data/boxes.csv: one frame table over the eight
datasets under oss://algorithm-datasets/pii/ (WOR-125). Stdlib only.

frames.csv  dataset,image,session_id,chunk,view,frame_idx,width,height,size,md5,role,n_boxes
boxes.csv   dataset,image,x1,y1,x2,y2,ignore

Image set, size and md5 come from data/oss_pii_keys.jsonl (never from the
images). Boxes come from, per dataset:

  face_mine_v1     /data/esteban/pii_backup/face-mine_labeled.csv (the human review;
                   datasets/face_mine_v1/boxes.jsonl holds the MACHINE miner
                   boxes and is deliberately not used)
  faceback_45      datasets/faceback_45/boxes.jsonl (human, normalized xywh)
  gt_bench_full    datasets/gt_bench_v1/labels/gt_bundle.json + uuid_map.json
  gt_bench_sparse  datasets/gt_bench_v1/labels/gt_eval_sparse.json
  face10k_v3, face10k_repair, pii_frames
                   the labelv2 blocks of training/manifests/train_Z2.txt
  wider_face       datasets/wider_face/wider_train_labelv2.txt

Boxes are pixel xyxy in the image's own space, one decimal. `ignore` is 1 for
the labelv2 5-field lines "x1 y1 x2 y2 1" (gt_bboxes_ignore regions, see
training/patches/.../retinaface.py::_parse_ann_line), else 0. n_boxes counts
every boxes.csv row of the image, ignore rows included.

role: face_mine_v1 and faceback_45 by session from data/episode_usage.csv,
asserted equal to data/splits/*.txt; gt_bench_* = eval; the rest = train,
except `unlabeled` (WOR-131): a frame of a dataset whose label source is a
labelv2 manifest (LABEL_SOURCE == "manifest") that is absent from that
manifest has no label at all, gets role=unlabeled and n_boxes=0, and is kept
out of every manifest rebuild. Datasets with a per-frame label file cover
every frame by construction; a gap there is an error, not an unlabeled row.

  python3 data/build_frame_table.py                       # build data/frames.csv + boxes.csv
  python3 data/build_frame_table.py --out-dir /tmp/run2   # build elsewhere (determinism check)
  python3 data/build_frame_table.py --check-manifest training/manifests/train_Z2.txt
                                                          # acceptance 2/3: rebuild vs reference
  python3 data/build_frame_table.py --rebuild-manifest /tmp/train_Z2_rebuilt.txt
  python3 data/build_frame_table.py --src faceback_jsonl=/tmp/copy.jsonl --out-dir /tmp/sab
                                                          # sabotage: build from an edited copy

Exit status: 0 ok; 1 on any assertion or check failure; 2 when a role in
episode_usage.csv disagrees with the split lists (reported, never guessed).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pii_root import CODE_ROOT, LEGACY_ROOT  # noqa: E402  (PII-1639: the store-local resolver)

# The label sources this table reads (face-mine_labeled.csv, gt_bench_v1/, the old dataset
# names) exist only in the pre-PII-1315 tree, /data/esteban/pii_backup after PII-1448.
DATA_ROOT = LEGACY_ROOT
DS_ROOT = os.path.join(DATA_ROOT, "datasets")

FRAME_COLS = ["dataset", "image", "session_id", "chunk", "view", "frame_idx",
              "width", "height", "size", "md5", "role", "n_boxes"]
BOX_COLS = ["dataset", "image", "x1", "y1", "x2", "y2", "ignore"]
DATASETS = ["face10k_repair", "face10k_v3", "face_mine_v1", "faceback_45",
            "gt_bench_full", "gt_bench_sparse", "pii_frames", "wider_face"]
FISHEYE = (2328, 1748)
KPS = " ".join(["-1.0"] * 15)
EXPECT_OSS = 192_701
ROLES = ("train", "eval", "unlabeled")
# How each dataset is labeled. "perframe": a label file with one record per
# image (a frame with zero faces is still a record), so every OSS image must
# appear in it. "manifest": the labelv2 blocks of a training manifest, which
# only list the images that were labeled; an OSS image absent from the blocks
# has no label at all and becomes role=unlabeled (WOR-131).
LABEL_SOURCE = {
    "face_mine_v1": "perframe", "faceback_45": "perframe",
    "gt_bench_full": "perframe", "gt_bench_sparse": "perframe",
    "face10k_v3": "manifest", "face10k_repair": "manifest",
    "pii_frames": "manifest", "wider_face": "manifest",
}
# the unlabeled rows the tables carry today: the 16 face10k_repair images the
# vendor never returned boxes for (WOR-131, user decision 2026-09-09: kept on
# disk and OSS, never trained on). Any change here must be a deliberate one.
EXPECT_UNLABELED = {"n": 16, "dataset": "face10k_repair",
                    "session_id": "20260725_055005_MPCWWA", "chunk": "0"}

SRC = {
    "keys": os.path.join(HERE, "oss_pii_keys.jsonl"),
    "usage": os.path.join(HERE, "episode_usage.csv"),
    "split_train": os.path.join(HERE, "splits", "train_sessions_v1.txt"),
    "split_eval": os.path.join(HERE, "splits", "eval_sessions_v1.txt"),
    "fb_train": os.path.join(HERE, "splits", "faceback_train_sessions_v1.txt"),
    "fb_eval": os.path.join(HERE, "splits", "faceback_eval_sessions_v1.txt"),
    "face_mine_csv": os.path.join(DATA_ROOT, "face-mine_labeled.csv"),
    "faceback_jsonl": os.path.join(DS_ROOT, "faceback_45", "boxes.jsonl"),
    "gt_bundle": os.path.join(DS_ROOT, "gt_bench_v1", "labels", "gt_bundle.json"),
    "uuid_map": os.path.join(DS_ROOT, "gt_bench_v1", "labels", "uuid_map.json"),
    "gt_sparse": os.path.join(DS_ROOT, "gt_bench_v1", "labels", "gt_eval_sparse.json"),
    "z2": os.path.join(CODE_ROOT, "training", "manifests", "train_Z2.txt"),
    "wider": os.path.join(DS_ROOT, "wider_face", "wider_train_labelv2.txt"),
}

SESSION = r"(\d{8}_\d{6}_[A-Z]{6})"
NAME_RE = {  # dataset -> (regex, group names)
    "face10k_v3": (re.compile(rf"^{SESSION}_(\d{{3}})_f(\d{{6}})\.jpg$"), ("session", "chunk", "frame")),
    "face10k_repair": (re.compile(rf"^{SESSION}_chunk_(\d{{3}})_(lview|rview)_f(\d{{6}})\.jpg$"), ("session", "chunk", "view", "frame")),
    "face_mine_v1": (re.compile(rf"^{SESSION}_c(\d{{3}})_f(\d{{6}})\.jpg$"), ("session", "chunk", "frame")),
    "faceback_45": (re.compile(rf"^{SESSION}_c(\d{{3}})_(left|right)_f(\d{{6}})\.jpg$"), ("session", "chunk", "view", "frame")),
    "gt_bench_sparse": (re.compile(rf"^{SESSION}_(\d{{3}})_f(\d{{6}})\.jpg$"), ("session", "chunk", "frame")),
    "gt_bench_full": (re.compile(rf"^{SESSION}_(\d{{3}})_f(\d{{6}})\.jpg$"), ("session", "chunk", "frame")),
    "pii_frames": (re.compile(rf"^{SESSION}_c(\d{{3}})_(lview|rview)_f(\d{{6}})\.jpg$"), ("session", "chunk", "view", "frame")),
    "wider_face": (re.compile(r"^[0-9A-Za-z_\-\.]+\.jpg$"), ()),
}
# labelv2 path prefix -> dataset; the manifest paths are relative to the mix
# root's images/ (training/mk_mix_Z2.py)
MANIFEST_DATASETS = ["pii_frames", "face10k_v3", "face10k_repair", "face_mine_v1", "faceback_45"]
PII_PATH = re.compile(rf"^pii/{SESSION}/chunk_(\d{{3}})/(lview|rview)/(f\d{{6}}\.jpg)$")


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def check(ok: bool, msg: str) -> None:
    if not ok:
        die(msg)


def fmt(v: float) -> str:
    return f"{v:.1f}"


def safe_field(s: str) -> bool:
    return s.isascii() and not any(c in s for c in ",\" \n\r\t")


def read_list(path: str) -> set[str]:
    s = {ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()}
    check(bool(s), f"empty list {path}")
    return s


def parse_name(dataset: str, name: str) -> dict:
    rx, groups = NAME_RE[dataset]
    m = rx.match(name)
    check(m is not None, f"{dataset}: name does not follow the dataset rule: {name}")
    d = dict(zip(groups, m.groups()))
    return {
        "session_id": d.get("session", ""),
        "chunk": str(int(d["chunk"])) if "chunk" in d else "",
        "view": d.get("view", ""),
        "frame_idx": str(int(d["frame"])) if "frame" in d else "",
    }


def manifest_path_to_key(path: str) -> tuple[str, str]:
    """labelv2 relative path -> (dataset, image basename as on OSS)."""
    if path.startswith("pii/"):
        m = PII_PATH.match(path)
        check(m is not None, f"unparseable pii path {path}")
        return "pii_frames", f"{m.group(1)}_c{m.group(2)}_{m.group(3)}_{m.group(4)}"
    for prefix, ds in (("face10k/images/", "face10k_v3"), ("face10k/repair_v1/", "face10k_repair"),
                       ("face_mine/", "face_mine_v1"), ("faceback/", "faceback_45"), ("wider/", "wider_face")):
        if path.startswith(prefix):
            name = path[len(prefix):]
            if ds == "wider_face":
                name = name.rsplit("/", 1)[-1]
            check("/" not in name, f"unexpected nesting in {path}")
            return ds, name
    die(f"manifest path with unknown prefix: {path}")


def key_to_manifest_path(dataset: str, image: str) -> str:
    if dataset == "pii_frames":
        p = parse_name(dataset, image)
        return f"pii/{p['session_id']}/chunk_{int(p['chunk']):03d}/{p['view']}/f{int(p['frame_idx']):06d}.jpg"
    return {"face10k_v3": "face10k/images/", "face10k_repair": "face10k/repair_v1/",
            "face_mine_v1": "face_mine/", "faceback_45": "faceback/"}[dataset] + image


def labelv2_blocks(path: str):
    """Yield (relpath, width, height, [(x1,y1,x2,y2,ignore) as strings]) in file order."""
    cur = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if ln.startswith("#"):
                if cur:
                    yield cur
                parts = ln.split()
                check(len(parts) == 4, f"bad header {ln!r}")
                cur = (parts[1], int(parts[2]), int(parts[3]), [])
            else:
                check(cur is not None, f"box line before any header: {ln!r}")
                v = ln.split()
                if len(v) == 19:
                    check(v[4:] == KPS.split(), f"non-placeholder keypoints: {ln!r}")
                    ignore = "0"
                elif len(v) == 5:
                    check(v[4] == "1", f"5-field line without ignore flag 1: {ln!r}")
                    ignore = "1"
                else:
                    die(f"box line with {len(v)} fields: {ln!r}")
                # re-format so a source written with a different float style still compares at 0.1 px
                cur[3].append(tuple(fmt(float(x)) for x in v[:4]) + (ignore,))
    if cur:
        yield cur


# ----------------------------------------------------------------- sources

def load_keys(path: str):
    """-> {(dataset, image): (size, md5)} in file order."""
    keys = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        ds, key = d["dataset"], d["oss_key"]
        check(ds in DATASETS, f"unknown dataset {ds}")
        check(key.startswith(f"pii/{ds}/"), f"key outside its prefix: {key}")
        image = key[len(f"pii/{ds}/"):]
        check("/" not in image and image.endswith(".jpg"), f"nested or non-jpg key {key}")
        check(safe_field(image), f"unsafe filename {image!r}")
        k = (ds, image)
        check(k not in keys, f"duplicate key {key}")
        keys[k] = (int(d["size"]), d["md5"])
    check(len(keys) == EXPECT_OSS, f"{len(keys)} keys != {EXPECT_OSS}")
    return keys


def load_roles(src: dict, sessions_by_ds: dict[str, set[str]]):
    """-> {(dataset, session): role} for face_mine_v1 / faceback_45 / gt_bench; exit 2 on disagreement."""
    usage: dict[str, tuple[set[str], str]] = {}
    with open(src["usage"], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["dataset"]:
                continue
            tags = set(r["dataset"].split("+"))
            prev = usage.setdefault(r["session_id"], (tags, r["role"]))
            check(prev == (tags, r["role"]), f"episode_usage: session {r['session_id']} has conflicting rows")
    lists = {
        "face_mine_v1": (read_list(src["split_train"]), read_list(src["split_eval"])),
        "faceback_45": (read_list(src["fb_train"]), read_list(src["fb_eval"])),
    }
    roles = {}
    disagreements = []
    for ds, (train, ev) in lists.items():
        check(not (train & ev), f"{ds}: {len(train & ev)} sessions in both split lists")
        for s in sorted(sessions_by_ds[ds]):
            in_list = ("train" if s in train else "") + ("eval" if s in ev else "")
            u = usage.get(s)
            u_role = u[1] if u and ds in u[0] else "(absent)"
            if in_list not in ("train", "eval") or u_role != in_list:
                disagreements.append((ds, s, f"split_list={in_list or '(absent)'}", f"episode_usage={u_role}"))
            roles[(ds, s)] = in_list
        listed_not_seen = (train | ev) - sessions_by_ds[ds]
        check(not listed_not_seen, f"{ds}: {len(listed_not_seen)} listed sessions have no frame, e.g. {sorted(listed_not_seen)[:3]}")
        print(f"roles {ds}: {len(train)} train + {len(ev)} eval sessions, all agree with episode_usage.csv")
    if disagreements:
        for d in disagreements:
            print("ROLE DISAGREEMENT:", *d, file=sys.stderr)
        die(f"{len(disagreements)} session(s) where episode_usage.csv and data/splits disagree; not guessing", code=2)
    for ds in ("gt_bench_sparse", "gt_bench_full"):
        for s in sorted(sessions_by_ds[ds]):
            u = usage.get(s)
            check(u is not None and "gt_bench_v1" in u[0] and u[1] == "eval",
                  f"{ds}: session {s} is not tagged gt_bench_v1/eval in episode_usage.csv ({u})")
            roles[(ds, s)] = "eval"
    return roles


def load_face_mine(path: str, W: int, H: int):
    boxes: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            fn = r["image_uri"].rsplit("/", 1)[-1]
            check((int(r["width"]), int(r["height"])) == (W, H), f"face_mine dims {fn}")
            lst = boxes.setdefault(fn, [])
            if r["box_i"] == "":
                check(int(r["n_boxes"]) == 0, f"face_mine: empty box_i but n_boxes != 0 for {fn}")
                continue
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            check(0 <= x and 0 <= y and x + w <= 1 and y + h <= 1 and w > 0 and h > 0, f"face_mine box out of range {fn}")
            lst.append((fmt(x * W), fmt(y * H), fmt((x + w) * W), fmt((y + h) * H), "0"))
    return boxes


def load_faceback(path: str, W: int, H: int):
    boxes: dict[str, list] = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        fn = r["image"].rsplit("/", 1)[-1]
        check((r["width"], r["height"]) == (W, H), f"faceback dims {fn}")
        check(fn not in boxes, f"faceback: duplicate frame {fn}")
        check(r["n_boxes"] == len(r["boxes"]), f"faceback: n_boxes mismatch {fn}")
        boxes[fn] = [(fmt(b["x"] * W), fmt(b["y"] * H), fmt((b["x"] + b["w"]) * W), fmt((b["y"] + b["h"]) * H), "0")
                     for b in r["boxes"]]
    return boxes


def load_gt_full(bundle_path: str, map_path: str, W: int, H: int):
    bundle = json.load(open(bundle_path, encoding="utf-8"))
    umap = json.load(open(map_path, encoding="utf-8"))
    boxes: dict[str, list] = {}
    for uuid, v in bundle.items():
        check(tuple(v["image_size"]) == (H, W), f"gt_bundle {uuid}: image_size {v['image_size']} is not (H, W) fisheye")
        if uuid not in umap:
            print(f"gt_bundle: uuid {uuid} absent from uuid_map.json: {len(v['frames'])} frames, "
                  f"{sum(len(b) for b in v['frames'].values())} boxes (not emitted)")
            continue
        session, chunk, _ = umap[uuid]
        for fidx, bl in v["frames"].items():
            fn = f"{session}_{int(chunk):03d}_f{int(fidx):06d}.jpg"
            check(fn not in boxes, f"gt_bundle: duplicate frame {fn}")
            boxes[fn] = [tuple(fmt(c) for c in b) + ("0",) for b in bl]
    return boxes


def load_gt_sparse(path: str):
    boxes: dict[str, list] = {}
    for e in json.load(open(path, encoding="utf-8")):
        fn = e["path"].rsplit("/", 1)[-1]
        check(fn not in boxes, f"gt_eval_sparse: duplicate frame {fn}")
        boxes[fn] = [tuple(fmt(c) for c in b) + ("0",) for b in e["boxes"]]
    return boxes


def load_labelv2(path: str, want: set[str]):
    """-> {dataset: {image: [boxes]}}, {(dataset,image): (w,h)} for the datasets in `want`."""
    boxes: dict[str, dict[str, list]] = defaultdict(dict)
    dims: dict[tuple[str, str], tuple[int, int]] = {}
    for relpath, w, h, bl in labelv2_blocks(path):
        ds, image = manifest_path_to_key(relpath)
        if ds not in want:
            continue
        check(image not in boxes[ds], f"{path}: duplicate frame {relpath}")
        boxes[ds][image] = bl
        dims[(ds, image)] = (w, h)
    return boxes, dims


# ----------------------------------------------------------------- build

def build(src: dict, out_dir: str) -> None:
    W, H = FISHEYE
    keys = load_keys(src["keys"])
    print(f"oss keys: {len(keys)} objects, " + ", ".join(f"{ds} {n}" for ds, n in sorted(Counter(k[0] for k in keys).items())))

    parsed = {k: parse_name(*k) for k in keys}
    sessions_by_ds: dict[str, set[str]] = defaultdict(set)
    for (ds, _), p in parsed.items():
        if p["session_id"]:
            sessions_by_ds[ds].add(p["session_id"])
    roles = load_roles(src, sessions_by_ds)

    # boxes per dataset
    boxes: dict[str, dict[str, list]] = {}
    boxes["face_mine_v1"] = load_face_mine(src["face_mine_csv"], W, H)
    boxes["faceback_45"] = load_faceback(src["faceback_jsonl"], W, H)
    boxes["gt_bench_full"] = load_gt_full(src["gt_bundle"], src["uuid_map"], W, H)
    boxes["gt_bench_sparse"] = load_gt_sparse(src["gt_sparse"])
    z2_boxes, z2_dims = load_labelv2(src["z2"], {"face10k_v3", "face10k_repair", "pii_frames"})
    boxes.update(z2_boxes)
    wider_boxes, wider_dims = load_labelv2(src["wider"], {"wider_face"})
    boxes.update(wider_boxes)
    for (ds, image), (w, h) in z2_dims.items():
        check((w, h) == (W, H), f"{ds}/{image}: manifest dims {w}x{h} != fisheye")

    # every labeled image must be an OSS object; every OSS image must be in its
    # label source, except that a "manifest" source may lack it (-> unlabeled)
    check(set(LABEL_SOURCE) == set(DATASETS), "LABEL_SOURCE must name every dataset once")
    unlabeled: dict[str, list[str]] = defaultdict(list)
    for ds in DATASETS:
        orphans = [n for n in boxes[ds] if (ds, n) not in keys]
        check(not orphans, f"{ds}: {len(orphans)} labeled images are not in oss_pii_keys.jsonl, e.g. {orphans[:3]}")
        missing = sorted(n for (d, n) in keys if d == ds and n not in boxes[ds])
        if LABEL_SOURCE[ds] == "perframe":
            check(not missing, f"{ds}: per-frame label source lacks {len(missing)} OSS images, "
                               f"e.g. {missing[:3]}; a per-frame source must cover every frame")
        else:
            unlabeled[ds] = missing
    unlabeled_set = {(ds, n) for ds, lst in unlabeled.items() for n in lst}
    unl_rows = sorted(unlabeled_set)
    print(f"unlabeled (in OSS, absent from the dataset's labelv2 manifest; role=unlabeled, n_boxes=0): {len(unl_rows)}")
    for ds, n in unl_rows:
        print(f"  {ds}/{n}")
    exp = EXPECT_UNLABELED
    check(len(unl_rows) == exp["n"], f"{len(unl_rows)} unlabeled rows != the expected {exp['n']}")
    for ds, n in unl_rows:
        p = parsed[(ds, n)]
        check(ds == exp["dataset"] and p["session_id"] == exp["session_id"] and p["chunk"] == exp["chunk"],
              f"unexpected unlabeled row {ds}/{n} (expected only {exp['dataset']} {exp['session_id']} chunk {exp['chunk']})")

    # rows
    frame_rows = []
    box_rows = []
    per_ds = Counter()
    per_ds_boxes = Counter()
    per_ds_ignore = Counter()
    per_ds_role = Counter()
    for (ds, image) in sorted(keys):
        size, md5 = keys[(ds, image)]
        p = parsed[(ds, image)]
        if ds == "wider_face":
            check((ds, image) in wider_dims, f"{ds}/{image}: no dims (wider_face dims come only from its labelv2 header)")
            w, h = wider_dims[(ds, image)]
        else:
            w, h = W, H
        if (ds, image) in unlabeled_set:
            role = "unlabeled"
        elif ds in ("face_mine_v1", "faceback_45", "gt_bench_sparse", "gt_bench_full"):
            role = roles[(ds, p["session_id"])]
        else:
            role = "train"
        check(role in ROLES, f"{ds}/{image}: role {role!r}")
        bl = boxes[ds].get(image, [])
        check(not (role == "unlabeled" and bl), f"{ds}/{image}: unlabeled but has boxes")
        bl = sorted(bl, key=lambda b: (float(b[0]), float(b[1]), float(b[2]), float(b[3]), b[4]))
        for b in bl:
            box_rows.append((ds, image) + b)
        row = [ds, image, p["session_id"], p["chunk"], p["view"], p["frame_idx"],
               str(w), str(h), str(size), md5, role, str(len(bl))]
        check(all(safe_field(x) for x in row), f"unsafe field in {row}")
        frame_rows.append(row)
        per_ds[ds] += 1
        per_ds_role[(ds, role)] += 1
        per_ds_boxes[ds] += len(bl)
        per_ds_ignore[ds] += sum(1 for b in bl if b[4] == "1")

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "frames.csv")
    bp = os.path.join(out_dir, "boxes.csv")
    with open(fp, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(FRAME_COLS) + "\n")
        for row in frame_rows:
            f.write(",".join(row) + "\n")
    with open(bp, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(BOX_COLS) + "\n")
        for row in box_rows:
            f.write(",".join(row) + "\n")
    for ds in DATASETS:
        roles_str = " ".join(f"{r} {per_ds_role[(ds, r)]}" for r in ROLES)
        print(f"{ds:16s} frames {per_ds[ds]:7d}  boxes {per_ds_boxes[ds]:7d}  (ignore {per_ds_ignore[ds]})  {roles_str}")
    print(f"wrote {fp}: {len(frame_rows)} rows; {bp}: {len(box_rows)} rows")


# ----------------------------------------------------------------- tables in

def load_tables(out_dir: str):
    fp = os.path.join(out_dir, "frames.csv")
    bp = os.path.join(out_dir, "boxes.csv")
    frames = {}
    with open(fp, encoding="utf-8") as f:
        check(f.readline().rstrip("\n") == ",".join(FRAME_COLS), f"{fp}: unexpected header")
        for ln in f:
            v = ln.rstrip("\n").split(",")
            check(len(v) == len(FRAME_COLS), f"{fp}: bad row {ln!r}")
            check(v[FRAME_COLS.index("role")] in ROLES, f"{fp}: unknown role in row {ln!r}")
            frames[(v[0], v[1])] = dict(zip(FRAME_COLS, v))
    boxes: dict[tuple[str, str], list] = defaultdict(list)
    with open(bp, encoding="utf-8") as f:
        check(f.readline().rstrip("\n") == ",".join(BOX_COLS), f"{bp}: unexpected header")
        for ln in f:
            v = ln.rstrip("\n").split(",")
            check(len(v) == len(BOX_COLS), f"{bp}: bad row {ln!r}")
            check((v[0], v[1]) in frames, f"{bp}: box for unknown frame {v[:2]}")
            boxes[(v[0], v[1])].append(tuple(v[2:]))
    for k, fr in frames.items():
        check(int(fr["n_boxes"]) == len(boxes.get(k, [])), f"n_boxes mismatch for {k}")
        check(not (fr["role"] == "unlabeled" and boxes.get(k)), f"{k}: role=unlabeled but boxes.csv has rows for it")
    return frames, boxes


def box_line(b: tuple) -> str:
    x1, y1, x2, y2, ign = b
    return f"{x1} {y1} {x2} {y2} 1" if ign == "1" else f"{x1} {y1} {x2} {y2} {KPS}"


def rebuild_blocks(frames, boxes):
    """Training manifest from the tables: every role=train frame of the five
    manifest datasets, datasets in mk_mix order, images sorted, boxes sorted."""
    out = []
    for ds in MANIFEST_DATASETS:
        for (d, image) in sorted(k for k in frames if k[0] == ds and frames[k]["role"] == "train"):
            fr = frames[(d, image)]
            out.append((key_to_manifest_path(ds, image), int(fr["width"]), int(fr["height"]),
                        [box_line(b) for b in boxes.get((d, image), [])]))
    return out


def blocks_text(blocks) -> str:
    lines = []
    for path, w, h, bl in blocks:
        lines.append(f"# {path} {w} {h}")
        lines.extend(bl)
    return "\n".join(lines) + "\n"


def check_manifest(ref_path: str, out_dir: str) -> None:
    frames, boxes = load_tables(out_dir)
    ref = list(labelv2_blocks(ref_path))
    ref_by_key = {}
    for relpath, w, h, bl in ref:
        k = manifest_path_to_key(relpath)
        check(k not in ref_by_key, f"reference has duplicate frame {relpath}")
        ref_by_key[k] = (relpath, w, h, bl)
    n_ref_boxes = sum(len(b[3]) for b in ref)
    print(f"reference {ref_path}: {len(ref)} frames / {n_ref_boxes} boxes")

    rebuilt = rebuild_blocks(frames, boxes)
    reb_by_key = {manifest_path_to_key(p): (p, w, h, bl) for p, w, h, bl in rebuilt}
    n_reb_boxes = sum(len(b[3]) for b in rebuilt)
    print(f"rebuild from tables: {len(rebuilt)} frames / {n_reb_boxes} boxes")

    failures = 0
    missing = sorted(set(ref_by_key) - set(reb_by_key))
    surplus = sorted(set(reb_by_key) - set(ref_by_key))
    if missing:
        failures += len(missing)
        print(f"FAIL: {len(missing)} reference frames absent from the rebuild, e.g. {missing[:5]}")
    if surplus:
        failures += len(surplus)
        print(f"FAIL: rebuild has {len(surplus)} role=train frames the reference lacks, e.g. {surplus[:5]}")
    else:
        print("rebuild surplus: 0 (every role=train frame of the manifest datasets is in the reference)")
    n_unl = Counter(k[0] for k in frames if frames[k]["role"] == "unlabeled")
    print(f"role=unlabeled frames kept out of the rebuild: {sum(n_unl.values())} {dict(sorted(n_unl.items()))}")

    # (image, box) pairs at 0.1 px, plus ignore flag and line bytes
    n_pairs = 0
    bad_pairs = 0
    bad_flags = 0
    bad_dims = 0
    bad_line_order = 0
    for k in sorted(set(ref_by_key) & set(reb_by_key)):
        rp, rw, rh, rbl = ref_by_key[k]
        bp_, bw, bh, bbl = reb_by_key[k]
        check(rp == bp_, f"path mapping not invertible: {rp} vs {bp_}")
        if (rw, rh) != (bw, bh):
            bad_dims += 1
        ref_pairs = sorted(b[:4] for b in rbl)
        reb_pairs = sorted(tuple(b[:4]) for b in boxes.get(k, []))
        n_pairs += len(ref_pairs)
        if ref_pairs != reb_pairs:
            bad_pairs += 1
            if bad_pairs <= 5:
                print(f"  box mismatch {k}: reference {ref_pairs} vs table {reb_pairs}")
            continue
        ref_lines = [box_line(b) for b in rbl]
        if sorted(ref_lines) != sorted(bbl):
            bad_flags += 1
        elif ref_lines != bbl:
            bad_line_order += 1
    failures += bad_pairs + bad_flags + bad_dims
    print(f"compared {len(set(ref_by_key) & set(reb_by_key))} frames / {n_pairs} boxes: "
          f"{bad_pairs} frames with a box-set difference at 0.1 px, {bad_flags} with an ignore-flag/format "
          f"difference, {bad_dims} with a width/height difference")

    # acceptance 3: no eval-role or unlabeled frame reaches the manifest
    leaked = Counter()
    for k in ref_by_key:
        if frames[k]["role"] != "train":
            leaked[(k[0], frames[k]["role"])] += 1
    if leaked:
        failures += sum(leaked.values())
        print(f"FAIL: non-train-role frames present in the reference manifest: {dict(leaked)}")
    for ds in ("face_mine_v1", "faceback_45"):
        ev = [k for k in frames if k[0] == ds and frames[k]["role"] == "eval"]
        in_ref = sum(1 for k in ev if k in ref_by_key)
        print(f"{ds}: {len(ev)} eval-role frames, {in_ref} of them in the reference manifest")

    # byte-identity level
    ref_text = open(ref_path, encoding="utf-8").read()
    reb_text = blocks_text(rebuilt)
    ordered = [reb_by_key[k] for k in (manifest_path_to_key(b[0]) for b in ref) if k in reb_by_key]
    ordered_text = blocks_text(ordered)
    ref_order = [b[0] for b in ref]
    reb_order = [b[0] for b in rebuilt if manifest_path_to_key(b[0]) in ref_by_key]
    print(f"byte identity: sorted rebuild == reference: {reb_text == ref_text}; "
          f"rebuild in reference block order == reference: {ordered_text == ref_text}; "
          f"blocks whose file position differs from the sorted rebuild: {sum(1 for a, b in zip(ref_order, reb_order) if a != b)}; "
          f"blocks whose box lines are not in (x1,y1,x2,y2) order: {bad_line_order}")
    if failures:
        die(f"check-manifest: {failures} difference(s) against {ref_path}")
    print("check-manifest OK: same (image, box) set at 0.1 px, same ignore flags, no eval-role frame in the manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=HERE, help="where frames.csv / boxes.csv are written or read (default data/)")
    ap.add_argument("--src", action="append", default=[], metavar="NAME=PATH",
                    help="override a source file; names: " + ", ".join(SRC))
    ap.add_argument("--check-manifest", metavar="LABELV2", help="compare a rebuild from the tables against this manifest")
    ap.add_argument("--rebuild-manifest", metavar="OUT", help="write the sorted rebuild of the training manifest")
    a = ap.parse_args()
    src = dict(SRC)
    for kv in a.src:
        name, _, path = kv.partition("=")
        check(name in SRC and path, f"bad --src {kv!r}")
        src[name] = path
    if a.check_manifest:
        check_manifest(a.check_manifest, a.out_dir)
    elif a.rebuild_manifest:
        frames, boxes = load_tables(a.out_dir)
        blocks = rebuild_blocks(frames, boxes)
        with open(a.rebuild_manifest, "w", encoding="utf-8", newline="\n") as f:
            f.write(blocks_text(blocks))
        print(f"wrote {a.rebuild_manifest}: {len(blocks)} frames / {sum(len(b[3]) for b in blocks)} boxes")
    else:
        build(src, a.out_dir)


if __name__ == "__main__":
    main()

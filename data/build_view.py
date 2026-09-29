"""Render a pii2 view (a training mix or an eval set) from its recipe.yaml (PII-1373).

A view is declared once under /data/esteban/pii/views/<name>/recipe.yaml and rendered to
every trainer format:

    recipe.yaml   hand written: name, role, sources [{dataset, boxes: vN, split, ...}], notes
                  a source may carry `min_side: S` (default 0, PII-1406): the long side floor
                  the legacy eval exports applied. It changes d2.json only, where a box under
                  the floor becomes iscrowd=1, exactly as evaluation/egoblur_d2/d2_dataset.py
                  did; scrfd.txt is untouched by it (mmdet applies its own floor at train
                  time), so a view's scrfd.txt is the same with and without a floor
    scrfd.txt     mmdet / SCRFD manifest, the format of training/manifests/*.txt:
                  "# <abs image path> <w> <h>" then one line per box,
                  "x1 y1 x2 y2" + 15 landmark placeholders (-1.0) for a positive,
                  "x1 y1 x2 y2 1" for an ignore region
    d2.json       detectron2 dict list, same content; evaluation/egoblur_d2/d2_dataset.py is
                  the reference for the format and for the ignore -> iscrowd=1 rule
    summary.json  per source frame / box / ignore counts (labelled, and `d2_*` after
                  min_side), md5 of the two outputs, timings, and the keys the egoblur eval
                  hook reads out of a legacy d2 summary

Image paths are absolute into /data/esteban/pii/datasets/<name>/images/; there is no symlink
tree. Nothing outside /data/esteban/pii/views/ is ever written.

    python3 data/build_view.py render <name> [<name> ...]   # or --all
    python3 data/build_view.py check  <name> [<name> ...]    # re-render to a temp dir and diff
    python3 data/build_view.py legacy <name> [<name> ...]    # compare with the legacy manifest
                                                             # or eval symlink dir of PII-1315

`legacy` is the reproduction proof: the legacy manifest paths
(views/mix_ds/images/<prefix>/...) are mapped to pii2 image names and the two files are
compared frame by frame and box by box; box order inside a frame and record order are
reported separately, since pii2 sorts boxes by (image, x1, y1) and the legacy blocks do not.

Stdlib only; PyYAML is used when it is importable, otherwise the small recipe subset is
parsed by the loader below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# PII-1449: the live store; data/pii_root.py resolves it (env PII_ROOT or PII2_ROOT,
# default /data/esteban/pii). A clone elsewhere (shang, fluence) sets the env var.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pii_root import LEGACY_ROOT, PII_ROOT  # noqa: E402

PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
DATASETS = Path(PII2_ROOT) / "datasets"
VIEWS = Path(PII2_ROOT) / "views"
DEFAULT_SIZE = (2328, 1748)          # every pii2 dataset but the two WIDER ones is raw fisheye
KPS = " ".join(["-1.0"] * 15)        # RetinaFace landmark placeholders, as in the legacy files
THING_CLASSES = ["face", "egoblur_class1"]   # d2_dataset.py: keep NUM_CLASSES=2


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ recipe (yaml subset)
def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return _mini_yaml(path.read_text(encoding="utf-8"))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scalar(tok: str):
    tok = tok.strip()
    if tok.startswith(("'", '"')) and tok.endswith(("'", '"')) and len(tok) > 1:
        return tok[1:-1]
    if tok in ("null", "~", ""):
        return None
    if tok in ("true", "false"):
        return tok == "true"
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    if tok.startswith("{") and tok.endswith("}"):     # flow mapping, strings only
        out = {}
        body = tok[1:-1].strip()
        if body:
            for part in body.split(","):
                k, _, v = part.partition(":")
                out[k.strip()] = _scalar(v)
        return out
    return tok


def _mini_yaml(text: str):
    """Enough YAML for these recipes: nested maps, lists of maps, scalars, `key: |` blocks."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip()), raw.rstrip()))

    def block(i: int, indent: int):
        node, i0 = None, i
        while i < len(lines):
            ind, raw = lines[i]
            if ind < indent:
                break
            body = raw.strip()
            if body.startswith("- "):
                if node is None:
                    node = []
                if not isinstance(node, list):
                    break
                inner = ind + 2
                lines[i] = (inner, " " * inner + body[2:])
                item, i = block(i, inner)
                node.append(item)
                continue
            if node is None:
                node = {}
            if not isinstance(node, dict):
                break
            key, _, rest = body.partition(":")
            key, rest = key.strip(), rest.strip()
            if rest == "|":
                buf, j = [], i + 1
                while j < len(lines) and lines[j][0] > ind:
                    buf.append(lines[j][1][ind + 2:])
                    j += 1
                node[key], i = "\n".join(buf) + "\n", j
                continue
            if rest == "":
                node[key], i = block(i + 1, ind + 1)
                continue
            node[key] = _scalar(rest)
            i += 1
        if node is None and i == i0:
            return None, i
        return node, i

    node, _ = block(0, 0)
    return node


# ------------------------------------------------------------------ dataset tables
def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dataset_frames(dataset: str):
    root = DATASETS / dataset
    if not root.is_dir():
        die(f"no such pii2 dataset: {root}")
    role = {r["session"]: r["role"] for r in read_csv(root / "split.csv")}
    rows = read_csv(root / "frames.csv")
    for r in rows:
        r["_role"] = role.get(r["session"], "?")
    return rows


def dataset_boxes(dataset: str, version: str):
    path = DATASETS / dataset / "boxes" / version / "boxes.csv"
    if not path.exists():
        die(f"{dataset} {version} has no boxes.csv (pending pass?): {path}")
    out: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["image"], []).append(
                (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), int(r["ignore"])))
    return out, path


def select(source: dict):
    """frames.csv rows of one source, filtered by split role and the optional `where` map."""
    rows = dataset_frames(source["dataset"])
    want = source.get("split", "all")
    if want not in ("all", "train", "eval"):
        die(f"bad split {want!r} in source {source}")
    if want != "all":
        rows = [r for r in rows if r["_role"] == want]
    for col, val in (source.get("where") or {}).items():
        if rows and col not in rows[0]:
            die(f"{source['dataset']}/frames.csv has no column {col!r}")
        rows = [r for r in rows if r[col] == str(val)]
    rows.sort(key=lambda r: r["image"])
    return rows


def size_of(row: dict) -> tuple[int, int]:
    if row.get("width") and row.get("height"):
        return int(row["width"]), int(row["height"])
    return DEFAULT_SIZE


# ------------------------------------------------------------------ render
def d2_crowd(x1: float, y1: float, x2: float, y2: float, ig: int, floor: float) -> int:
    """d2.json iscrowd, the d2_dataset.py rule (PII-1406): a stored ignore row, or a box whose
    long side is below the view's min_side floor, is an ignore region (iscrowd 1)."""
    return 1 if (ig or (floor > 0 and max(x2 - x1, y2 - y1) < floor)) else 0


def build_records(recipe: dict):
    """(records, per source counters). One record per frame, in recipe source order.

    Boxes are stored as labelled; `min_side` never rewrites them. It is carried on each record
    and applied by write_d2 alone, so scrfd.txt is byte identical with and without a floor."""
    recs, per_src = [], []
    for source in recipe["sources"]:
        ds, ver = source["dataset"], source["boxes"]
        floor = float(source.get("min_side") or 0)
        img_dir = DATASETS / ds / "images"
        boxes, boxes_csv = dataset_boxes(ds, ver)
        rows = select(source)
        n_box = n_ign = d2_box = d2_ign = d2_empty = 0
        for r in rows:
            bs = boxes.get(r["image"], [])
            n_ign += sum(b[4] for b in bs)
            n_box += len(bs) - sum(b[4] for b in bs)
            crowd = [d2_crowd(*b, floor) for b in bs]
            d2_ign += sum(crowd)
            d2_box += len(crowd) - sum(crowd)
            d2_empty += int(all(crowd) if crowd else True)
            w, h = size_of(r)
            recs.append({"path": str(img_dir / r["image"]), "w": w, "h": h, "boxes": bs,
                         "min_side": floor})
        per_src.append({"dataset": ds, "boxes_version": ver, "split": source.get("split", "all"),
                        "min_side": floor,
                        "frames": len(rows), "box_rows": n_box + n_ign,
                        "boxes": n_box, "ignore": n_ign,
                        "d2_boxes": d2_box, "d2_ignore": d2_ign, "d2_empty_frames": d2_empty,
                        "boxes_csv": str(boxes_csv), "boxes_md5": md5_file(boxes_csv)})
    return recs, per_src


def write_scrfd(recs, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(f"# {rec['path']} {rec['w']} {rec['h']}\n")
            for x1, y1, x2, y2, ig in rec["boxes"]:
                head = f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                f.write(f"{head} 1\n" if ig else f"{head} {KPS}\n")


def write_d2(recs, path: Path) -> None:
    out = []
    for i, rec in enumerate(recs):
        floor = rec.get("min_side") or 0.0
        ann = [{"bbox": [x1, y1, x2, y2], "bbox_mode": 0, "category_id": 0,
                "iscrowd": d2_crowd(x1, y1, x2, y2, ig, floor)}
               for x1, y1, x2, y2, ig in rec["boxes"]]
        out.append({"file_name": rec["path"], "width": rec["w"], "height": rec["h"],
                    "image_id": i, "annotations": ann})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def render(name: str, out_dir: Path | None = None, quiet: bool = False) -> dict:
    vdir = VIEWS / name
    recipe_path = vdir / "recipe.yaml"
    if not recipe_path.exists():
        die(f"no recipe: {recipe_path}")
    recipe = load_yaml(recipe_path)
    if recipe.get("name") != name:
        die(f"{recipe_path}: name {recipe.get('name')!r} != directory {name!r}")
    out_dir = out_dir or vdir
    if out_dir == vdir and not str(out_dir.resolve()).startswith(str(VIEWS)):
        die(f"refusing to write outside {VIEWS}: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    recs, per_src = build_records(recipe)
    t_build = time.time() - t0
    t1 = time.time()
    write_scrfd(recs, out_dir / "scrfd.txt")
    t_scrfd = time.time() - t1
    t2 = time.time()
    write_d2(recs, out_dir / "d2.json")
    t_d2 = time.time() - t2

    floors = sorted({s["min_side"] for s in per_src})
    summary = {
        "name": name,
        "role": recipe.get("role"),
        "recipe_md5": md5_file(recipe_path),
        "frames": len(recs),
        "boxes": sum(s["boxes"] for s in per_src),
        "ignore": sum(s["ignore"] for s in per_src),
        "empty_frames": sum(1 for r in recs if not any(b[4] == 0 for b in r["boxes"])),
        # d2_* are the same counts after min_side: what d2.json actually carries.
        "min_side": floors[0] if len(floors) == 1 else floors,
        "d2_boxes": sum(s["d2_boxes"] for s in per_src),
        "d2_ignore": sum(s["d2_ignore"] for s in per_src),
        "d2_empty_frames": sum(s["d2_empty_frames"] for s in per_src),
        "sources": per_src,
        "scrfd_txt": str(out_dir / "scrfd.txt"),
        "scrfd_md5": md5_file(out_dir / "scrfd.txt"),
        "d2_json": str(out_dir / "d2.json"),
        "d2_md5": md5_file(out_dir / "d2.json"),
        # keys the egoblur eval hook reads out of a legacy d2 *.summary.json
        # (evaluation/egoblur_d2/train_egoblur.py: n_images, boxes, ignore, json_md5 of the
        # train set; min_side, dataset, role, boxes_version, boxes_csv, boxes_md5 of the eval
        # set). dataset level keys are written only for a single source view.
        "n_images": len(recs),
        "json": str(out_dir / "d2.json"),
        "json_md5": md5_file(out_dir / "d2.json"),
        **({"dataset": per_src[0]["dataset"], "boxes_version": per_src[0]["boxes_version"],
            "boxes_csv": per_src[0]["boxes_csv"], "boxes_md5": per_src[0]["boxes_md5"]}
           if len(per_src) == 1 else {}),
        "thing_classes": THING_CLASSES,
        "seconds": {"build": round(t_build, 1), "scrfd": round(t_scrfd, 1),
                    "d2": round(t_d2, 1), "total": round(time.time() - t0, 1)},
        "rendered_by": "data/build_view.py render (PII-1373)",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    if not quiet:
        print(f"{name}: {summary['frames']} frames / {summary['boxes']} boxes / "
              f"{summary['ignore']} ignore in {summary['seconds']['total']}s -> {out_dir}")
    return summary


# ------------------------------------------------------------------ check
def check(name: str) -> bool:
    vdir = VIEWS / name
    for f in ("scrfd.txt", "d2.json", "summary.json"):
        if not (vdir / f).exists():
            die(f"{name}: {f} missing, render first")
    tmp = Path(tempfile.mkdtemp(prefix=f"view_{name}_"))
    try:
        fresh = render(name, out_dir=tmp, quiet=True)
        ok = True
        for f in ("scrfd.txt", "d2.json"):
            a, b = md5_file(vdir / f), md5_file(tmp / f)
            if a != b:
                print(f"{name}: {f} DIFFERS ({a} on disk, {b} re-rendered)")
                ok = False
        stored = json.load(open(vdir / "summary.json"))
        for k in ("frames", "boxes", "ignore", "min_side", "d2_boxes", "d2_ignore",
                  "scrfd_md5", "d2_md5", "recipe_md5"):
            if stored.get(k) != fresh.get(k):
                print(f"{name}: summary.json {k} {stored.get(k)} != {fresh.get(k)}")
                ok = False
        print(f"{name}: {'OK' if ok else 'MISMATCH'} "
              f"({fresh['frames']} frames, {fresh['boxes']} boxes, {fresh['ignore']} ignore)")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ legacy proof
PII_CHUNK = re.compile(r"^(.+)/chunk_(\d+)/(\w+)/f(\d+)\.jpg$")


def legacy_name(source: dict, rest: str) -> str:
    """Legacy path below views/mix_ds/images/<prefix>/ -> the pii2 frames.csv image name."""
    how = source.get("legacy_path", "flat")
    if how == "flat":
        return rest
    if how == "basename":
        return rest.rsplit("/", 1)[-1]
    if how == "chunked":                      # pii/<session>/chunk_NNN/<view>/fNNNNNN.jpg
        m = PII_CHUNK.match(rest)
        if not m:
            die(f"cannot map legacy path {rest!r}")
        return f"{m.group(1)}_c{m.group(2)}_{m.group(3)}_f{m.group(4)}.jpg"
    die(f"bad legacy_path {how!r}")


def parse_scrfd(path: Path):
    """-> list of (path, w, h, [(x1,y1,x2,y2,ignore)]), in file order."""
    recs, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                p = line[1:].split()
                cur = (p[0], int(p[1]), int(p[2]), [])
                recs.append(cur)
                continue
            v = line.split()
            if len(v) < 4:
                continue
            ig = 1 if (len(v) == 5 and float(v[4]) == 1) else 0
            cur[3].append((round(float(v[0]), 1), round(float(v[1]), 1),
                           round(float(v[2]), 1), round(float(v[3]), 1), ig))
    return recs


def legacy(name: str) -> bool:
    vdir = VIEWS / name
    recipe = load_yaml(vdir / "recipe.yaml")
    leg = recipe.get("legacy")
    if not leg or not leg.get("manifest"):
        print(f"{name}: no legacy manifest declared "
              f"({(leg or {}).get('note', 'nothing to reproduce')})")
        return True
    src_by_prefix = {s["legacy_prefix"]: s for s in recipe["sources"] if s.get("legacy_prefix")}
    # legacy name -> pii2 image name, per source (faceight rounds carry the legacy name in a column)
    alias = {}
    for prefix, s in src_by_prefix.items():
        col = s.get("legacy_name_col")
        if col:
            alias[prefix] = {r[col]: r["image"] for r in dataset_frames(s["dataset"])}

    mine_recs = parse_scrfd(vdir / "scrfd.txt")
    mine = {r[0]: r for r in mine_recs}
    mine_order = [r[0] for r in mine_recs]
    theirs, their_order = {}, []
    for path, w, h, bs in parse_scrfd(Path(leg["manifest"])):
        prefix, _, rest = path.partition("/")
        s = src_by_prefix.get(prefix)
        if s is None:
            die(f"{name}: legacy prefix {prefix!r} is not a source of this recipe")
        img = legacy_name(s, rest)
        img = alias.get(prefix, {}).get(img, img)
        full = str(DATASETS / s["dataset"] / "images" / img)
        theirs[full] = (full, w, h, bs)
        their_order.append(full)

    only_mine = sorted(set(mine) - set(theirs))
    only_theirs = sorted(set(theirs) - set(mine))
    diff_boxes = [p for p in set(mine) & set(theirs)
                  if sorted(mine[p][3]) != sorted(theirs[p][3])]
    diff_size = [p for p in set(mine) & set(theirs) if mine[p][1:3] != theirs[p][1:3]]
    same_order = mine_order == their_order
    sorted_order = sorted(mine_order) == sorted(their_order)
    box_order_diff = sum(1 for p in set(mine) & set(theirs) if mine[p][3] != theirs[p][3])
    ok = not (only_mine or only_theirs or diff_boxes or diff_size)
    print(f"{name}: legacy {leg['manifest']}")
    print(f"  frames rendered {len(mine)} / legacy {len(theirs)}; "
          f"only rendered {len(only_mine)}, only legacy {len(only_theirs)}")
    print(f"  frames with different boxes (1 decimal, sorted): {len(diff_boxes)}; "
          f"different w/h: {len(diff_size)}")
    print(f"  record order identical: {same_order} (identical once both are sorted by path: "
          f"{sorted_order}); frames whose box order differs: {box_order_diff}")
    for p in (only_mine[:3] + only_theirs[:3] + diff_boxes[:3]):
        print(f"    e.g. {p}")
    print(f"  {'REPRODUCED' if ok else 'DIFFERS'}")
    return ok


def legacy_dir(name: str) -> bool:
    """Eval views whose legacy form is a symlink directory, not a manifest: compare frame sets."""
    vdir = VIEWS / name
    recipe = load_yaml(vdir / "recipe.yaml")
    leg = recipe.get("legacy") or {}
    d = leg.get("dir")
    if not d:
        return True
    s = recipe["sources"][0]
    col = s.get("legacy_name_col")
    frames = dataset_frames(s["dataset"])
    alias = {r[col]: r["image"] for r in frames} if col else {r["image"]: r["image"] for r in frames}
    want = {alias.get(n, n) for n in os.listdir(d)}
    got = {Path(r[0]).name for r in parse_scrfd(vdir / "scrfd.txt")}
    print(f"{name}: legacy dir {d}: {len(want)} links vs {len(got)} rendered frames; "
          f"only legacy {len(want - got)}, only rendered {len(got - want)} -> "
          f"{'SAME FRAME SET' if want == got else 'DIFFERS'}")
    return want == got


# ------------------------------------------------------------------ cli
def view_names(args) -> list[str]:
    if args.all:
        return sorted(p.name for p in VIEWS.iterdir() if (p / "recipe.yaml").exists())
    if not args.names:
        die("give one or more view names, or --all")
    return args.names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("render", "check", "legacy"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ok = True
    for name in view_names(args):
        if args.cmd == "render":
            render(name)
        elif args.cmd == "check":
            ok &= check(name)
        else:
            recipe = load_yaml(VIEWS / name / "recipe.yaml")
            leg = recipe.get("legacy") or {}
            ok &= legacy_dir(name) if leg.get("dir") else legacy(name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

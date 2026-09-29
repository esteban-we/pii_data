#!/usr/bin/env python3
"""Build pii-data datasets/faceback_hq/boxes/v2.csv from the round-2 vendor review (PII-960).

v2 is a full replacement of v1, not a patch: the 4,023 frames the vendor re-reviewed take
the boxes the drop returns, every other frame of the 16,940 keeps its v1 rows byte for
byte. v1 is never touched.

Input
  /data/esteban/tmp/pii/fb_hq_2/face_boxes_faceback_hq_2-{left,right}.jsonl
  2,110 + 1,913 frame records plus one {"kind": "end"} trailer each, 12,786 boxes,
  review_round 2, machine_src faceback_hq_2/gt+armAE34@1. Boxes are normalised top-left
  x,y,w,h in 2328x1748 (the PII-99 / PII-176 faceback convention; `check` re-establishes
  it by counting boxes that coincide with a v1 box at 0.1 px under each candidate).

Frame identity
  The drop keys by (session, chunk, view, frame_idx) and its image_uri basename carries no
  eye; the pii-data image is <session>_c<chunk>_<left|right>_f<idx:06d>.jpg with left for
  view "lview" and right for "rview" (PII-176). Every mapped name must be a faceback_hq
  row of frames.csv and must agree with that row's session_id, chunk, view, frame_idx,
  width and height.

ignore
  v1 carries ignore=0 on all 23,016 rows: the 40 px long-side floor is applied by the
  eval (min_side 40), it was never written into this column, and 3,969 v1 rows are under
  it. `check` prints both candidate rules; v2 keeps v1's, ignore=0, so that reviewed and
  untouched frames stay in one convention (PII-960).

Layout
  boxes/v2.frames.txt lists the 4,023 frames the pass covered, one image name per line,
  sorted, as the repo README requires of a partial box version (face_mine_v1 v2 is the
  precedent). Same columns and same order as v1: image,x1,y1,x2,y2,ignore, 1-decimal pixel corners,
  rows sorted by (image, x1, y1) with x1/y1 compared as numbers and the image name as a
  string, which is exactly how v1 is sorted.

usage: python3 mining/faceback_hq_boxes_v2.py build   [--dry-run]
       python3 mining/faceback_hq_boxes_v2.py check
       python3 mining/faceback_hq_boxes_v2.py report  [--out FILE]
Never commits.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(*([os.path.dirname(os.path.abspath(__file__))]
                                 + [os.pardir] * 3 + ["data"])))
from pii_root import dataset as pii_dataset  # noqa: E402

# PII-1449: pii-data is superseded by the PII-1315 store, where this dataset is
# faceback_45 and these boxes are boxes/v3/boxes.csv. The v1.csv / v2.csv paths
# below are the pii-data file layout, so --root still defaults to that checkout;
# porting the reader to boxes/vN/boxes.csv is a separate change.
DATA_ROOT = "/home/esteban/repos/pii-data"
DROP_DIR = "/data/esteban/tmp/pii/fb_hq_2"
_PRELABELS = pii_dataset("faceback_45") / "boxes" / "v3" / "job" / "prelabels"
ADDITIONS = str(_PRELABELS / "additions.jsonl")
IMPORT_JSONL = str(_PRELABELS / "import.jsonl")
DATASET = "faceback_hq"
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
EYE_OF_VIEW = {"lview": "left", "rview": "right"}
NO_EYE_NAME = re.compile(r"^(.+)_f(\d+)\.jpg$")
EXPECT_FRAMES = 16_940
EXPECT_REVIEWED = 4_023
EXPECT_DROP_BOXES = 12_786
EXPECT_V1_ROWS = 23_016
SCORE_BANDS = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
SIDE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 1e9)]


def v1_path(root): return os.path.join(root, "datasets", DATASET, "boxes", "v1.csv")
def v2_path(root): return os.path.join(root, "datasets", DATASET, "boxes", "v2.csv")
def v2_frames_path(root): return os.path.join(root, "datasets", DATASET, "boxes", "v2.frames.txt")


def read_drop(path):
    """(frame records, trailers) of one vendor jsonl; a line without `session` is a trailer."""
    recs, trailers = [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            (recs if "session" in d else trailers).append(d)
    if len(trailers) != 1 or trailers[0].get("kind") != "end" or trailers[0].get("frames") != len(recs):
        sys.exit(f"{path}: trailers {trailers} do not match {len(recs)} frame records")
    for d in recs:
        if d["n_boxes"] != len(d["boxes"]):
            sys.exit(f"{path}: {d['image_uri']} n_boxes {d['n_boxes']} != {len(d['boxes'])}")
    return recs, trailers[0]


def load_drop(drop_dir=DROP_DIR, verbose=False):
    recs = []
    for eye in ("left", "right"):
        p = os.path.join(drop_dir, f"face_boxes_faceback_hq_2-{eye}.jsonl")
        rs, tr = read_drop(p)
        if verbose:
            print(f"{os.path.basename(p)}: {len(rs)} frames, {sum(r['n_boxes'] for r in rs)} boxes, "
                  f"trailer {tr}, review_round {dict(Counter(r.get('review_round') for r in rs))}, "
                  f"machine_src {sorted({r.get('machine_src') for r in rs})}, "
                  f"{len({r.get('labeled_by') for r in rs})} labelers")
        recs += rs
    return recs


def image_of(d):
    """pii-data image name of a drop record (PII-176 owner-name-from-view mapping)."""
    base = d["image_uri"].rsplit("/", 1)[-1]
    eye = EYE_OF_VIEW.get(d.get("view"))
    if eye is None:
        sys.exit(f"{base}: view {d.get('view')!r} is not one of {sorted(EYE_OF_VIEW)}")
    m = NO_EYE_NAME.match(base)
    if not m:
        sys.exit(f"{base}: basename is not <stem>_f<idx>.jpg; cannot insert the eye")
    return f"{m.group(1)}_{eye}_f{m.group(2)}.jpg"


def to_px(b, W, H):
    """Normalised top-left x,y,w,h -> 1-decimal pixel xyxy."""
    x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate box {b}")
    return (round(x * W, 1), round(y * H, 1), round((x + w) * W, 1), round((y + h) * H, 1))


def read_boxes(path):
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        if hdr != BOX_COLS:
            sys.exit(f"{path}: unexpected header {hdr}")
        return [r for r in rd]


def frames_of(root, dataset=DATASET):
    out = {}
    with open(os.path.join(root, "frames.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset:
                out[r["image"]] = r
    return out


def resolve(recs, frames):
    """image name -> record, with every identity assertion of the docstring."""
    by_image = {}
    for d in recs:
        image = image_of(d)
        if image in by_image:
            sys.exit(f"{image}: mapped by two drop records")
        o = frames.get(image)
        if o is None:
            sys.exit(f"{image}: not a {DATASET} row of frames.csv")
        want = (d["session"], int(d["chunk"]), EYE_OF_VIEW[d["view"]], int(d["frame_idx"]))
        have = (o["session_id"], int(o["chunk"]), o["view"], int(o["frame_idx"]))
        if want != have:
            sys.exit(f"{image}: record {want} != frames.csv row {have}")
        if (int(d["width"]), int(d["height"])) != (int(o["width"]), int(o["height"])):
            sys.exit(f"{image}: drop {d['width']}x{d['height']} != frames.csv "
                     f"{o['width']}x{o['height']}")
        by_image[image] = d
    return by_image


def build_rows(root):
    """The v2 rows, as lists of strings. Pure: reads v1 and the drop, writes nothing."""
    frames = frames_of(root)
    recs = load_drop()
    by_image = resolve(recs, frames)
    v1 = read_boxes(v1_path(root))
    per_image = defaultdict(list)
    for r in v1:
        per_image[r[0]].append(r)
    if set(per_image) - set(frames):
        sys.exit("v1 has images that are not frames.csv faceback_hq rows")

    for image, d in by_image.items():
        W, H = int(d["width"]), int(d["height"])
        px = sorted(to_px(b, W, H) for b in d["boxes"])
        per_image[image] = [[image, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", "0"]
                            for x1, y1, x2, y2 in px]
        if not px:
            del per_image[image]
    rows = []
    for image in sorted(per_image):
        rows += per_image[image]
    if rows != sorted(rows, key=lambda r: (r[0], float(r[1]), float(r[2]))):
        sys.exit("internal: rows are not in v1's (image, x1, y1) order")
    return rows, by_image, v1


def write_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(BOX_COLS)
        w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------- geometry helpers
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(old, new, thr=0.5):
    """Greedy best-IoU matching. -> (pairs, unmatched_old_idx, unmatched_new_idx)."""
    cand = sorted(((iou(o, n), i, j) for i, o in enumerate(old) for j, n in enumerate(new)),
                  key=lambda t: (-t[0], t[1], t[2]))
    used_o, used_n, pairs = set(), set(), []
    for v, i, j in cand:
        if v < thr or i in used_o or j in used_n:
            continue
        used_o.add(i); used_n.add(j); pairs.append((i, j, v))
    return pairs, [i for i in range(len(old)) if i not in used_o], \
                  [j for j in range(len(new)) if j not in used_n]


def band(value, bands):
    for lo, hi in bands:
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
    return "out"


def xyxy(r): return (float(r[1]), float(r[2]), float(r[3]), float(r[4]))


# ---------------------------------------------------------------- commands
def cmd_check(args):
    root = args.root
    frames = frames_of(root)
    print(f"frames.csv {DATASET} rows: {len(frames)} "
          f"(roles {dict(Counter(r['role'] for r in frames.values()))})")
    assert len(frames) == EXPECT_FRAMES, len(frames)

    recs = load_drop(verbose=True)
    print(f"drop: {len(recs)} frame records, {sum(r['n_boxes'] for r in recs)} boxes")
    assert len(recs) == EXPECT_REVIEWED and sum(r["n_boxes"] for r in recs) == EXPECT_DROP_BOXES
    by_image = resolve(recs, frames)
    print(f"identity: all {len(by_image)} drop frames resolve to distinct {DATASET} frames.csv rows "
          f"and agree on session, chunk, view, frame_idx, width, height")

    v1 = read_boxes(v1_path(root))
    print(f"v1: {len(v1)} rows over {len({r[0] for r in v1})} images, "
          f"ignore {dict(Counter(r[5] for r in v1))}")
    assert len(v1) == EXPECT_V1_ROWS

    # the decisive ignore check: which rule reproduces v1's own flags from v1's coordinates
    def long_side(r):
        x1, y1, x2, y2 = xyxy(r)
        return max(x2 - x1, y2 - y1)
    floor = sum(1 for r in v1 if (long_side(r) < 40) != (r[5] == "1"))
    const = sum(1 for r in v1 if r[5] != "0")
    print(f"ignore rule on v1's own coordinates: 'long side < 40 -> 1' mismatches {floor} rows; "
          f"'always 0' mismatches {const}; v1 rows under 40 px: "
          f"{sum(1 for r in v1 if long_side(r) < 40)}")
    if const != 0:
        sys.exit("v1's ignore column is not constant 0; the v2 rule must be re-derived")

    # coordinate convention, re-established on this drop
    W, H = 2328, 1748
    per_image = defaultdict(set)
    for r in v1:
        per_image[r[0]].add(xyxy(r))
    def conv(b, kind):
        x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
        if kind == "tlwh":
            return (round(x * W, 1), round(y * H, 1), round((x + w) * W, 1), round((y + h) * H, 1))
        if kind == "cxcywh":
            return (round((x - w / 2) * W, 1), round((y - h / 2) * H, 1),
                    round((x + w / 2) * W, 1), round((y + h / 2) * H, 1))
        return (round(x * W, 1), round(y * H, 1), round(w * W, 1), round(h * H, 1))
    for kind in ("tlwh", "cxcywh", "xyxy"):
        hit = sum(conv(b, kind) in per_image.get(im, ()) for im, d in by_image.items() for b in d["boxes"])
        print(f"convention {kind}: {hit}/{EXPECT_DROP_BOXES} drop boxes coincide with a v1 box at 0.1 px")

    p2 = v2_path(root)
    if not os.path.exists(p2):
        print(f"{p2}: absent, nothing further to check")
        return
    v2 = read_boxes(p2)
    rows, _, _ = build_rows(root)
    print(f"v2: {len(v2)} rows over {len({r[0] for r in v2})} images; "
          f"rebuild identical: {rows == v2}")
    if rows != v2:
        want, have = defaultdict(list), defaultdict(list)
        for r in rows: want[r[0]].append(r)
        for r in v2: have[r[0]].append(r)
        bad = sorted(im for im in set(want) | set(have) if want.get(im, []) != have.get(im, []))
        sys.exit(f"v2.csv is not what a rebuild produces: {len(bad)} frames differ, "
                 f"first {bad[:4]}")

    v1_by, v2_by = defaultdict(list), defaultdict(list)
    for r in v1: v1_by[r[0]].append(r)
    for r in v2: v2_by[r[0]].append(r)
    untouched = set(frames) - set(by_image)
    bad = [im for im in untouched if v1_by.get(im, []) != v2_by.get(im, [])]
    print(f"untouched frames: {len(untouched)}; rows differing from v1: {len(bad)}")
    assert not bad, bad[:5]
    stray = set(v2_by) - set(frames)
    print(f"v2 images outside the {EXPECT_FRAMES} frames of the set: {len(stray)}")
    assert not stray
    print(f"coverage: {len(v2_by)} frames with >= 1 row + {EXPECT_FRAMES - len(v2_by)} with none "
          f"= {EXPECT_FRAMES}")
    print("ignore column:", dict(Counter(r[5] for r in v2)))
    listing = v2_frames_path(root)
    if os.path.exists(listing):
        names = [ln.rstrip("\n") for ln in open(listing)]
        print(f"v2.frames.txt: {len(names)} names, sorted {names == sorted(names)}, "
              f"equals the drop frames {set(names) == set(by_image)}")
        assert len(names) == EXPECT_REVIEWED and names == sorted(names) and set(names) == set(by_image)
    else:
        sys.exit(f"{listing} is missing")
    print("OK")


def cmd_build(args):
    root = args.root
    rows, by_image, v1 = build_rows(root)
    out = v2_path(root)
    print(f"v1 {len(v1)} rows over {len({r[0] for r in v1})} images -> "
          f"v2 {len(rows)} rows over {len({r[0] for r in rows})} images "
          f"({len(by_image)} frames re-reviewed)")
    listing = v2_frames_path(root)
    if args.dry_run:
        print(f"--dry-run: {out} and {listing} not written")
        return
    write_csv(out, rows)
    with open(listing, "w") as f:
        f.write("".join(im + "\n" for im in sorted(by_image)))
    print(f"wrote {out} and {listing} ({len(by_image)} frames sent for the pass)")


def load_additions():
    """The machine additions PII-946 proposed: image -> [(xyxy, score)]."""
    out = {}
    with open(ADDITIONS) as f:
        for ln in f:
            d = json.loads(ln)
            out[d["file"]] = [(tuple(b["xyxy"]), b["score"]) for b in d["boxes"]]
    return out


def cmd_report(args):
    root = args.root
    v1, v2 = read_boxes(v1_path(root)), read_boxes(v2_path(root))
    recs = load_drop()
    frames = frames_of(root)
    reviewed = set(resolve(recs, frames))
    v1_by, v2_by = defaultdict(list), defaultdict(list)
    for r in v1: v1_by[r[0]].append(xyxy(r))
    for r in v2: v2_by[r[0]].append(xyxy(r))

    n1 = sum(len(v1_by.get(i, ())) for i in reviewed)
    n2 = sum(len(v2_by.get(i, ())) for i in reviewed)
    identical = moved = added = removed = 0
    for im in reviewed:
        old, new = v1_by.get(im, []), v2_by.get(im, [])
        pairs, un_o, un_n = match(old, new)
        for i, j, _ in pairs:
            if old[i] == new[j]:
                identical += 1
            else:
                moved += 1
        added += len(un_n)
        removed += len(un_o)
    print(f"rows: v1 {len(v1)} / v2 {len(v2)} overall; on the {len(reviewed)} reviewed frames "
          f"v1 {n1} / v2 {n2}")
    print(f"on the reviewed frames: identical {identical}, moved {moved}, added {added}, "
          f"removed {removed}")
    print(f"frames with >= 1 row: v1 {len(v1_by)} / v2 {len(v2_by)}; "
          f"face-free: v1 {EXPECT_FRAMES - len(v1_by)} / v2 {EXPECT_FRAMES - len(v2_by)}")

    # additions kept vs dropped
    adds = load_additions()
    kept_s, drop_s, kept_l, drop_l = Counter(), Counter(), Counter(), Counter()
    kept = dropped = 0
    loose = 0          # dropped at IoU 0.5 but still overlapping a v2 box at IoU >= 0.2
    add_matched = {}   # image -> set of v2 box indices explained by a kept addition
    for im, boxes in adds.items():
        new = v2_by.get(im, [])
        pairs, _, _ = match([b for b, _ in boxes], new)
        hit = {i for i, _, _ in pairs}
        add_matched[im] = {j for _, j, _ in pairs}
        for i, (b, _) in enumerate(boxes):
            if i not in hit and new and max(iou(b, n) for n in new) >= 0.2:
                loose += 1
        for i, (b, sc) in enumerate(boxes):
            ls = max(b[2] - b[0], b[3] - b[1])
            if i in hit:
                kept += 1; kept_s[band(sc, SCORE_BANDS)] += 1; kept_l[band(ls, SIDE_BANDS)] += 1
            else:
                dropped += 1; drop_s[band(sc, SCORE_BANDS)] += 1; drop_l[band(ls, SIDE_BANDS)] += 1
    total = kept + dropped
    print(f"\nmachine additions proposed {total}: kept {kept} ({kept/total:.1%}), dropped {dropped}")
    for name, k, d, bands in (("score", kept_s, drop_s, SCORE_BANDS), ("long side", kept_l, drop_l, SIDE_BANDS)):
        print(f"  by {name}:")
        for lo, hi in bands:
            key = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
            tot = k[key] + d[key]
            print(f"    {key:>10}  proposed {tot:5d}  kept {k[key]:5d}  dropped {d[key]:5d}"
                  + (f"  {k[key]/tot:.1%}" if tot else ""))

    # v1 GT boxes deleted on the reviewed frames
    gt_del = 0
    for im in reviewed:
        _, un_o, _ = match(v1_by.get(im, []), v2_by.get(im, []))
        gt_del += len(un_o)
    print(f"  dropped at IoU 0.5 but still overlapping a v2 box at IoU >= 0.2: {loose}")

    # where do the v2 boxes of the reviewed frames come from
    from_v1 = from_add = fresh = 0
    for im in reviewed:
        old, new = v1_by.get(im, []), v2_by.get(im, [])
        pairs, _, _ = match(old, new)
        m1 = {j for _, j, _ in pairs}
        ma = add_matched.get(im, set())
        for j in range(len(new)):
            if j in m1: from_v1 += 1
            elif j in ma: from_add += 1
            else: fresh += 1
    print(f"\nv2 boxes on the reviewed frames {n2}: {from_v1} match a v1 GT box, "
          f"{from_add} a proposed addition, {fresh} neither (drawn fresh by the humans)")
    print(f"\nv1 GT boxes on the reviewed frames: {n1}; deleted by the humans: {gt_del}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "check", "report"])
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the faceight_b / faceight_c Verdict packages (WOR-195): import.jsonl + README.md.

Set: the rows of frames.csv at annotation_round == --round (2 = faceight_b, 60,000 frames;
3 = faceight_c, 40,000; both eyes; drawn in WOR-191 by cell shares, seed 19760703). Each frame
is put in its cell by faceight_cells.cell_of (imported, not restated): sW / sA = max box score
of armW / armAA34 in the four JSONLs of --base, band = band(max(sW, sA)) in 0.6+ / [0.3,0.6) /
[0.1,0.3) / faceless, class both / armW only / AA34 only within the band.

Prelabel rule (user, 2026-09-12): per frame exactly ONE detector, chosen by the cell:
  armW only cells          -> armW boxes (det_10g_armW)
  both and AA34 only cells -> armAA34 boxes (det_34g_armAA34)
  faceless                 -> no boxes
and only that detector's boxes with score >= the cell's band edge (0.6+ -> 0.6, [0.3,0.6) -> 0.3,
[0.1,0.3) -> 0.1; the edge is faceight_cells.LO[band]). Nothing else is written: no second
detector, no sub-edge box, no score.

Record (one line per frame, the faceight_a / faceback-45 importer shape):
  session, chunk ("%03d"), view ("lview" / "rview"), frame_idx, width 2328, height 1748,
  boxes [{x, y, w, h}] normalized to the image, 4 dp, clipped to [0, 1], zero-area dropped
  (exactly as faceight_prep.py did for round 1), image "<session>_c<chunk>_f<frame_idx:06d>.jpg".
The image name carries no eye token (the faceback-45 importer convention: import_right.jsonl
there has view "rview" and an eye-less image); the eye is the view field. The script refuses to
write if a name repeats within the round or is one of the faceight_a names (<base>/verdict/import_left.jsonl).

    faceight_verdict_prep.py --round {2,3} [--out-dir DIR] [--dry-run]
        [--base /data/esteban/faceight] [--expect-md5 MD5] [--draw-csv CSV] [--no-disk-check]

Defaults: --out-dir <base>/verdict_<b|c>; frames.csv, the JSONLs, round<R>_select.txt,
round23_draw.csv and verdict/import_left.jsonl are read from --base. Checks (all fatal):
frames.csv md5 == --expect-md5; the round's row set == round<R>_select.txt; every frame has one
armW and one armAA34 record with w 2328 x h 1748 and the same frame_idx as frames.csv; the cell
== round23_draw.csv's cell for every frame; every jpg exists under frames/ or frames_right/.
Stdlib + numpy (faceight_cells needs numpy); run on shang with /data/esteban/armw/venv/bin/python.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faceight_buckets import EYES, JSONLS, md5  # noqa: E402
from faceight_cells import CELLS, FACELESS, LO, cell_name, cell_of  # noqa: E402
from faceight_round import HEADERS, read_table  # noqa: E402

BASE = "/data/esteban/faceight"
ROUNDS = {2: "b", 3: "c"}
EXPECT_MD5 = "bb56ee8f767a55239700f717f75361ed"  # frames.csv after WOR-191
EXPECT_ROWS = {2: 60000, 3: 40000}
W, H = 2328, 1748
DETECTOR = {"armW only": "armW", "both": "AA34", "AA34 only": "AA34"}
MODEL = {"armW": "det_10g_armW", "AA34": "det_34g_armAA34"}
VIEW = {"vst_left": "lview", "vst_right": "rview"}
FRAMES_DIR = {"vst_left": "frames", "vst_right": "frames_right"}
BUCKET = "oss://we-vlm-annotation-data-sh"


def image_name(session, chunk, frame_idx):
    return f"{session}_c{int(chunk):03d}_f{int(frame_idx):06d}.jpg"


def norm_box(x1, y1, x2, y2):
    """faceight_prep.py's conversion: clip to [0, 1], xywh, 4 dp; None when the rounded area is zero."""
    x = max(0, min(1, x1 / W))
    y = max(0, min(1, y1 / H))
    w = max(0, min(1, x2 / W)) - x
    h = max(0, min(1, y2 / H)) - y
    b = {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}
    if b["w"] <= 0 or b["h"] <= 0:
        return None
    return b


def select_rows(csv_path, rnd):
    _, header, rows = read_table(csv_path)
    if header != HEADERS[-1]:
        sys.exit(f"{csv_path}: need the 14-column header (view, n_faces_aa34), got {header}")
    col = {c: i for i, c in enumerate(header)}
    out = []
    for r in rows:
        if r[col["annotation_round"]] != str(rnd):
            continue
        d = {c: r[col[c]] for c in ("file", "episode_id", "session_id", "chunk", "t_ms", "frame_idx", "view")}
        if d["view"] not in EYES:
            sys.exit(f"{d['file']}: view {d['view']!r} is not one of {EYES}")
        out.append(d)
    return out


def load_records(jsonl_dir, files):
    """detector -> {file: record} for the wanted files only; every wanted file must have one record per detector."""
    want = set(files)
    out = {}
    for det, names in JSONLS.items():
        m = {}
        for name in names:
            path = os.path.join(jsonl_dir, name)
            with open(path) as f:
                for ln, line in enumerate(f, 1):
                    r = json.loads(line)
                    fn = r["file"]
                    if fn not in want:
                        continue
                    if fn in m:
                        sys.exit(f"{path} line {ln}: file {fn!r} already seen for {det}")
                    if r.get("model") != MODEL[det]:
                        sys.exit(f"{path} line {ln}: model {r.get('model')!r}, expected {MODEL[det]!r}")
                    if r.get("w") != W or r.get("h") != H:
                        sys.exit(f"{path} line {ln}: {fn} is {r.get('w')} x {r.get('h')}, expected {W} x {H}")
                    boxes = r.get("boxes")
                    if not isinstance(boxes, list) or any(len(b) != 5 for b in boxes):
                        sys.exit(f"{path} line {ln}: bad boxes list ({fn})")
                    m[fn] = r
            print(f"{det}: {name}: {len(m):,} wanted records so far", file=sys.stderr)
        miss = [f for f in files if f not in m]
        if miss:
            sys.exit(f"{len(miss)} frames have no {det} record in {jsonl_dir}, e.g. {miss[:5]}")
        out[det] = m
    return out


def prelabel(cell, recs):
    """(boxes, n_dropped) for one frame: the chosen detector's boxes at score >= the band edge, normalized."""
    cls, band = cell
    if cls == FACELESS:
        return [], 0
    det, edge = DETECTOR[cls], LO[band]
    boxes, dropped = [], 0
    for x1, y1, x2, y2, sc in recs[det]["boxes"]:
        if sc < edge:
            continue
        b = norm_box(x1, y1, x2, y2)
        if b is None:
            dropped += 1
            continue
        boxes.append(b)
    return boxes, dropped


def build(rnd, base, expect_md5, draw_csv, disk_check):
    tag = ROUNDS[rnd]
    csv_path = os.path.join(base, "frames.csv")
    src = {"frames.csv": md5(csv_path)}
    if expect_md5 and src["frames.csv"] != expect_md5:
        sys.exit(f"{csv_path}: md5 {src['frames.csv']} != expected {expect_md5} (pass --expect-md5 to override)")
    rows = select_rows(csv_path, rnd)
    if len(rows) != EXPECT_ROWS[rnd]:
        sys.exit(f"round {rnd}: {len(rows):,} rows, expected {EXPECT_ROWS[rnd]:,}")
    files = [r["file"] for r in rows]
    sel_path = os.path.join(base, f"round{rnd}_select.txt")
    with open(sel_path) as f:
        sel = [l.strip() for l in f if l.strip()]
    if set(sel) != set(files) or len(sel) != len(files):
        sys.exit(f"round {rnd} rows != {sel_path} ({len(sel):,} names, {len(set(sel) & set(files)):,} in common)")
    src[os.path.basename(sel_path)] = md5(sel_path)
    for names in JSONLS.values():
        for name in names:
            src[name] = md5(os.path.join(base, name))
    recs = load_records(base, files)

    # Cells (faceight_cells.cell_of), cross-checked against the WOR-191 audit trail when present.
    stated = {}
    if draw_csv and os.path.exists(draw_csv):
        with open(draw_csv) as f:
            for r in csv.DictReader(f):
                if r["round"] == str(rnd):
                    stated[r["file"]] = r["cell"]
        src[os.path.basename(draw_csv)] = md5(draw_csv)
        if set(stated) != set(files):
            sys.exit(f"{draw_csv}: round {rnd} files differ from frames.csv ({len(stated):,} vs {len(files):,})")

    a_path = os.path.join(base, "verdict", "import_left.jsonl")
    a_names = set()
    if os.path.exists(a_path):
        with open(a_path) as f:
            a_names = {json.loads(l)["image"] for l in f}
        src["verdict/import_left.jsonl"] = md5(a_path)

    out, seen = [], {}
    stats = Counter()  # (cell name, eye, metric) -> n
    for r in rows:
        fn, eye = r["file"], r["view"]
        rw, ra = recs["armW"][fn], recs["AA34"][fn]
        for det, rec in (("armW", rw), ("AA34", ra)):
            if rec["session_id"] != r["session_id"] or int(rec["chunk"]) != int(r["chunk"]) \
                    or int(rec["frame_idx"]) != int(r["frame_idx"]) or rec["view"] != eye:
                sys.exit(f"{fn}: {det} record (session, chunk, frame_idx, view) differs from frames.csv")
        sw = max((b[4] for b in rw["boxes"]), default=0.0)
        sa = max((b[4] for b in ra["boxes"]), default=0.0)
        cell = cell_of(sw, sa)
        cn = cell_name(*cell)
        if stated and stated[fn] != cn:
            sys.exit(f"{fn}: cell {cn!r} != {stated[fn]!r} in {draw_csv}")
        boxes, dropped = prelabel(cell, {"armW": rw, "AA34": ra})
        img = image_name(r["session_id"], r["chunk"], r["frame_idx"])
        if img in seen:
            sys.exit(f"image name collision within round {rnd}: {img} from {fn} and {seen[img]}")
        if img in a_names:
            sys.exit(f"image name {img} ({fn}) collides with a faceight_a name in {a_path}")
        seen[img] = fn
        if disk_check and not os.path.isfile(os.path.join(base, FRAMES_DIR[eye], fn)):
            sys.exit(f"{fn}: missing under {os.path.join(base, FRAMES_DIR[eye])}")
        out.append({"session": r["session_id"], "chunk": f"{int(r['chunk']):03d}", "view": VIEW[eye],
                    "frame_idx": int(r["frame_idx"]), "width": W, "height": H, "boxes": boxes, "image": img})
        stats[(cn, eye, "frames")] += 1
        stats[(cn, eye, "boxes")] += len(boxes)
        stats[(cn, eye, "empty")] += int(not boxes)
        stats[(cn, eye, "dropped")] += dropped
    return tag, out, stats, src, len(a_names)


def readme(rnd, tag, out, stats, src, n_a):
    n = len(out)
    names = [cell_name(c, b) for c, b in CELLS]
    per_eye = Counter(o["view"] for o in out)
    tot = lambda cn, m: sum(stats[(cn, e, m)] for e in EYES)  # noqa: E731
    all_ = lambda m: sum(stats[(cn, e, m)] for cn in names for e in EYES)  # noqa: E731
    lr = lambda cn, m: f"{tot(cn, m):,} ({stats[(cn, 'vst_left', m)]:,} / {stats[(cn, 'vst_right', m)]:,})"  # noqa: E731
    L = [f"# faceight_{tag}: pre-labels for Verdict import (WOR-195)", ""]
    L.append(f"Set: faceight_{tag} = round {rnd} of `frames.csv` (rows with annotation_round == {rnd}): {n:,} frames, "
             f"both eyes ({per_eye['lview']:,} lview, {per_eye['rview']:,} rview), one frame per line. Drawn in WOR-191 by "
             f"cell shares over the 10 cells of `mining/faceight_cells.py` with `numpy.random.default_rng(19760703)`, "
             f"breadth-first over episodes; round 2 is the first 60% of each cell's draw order, round 3 the rest. "
             f"Frames are raw fisheye {W} x {H} jpg with unblurred faces of real people (PII): same handling rules as "
             f"any PII batch.")
    L.append("")
    L.append(f"Files: `import.jsonl` (one line per frame, the faceight round-1 / faceback-45 importer record: session, "
             f"chunk \"%03d\", view lview / rview, frame_idx, width, height, boxes as normalized xywh with 4 decimals, "
             f"image) and this README. OSS: `{BUCKET}/faceight_{tag}/images/<image>` and "
             f"`{BUCKET}/faceight_{tag}/prelabels/{{import.jsonl,README.md}}`. The image name is "
             f"`<session>_c<chunk>_f<frame_idx:06d>.jpg` with no eye token, as the faceback-45 importer files "
             f"(`import_right.jsonl` there: view \"rview\", eye-less image) and round 1 (`faceight/`); the eye is the "
             f"`view` field. Names are unique within this set and disjoint from the {n_a:,} faceight_a names "
             f"(asserted by `mining/faceight_verdict_prep.py`). faceback-45 put each eye in its own dataset prefix "
             f"(`faceback-45-left/`, `faceback-45-right/`); here both eyes share one prefix, so split `import.jsonl` "
             f"by `view` if the importer wants one dataset per eye.")
    L.append("")
    L.append("Prelabel rule: per frame exactly ONE detector, chosen by the frame's cell: armW only cells -> armW boxes "
             "(det_10g_armW); both and AA34 only cells -> armAA34 boxes (det_34g_armAA34); faceless -> no boxes "
             "(`boxes: []`, the \"machine looked, no face here\" state). Only that detector's boxes with score >= the "
             "cell's band edge are kept: 0.6+ cells -> 0.6, [0.3,0.6) cells -> 0.3, [0.1,0.3) cells -> 0.1. Nothing else "
             "is in the package: no second detector, no box under the edge, no scores. Boxes are converted as in round 1: "
             "x = x1 / width, y = y1 / height, w = x2 / width - x, h = y2 / height - y, clipped to [0, 1], 4 decimals, "
             "zero-area boxes dropped.")
    L.append("")
    L.append("Cell rule (`mining/faceight_cells.py`, WOR-190): sW / sA = max box score of armW / armAA34 on the frame "
             "(0 without a box); band = band(max(sW, sA)) in 0.6+ / [0.3,0.6) / [0.1,0.3), faceless when neither "
             "detector has a box (every stored box is >= 0.1); within a band with lower edge lo: both = sW >= lo and "
             "sA >= lo, armW only = sW >= lo and sA < lo, AA34 only = sA >= lo and sW < lo.")
    L.append("")
    L.append("## Counts per cell (total (lview / rview))")
    L.append("")
    L.append("| cell | detector | edge | frames | boxes kept | frames with 0 boxes | zero-area dropped |")
    L.append("|---|---|---|---|---|---|---|")
    for c, b in CELLS:
        cn = cell_name(c, b)
        det = "none" if c == FACELESS else MODEL[DETECTOR[c]]
        edge = "" if c == FACELESS else f"{LO[b]:g}"
        L.append(f"| {cn} | {det} | {edge} | {lr(cn, 'frames')} | {lr(cn, 'boxes')} | {lr(cn, 'empty')} | {lr(cn, 'dropped')} |")
    L.append(f"| **total** | | | {all_('frames'):,} ({per_eye['lview']:,} / {per_eye['rview']:,}) | "
             f"{all_('boxes'):,} ({sum(stats[(cn, 'vst_left', 'boxes')] for cn in names):,} / "
             f"{sum(stats[(cn, 'vst_right', 'boxes')] for cn in names):,}) | "
             f"{all_('empty'):,} ({sum(stats[(cn, 'vst_left', 'empty')] for cn in names):,} / "
             f"{sum(stats[(cn, 'vst_right', 'empty')] for cn in names):,}) | {all_('dropped'):,} |")
    L.append("")
    L.append("## Sources (md5)")
    L.append("")
    for k, v in src.items():
        L.append(f"- `{k}`: {v}")
    L.append("")
    L.append("Built by `mining/faceight_verdict_prep.py --round " + str(rnd) + "` (repo pii, WOR-195).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, help="2 (faceight_b) or 3 (faceight_c); b / c also accepted")
    ap.add_argument("--out-dir", help="default <base>/verdict_<b|c>")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--expect-md5", default=EXPECT_MD5, help="frames.csv md5 to insist on ('' to skip)")
    ap.add_argument("--draw-csv", default=None, help="default <base>/round23_draw.csv; cells cross-checked when present")
    ap.add_argument("--no-disk-check", action="store_true", help="do not stat every jpg")
    ap.add_argument("--dry-run", action="store_true", help="build and print the README, write nothing")
    a = ap.parse_args()
    rnd = {"2": 2, "3": 3, "b": 2, "c": 3}.get(a.round)
    if rnd is None:
        sys.exit(f"--round {a.round!r}: expected 2, 3, b or c")
    draw_csv = a.draw_csv or os.path.join(a.base, "round23_draw.csv")
    tag, out, stats, src, n_a = build(rnd, a.base, a.expect_md5, draw_csv, not a.no_disk_check)
    text = readme(rnd, tag, out, stats, src, n_a)
    out_dir = a.out_dir or os.path.join(a.base, f"verdict_{tag}")
    print(text)
    if a.dry_run:
        print(f"dry run: {len(out):,} lines not written to {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "import.jsonl")
    with open(p + ".tmp", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    os.replace(p + ".tmp", p)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)
    print(f"wrote {p} ({len(out):,} lines, md5 {md5(p)}) and README.md")


if __name__ == "__main__":
    main()

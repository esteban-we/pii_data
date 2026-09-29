#!/usr/bin/env python3
"""Disjoint band cells over the unannotated rows of data/faceight/frames.csv (WOR-190).

Read-only. Scope = rows with annotation_round == 0 (701,904 rows). For each
in-scope frame sW = max box score of det_10g_armW and sA = max box score of
det_34g_armAA34, each 0 when the detector has no box; both come from the JSONLs
of --jsonl-dir (armW: faceight_armW.jsonl + faceight_armW_right.jsonl, AA34:
faceight_armAA34.jsonl + faceight_armAA34_right.jsonl, boxes [x1, y1, x2, y2,
score], every box >= 0.1), joined on `file`.

Cell rule (exactly one cell per frame):
  band = band(max(sW, sA)) with 0.6+ (>= 0.6), [0.3,0.6), [0.1,0.3), faceless (< 0.1,
  i.e. no box at all). Within a band with lower edge lo (0.6, 0.3, 0.1):
  both      = sW >= lo and sA >= lo
  armW only = sW >= lo and sA <  lo
  AA34 only = sA >= lo and sW <  lo
  (the band is set by the larger score, so at least one detector reaches lo).
Shares (percent of the round-2 sample): armW only 2 / 4 / 2, both 4 / 8 / 4,
AA34 only 16 / 32 / 16 in band order 0.6+, [0.3,0.6), [0.1,0.3); faceless 12.
N_max = min over cells of floor(episodes(cell) x 100 / share(cell)), the largest N
whose per-cell quota N x share / 100 fits under one frame per episode everywhere.

    faceight_cells.py --jsonl-dir DIR [--csv data/faceight/frames.csv] [--out reports/x.md]
                      [--seed 190]

Nothing is selected and frames.csv is never written.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faceight_buckets import EXPECT, EYES, JSONLS, md5, pct, table  # noqa: E402
from faceight_round import HEADERS, read_table  # noqa: E402

BANDS = ["0.6+", "[0.3,0.6)", "[0.1,0.3)"]
LO = {"0.6+": 0.6, "[0.3,0.6)": 0.3, "[0.1,0.3)": 0.1}
CLASSES = ["armW only", "both", "AA34 only"]
FACELESS = "faceless"
CELLS = [(c, b) for b in BANDS for c in CLASSES] + [(FACELESS, "")]
SHARE = {("armW only", "0.6+"): 2, ("armW only", "[0.3,0.6)"): 4, ("armW only", "[0.1,0.3)"): 2,
         ("both", "0.6+"): 4, ("both", "[0.3,0.6)"): 8, ("both", "[0.1,0.3)"): 4,
         ("AA34 only", "0.6+"): 16, ("AA34 only", "[0.3,0.6)"): 32, ("AA34 only", "[0.1,0.3)"): 16,
         (FACELESS, ""): 12}
N_LIST = [200, 1000, 5000, 20000]
MIN_SCORE = 0.1
# Earlier reports, used as external anchors for the sanity checks.
# WOR-187 (c) at 0.6: either / both / armW only / AA34 only
AT06 = {"all": (83899, 60739, 5456, 17704), "vst_left": (30568, 20302, 1812, 8454),
        "vst_right": (53331, 40437, 3644, 9250)}
# WOR-189 (c) either at 0.3; WOR-188 (c) either and neither at 0.1
EITHER03 = {"all": 201779, "vst_left": 88348, "vst_right": 113431}
EITHER01 = {"all": 564120, "vst_left": 263631, "vst_right": 300489}
NEITHER01 = {"all": 137784, "vst_left": 66129, "vst_right": 71655}


def cell_name(c, b):
    return c if c == FACELESS else f"{c} {b}"


def band_of(s):
    if s >= 0.6:
        return "0.6+"
    if s >= 0.3:
        return "[0.3,0.6)"
    if s >= MIN_SCORE:
        return "[0.1,0.3)"
    return FACELESS


def cell_of(sw, sa):
    """(class, band) of one frame from its two max scores; the rule in the module docstring."""
    b = band_of(max(sw, sa))
    if b == FACELESS:
        return (FACELESS, "")
    lo = LO[b]
    if sw >= lo and sa >= lo:
        return ("both", b)
    if sw >= lo:
        return ("armW only", b)
    return ("AA34 only", b)


def max_scores(jsonl_dir):
    """detector -> {file: max box score (0.0 if no box)}; also the min box score seen and record counts."""
    out, lines, smin = {}, {}, {}
    for det, names in JSONLS.items():
        m, lo = {}, 1.0
        for name in names:
            path = os.path.join(jsonl_dir, name)
            n = 0
            with open(path) as f:
                for ln, line in enumerate(f, 1):
                    r = json.loads(line)
                    fn = r["file"]
                    if fn in m:
                        sys.exit(f"{path} line {ln}: file {fn!r} already seen for {det}")
                    boxes = r.get("boxes")
                    if not isinstance(boxes, list):
                        sys.exit(f"{path} line {ln}: no boxes list ({fn})")
                    for b in boxes:
                        if len(b) != 5:
                            sys.exit(f"{path} line {ln}: box with {len(b)} values ({fn})")
                    scores = [float(b[4]) for b in boxes]
                    m[fn] = max(scores) if scores else 0.0
                    if scores:
                        lo = min(lo, min(scores))
                    n += 1
            lines[name] = n
            print(f"{det}: {name}: {n:,} records", file=sys.stderr)
        out[det], smin[det] = m, lo
    return out, lines, smin


def load(csv_path, jsonl_dir):
    _, header, rows = read_table(csv_path)
    if header != HEADERS[-1]:
        sys.exit(f"{csv_path}: need the 14-column header (view, n_faces_aa34), got {header}")
    col = {c: i for i, c in enumerate(header)}
    keep = [r for r in rows if r[col["annotation_round"]] == "0"]
    files = [r[col["file"]] for r in keep]
    ms, lines, smin = max_scores(jsonl_dir)
    for det in JSONLS:
        miss = [f for f in files if f not in ms[det]]
        if miss:
            sys.exit(f"{len(miss)} in-scope rows have no {det} record in {jsonl_dir}, e.g. {miss[:5]}")
    d = {
        "file": np.array(files),
        "view": np.array([r[col["view"]] for r in keep]),
        "episode": np.array([r[col["episode_id"]] for r in keep]),
        "session": np.array([r[col["session_id"]] for r in keep]),
        "sW": np.array([ms["armW"][f] for f in files]),
        "sA": np.array([ms["AA34"][f] for f in files]),
    }
    src = {"dir": jsonl_dir, "lines": lines, "smin": smin,
           "unused": {det: len(set(ms[det]) - set(files)) for det in JSONLS}}
    return d, len(rows), src


def report(csv_path, jsonl_dir, seed):
    d, n_all, src = load(csv_path, jsonl_dir)
    n = len(d["view"])
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        if not ok:
            sys.exit(f"sanity check failed: {name}")

    check(f"in-scope rows == {EXPECT['all']:,}", n == EXPECT["all"])
    per_eye = {e: int((d["view"] == e).sum()) for e in EYES}
    for e in EYES:
        check(f"{e} rows == {EXPECT[e]:,}", per_eye[e] == EXPECT[e])
    for det in JSONLS:
        check(f"every in-scope row has an {det} record in the JSONLs", True)  # load() exited otherwise
        check(f"every {det} box has score >= {MIN_SCORE:g}", src["smin"][det] >= MIN_SCORE)
    check(f"shares sum to 100", sum(SHARE.values()) == 100)

    # Cell per frame (vectorised; the scalar rule in cell_of is checked against it on a sample below).
    smax = np.maximum(d["sW"], d["sA"])
    cell = np.empty(n, dtype=object)
    cell[:] = FACELESS
    for b in BANDS:
        lo = LO[b]
        nxt = 0.6 if b == "[0.3,0.6)" else (0.3 if b == "[0.1,0.3)" else np.inf)
        in_b = (smax >= lo) & (smax < nxt)
        w, a = d["sW"] >= lo, d["sA"] >= lo
        cell[in_b & w & a] = cell_name("both", b)
        cell[in_b & w & ~a] = cell_name("armW only", b)
        cell[in_b & ~w & a] = cell_name("AA34 only", b)
        check(f"band {b}: every frame is both / armW only / AA34 only", int((in_b & ~w & ~a).sum()) == 0)
    rng = random.Random(seed)
    sample = rng.sample(range(n), 2000)
    check("vectorised cell == cell_of(sW, sA) on 2,000 random frames",
          all(cell[i] == cell_name(*cell_of(float(d["sW"][i]), float(d["sA"][i]))) for i in sample))

    names = [cell_name(c, b) for c, b in CELLS]
    groups = {"vst_left": d["view"] == "vst_left", "vst_right": d["view"] == "vst_right",
              "all": np.ones(n, bool)}
    stats = {}  # (group, cell name) -> (frames, episodes, sessions)
    for g, m in groups.items():
        tot = 0
        for nm in names:
            mc = m & (cell == nm)
            fr = int(mc.sum())
            stats[(g, nm)] = (fr, len(set(d["episode"][mc].tolist())), len(set(d["session"][mc].tolist())))
            tot += fr
        check(f"{g}: the 10 cells partition the rows ({tot:,})", tot == (n if g == "all" else per_eye[g]))
        e06, b06, w06, a06 = AT06[g]
        s = lambda c, b: stats[(g, cell_name(c, b))][0]  # noqa: E731
        check(f"{g}: 0.6+ cells sum {s('armW only', '0.6+') + s('both', '0.6+') + s('AA34 only', '0.6+'):,} == WOR-187 either at 0.6 ({e06:,})",
              s("armW only", "0.6+") + s("both", "0.6+") + s("AA34 only", "0.6+") == e06)
        check(f"{g}: both 0.6+ == WOR-187 both at 0.6 ({b06:,})", s("both", "0.6+") == b06)
        check(f"{g}: armW only 0.6+ == WOR-187 armW only at 0.6 ({w06:,})", s("armW only", "0.6+") == w06)
        check(f"{g}: AA34 only 0.6+ == WOR-187 AA34 only at 0.6 ({a06:,})", s("AA34 only", "0.6+") == a06)
        b03 = sum(s(c, b) for c in CLASSES for b in ("0.6+", "[0.3,0.6)"))
        check(f"{g}: 0.6+ and [0.3,0.6) cells sum {b03:,} == WOR-189 either at 0.3 ({EITHER03[g]:,})", b03 == EITHER03[g])
        b01 = sum(s(c, b) for c in CLASSES for b in BANDS)
        check(f"{g}: non-faceless cells sum {b01:,} == WOR-188 either at 0.1 ({EITHER01[g]:,})", b01 == EITHER01[g])
        check(f"{g}: faceless {s(FACELESS, ''):,} == WOR-188 neither at 0.1 ({NEITHER01[g]:,})", s(FACELESS, "") == NEITHER01[g])

    # N_max per group
    nmax, binding, ratios = {}, {}, {}
    for g in groups:
        r = {nm: stats[(g, nm)][1] * 100 // SHARE[cb] for cb, nm in zip(CELLS, names)}
        ratios[g] = r
        nm_min = min(names, key=lambda nm: (r[nm], names.index(nm)))
        nmax[g], binding[g] = r[nm_min], nm_min
        for cb, nm in zip(CELLS, names):
            check(f"{g}: N_max quota fits in {nm} (ceil({nmax[g]:,} x {SHARE[cb]} / 100) <= {stats[(g, nm)][1]:,} episodes)",
                  math.ceil(nmax[g] * SHARE[cb] / 100) <= stats[(g, nm)][1])
        cb = CELLS[names.index(nm_min)]
        check(f"{g}: N_max + 1 does not fit in the binding cell {nm_min}",
              (nmax[g] + 1) * SHARE[cb] / 100 > stats[(g, nm_min)][1])

    L = []
    L.append("# faceight round 2: episodes and frames per disjoint cell, N_max (WOR-190)\n")
    L.append("**Cell rule (exactly one cell per frame).** For each in-scope frame sW = max box score of armW "
             "(det_10g_armW; 0 if no box) and sA = max box score of AA34 (det_34g_armAA34; 0 if no box). "
             "Band of the frame = band(max(sW, sA)) with bands 0.6+ (>= 0.6), [0.3,0.6), [0.1,0.3) and faceless "
             "(< 0.1, i.e. no box at all by either detector; every stored box has score >= 0.1). Within band b with "
             "lower edge lo(b) in {0.6, 0.3, 0.1}: **both** = sW >= lo(b) and sA >= lo(b); **armW only** = sW >= lo(b) "
             "and sA < lo(b); **AA34 only** = sA >= lo(b) and sW < lo(b). The band is set by the larger of the two "
             "scores, so at least one detector always reaches lo(b) and the three classes cover the band. "
             "Consequences worth knowing: a frame with sW = 0.7 and sA = 0.4 is armW only 0.6+ (not both), and a "
             "frame with sW = 0.25 and sA = 0.35 is AA34 only [0.3,0.6).\n")
    L.append("**Shares** (percent of a round-2 sample of N frames), band order 0.6+ / [0.3,0.6) / [0.1,0.3): "
             "armW only 2 / 4 / 2, both 4 / 8 / 4, AA34 only 16 / 32 / 16; faceless 12 (sum 100). "
             "quota(cell, N) = ceil(N x share / 100). N_max = min over cells of floor(episodes(cell) x 100 / share(cell)): "
             "the largest N whose quota fits under one frame per episode in every cell.\n")
    L.append(f"Input: `data/faceight/frames.csv`, md5 `{md5(csv_path)}`, {n_all:,} rows; in scope (annotation_round == 0): "
             f"{n:,} rows ({', '.join(f'{e} {per_eye[e]:,}' for e in EYES)}). Scores: per-file max over the `boxes` "
             f"lists of `{src['dir']}`: " + ", ".join(f"`{k}` ({v:,} records)" for k, v in src["lines"].items())
             + f"; every in-scope row found a record for both detectors; "
             + ", ".join(f"{v:,} {k} records" for k, v in src["unused"].items())
             + " name files that are not in-scope rows (round-1 frames and their right twins) and are ignored. "
             + "Smallest box score seen: " + ", ".join(f"{k} {v:.4f}" for k, v in src["smin"].items())
             + ". Nothing is selected and frames.csv is not written.\n")

    # 1. Tables
    L.append("## 1. Frames, episodes and sessions per cell\n")
    L.append("Episodes and sessions are distinct counts within the cell; an episode with frames in several cells is "
             "counted in each, so the episode column does not sum to the distinct total (given in the last row). In the "
             "both-eyes table an episode with frames in both eyes counts once per cell. Sessions equal episodes because "
             "each session holds one episode in this table.\n")
    for g, title in (("vst_left", "vst_left"), ("vst_right", "vst_right"), ("all", "both eyes (total)")):
        m = groups[g]
        ep_all, se_all = len(set(d["episode"][m].tolist())), len(set(d["session"][m].tolist()))
        rows_md = []
        for cb, nm in zip(CELLS, names):
            fr, ep, se = stats[(g, nm)]
            rows_md.append([nm, f"{fr:,}", pct(fr, m.sum()), f"{ep:,}", f"{se:,}", f"{SHARE[cb]}%",
                            f"{ratios[g][nm]:,}" + (" **(binding)**" if nm == binding[g] else "")])
        rows_md.append(["**total**", f"{int(m.sum()):,}", "100.0%", f"{ep_all:,} distinct", f"{se_all:,} distinct", "100%", ""])
        L.append(f"### {title}: {int(m.sum()):,} frames, {ep_all:,} episodes, {se_all:,} sessions\n")
        L.append(table(["cell", "frames", "% of frames", "episodes", "sessions", "share", "floor(episodes x 100 / share)"], rows_md))
        L.append("")
        # frames grid with row and column sums
        grid = []
        col_tot = Counter()
        for b in BANDS:
            cells = [f"{stats[(g, cell_name(c, b))][0]:,}" for c in CLASSES]
            rs = sum(stats[(g, cell_name(c, b))][0] for c in CLASSES)
            for c in CLASSES:
                col_tot[c] += stats[(g, cell_name(c, b))][0]
            grid.append([f"band {b}"] + cells + [f"{rs:,}"])
        fl = stats[(g, FACELESS)][0]
        grid.append(["faceless", "", "", "", f"{fl:,}"])
        grid.append(["**column sum**"] + [f"{col_tot[c]:,}" for c in CLASSES]
                    + [f"{sum(col_tot.values()) + fl:,}"])
        check(f"{g}: frames grid row sums + faceless == rows", sum(col_tot.values()) + fl == int(m.sum()))
        L.append(f"Frames grid, {title} (row sums = frames per band, column sums = frames per class):\n")
        L.append(table(["band \\ class"] + CLASSES + ["row sum"], grid))
        L.append("")

    # 2. N_max and quotas
    L.append("## 2. N_max and per-cell quotas\n")
    for g, title in (("all", "both eyes (total)"), ("vst_left", "vst_left"), ("vst_right", "vst_right")):
        L.append(f"- **{title}: N_max = {nmax[g]:,}**, binding cell {binding[g]} "
                 f"({stats[(g, binding[g])][1]:,} episodes at {SHARE[CELLS[names.index(binding[g])]]}%). "
                 f"At N_max + 1 = {nmax[g] + 1:,} the binding quota would be "
                 f"{(nmax[g] + 1) * SHARE[CELLS[names.index(binding[g])]] / 100:,.2f} > {stats[(g, binding[g])][1]:,}.")
    L.append("")
    for g, title in (("all", "both eyes (total)"), ("vst_left", "vst_left"), ("vst_right", "vst_right")):
        Ns = N_LIST + [nmax[g]]
        rows_md = []
        for cb, nm in zip(CELLS, names):
            ep = stats[(g, nm)][1]
            cells = [nm, f"{SHARE[cb]}%", f"{ep:,}"]
            for N in Ns:
                q = math.ceil(N * SHARE[cb] / 100)
                cells.append(f"{q:,} {'fits' if q <= ep else 'NO'}")
            rows_md.append(cells)
        fits = []
        for N in Ns:
            ok = all(math.ceil(N * SHARE[cb] / 100) <= stats[(g, nm)][1] for cb, nm in zip(CELLS, names))
            fits.append(f"{N:,}: {'fits' if ok else 'does not fit'}")
        rows_md.append(["**all cells**", "100%", ""] + [f.split(': ')[1] for f in fits])
        L.append(f"### {title}: quota = ceil(N x share / 100) and whether quota <= episodes(cell)\n")
        L.append(table(["cell", "share", "episodes"] + [f"N = {N:,}" + (" (N_max)" if N == nmax[g] else "") for N in Ns], rows_md))
        L.append("")

    # 3. Sanity and spot checks
    L.append("## 3. Sanity\n")
    L.append(f"- Partition: the 10 cells sum to {per_eye['vst_left']:,} (vst_left), {per_eye['vst_right']:,} (vst_right), {n:,} (total).")
    e06 = {g: sum(stats[(g, cell_name(c, '0.6+'))][0] for c in CLASSES) for g in groups}
    L.append(f"- Union of the three 0.6+ cells = {e06['all']:,} total ({e06['vst_left']:,} left, {e06['vst_right']:,} right) "
             f"== WOR-187 either at 0.6 ({AT06['all'][0]:,}: {AT06['vst_left'][0]:,} left, {AT06['vst_right'][0]:,} right); "
             f"cell by cell the 0.6+ row equals the WOR-187 (c) both / armW only / AA34 only split.")
    L.append(f"- Faceless = {stats[('all', FACELESS)][0]:,} total ({stats[('vst_left', FACELESS)][0]:,} left, "
             f"{stats[('vst_right', FACELESS)][0]:,} right) == WOR-188 neither at 0.1 ({NEITHER01['all']:,}: "
             f"{NEITHER01['vst_left']:,} left, {NEITHER01['vst_right']:,} right).")
    L.append(f"- Also: 0.6+ plus [0.3,0.6) cells == WOR-189 either at 0.3 per eye and total; non-faceless cells == WOR-188 "
             f"either at 0.1 per eye and total (asserted below).")
    L.append("")
    # spot-check candidates: one frame from each of three different cells, drawn with the seed
    picks = []
    pool = [nm for nm in names if stats[("all", nm)][0] > 0]
    for nm in rng.sample(pool, 3):
        idx = np.flatnonzero(cell == nm)
        i = int(idx[rng.randrange(len(idx))])
        picks.append((nm, i))
    L.append(f"Spot-check candidates (seed {seed}; one frame from each of three different cells, to be verified by hand "
             f"against the raw JSONL records):\n")
    L.append(table(["cell", "file", "view", "episode", "sW", "sA"],
                   [[nm, f"`{d['file'][i]}`", d["view"][i], d["episode"][i], f"{d['sW'][i]:.4f}", f"{d['sA'][i]:.4f}"]
                    for nm, i in picks]))
    L.append("")
    L.append("## Sanity checks (all asserted by `mining/faceight_cells.py`; the script exits on the first failure)\n")
    L += [f"- {name}: ok" for name, ok in checks]
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    ap.add_argument("--csv", default=os.path.join(root, "data", "faceight", "frames.csv"))
    ap.add_argument("--jsonl-dir", required=True,
                    help="directory holding faceight_armW{,_right}.jsonl and faceight_armAA34{,_right}.jsonl")
    ap.add_argument("--out", default=None, help="markdown file to write (default: stdout)")
    ap.add_argument("--seed", type=int, default=190, help="seed for the spot-check draw")
    args = ap.parse_args()
    text = report(args.csv, args.jsonl_dir, args.seed)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

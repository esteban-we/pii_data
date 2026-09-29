#!/usr/bin/env python3
"""Draw the faceight round-2 / round-3 frames by cell shares (WOR-191).

Pool = rows of data/faceight/frames.csv with annotation_round == 0 (701,904 rows,
both eyes). Each pool frame is put in exactly one of the 10 cells of
faceight_cells.py (cell rule imported from there, not restated here): class
armW only / both / AA34 only x band 0.6+ / [0.3,0.6) / [0.1,0.3), plus faceless.

Shares (percent of N) in the share-table order used everywhere in this script
(class-major: armW only 0.6+, armW only [0.3,0.6), armW only [0.1,0.3), both
0.6+, ..., AA34 only [0.1,0.3), faceless):
    armW only 1 / 2 / 1, both 4 / 8 / 4, AA34 only 16 / 32 / 16, faceless 16.
quota(cell) = share x N / 100, which must be a whole number and must not exceed
the frames in the cell.

Draw, breadth-first over episodes. One numpy.random.default_rng(--seed) is
created once and consumed in the share-table order. Per cell: the cell's
distinct episode_ids are sorted and shuffled once (rng.permutation); pass 1
walks that order and takes one uniformly chosen unused frame of the episode in
the cell (rng.integers over the episode's unused frames, listed in file-name
order) until the quota is met; if the quota exceeds the episode count, pass 2
walks the same order taking a second frame from every episode that still has
unused frames, and so on. Both eyes are pooled. A cell whose frames run out
before the quota is an error.

Round split: within each cell, in draw order, the first share x N x --split
/ 100 frames (whole number) get annotation_round 2 and the rest round 3, so
round 2 holds the front of the breadth-first order.

Outputs (--out-dir): round2_select.txt and round3_select.txt (one file name per
line, draw order, disjoint; inputs for `faceight_round.py --round N
--select-file`) and round23_draw.csv (file, round, cell, pass, episode_id,
session_id, view, sW, sA: the audit trail the verifier checks against).
--report writes the markdown report. --dry-run prints the report and writes
nothing. Nothing here writes frames.csv.

    faceight_round2_select.py --jsonl-dir /data/esteban/faceight \\
        [--csv data/faceight/frames.csv] [--n 100000] [--seed 19760703]
        [--split 0.6] [--shares 1,2,1,4,8,4,16,32,16,16]
        [--out-dir /data/esteban/faceight] [--report reports/faceight_round2_draw.md]
        [--episode-usage data/episode_usage.csv] [--expect-md5 MD5] [--expect-pool 701904] [--dry-run]

Stdlib + numpy (run on shang with /data/esteban/armw/venv/bin/python).
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faceight_buckets import EXPECT, EYES, md5, pct, table  # noqa: E402
from faceight_cells import BANDS, CLASSES, FACELESS, cell_name, cell_of, load  # noqa: E402

SEED = 19760703
N = 100000
SPLIT = 0.6
# Share-table order: class-major (each class over the bands 0.6+, [0.3,0.6), [0.1,0.3)), then faceless.
ORDER = [(c, b) for c in CLASSES for b in BANDS] + [(FACELESS, "")]
SHARES = [1, 2, 1, 4, 8, 4, 16, 32, 16, 16]
EXPECT_MD5 = "b2acba9162ecf4088937b918f27f960e"
ROUNDS = (2, 3)


def whole(x, what):
    """x as an int; error when x is not a whole number."""
    if abs(x - round(x)) > 1e-9:
        sys.exit(f"{what} = {x} is not a whole number")
    return int(round(x))


def draw_cell(rng, frames, episode, file, quota):
    """Breadth-first draw of `quota` frames from `frames` (indices into the pool).
    Returns [(index, pass)] in draw order and the shuffled episode order."""
    by_ep = defaultdict(list)
    for i in frames:
        by_ep[episode[i]].append(i)
    eps = sorted(by_ep)
    for e in eps:
        by_ep[e].sort(key=lambda i: file[i])
    order = [eps[j] for j in rng.permutation(len(eps))]
    remaining = {e: list(by_ep[e]) for e in eps}
    drawn, npass = [], 0
    while len(drawn) < quota:
        npass += 1
        took = 0
        for e in order:
            rem = remaining[e]
            if not rem:
                continue
            k = int(rng.integers(len(rem)))
            drawn.append((rem.pop(k), npass))
            took += 1
            if len(drawn) == quota:
                break
        if took == 0:
            sys.exit(f"cell exhausted after {len(drawn)} of {quota} frames")
    return drawn, order


def scene_table(episode_usage, sets):
    """Rows of the scene_category mix: one column per (label, episode set)."""
    eu = {}
    with open(episode_usage, newline="") as f:
        for r in csv.DictReader(f):
            eu[r["episode_id"]] = r
    corpus = {e for e, r in eu.items() if r["faceight"] == "1"}
    cols = [("faceight corpus", corpus)] + sets
    for label, s in cols:
        miss = [e for e in s if e not in eu]
        if miss:
            sys.exit(f"{label}: {len(miss)} episodes missing from {episode_usage}, e.g. {miss[:3]}")
    counts = {label: Counter(eu[e]["scene_category"] for e in s) for label, s in cols}
    cats = [c for c, _ in counts["faceight corpus"].most_common()]
    for label, _ in cols:
        for c in counts[label]:
            if c not in cats:
                cats.append(c)
    rows = []
    for c in cats:
        rows.append([c] + [f"{counts[l][c]:,} ({pct(counts[l][c], len(s))})" for l, s in cols])
    rows.append(["**total**"] + [f"{len(s):,}" for _, s in cols])
    return table(["scene_category"] + [l for l, _ in cols], rows)


def stats(picks, d):
    """(frames, distinct episodes, episodes with >= 2 frames, max frames per episode) of a list of indices."""
    c = Counter(d["episode"][i] for i in picks)
    return (len(picks), len(c), sum(1 for v in c.values() if v >= 2), max(c.values()) if c else 0)


def fmt_stats(s):
    return [f"{s[0]:,}", f"{s[1]:,}", f"{s[2]:,}", f"{s[3]:,}"]


def run(a):
    checks = []

    def check(name, ok):
        checks.append(name)
        if not ok:
            sys.exit(f"check failed: {name}")

    shares = [int(x) for x in a.shares.split(",")]
    if len(shares) != len(ORDER):
        sys.exit(f"--shares needs {len(ORDER)} values in the share-table order, got {len(shares)}")
    check("shares sum to 100", sum(shares) == 100)
    share = dict(zip(ORDER, shares))
    quota = {cb: whole(share[cb] * a.n / 100, f"quota({cell_name(*cb)})") for cb in ORDER}
    q2 = {cb: whole(share[cb] * a.n * a.split / 100, f"round-2 quota({cell_name(*cb)})") for cb in ORDER}
    q3 = {cb: quota[cb] - q2[cb] for cb in ORDER}
    check(f"quotas sum to N = {a.n:,}", sum(quota.values()) == a.n)
    n2, n3 = sum(q2.values()), sum(q3.values())

    md5_before = md5(a.csv)
    if a.expect_md5:
        check(f"frames.csv md5 == {a.expect_md5}", md5_before == a.expect_md5)
    d, n_all, src = load(a.csv, a.jsonl_dir)
    n = len(d["file"])
    if a.expect_pool:
        check(f"pool (annotation_round == 0) == {a.expect_pool:,} rows", n == a.expect_pool)
    d["cell"] = np.array([cell_name(*cell_of(float(w), float(s))) for w, s in zip(d["sW"], d["sA"])])
    idx_of = {cell_name(*cb): np.flatnonzero(d["cell"] == cell_name(*cb)) for cb in ORDER}
    check("the cells partition the pool", sum(len(v) for v in idx_of.values()) == n)
    ep_in = {cb: len(set(d["episode"][idx_of[cell_name(*cb)]].tolist())) for cb in ORDER}
    for cb in ORDER:
        nm = cell_name(*cb)
        check(f"quota {quota[cb]:,} <= frames in {nm} ({len(idx_of[nm]):,})", quota[cb] <= len(idx_of[nm]))

    rng = np.random.default_rng(a.seed)
    picks = {}      # cell -> [(index, pass)] in draw order
    rnd_of = {}     # index -> round
    first_eps = {}  # cell -> first 3 episodes of the shuffled order (for the report)
    for cb in ORDER:
        nm = cell_name(*cb)
        drawn, order = draw_cell(rng, idx_of[nm].tolist(), d["episode"], d["file"], quota[cb])
        picks[cb] = drawn
        first_eps[cb] = order[:3]
        for j, (i, _) in enumerate(drawn):
            rnd_of[i] = 2 if j < q2[cb] else 3
    sel = {r: [i for cb in ORDER for i, _ in picks[cb] if rnd_of[i] == r] for r in ROUNDS}
    all_idx = [i for cb in ORDER for i, _ in picks[cb]]
    check(f"round-2 selection has {n2:,} frames", len(sel[2]) == n2)
    check(f"round-3 selection has {n3:,} frames", len(sel[3]) == n3)
    check("no frame drawn twice", len(set(all_idx)) == len(all_idx) == a.n)
    files = {r: [d["file"][i] for i in sel[r]] for r in ROUNDS}
    check("no file in both select files", not set(files[2]) & set(files[3]))
    for cb in ORDER:
        nm = cell_name(*cb)
        both = stats([i for i, _ in picks[cb]], d)
        r2 = stats([i for i, _ in picks[cb] if rnd_of[i] == 2], d)
        check(f"{nm}: distinct episodes over both rounds {both[1]:,} == min(quota, episodes) = {min(quota[cb], ep_in[cb]):,}",
              both[1] == min(quota[cb], ep_in[cb]))
        check(f"{nm}: round-2 distinct episodes {r2[1]:,} == min(quota_2, episodes) = {min(q2[cb], ep_in[cb]):,}",
              r2[1] == min(q2[cb], ep_in[cb]))
        check(f"{nm}: every drawn frame is in the cell", all(d["cell"][i] == nm for i, _ in picks[cb]))
        check(f"{nm}: per-round counts == quotas ({q2[cb]:,} / {q3[cb]:,})",
              (r2[0], both[0] - r2[0]) == (q2[cb], q3[cb]))

    # ---- report
    L = [f"# faceight rounds 2 and 3: the draw (WOR-191)\n"]
    L.append(f"Input: `{os.path.abspath(a.csv)}` md5 `{md5_before}`, {n_all:,} rows, pool (annotation_round == 0) "
             f"{n:,} rows ({', '.join(f'{e} {int((d['view'] == e).sum()):,}' for e in EYES)}). Scores from the JSONLs of "
             f"`{src['dir']}` (" + ", ".join(f"{k} {v:,} records" for k, v in src["lines"].items())
             + "). Cell rule: `mining/faceight_cells.py` (WOR-190, restated in `reports/faceight_round2_cells.md`). "
             f"N = {a.n:,}, seed {a.seed} (`numpy.random.default_rng`, created once, consumed in the share-table order "
             f"below), split {a.split:g} (round 2 = the first share x {a.n:,} x {a.split:g} / 100 frames of each cell "
             f"in draw order, round 3 = the rest; totals {n2:,} and {n3:,}). "
             + ("Dry run: nothing written." if a.dry_run else
                f"Select files: `{os.path.join(a.out_dir, 'round2_select.txt')}`, `{os.path.join(a.out_dir, 'round3_select.txt')}`; "
                f"audit trail `{os.path.join(a.out_dir, 'round23_draw.csv')}`.") + "\n")
    L.append("Draw: per cell, the distinct episode_ids are sorted and shuffled once; pass 1 walks that order taking one "
             "uniformly chosen unused frame of the episode in the cell until the quota is met; when the quota exceeds "
             "the episode count, pass 2 walks the same order taking a second frame from each episode that still has "
             "unused frames, and so on (breadth-first: every episode of the cell is drawn once before any is drawn "
             "twice). Both eyes pooled.\n")
    L.append("## 1. Shares, quotas and the cells\n")
    rows = []
    for cb in ORDER:
        nm = cell_name(*cb)
        rows.append([nm, f"{share[cb]}%", f"{quota[cb]:,}", f"{q2[cb]:,}", f"{q3[cb]:,}", f"{len(idx_of[nm]):,}",
                     f"{ep_in[cb]:,}", "fits" if quota[cb] <= ep_in[cb] else f"{quota[cb] - ep_in[cb]:,} repeats needed"])
    rows.append(["**total**", "100%", f"{a.n:,}", f"{n2:,}", f"{n3:,}", f"{n:,}",
                 f"{len(set(d['episode'].tolist())):,} distinct", ""])
    L.append(table(["cell", "share", "quota", "round 2", "round 3", "frames in cell", "episodes in cell",
                    "quota vs episodes"], rows))
    L.append("")
    L.append("## 2. Realised counts per cell\n")
    L.append("frames / episodes = distinct episodes / >= 2 = episodes with two or more drawn frames / max = most "
             "frames drawn from one episode. Per eye, an episode drawn in both eyes counts in each eye.\n")
    for label, rset in (("both rounds", ROUNDS), ("round 2", (2,)), ("round 3", (3,))):
        rows = []
        tot = []
        for cb in ORDER:
            nm = cell_name(*cb)
            ids = [i for i, _ in picks[cb] if rnd_of[i] in rset]
            tot += ids
            cells = [nm] + fmt_stats(stats(ids, d))
            for e in EYES:
                cells += fmt_stats(stats([i for i in ids if d["view"][i] == e], d))
            rows.append(cells)
        cells = ["**all cells**"] + fmt_stats(stats(tot, d))
        for e in EYES:
            cells += fmt_stats(stats([i for i in tot if d["view"][i] == e], d))
        rows.append(cells)
        hdr = ["cell"] + [f"{g} {k}" for g in ["both eyes"] + EYES for k in ("frames", "episodes", ">= 2", "max")]
        L.append(f"### {label}\n")
        L.append(table(hdr, rows))
        L.append("")
    L.append("### Passes per cell (frames taken in pass 1 / 2 / ...; pass k = the k-th frame of an episode)\n")
    rows = []
    for cb in ORDER:
        pc = Counter(p for _, p in picks[cb])
        pc2 = Counter(p for i, p in picks[cb] if rnd_of[i] == 2)
        rows.append([cell_name(*cb), " / ".join(f"{pc[p]:,}" for p in sorted(pc)),
                     " / ".join(f"{pc2[p]:,}" for p in sorted(pc2)), max(pc)])
    L.append(table(["cell", "both rounds", "round 2", "passes"], rows))
    L.append("")
    L.append("## 3. Episodes and sessions per round\n")
    rows = []
    sets = []
    for label, ids in (("round 2", sel[2]), ("round 3", sel[3]), ("both rounds", all_idx)):
        eps = set(d["episode"][ids].tolist())
        ses = set(d["session"][ids].tolist())
        sets.append((label, eps))
        cells = [label, f"{len(ids):,}", f"{len(eps):,}", f"{len(ses):,}"]
        for e in EYES:
            m = [i for i in ids if d["view"][i] == e]
            cells += [f"{len(m):,}", f"{len(set(d['episode'][m].tolist())):,}"]
        rows.append(cells)
    ov = set(d["episode"][sel[2]].tolist()) & set(d["episode"][sel[3]].tolist())
    L.append(table(["round", "frames", "episodes", "sessions", "vst_left frames", "vst_left episodes",
                    "vst_right frames", "vst_right episodes"], rows))
    L.append(f"\nEpisodes in both rounds: {len(ov):,}. Pool: {len(set(d['episode'].tolist())):,} episodes, "
             f"{len(set(d['session'].tolist())):,} sessions.\n")
    L.append("## 4. scene_category of the drawn episodes vs the faceight corpus\n")
    L.append(f"From `{os.path.abspath(a.episode_usage)}`: faceight corpus = episodes with faceight == 1; pool = the "
             f"distinct episodes of the {n:,} pool frames.\n")
    L.append(scene_table(a.episode_usage, [("pool", set(d["episode"].tolist()))] + sets))
    L.append("")
    L.append("## 5. Draw order anchors (first three shuffled episodes per cell)\n")
    L.append(table(["cell", "episodes"], [[cell_name(*cb), ", ".join(first_eps[cb])] for cb in ORDER]))
    L.append("")
    L.append("## 6. Checks asserted by the script (exit on the first failure)\n")
    L += [f"- {c}: ok" for c in checks]
    L.append("")
    text = "\n".join(L)

    if a.dry_run:
        sys.stdout.write(text)
        print("dry run, nothing written", file=sys.stderr)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    for r in ROUNDS:
        p = os.path.join(a.out_dir, f"round{r}_select.txt")
        with open(p, "w") as f:
            f.write("".join(fn + "\n" for fn in files[r]))
        print(f"wrote {p}: {len(files[r]):,} files, md5 {md5(p)}", file=sys.stderr)
    p = os.path.join(a.out_dir, "round23_draw.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "round", "cell", "pass", "episode_id", "session_id", "view", "sW", "sA"])
        for cb in ORDER:
            for i, ps in picks[cb]:
                w.writerow([d["file"][i], rnd_of[i], cell_name(*cb), ps, d["episode"][i], d["session"][i],
                            d["view"][i], f"{d['sW'][i]:.4f}", f"{d['sA'][i]:.4f}"])
    print(f"wrote {p}: md5 {md5(p)}", file=sys.stderr)
    if a.report:
        with open(a.report, "w") as f:
            f.write(text)
        print(f"wrote {a.report}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    ap.add_argument("--csv", default=os.path.join(root, "data", "faceight", "frames.csv"))
    ap.add_argument("--jsonl-dir", required=True,
                    help="directory holding faceight_armW{,_right}.jsonl and faceight_armAA34{,_right}.jsonl")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--split", type=float, default=SPLIT, help="fraction of each cell's quota that goes to round 2")
    ap.add_argument("--shares", default=",".join(str(s) for s in SHARES),
                    help="10 percentages in the share-table order: " + ", ".join(cell_name(*cb) for cb in ORDER))
    ap.add_argument("--out-dir", default="/data/esteban/faceight")
    ap.add_argument("--report", default=None, help="markdown report to write (default: stdout)")
    ap.add_argument("--episode-usage", default=os.path.join(root, "data", "episode_usage.csv"))
    ap.add_argument("--expect-md5", default=EXPECT_MD5, help="required md5 of --csv ('' to skip)")
    ap.add_argument("--expect-pool", type=int, default=EXPECT["all"], help="required pool size (0 to skip)")
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

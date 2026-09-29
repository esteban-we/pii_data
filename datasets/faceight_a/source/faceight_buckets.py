#!/usr/bin/env python3
"""Face-count buckets over the unannotated rows of data/faceight/frames.csv (WOR-187).

Read-only. Scope = rows with annotation_round == 0 (round-1 right twins are not
rows, so nothing else is excluded). Prints a markdown report with, per eye x
batch x bucket (0 / 1 / 2 / 3 / 4+ faces), the counts by armW (n_faces) and by
armAA34 (n_faces_aa34), the armW x AA34 cross-table per eye, the either / both /
armW only / AA34 only / neither split, both-eye totals, and per eye the episodes,
sessions and residues with at least one face by either detector. No selection is
made and frames.csv is never written.

    faceight_buckets.py [--csv data/faceight/frames.csv] [--out reports/x.md]
                        [--thr T --jsonl-dir DIR]

With --thr T --jsonl-dir DIR (WOR-188) the face counts are not read from the
n_faces / n_faces_aa34 columns but recounted per file as the boxes with score
>= T in the JSONLs of DIR (armW: faceight_armW.jsonl + faceight_armW_right.jsonl,
AA34: faceight_armAA34.jsonl + faceight_armAA34_right.jsonl), joined on `file`;
every in-scope row must get a count from both detectors. The report then also
opens with the (c) totals at 0.6 (the csv columns) against T. Without the two
flags the script behaves exactly as before.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faceight_round import BUCKETS, HEADERS, bucket, open_index, read_table  # noqa: E402

EYES = ["vst_left", "vst_right"]
SPLIT = ["either", "both", "armW only", "AA34 only", "neither"]
EXPECT = {"all": 701904, "vst_left": 329760, "vst_right": 372144}
JSONLS = {"armW": ["faceight_armW.jsonl", "faceight_armW_right.jsonl"],
          "AA34": ["faceight_armAA34.jsonl", "faceight_armAA34_right.jsonl"]}
CSV_THR = 0.6  # the threshold behind n_faces / n_faces_aa34 (faceight_pull.py --face-thr)


def md5(path):
    """md5 of the index's bytes, read through open_index so a .gz-only checkout
    (PII-1639) still reports the plain file's hash."""
    h = hashlib.md5()
    with open_index(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(n, d):
    return "0.0%" if d == 0 else f"{100.0 * n / d:.1f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bucket_row(label, bk, total):
    """Cells for one row of a bucket table: count and row percentage per bucket, total."""
    cells = [label]
    for b in BUCKETS:
        n = int(bk.get(b, 0))
        cells.append(f"{n:,} ({pct(n, total)})")
    cells.append(f"{total:,}")
    return cells


def recount(jsonl_dir, thr):
    """detector -> {file: number of boxes with score >= thr}, plus per-file line counts.
    Exits on a duplicate file within a detector or a record without a boxes list."""
    counts, lines = {}, {}
    for det, names in JSONLS.items():
        c = {}
        for name in names:
            path = os.path.join(jsonl_dir, name)
            n = 0
            with open(path) as f:
                for ln, line in enumerate(f, 1):
                    r = json.loads(line)
                    fn = r["file"]
                    if fn in c:
                        sys.exit(f"{path} line {ln}: file {fn!r} already seen for {det}")
                    boxes = r.get("boxes")
                    if not isinstance(boxes, list):
                        sys.exit(f"{path} line {ln}: no boxes list ({fn})")
                    c[fn] = sum(1 for b in boxes if b[4] >= thr)
                    n += 1
            lines[name] = n
            print(f"{det}: {name}: {n:,} records", file=sys.stderr)
        counts[det] = c
    return counts, lines


def load(csv_path, thr=None, jsonl_dir=None):
    lines, header, rows = read_table(csv_path)
    if header != HEADERS[-1]:
        sys.exit(f"{csv_path}: need the 14-column header (view, n_faces_aa34), got {header}")
    col = {c: i for i, c in enumerate(header)}
    twins = {r[col["file_right"]] for r in rows if r[col["annotation_round"]] != "0"
             and r[col["file_right"]]}
    keep = [r for r in rows if r[col["annotation_round"]] == "0"]
    files = {r[col["file"]] for r in keep}
    n_twin_rows = len(files & twins)
    empty_aa = sum(1 for r in keep if r[col["n_faces_aa34"]] == "")
    if empty_aa:
        sys.exit(f"{empty_aa} in-scope rows have an empty n_faces_aa34")
    d = {
        "view": np.array([r[col["view"]] for r in keep]),
        "batch": np.array([int(r[col["batch"]]) for r in keep]),
        "s": np.array([int(r[col["s"]]) for r in keep]),
        "episode": np.array([r[col["episode_id"]] for r in keep]),
        "session": np.array([r[col["session_id"]] for r in keep]),
        "nW": np.array([int(r[col["n_faces"]]) for r in keep]),
        "nA": np.array([int(r[col["n_faces_aa34"]]) for r in keep]),
    }
    src = None
    if thr is not None:
        counts, jl = recount(jsonl_dir, thr)
        d["nW_csv"], d["nA_csv"] = d["nW"], d["nA"]
        miss = {det: [f for f in (r[col["file"]] for r in keep) if f not in counts[det]]
                for det in JSONLS}
        for det, m in miss.items():
            if m:
                sys.exit(f"{len(m)} in-scope rows have no {det} record in {jsonl_dir}, "
                         f"e.g. {m[:5]}")
        d["nW"] = np.array([counts["armW"][r[col["file"]]] for r in keep])
        d["nA"] = np.array([counts["AA34"][r[col["file"]]] for r in keep])
        src = {"thr": thr, "dir": jsonl_dir, "lines": jl,
               "unused": {det: len(set(counts[det]) - files) for det in JSONLS}}
    d["bW"] = np.array([bucket(n) for n in d["nW"]])
    d["bA"] = np.array([bucket(n) for n in d["nA"]])
    rounds = Counter(r[col["annotation_round"]] for r in rows)
    return d, len(rows), rounds, len(twins), n_twin_rows, src


def split_counts(w, a):
    """either / both / armW only / AA34 only / neither over boolean arrays."""
    return {"either": int((w | a).sum()), "both": int((w & a).sum()),
            "armW only": int((w & ~a).sum()), "AA34 only": int((~w & a).sum()),
            "neither": int((~w & ~a).sum())}


def split_line(name, sc):
    return f"{name} " + " / ".join(f"{k} {sc[k]:,}" for k in SPLIT)


def report(csv_path, thr=None, jsonl_dir=None):
    d, n_all, rounds, n_twins, n_twin_rows, src = load(csv_path, thr, jsonl_dir)
    n = len(d["view"])
    per_eye = {e: int((d["view"] == e).sum()) for e in EYES}
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        if not ok:
            sys.exit(f"sanity check failed: {name}")

    check(f"in-scope rows == {EXPECT['all']:,}", n == EXPECT["all"])
    for e in EYES:
        check(f"{e} rows == {EXPECT[e]:,}", per_eye[e] == EXPECT[e])
    check("no in-scope row is the file_right of a round-1 row", n_twin_rows == 0)
    check("rounds present are only 0 and 1", set(rounds) == {"0", "1"})
    if src:
        for det in JSONLS:
            check(f"every in-scope row has an {det} record in the JSONLs", True)  # load() exited otherwise
        same_w = int((d["nW"] == d["nW_csv"]).sum())
        same_a = int((d["nA"] == d["nA_csv"]).sum())
        if thr == CSV_THR:
            check("recount at 0.6 == n_faces on every in-scope row", same_w == n)
            check("recount at 0.6 == n_faces_aa34 on every in-scope row", same_a == n)

    L = []
    L.append("# faceight round 2: face-count buckets over the unannotated frames "
             + ("(WOR-187)" if not src else f"(WOR-188, score threshold {thr:g})") + "\n")
    if src:
        parts = []
        for e in EYES + ["both eyes"]:
            m_e = np.ones(n, bool) if e == "both eyes" else d["view"] == e
            at_csv = split_counts(d["nW_csv"][m_e] >= 1, d["nA_csv"][m_e] >= 1)
            at_thr = split_counts(d["nW"][m_e] >= 1, d["nA"][m_e] >= 1)
            parts.append(f"{e}: at {CSV_THR:g} " + " / ".join(f"{at_csv[k]:,}" for k in SPLIT)
                         + f", at {thr:g} " + " / ".join(f"{at_thr[k]:,}" for k in SPLIT))
        L.append(f"**(c) totals at {CSV_THR:g} vs {thr:g}** (either / both / armW only / AA34 only / neither): "
                 + "; ".join(parts) + ".\n")
    L.append(f"Input: `{os.path.relpath(csv_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))}`, "
             f"md5 `{md5(csv_path)}`, {n_all:,} rows "
             f"(annotation_round counts: {', '.join(f'{k}={v:,}' for k, v in sorted(rounds.items()))}).")
    if src:
        L.append(f"Face counts: recounted per file from the `boxes` lists (all boxes at score >= 0.1 after NMS) in "
                 f"`{src['dir']}`: " + ", ".join(f"`{k}` ({v:,} records)" for k, v in src["lines"].items())
                 + f"; a face = a box with score >= {thr:g}. Every in-scope row found a record for both detectors; "
                 + ", ".join(f"{v:,} {k} records" for k, v in src["unused"].items())
                 + " name files that are not in-scope rows (round-1 frames and their right twins) and are ignored. "
                 f"The recount agrees with the csv columns (score >= {CSV_THR:g}) on {same_w:,} rows for armW and "
                 f"{same_a:,} rows for AA34.")
    L.append(f"Exclusion rule: every row with `annotation_round != 0` is dropped ({n_all - n:,} rows). "
             f"Round-1 right twins are not rows of the table (they are only `file_right` values on "
             f"{n_twins:,} round-1 rows; {n_twin_rows} in-scope rows carry such a name), so nothing else is excluded. "
             f"In scope: {n:,} rows ({', '.join(f'{e} {per_eye[e]:,}' for e in EYES)}).")
    if src:
        L.append(f"Buckets: 0 / 1 / 2 / 3 / 4+ faces. armW = det_10g_armW boxes at score >= {thr:g}; "
                 f"AA34 = det_34g_armAA34 boxes at score >= {thr:g}. Row percentages are within the row. "
                 "No selection is made here and frames.csv is not written.\n")
    else:
        L.append("Buckets: 0 / 1 / 2 / 3 / 4+ faces. armW = `n_faces` (det_10g_armW at score >= 0.6); "
                 "AA34 = `n_faces_aa34` (det_34g_armAA34 at score >= 0.6). Row percentages are within the row. "
                 "No selection is made here and frames.csv is not written.\n")

    # (a) + (d): per eye x batch x bucket, by armW and by AA34
    a_eye_tot = {}
    for name, key in (("armW", "bW"), ("AA34", "bA")):
        L.append(f"## (a) Per eye x batch x bucket, by {name}\n")
        rows_md = []
        tot_all = Counter()
        for e in EYES:
            m_e = d["view"] == e
            tot_e = Counter()
            for b in sorted(set(d["batch"][m_e].tolist())):
                m = m_e & (d["batch"] == b)
                bk = Counter(d[key][m].tolist())
                rows_md.append(bucket_row(f"{e} batch {b}", bk, int(m.sum())))
                tot_e.update(bk)
            check(f"(a) {name} {e} total == rows", sum(tot_e.values()) == per_eye[e])
            rows_md.append(bucket_row(f"**{e} total**", tot_e, per_eye[e]))
            tot_all.update(tot_e)
            a_eye_tot[(name, e)] = tot_e
        check(f"(a) {name} both-eye total == {n:,}", sum(tot_all.values()) == n)
        rows_md.append(bucket_row("**both eyes**", tot_all, n))
        L.append(table(["eye / batch"] + [f"{b} faces" for b in BUCKETS] + ["rows"], rows_md))
        L.append("")

    # (b): cross-table per eye
    L.append("## (b) Per eye: armW bucket (rows) x AA34 bucket (columns)\n")
    for e in EYES:
        m_e = d["view"] == e
        rows_md = []
        col_tot = Counter()
        for bw in BUCKETS:
            m = m_e & (d["bW"] == bw)
            bk = Counter(d["bA"][m].tolist())
            col_tot.update(bk)
            cells = [f"armW {bw}"] + [f"{int(bk.get(ba, 0)):,}" for ba in BUCKETS] + [f"{int(m.sum()):,}"]
            rows_md.append(cells)
            check(f"(b) {e} armW {bw} row marginal == (a) armW", int(m.sum()) == a_eye_tot[("armW", e)].get(bw, 0))
        for ba in BUCKETS:
            check(f"(b) {e} AA34 {ba} column marginal == (a) AA34", col_tot.get(ba, 0) == a_eye_tot[("AA34", e)].get(ba, 0))
        check(f"(b) {e} total == rows", sum(col_tot.values()) == per_eye[e])
        rows_md.append(["**AA34 total**"] + [f"{int(col_tot.get(ba, 0)):,}" for ba in BUCKETS] + [f"{per_eye[e]:,}"])
        agree = int((m_e & (d["bW"] == d["bA"])).sum())
        more = int((m_e & (d["nA"] > d["nW"])).sum())
        fewer = int((m_e & (d["nA"] < d["nW"])).sum())
        L.append(f"### {e} ({per_eye[e]:,} rows; same bucket {agree:,} = {pct(agree, per_eye[e])}; "
                 f"AA34 counts more on {more:,} = {pct(more, per_eye[e])}, fewer on {fewer:,} = {pct(fewer, per_eye[e])})\n")
        L.append(table(["armW \\ AA34"] + [f"AA34 {b}" for b in BUCKETS] + ["armW total"], rows_md))
        L.append("")

    # (c) + (d): either / both / only / neither
    L.append("## (c) Frames with at least one face, per eye and both eyes\n")
    rows_md = []
    c_all = Counter()
    for e in EYES:
        m_e = d["view"] == e
        sc = split_counts(d["nW"][m_e] >= 1, d["nA"][m_e] >= 1)
        check(f"(c) {e} both + armW only + AA34 only + neither == rows",
              sc["both"] + sc["armW only"] + sc["AA34 only"] + sc["neither"] == per_eye[e])
        check(f"(c) {e} either == both + armW only + AA34 only",
              sc["either"] == sc["both"] + sc["armW only"] + sc["AA34 only"])
        rows_md.append([e] + [f"{sc[k]:,} ({pct(sc[k], per_eye[e])})" for k in SPLIT] + [f"{per_eye[e]:,}"])
        c_all.update(sc)
    check("(c) both-eye split sums to rows",
          c_all["both"] + c_all["armW only"] + c_all["AA34 only"] + c_all["neither"] == n)
    rows_md.append(["**both eyes**"] + [f"{c_all[k]:,} ({pct(c_all[k], n)})" for k in SPLIT] + [f"{n:,}"])
    L.append(table(["eye"] + SPLIT + ["rows"], rows_md))
    L.append("")

    # (e): episodes, sessions, residues with a face by either detector
    L.append("## (e) Episodes, sessions and residues with at least one face by either detector\n")
    rows_md = []
    for e in EYES:
        m_e = d["view"] == e
        m_f = m_e & ((d["nW"] >= 1) | (d["nA"] >= 1))
        ep_all, ep_f = len(set(d["episode"][m_e].tolist())), len(set(d["episode"][m_f].tolist()))
        se_all, se_f = len(set(d["session"][m_e].tolist())), len(set(d["session"][m_f].tolist()))
        rows_md.append([e, f"{ep_f:,} of {ep_all:,} ({pct(ep_f, ep_all)})", f"{se_f:,} of {se_all:,} ({pct(se_f, se_all)})"])
    L.append(table(["eye", "episodes with a face", "sessions with a face"], rows_md))
    L.append("")
    for e in EYES:
        m_e = d["view"] == e
        m_f = m_e & ((d["nW"] >= 1) | (d["nA"] >= 1))
        rows_md = []
        tot_rows = tot_f = 0
        for s in sorted(set(d["s"][m_e].tolist())):
            m_s = m_e & (d["s"] == s)
            n_s, f_s = int(m_s.sum()), int((m_f & (d["s"] == s)).sum())
            tot_rows += n_s
            tot_f += f_s
            rows_md.append([f"{s}", f"{n_s:,}", f"{f_s:,}", pct(f_s, n_s)])
        check(f"(e) {e} residue rows sum == rows", tot_rows == per_eye[e])
        check(f"(e) {e} residue face sum == (c) either", tot_f == split_counts(d["nW"][m_e] >= 1, d["nA"][m_e] >= 1)["either"])
        rows_md.append(["**total**", f"{tot_rows:,}", f"{tot_f:,}", pct(tot_f, tot_rows)])
        L.append(f"### {e}: frames per residue s (batches "
                 f"{', '.join(str(b) for b in sorted(set(d['batch'][m_e].tolist())))})\n")
        L.append(table(["s", "frames", "with a face (either)", "share"], rows_md))
        L.append("")

    L.append("## Notes\n")
    m12 = (d["view"] == "vst_left") & (d["batch"] <= 2)
    ab12 = int((m12 & (d["nA"] >= 1)).sum())
    wb12 = int((m12 & (d["nW"] >= 1)).sum())
    if wb12 == 0:
        L.append("- vst_left batches 1 and 2 are 100% armW 0 faces because round 1 took every armW `n_faces >= 1` frame of "
                 f"those batches (WOR-94 rule); the {ab12:,} AA34-positive frames left there are frames armW scored 0 "
                 "(AA34-only candidates that round 1 could not see).")
        L.append("- The per-residue share is ~14% on every residue of batches 3, 4 and 5 and ~2.7% on the batch 1/2 "
                 "residues for the same reason; no single second stands out.")
    else:
        L.append(f"- Round 1 took every frame of vst_left batches 1 and 2 with an armW box at score >= {CSV_THR:g} "
                 f"(WOR-94 rule), so every armW face counted there at {thr:g} ({wb12:,} frames) comes from a box "
                 f"scored in [{thr:g}, {CSV_THR:g}); {ab12:,} frames there have an AA34 face at {thr:g}.")
        L.append("- The per-residue shares stay lower on the batch 1/2 residues for the same reason.")
    L.append("- Episode and session counts coincide because each session holds one episode in this table; "
             "vst_right has 2 fewer because of the 2 videos with permanent 404 (WOR-74).\n")
    L.append("## Sanity checks (all asserted by `mining/faceight_buckets.py`; the script exits on the first failure)\n")
    L += [f"- {name}: ok" for name, ok in checks]
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    ap.add_argument("--csv", default=os.path.join(root, "data", "faceight", "frames.csv"))
    ap.add_argument("--out", default=None, help="markdown file to write (default: stdout)")
    ap.add_argument("--thr", type=float, default=None,
                    help="recount faces as boxes with score >= THR from the JSONLs (needs --jsonl-dir)")
    ap.add_argument("--jsonl-dir", default=None,
                    help="directory holding faceight_armW{,_right}.jsonl and faceight_armAA34{,_right}.jsonl")
    args = ap.parse_args()
    if (args.thr is None) != (args.jsonl_dir is None):
        ap.error("--thr and --jsonl-dir go together")
    text = report(args.csv, args.thr, args.jsonl_dir)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

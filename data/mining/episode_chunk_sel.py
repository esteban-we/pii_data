#!/usr/bin/env python3
"""Rebuild the chunk_sel column of data/episode_usage.csv from a chunk-probe JSONL.

chunk_sel = index of the LATEST chunk whose duration is >= MIN_DUR_S (240 s by
default), -1 when the session is unprobed, errored, or has no such chunk
(WOR-52 rule, WOR-66). Idempotent: an existing chunk_sel column is replaced.
Every other column is left byte-identical (the CSV is rewritten line by line;
it has no quoting and no CR).

JSONL record (mining/episode_chunks_probe.py, WOR-62), one per session; the
last record per session wins:
  {session_id, episode_id, faceight, n_chunks, total_dur_s,
   chunks: [{i, bytes, dur_ms, nb, fps}], err}

Usage:
  python mining/episode_chunk_sel.py PROBE.jsonl [--csv data/episode_usage.csv]
                                     [--min-dur 240] [--stats]
"""
import argparse
import collections
import json
import os
import sys

COL = "chunk_sel"


def load_probe(path):
    """session_id -> record (last one wins); unparsable lines are skipped."""
    recs = {}
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1  # e.g. a line still being written when copied
                continue
            recs[r["session_id"]] = r
    return recs, bad


def chunk_sel(rec, min_dur_s):
    """Latest chunk index with dur_ms >= min_dur_s*1000, else -1."""
    if rec is None or rec.get("err"):
        return -1
    best = -1
    for c in rec.get("chunks") or []:
        if c.get("dur_ms") is None:
            continue
        if c["dur_ms"] >= min_dur_s * 1000.0 and c["i"] > best:
            best = c["i"]
    return best


def rebuild(csv_path, recs, min_dur_s):
    """Rewrite csv_path with COL as last column. Returns per-row (faceight, sel, probed)."""
    tmp = csv_path + ".tmp"
    out_rows = []
    with open(csv_path) as fin, open(tmp, "w") as fout:
        header = fin.readline().rstrip("\n").split(",")
        k_present = COL in header
        if k_present:
            k = header.index(COL)
            if k != len(header) - 1:
                sys.exit(f"{COL} is column {k}, not last; refusing (columns after it would shift)")
            header = header[:k]
        i_sess = header.index("session_id")
        i_f8 = header.index("faceight") if "faceight" in header else None
        fout.write(",".join(header) + f",{COL}\n")
        for line in fin:
            line = line.rstrip("\n")
            base = line.rsplit(",", 1)[0] if k_present else line
            fields = base.split(",")
            rec = recs.get(fields[i_sess])
            sel = chunk_sel(rec, min_dur_s)
            fout.write(f"{base},{sel}\n")
            out_rows.append((fields[i_f8] if i_f8 is not None else "", sel, rec is not None))
    os.replace(tmp, csv_path)
    return out_rows


def stats(rows, recs, bad):
    print(f"probe records: {len(recs)} sessions ({bad} unparsable lines skipped)")
    for name, pick in (("faceight", lambda f, p: f == "1"),
                       ("corpus (probed, non-faceight)", lambda f, p: f != "1" and p),
                       ("corpus (all non-faceight rows)", lambda f, p: f != "1")):
        sub = [s for f, s, p in rows if pick(f, p)]
        if not sub:
            continue
        h = collections.Counter(sub)
        n = len(sub)
        neg = h.get(-1, 0)
        print(f"[{name}] rows {n}, chunk_sel=-1 {neg} ({100*neg/n:.1f}%), "
              f">=0 {n-neg} ({100*(n-neg)/n:.1f}%)")
        print("  hist:", ", ".join(f"{k}:{h[k]}" for k in sorted(h)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "episode_usage.csv"))
    ap.add_argument("--min-dur", type=float, default=240.0, help="seconds (default 240)")
    ap.add_argument("--stats", action="store_true", help="print coverage and histogram")
    a = ap.parse_args()
    recs, bad = load_probe(a.jsonl)
    rows = rebuild(a.csv, recs, a.min_dur)
    print(f"wrote {a.csv}: {len(rows)} rows, column {COL}")
    if a.stats:
        stats(rows, recs, bad)


if __name__ == "__main__":
    main()

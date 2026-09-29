#!/usr/bin/env python3
"""Build data/faceight/frames.csv: one row per pulled faceight frame of either
eye, with an annotation_round column (WOR-94, WOR-106).

Columns: file, episode_id, session_id, chunk, s, batch, t_ms, frame_idx,
n_faces, annotation_round, then file_right, n_faces_right (WOR-96), then view,
n_faces_aa34 (WOR-106). `file` is the key: the lview name for left frames and
the rview name for right frames. Boxes are NOT copied; they stay in the JSONL
on shang. The initial build writes rows sorted by file; rows added later by
--append go at the end (see below), so file order is not a key.

Every JSONL record is checked on read: t_ms == 1000 * s + PHASE and file ==
<session>_c<chunk:03d>_<lview|rview>_t<t_ms>.jpg for the record's view; a
record that fails stops the script.

Round 1 rule (built in, used when --round 1 and no --select-file): every frame
with n_faces >= 1, plus N_ZERO frames drawn uniformly without replacement from
the n_faces == 0 frames with numpy.random.default_rng(SEED) over the zero-face
file names sorted. The draw is frozen: once the output has round-1 rows the
built-in rule may not fill any further row (it would, after --append, since
the zero-face set has grown); rerunning it on an unchanged file is a no-op
(nothing is written), and on a grown file it exits with an error. Later
assignments go through --select-file.

Append-only: if the output exists, every nonzero annotation_round is kept as
is; the script only ever fills rows still at 0. For --round N >= 2 there is no
built-in rule: without --select-file the script is a no-op that prints how many
rows are still at 0 and exits without writing. With --select-file (one file
name per line) the listed rows still at 0 get round N; listed rows already
nonzero are left alone and reported. This mode edits the annotation_round
field of the listed lines in place (line-level, no re-serialisation, no JSONL
needed); a table with the view column is never re-serialised.

--append (WOR-95, WOR-106): the JSONL has grown (new residues in residues.csv);
rows for JSONL files absent from the existing frames.csv are appended at the
end of the file, sorted by file among themselves, with annotation_round 0,
empty file_right / n_faces_right, view = --view and n_faces_aa34 from
--aa34-jsonl (empty where that JSONL has no record). --view names the eye of
the JSONL (required once the view column exists; every record must carry it).
A right-eye record whose file equals an existing row's file_right (the frozen
round-1 twins) is NOT appended: it is already represented by its left row; such
records are counted and reported as skipped. Every existing line is copied
byte for byte (line-level, like --merge-right). A frames.csv row of the same
view whose file is missing from the JSONL is an error (frames never
disappear); an --aa34-jsonl record whose file is in neither the JSONL nor the
table is an error. Rerun is a no-op; --dry-run prints the counts and writes
nothing.

--add-view-aa34 (WOR-106): appends the two columns view and n_faces_aa34 to an
existing frames.csv (12 columns). view comes from the file name (_lview_ ->
vst_left, _rview_ -> vst_right), n_faces_aa34 from --aa34-jsonl, keyed by file,
empty where there is no record. Every existing byte stays in place (line-level
append); an --aa34-jsonl record whose file is in neither the table nor --jsonl
(the armW JSONL of the same eye; frames not appended yet) is an error. On a
table that already has the columns, a value already present is kept where the
given JSONLs have no record (a rerun with one eye's JSONL is a no-op) and a
record that would change a present value is an error.

--merge-right JSONL (WOR-96): fills the columns file_right and n_faces_right of
an existing frames.csv from the right-view armW JSONL (faceight_armW_right.jsonl),
matched on the left file name (session, chunk, t_ms; `_rview_` -> `_lview_`),
appending the two columns when the header does not have them yet. Rows
without a right record get empty values. Every other column is copied byte for
byte (line-level, no re-serialisation); rerunning replaces the two columns. A
right record without a left row must be a row of its own (view vst_right),
otherwise it is an error. Nothing else happens in that mode.

Stdlib + numpy only (run on shang with /data/esteban/armw/venv/bin/python).
"""
import argparse
import csv
import io
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
from pii_root import open_index  # noqa: E402  (frames.csv is tracked gzipped, PII-1639)

SEED = 19760703
N_ZERO = 20000
PHASE = 233  # t_ms = 1000 * s + PHASE (residues.csv rule, .knuth/docs/faceight.md)
COLS = ["file", "episode_id", "session_id", "chunk", "s", "batch", "t_ms",
        "frame_idx", "n_faces", "annotation_round"]
EXTRA = ["file_right", "n_faces_right"]  # appended by --merge-right (WOR-96)
VIEWCOLS = ["view", "n_faces_aa34"]      # appended by --add-view-aa34 (WOR-106)
HEADERS = [COLS, COLS + EXTRA, COLS + EXTRA + VIEWCOLS]
VIEWS = {"vst_left": "lview", "vst_right": "rview"}
BUCKETS = ["0", "1", "2", "3", "4+"]


def bucket(n):
    return "4+" if n >= 4 else str(n)


def view_of(fn):
    """vst_left / vst_right from the file name."""
    for v, tag in VIEWS.items():
        if f"_{tag}_" in fn:
            return v
    sys.exit(f"file {fn!r} is neither a _lview_ nor a _rview_ name")


def check_record(r, where):
    """t_ms follows the residue rule and the file name is the name of that frame for its view."""
    want_t = 1000 * int(r["s"]) + PHASE
    if int(r["t_ms"]) != want_t:
        sys.exit(f"{where}: t_ms {r['t_ms']} != 1000*s+{PHASE} = {want_t} ({r['file']})")
    view = r.get("view", "vst_left")
    if view not in VIEWS:
        sys.exit(f"{where}: unknown view {view!r} ({r['file']})")
    want_f = f"{r['session_id']}_c{int(r['chunk']):03d}_{VIEWS[view]}_t{int(r['t_ms'])}.jpg"
    if r["file"] != want_f:
        sys.exit(f"{where}: file {r['file']!r} != expected {want_f!r}")


def read_jsonl(path, view=None):
    """file -> record fields (COLS without annotation_round, plus view). With view, every
    record must carry that view; without, the file must hold a single view."""
    rows, views = {}, set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["file"] in rows:
                sys.exit(f"duplicate file at line {ln}: {r['file']}")
            check_record(r, f"{path} line {ln}")
            v = r.get("view", "vst_left")
            if view and v != view:
                sys.exit(f"{path} line {ln}: view {v!r} != --view {view!r} ({r['file']})")
            views.add(v)
            d = {c: r[c] for c in COLS if c != "annotation_round"}
            d["view"] = v
            rows[r["file"]] = d
    if len(views) > 1:
        sys.exit(f"{path}: mixed views {sorted(views)}; one JSONL per eye")
    return rows


def read_aa34(paths, known, view=None):
    """file -> n_faces from armAA34 JSONLs; every file must be in `known` (and of `view`)."""
    aa = {}
    for path in paths:
        read_aa34_one(path, known, view, aa)
    return aa


def read_aa34_one(path, known, view, aa):
    unknown = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["file"] in aa:
                sys.exit(f"{path} line {ln}: duplicate file {r['file']}")
            check_record(r, f"{path} line {ln}")
            if view and r.get("view") != view:
                sys.exit(f"{path} line {ln}: view {r.get('view')!r} != --view {view!r} ({r['file']})")
            if r["file"] not in known:
                unknown.append(r["file"])
            aa[r["file"]] = str(int(r["n_faces"]))
    if unknown:
        sys.exit(f"{path}: {len(unknown)} records whose file is unknown (in neither the table nor "
                 f"the JSONL), e.g. {unknown[:3]}; refusing")
    print(f"aa34 jsonl {path}: {len(aa)} records so far")


def read_lines(csv_path):
    """Raw lines of an existing frames.csv plus its header (one of HEADERS)."""
    with open_index(csv_path) as f:
        lines = f.readlines()
    if not lines:
        sys.exit(f"{csv_path}: empty file")
    header = lines[0].rstrip("\r\n").split(",")
    if header not in HEADERS:
        sys.exit(f"{csv_path}: unexpected header {header!r}")
    return lines, header


def split_body(line, ncol, where):
    """Fields of a raw line; the table never carries quotes or commas inside a field."""
    body = line.rstrip("\r\n")
    if not body:
        sys.exit(f"{where}: empty line")
    if '"' in body:
        sys.exit(f"{where}: quoted field, not a line-level table")
    fields = body.split(",")
    if len(fields) != ncol:
        sys.exit(f"{where}: {len(fields)} fields, header has {ncol}")
    return body, line[len(body):], fields


def read_table(csv_path):
    """lines, header, and per-row field lists (rows[i] belongs to lines[i + 1])."""
    lines, header = read_lines(csv_path)
    rows = [split_body(line, len(header), f"{csv_path} line {i}")[2]
            for i, line in enumerate(lines[1:], 2)]
    files = [r[0] for r in rows]
    if len(set(files)) != len(files):
        dup = [f for f, n in Counter(files).items() if n > 1]
        sys.exit(f"{csv_path}: {len(dup)} duplicate file names, e.g. {dup[:3]}")
    return lines, header, rows


def read_existing(path):
    """file -> annotation_round, file -> extra values (columns beyond COLS), header, twin files."""
    rounds, extras, twins = {}, {}, set()
    with open_index(path) as f:
        rd = csv.DictReader(f)
        if rd.fieldnames not in HEADERS:
            sys.exit(f"{path}: unexpected header {rd.fieldnames}")
        extra_cols = rd.fieldnames[len(COLS):]
        for r in rd:
            rounds[r["file"]] = int(r["annotation_round"])
            extras[r["file"]] = [r[c] for c in extra_cols]
            if r.get("file_right"):
                twins.add(r["file_right"])
    return rounds, extras, list(rd.fieldnames), twins


def write_lines(out_path, lines):
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.writelines(lines)
    os.replace(tmp, out_path)


def merge_right(csv_path, jsonl_path, out_path, dry_run):
    """Fill / append EXTRA columns from the right-view JSONL; every other byte untouched."""
    right = {}
    with open(jsonl_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            want = f"{r['session_id']}_c{int(r['chunk']):03d}_rview_t{int(r['t_ms'])}.jpg"
            if r.get("view") != "vst_right" or r["file"] != want:
                sys.exit(f"{jsonl_path} line {ln}: not a right-view record ({r.get('view')}, {r['file']})")
            left = r["file"].replace("_rview_", "_lview_", 1)
            if left in right:
                sys.exit(f"{jsonl_path} line {ln}: duplicate right record for {left}")
            right[left] = (r["file"], str(int(r["n_faces"])))
    print(f"right jsonl: {len(right)} records")

    lines, header, rows = read_table(csv_path)
    has = header[len(COLS):len(COLS) + 2] == EXTRA
    if has:
        print(f"{csv_path}: already has {EXTRA}, replacing them")
        new_header = header
    else:
        new_header = header + EXTRA
    files = {r[0] for r in rows}
    unmatched = sorted(fr for lf, (fr, _) in right.items() if lf not in files and fr not in files)
    if unmatched:
        sys.exit(f"{len(unmatched)} right records have neither a left row nor a row of their own in "
                 f"{csv_path}, e.g. {unmatched[:3]}")
    own = sum(1 for lf, (fr, _) in right.items() if lf not in files and fr in files)
    filled, out = 0, [",".join(new_header) + lines[0][len(lines[0].rstrip("\r\n")):]]
    for line, fields in zip(lines[1:], rows):
        eol = line[len(line.rstrip("\r\n")):]
        fr, n = right.get(fields[0], ("", ""))
        filled += bool(fr)
        if has:
            fields = fields[:len(COLS)] + [fr, n] + fields[len(COLS) + 2:]
        else:
            fields = fields + [fr, n]
        out.append(",".join(fields) + eol)
    print(f"{csv_path}: {len(rows)} rows, {filled} with file_right, {len(rows) - filled} empty, "
          f"{own} right records are rows of their own (no twin link)")
    if dry_run:
        print("dry run, not writing")
        return
    write_lines(out_path, out)
    print(f"wrote {out_path}: {len(out) - 1} rows")


def add_view_aa34(csv_path, jsonl_path, aa34_paths, out_path, dry_run):
    """--add-view-aa34: append view and n_faces_aa34 to every line; old bytes untouched.
    An AA34 file must be in the table or in the armW JSONL (frames not appended yet)."""
    lines, header, rows = read_table(csv_path)
    if header == COLS + EXTRA + VIEWCOLS:
        print(f"{csv_path}: already has {VIEWCOLS}, replacing them")
        base = len(COLS) + len(EXTRA)
    elif header == COLS + EXTRA:
        base = len(header)
    else:
        sys.exit(f"{csv_path}: --add-view-aa34 needs the {len(COLS) + len(EXTRA)}-column table "
                 f"(run --merge-right first), header has {len(header)} columns")
    files = {r[0] for r in rows}
    known = files | set(read_jsonl(jsonl_path))
    aa = read_aa34(aa34_paths, known)
    later = len(set(aa) - files)
    print(f"aa34 records not in the table (frames not appended yet): {later}")
    views, filled, kept, changed = Counter(), 0, 0, []
    out = [",".join(COLS + EXTRA + VIEWCOLS) + lines[0][len(lines[0].rstrip("\r\n")):]]
    for line, fields in zip(lines[1:], rows):
        eol = line[len(line.rstrip("\r\n")):]
        v = view_of(fields[0])
        views[v] += 1
        old = fields[base + 1] if base < len(fields) else ""
        n = aa.get(fields[0], old)  # no record: keep the value already there (rerun with one eye)
        if old and n != old:
            changed.append((fields[0], old, n))
        kept += bool(old) and fields[0] not in aa
        filled += bool(n)
        out.append(",".join(fields[:base] + [v, n]) + eol)
    if changed:
        sys.exit(f"{len(changed)} rows would change n_faces_aa34, e.g. {changed[:3]}; refusing "
                 f"(the column is append-only, rebuild deliberately)")
    print(f"{csv_path}: {len(rows)} rows; view {dict(sorted(views.items()))}; n_faces_aa34 filled "
          f"{filled}, empty {len(rows) - filled}, kept from the file {kept}")
    if dry_run:
        print("dry run, not writing")
        return
    write_lines(out_path, out)
    print(f"wrote {out_path}: {len(out) - 1} rows, {len(COLS + EXTRA + VIEWCOLS)} columns")


def append_rows(csv_path, jsonl_path, view, aa34_paths, out_path, dry_run):
    """--append: add rows for JSONL files absent from csv_path at annotation_round 0.
    Existing lines are copied byte for byte; new rows go at the end, sorted by file."""
    lines, header, table = read_table(csv_path)
    has_view = header == COLS + EXTRA + VIEWCOLS
    if has_view and not view:
        sys.exit(f"{csv_path} has a view column: --append needs --view {{{','.join(VIEWS)}}}")
    if has_view and not aa34_paths:
        sys.exit(f"{csv_path} has n_faces_aa34: --append needs --aa34-jsonl (the armAA34 JSONL of "
                 f"the same eye)")
    rows = read_jsonl(jsonl_path, view)
    if not view:
        view = next(iter(rows.values()))["view"] if rows else "vst_left"
    seen = [r[0] for r in table]
    seen_set = set(seen)
    twins = set()
    if EXTRA[0] in header:
        i_fr = header.index(EXTRA[0])
        twins = {r[i_fr] for r in table if r[i_fr]}
    same_view = {f for f in seen if view_of(f) == view}
    gone = sorted(same_view - set(rows))
    if gone:
        sys.exit(f"{len(gone)} {view} rows of {csv_path} have no record in the JSONL (frames never "
                 f"disappear), e.g. {gone[:3]}; refusing")
    aa = read_aa34(aa34_paths, seen_set | set(rows), view) if aa34_paths else {}
    absent = set(rows) - seen_set
    skipped = sorted(absent & twins)
    new = sorted(absent - twins)
    print(f"{csv_path}: {len(seen)} rows; jsonl: {len(rows)} {view} records; {len(new)} to append "
          f"at annotation_round 0; {len(skipped)} skipped as round-1 twins (file_right of an "
          f"existing row)")
    if not new:
        print("nothing to append, file unchanged")
        return
    n_aa = sum(1 for f in new if f in aa)
    print(f"to append: {dist(rows, new)}; n_faces_aa34 filled on {n_aa}, empty on {len(new) - n_aa}")
    if dry_run:
        print("dry run, not writing")
        return
    eol = "\r\n" if lines[0].endswith("\r\n") else "\n"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol)
    for fn in new:
        r = rows[fn]
        vals = {c: r[c] for c in COLS[:-1]}
        vals.update({"annotation_round": 0, "file_right": "", "n_faces_right": "",
                     "view": view, "n_faces_aa34": aa.get(fn, "")})
        w.writerow([vals[c] for c in header])
    tail = "" if lines[-1].endswith(("\n", "\r")) else eol
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.writelines(lines)
        f.write(tail + buf.getvalue())
    os.replace(tmp, out_path)
    print(f"wrote {out_path}: {len(seen)} rows kept, {len(new)} appended, {len(seen) + len(new)} total")


def select_rows(csv_path, rnd, select_path, out_path, dry_run):
    """--round N --select-file: set annotation_round on the listed rows still at 0, in place."""
    lines, header, table = read_table(csv_path)
    with open(select_path) as f:
        sel = {ln.strip() for ln in f if ln.strip()}
    files = {r[0] for r in table}
    unknown = sel - files
    if unknown:
        sys.exit(f"--select-file names {len(unknown)} unknown files, e.g. {sorted(unknown)[:3]}")
    print(f"round {rnd}: --select-file lists {len(sel)} frames")
    i_r, i_n = header.index("annotation_round"), header.index("n_faces")
    filled, kept, out = 0, 0, [lines[0]]
    by_round = {}
    for line, fields in zip(lines[1:], table):
        if fields[0] in sel:
            if int(fields[i_r]) == 0:
                fields[i_r] = str(rnd)
                filled += 1
                line = ",".join(fields) + line[len(line.rstrip("\r\n")):]
            else:
                kept += 1
        by_round.setdefault(int(fields[i_r]), []).append(int(fields[i_n]))
        out.append(line)
    print(f"round {rnd}: filled {filled} rows at 0; {kept} selected rows already nonzero, left as is")
    for r in sorted(by_round):
        print(f"annotation_round {r}: {dist_n(by_round[r])}")
    if not filled:
        print("nothing to fill, file unchanged")
        return
    if dry_run:
        print("dry run, not writing")
        return
    write_lines(out_path, out)
    print(f"wrote {out_path}: {len(out) - 1} rows, {filled} changed")


def round1_selection(rows, n_zero, seed):
    sel = {f for f, r in rows.items() if r["n_faces"] >= 1}
    zero = sorted(f for f, r in rows.items() if r["n_faces"] == 0)
    if n_zero > len(zero):
        sys.exit(f"asked for {n_zero} zero-face frames, only {len(zero)} exist")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(zero), size=n_zero, replace=False)
    sel.update(zero[i] for i in idx)
    return sel


def dist_n(ns):
    c = Counter(bucket(n) for n in ns)
    return " / ".join(f"{b}={c[b]}" for b in BUCKETS) + f"  (total {sum(c.values())})"


def dist(rows, files):
    return dist_n(rows[f]["n_faces"] for f in files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default="/data/esteban/faceight/faceight_armW.jsonl")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "faceight", "frames.csv"))
    ap.add_argument("--round", type=int, default=1, dest="rnd")
    ap.add_argument("--select-file", default=None,
                    help="file names (one per line) to assign --round to, if still at 0")
    ap.add_argument("--n-zero", type=int, default=N_ZERO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--append", action="store_true",
                    help="append rows for JSONL files absent from --out at annotation_round 0; "
                         "existing lines untouched; then exit")
    ap.add_argument("--view", default=None, choices=sorted(VIEWS),
                    help="eye of --jsonl for --append (required once --out has the view column)")
    ap.add_argument("--aa34-jsonl", default=None, metavar="JSONL", nargs="+",
                    help="armAA34 JSONL(s): source of n_faces_aa34 for --append (the eye of --view) "
                         "and --add-view-aa34 (any eyes; values already in the table are kept)")
    ap.add_argument("--add-view-aa34", action="store_true",
                    help="append the view and n_faces_aa34 columns to --out (from --aa34-jsonl) and exit")
    ap.add_argument("--merge-right", default=None, metavar="JSONL",
                    help="fill file_right,n_faces_right in --out from this right-view armW JSONL and exit")
    a = ap.parse_args()
    modes = [bool(a.merge_right), a.append, a.add_view_aa34]
    if sum(modes) > 1:
        sys.exit("--append, --merge-right and --add-view-aa34 are separate modes; run them one at a time")
    if a.merge_right:
        if not os.path.exists(a.out):
            sys.exit(f"--merge-right needs an existing {a.out}")
        merge_right(a.out, a.merge_right, a.out, a.dry_run)
        return
    if a.add_view_aa34:
        if not os.path.exists(a.out):
            sys.exit(f"--add-view-aa34 needs an existing {a.out}")
        if not a.aa34_jsonl:
            sys.exit("--add-view-aa34 needs --aa34-jsonl")
        add_view_aa34(a.out, a.jsonl, a.aa34_jsonl, a.out, a.dry_run)
        return
    if a.append:
        if not os.path.exists(a.out):
            sys.exit(f"--append needs an existing {a.out}; build it first with --round 1")
        append_rows(a.out, a.jsonl, a.view, a.aa34_jsonl, a.out, a.dry_run)
        return
    if a.rnd < 1:
        sys.exit("--round must be >= 1")
    if a.select_file:
        if not os.path.exists(a.out):
            sys.exit(f"--select-file needs an existing {a.out}")
        select_rows(a.out, a.rnd, a.select_file, a.out, a.dry_run)
        return

    rows = read_jsonl(a.jsonl, a.view)
    files = sorted(rows)
    jview = next(iter(rows.values()))["view"] if rows else "vst_left"
    print(f"jsonl: {len(rows)} {jview} frames  n_faces buckets: {dist(rows, files)}")

    rounds, extras, old, header = {f: 0 for f in files}, {}, {}, COLS
    if os.path.exists(a.out):
        old, extras, header, twins = read_existing(a.out)
        missing = set(files) - set(old) - twins
        extra = {f for f in old if header != HEADERS[2] or view_of(f) == jview} - set(files)
        if missing or extra:
            sys.exit(f"{a.out} does not cover the JSONL: {len(missing)} missing, "
                     f"{len(extra)} extra rows; refusing (use --append for a grown JSONL)")
        rounds.update({f: old[f] for f in files if f in old})
        print(f"existing {a.out}: rounds {dict(sorted(Counter(old.values()).items()))}")

    # selection rule
    if a.rnd == 1:
        sel = round1_selection(rows, a.n_zero, a.seed)
        print(f"round 1 rule: all n_faces>=1 plus {a.n_zero} zero-face frames, seed {a.seed}: "
              f"{len(sel)} frames")
    else:
        still0 = sum(1 for v in rounds.values() if v == 0)
        print(f"round {a.rnd}: no built-in rule and no --select-file; no-op. "
              f"{still0} rows still at annotation_round 0 would be eligible. Not writing.")
        return

    filled, kept = [], []
    for f in sorted(sel):
        if rounds[f] == 0:
            rounds[f] = a.rnd
            filled.append(f)
        else:
            kept.append(f)
    print(f"round {a.rnd}: filled {len(filled)} rows at 0; {len(kept)} selected rows already "
          f"nonzero, left as is")
    if filled and any(v == 1 for v in old.values()):
        sys.exit(f"{a.out} already has round-1 rows and the built-in round-1 rule would fill "
                 f"{len(filled)} more (the JSONL has grown since the draw); the round-1 draw is "
                 f"frozen, assign rounds with --select-file. Not writing.")
    if filled and header == HEADERS[2]:
        sys.exit(f"{a.out} has the view column (one row per frame of either eye) and is never "
                 f"re-serialised from one JSONL; assign rounds with --select-file. Not writing.")

    by_round = {}
    for f in files:
        by_round.setdefault(rounds[f], []).append(f)
    for r in sorted(by_round):
        print(f"annotation_round {r}: {dist(rows, by_round[r])}")

    if old and not filled:
        print("nothing to fill, file unchanged")
        return
    if a.dry_run:
        print("dry run, not writing")
        return
    tmp = a.out + ".tmp"
    n_extra = len(header) - len(COLS)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for fn in files:
            r = rows[fn]
            w.writerow([r[c] for c in COLS[:-1]] + [rounds[fn]] + extras.get(fn, [""] * n_extra))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(files)} rows")


if __name__ == "__main__":
    main()

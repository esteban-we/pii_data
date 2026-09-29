#!/usr/bin/env python3
"""Generate the frozen faceback_45 train / eval_faceback_v1 session split (WOR-102).

Sibling of face_mine_split.py. Session-level 80/20 split: both eyes and every
chunk of a session land on one side (chunk-safe by construction: a chunk name
is <session>_c<idx>, so no chunk can cross sides). Sessions already in the
frozen face_mine_v1 lists keep their side (eval_sessions_v1 -> eval,
train_sessions_v1 -> train); the remaining sessions are assigned by a seeded
stratified draw (month x face-free bin x small-face bin) followed by a
deterministic hill-climb, so that sessions, frames and boxes are each within
TOL of EVAL_FRAC and month / face-free share / small-face share are balanced.

Per-session box mass is very skewed (median 54, max 4,049), so a plain
stratified draw like face_mine_split.py cannot hit +-1 pt on boxes; hence the
restarts + local search. Deterministic: SEED, sorted iteration everywhere.

Usage:
  faceback_split.py                 # search, print table, write the lists
  faceback_split.py --write-csv     # ... and update data/episode_usage.csv
  faceback_split.py --dry-run       # search + checks only, write nothing
  --manifest/--eval-list/--train-list/--csv/--out-dir override the inputs and
  the repo output dir (used by the sabotage runs).
Exit 1 on any failed check.
"""
import argparse
import collections
import hashlib
import random
import re
import sys
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import CODE_ROOT, dataset as pii_dataset  # noqa: E402

# PII-1449: the live store keeps the frozen split as datasets/faceback_45/split.csv;
# the splits/ mirror this script writes is the older pair of .txt lists.
DATASET = pii_dataset("faceback_45")
MANIFEST = Path(CODE_ROOT) / "training" / "manifests" / "faceback_45_labelv2.txt"
CSV = STORE / "data" / "episode_usage.csv"
SPLITS = STORE / "data" / "splits"
DATASET_NAME = "faceback_45"
FROZEN = "face_mine_v1"          # the dataset whose split we must agree with

SEED = 20260909
EVAL_FRAC = 0.20
TOL = 0.01                       # +-1 pt on sessions, frames and boxes
SMALL_PX = 64                    # box height in the raw 2328x1748 frame
RESTARTS = 200
MAX_PASSES = 50                  # hill-climb sweeps per restart

NAME_RE = re.compile(r"^(\d{8}_\d{6}_[A-Z]{6})_c(\d{3})_(left|right)_f(\d{6})\.jpg$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- inputs

def session_features(manifest):
    """Per-session frames / boxes / face-free frames / small boxes / chunks."""
    S = collections.defaultdict(
        lambda: dict(frames=0, boxes=0, ff=0, small=0, chunks=set()))
    cur = None
    nb = 0
    n_headers = n_boxes = 0
    with open(manifest) as f:
        for ln, line in enumerate(f, 1):
            if line.startswith("#"):
                if cur is not None and nb == 0:
                    cur["ff"] += 1
                parts = line[1:].split()
                if len(parts) != 3 or not parts[0].startswith("faceback/"):
                    fail(f"{manifest}:{ln}: bad header {line.rstrip()!r}")
                m = NAME_RE.match(parts[0][len("faceback/"):])
                if not m:
                    fail(f"{manifest}:{ln}: bad image name {parts[0]!r}")
                sess, chunk = m.group(1), m.group(2)
                cur = S[sess]
                cur["frames"] += 1
                cur["chunks"].add(f"{sess}_c{chunk}")
                nb = 0
                n_headers += 1
            else:
                if cur is None:
                    fail(f"{manifest}:{ln}: box line before any header")
                v = line.split()
                if len(v) < 4:
                    fail(f"{manifest}:{ln}: bad box line")
                x1, y1, x2, y2 = map(float, v[:4])
                if not (x2 > x1 and y2 > y1):
                    fail(f"{manifest}:{ln}: degenerate box")
                cur["boxes"] += 1
                if y2 - y1 < SMALL_PX:
                    cur["small"] += 1
                nb += 1
                n_boxes += 1
    if cur is not None and nb == 0:
        cur["ff"] += 1
    if n_headers == 0:
        fail(f"{manifest}: no image headers")
    for d in S.values():
        d["ff_share"] = d["ff"] / d["frames"]
        d["small_share"] = d["small"] / d["boxes"] if d["boxes"] else 0.0
    return dict(S), n_headers, n_boxes


def read_list(path):
    sessions = [l.strip() for l in open(path) if l.strip()]
    if len(set(sessions)) != len(sessions):
        fail(f"{path}: duplicate sessions")
    return set(sessions)


def csv_roles(csv_path):
    """session_id -> (dataset, role) from episode_usage.csv (no quoting used)."""
    roles = {}
    with open(csv_path, newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        if '"' in "".join(header):
            fail(f"{csv_path}: quoted header, line-level edit unsafe")
        col = {c: i for i, c in enumerate(header)}
        for need in ("session_id", "dataset", "role"):
            if need not in col:
                fail(f"{csv_path}: missing column {need}")
        for line in f:
            if '"' in line:
                fail(f"{csv_path}: quoted field, line-level edit unsafe")
            r = line.rstrip("\n").split(",")
            sid = r[col["session_id"]]
            if not sid:
                continue
            if sid in roles:
                fail(f"{csv_path}: duplicate session_id {sid}")
            roles[sid] = (r[col["dataset"]], r[col["role"]])
    return roles, col


def forced_sides(S, eval_v1, train_v1, roles):
    """Sessions pinned by the frozen face_mine_v1 split, with consistency checks."""
    both = eval_v1 & train_v1
    if both:
        fail(f"{len(both)} sessions in both {FROZEN} lists: {sorted(both)[:5]}")
    forced = {}
    for s in sorted(S):
        if s in eval_v1:
            forced[s] = "eval"
        elif s in train_v1:
            forced[s] = "train"
    # Cross-dataset consistency: the lists must agree with what episode_usage.csv
    # already records for face_mine_v1, in both directions.
    for s in sorted(S):
        if s not in roles:
            fail(f"{s} missing from episode_usage.csv")
        ds, role = roles[s]
        in_csv = FROZEN in ds.split("+")
        if in_csv and s not in forced:
            fail(f"{s}: csv says {ds}/{role} but it is in neither {FROZEN} list")
        if not in_csv and s in forced:
            fail(f"{s}: in {FROZEN} {forced[s]} list but csv dataset is {ds!r}")
        if in_csv and role != forced[s]:
            fail(f"{s}: csv role {role} != {FROZEN} list side {forced[s]}")
        claimed = DATASET_NAME in ds.split("+")     # idempotent re-run after --write-csv
        if not in_csv and not claimed and (ds or role != "unused"):
            fail(f"{s}: csv dataset {ds!r} role {role}: not unused, cannot claim")
    return forced


# ---------------------------------------------------------------- search

def stratum(sess, d):
    month = sess[:6]
    fb = 0 if d["ff_share"] <= 0.33 else (1 if d["ff_share"] <= 0.66 else 2)
    sb = (-1 if d["boxes"] == 0
          else 0 if d["small_share"] <= 0.2
          else 1 if d["small_share"] <= 0.4 else 2)
    return (month, fb, sb)


class Scorer:
    """Per-session integer feature vectors so a move is a vector add/subtract:
    [sessions, frames, boxes, face-free frames, small boxes, frames per month...]."""

    def __init__(self, S):
        self.S = S
        self.months = sorted({s[:6] for s in S})
        self.mi = {m: 5 + i for i, m in enumerate(self.months)}
        self.n = 5 + len(self.months)
        self.vec = {s: self._vec(s) for s in S}
        self.tot = self.sum_vec(S)

    def _vec(self, s):
        d = self.S[s]
        v = [0] * self.n
        v[0], v[1], v[2], v[3], v[4] = 1, d["frames"], d["boxes"], d["ff"], d["small"]
        v[self.mi[s[:6]]] = d["frames"]
        return v

    def sum_vec(self, keys):
        v = [0] * self.n
        for s in keys:
            for i, x in enumerate(self.vec[s]):
                v[i] += x
        return v

    def score_vec(self, e):
        t = self.tot
        hard = max(abs(e[k] / t[k] - EVAL_FRAC) for k in (0, 1, 2))
        tr = [t[i] - e[i] for i in range(self.n)]
        soft = abs(e[3] / e[1] - tr[3] / tr[1])
        soft += abs(e[4] / max(1, e[2]) - tr[4] / max(1, tr[2]))
        for i in range(5, self.n):
            soft += abs(e[i] / e[1] - tr[i] / tr[1])
        return hard * 100 + soft, hard, soft

    def score(self, eval_set):
        return self.score_vec(self.sum_vec(eval_set))

    def table(self, eval_set):
        rows = []
        train_set = set(self.S) - eval_set
        for name, keys in (("ALL", set(self.S)), ("TRAIN", train_set), ("EVAL", eval_set)):
            a = self.sum_vec(keys)
            rows.append(dict(
                name=name, sess=a[0], frames=a[1], boxes=a[2],
                sess_pct=a[0] / self.tot[0], frames_pct=a[1] / self.tot[1],
                boxes_pct=a[2] / self.tot[2], bpi=a[2] / a[1],
                ff=a[3] / a[1], small=a[4] / max(1, a[2]),
                months={m: a[self.mi[m]] / a[1] for m in self.months}))
        return rows


def stratified_start(S, free, forced, rng):
    """One seeded stratified draw over the free sessions (face_mine_split style)."""
    strata = collections.defaultdict(list)
    for s in sorted(S):
        strata[stratum(s, S[s])].append(s)
    eval_set = {s for s, side in forced.items() if side == "eval"}
    for _, sessions in sorted(strata.items()):
        target = EVAL_FRAC * sum(S[s]["frames"] for s in sessions)
        acc = sum(S[s]["frames"] for s in sessions if forced.get(s) == "eval")
        movable = [s for s in sessions if s in free]
        rng.shuffle(movable)
        for s in movable:
            if acc >= target:
                break
            eval_set.add(s)
            acc += S[s]["frames"]
    return eval_set


def hill_climb(scorer, eval_set, free, rng):
    """Single moves then pair swaps over the free sessions, first-improvement
    in a shuffled order; stops when a full pass finds nothing."""
    V = scorer.vec
    free_sorted = sorted(free)
    e = scorer.sum_vec(eval_set)
    cur = scorer.score_vec(e)[0]
    evaluated = 0
    for _ in range(MAX_PASSES):
        improved = False
        order = free_sorted[:]
        rng.shuffle(order)
        for s in order:                              # single moves
            sign = -1 if s in eval_set else 1
            trial = [x + sign * y for x, y in zip(e, V[s])]
            sc = scorer.score_vec(trial)[0]
            evaluated += 1
            if sc < cur - 1e-12:
                e, cur, improved = trial, sc, True
                eval_set.symmetric_difference_update({s})
        ev = [s for s in order if s in eval_set]
        tr = [s for s in order if s not in eval_set]
        for a in ev:                                  # pair swaps
            if a not in eval_set:
                continue
            for b in tr:
                if b in eval_set:
                    continue
                trial = [x - y + z for x, y, z in zip(e, V[a], V[b])]
                sc = scorer.score_vec(trial)[0]
                evaluated += 1
                if sc < cur - 1e-12:
                    e, cur, improved = trial, sc, True
                    eval_set.remove(a)
                    eval_set.add(b)
                    break
        if not improved:
            break
    return eval_set, cur, evaluated


def search(S, forced):
    free = set(S) - set(forced)
    scorer = Scorer(S)
    rng = random.Random(SEED)
    best = None
    n_eval = 0
    for r in range(RESTARTS):
        start = stratified_start(S, free, forced, rng)
        cand, sc, ne = hill_climb(scorer, set(start), free, rng)
        n_eval += ne
        if best is None or sc < best[1] - 1e-12:
            best = (frozenset(cand), sc, r)
    eval_set, sc, r = best
    _, hard, soft = scorer.score(eval_set)
    return set(eval_set), scorer, dict(score=sc, hard=hard, soft=soft, best_restart=r,
                                       restarts=RESTARTS, evaluated=n_eval)


# ---------------------------------------------------------------- checks / outputs

def verify(S, eval_set, train_set, forced, scorer, roles):
    if eval_set & train_set:
        fail("session on both sides")
    if eval_set | train_set != set(S):
        fail("split does not cover the manifest sessions")
    for s, side in forced.items():
        if (s in eval_set) != (side == "eval"):
            fail(f"{s} pinned to {side} by {FROZEN} but landed on the other side")
    chunk_side = {}
    for s in S:
        for c in S[s]["chunks"]:
            side = "eval" if s in eval_set else "train"
            if chunk_side.setdefault(c, side) != side:
                fail(f"chunk {c} crosses sides")
    _, hard, _ = scorer.score(eval_set)
    if hard > TOL:
        fail(f"hard balance {hard:.4f} exceeds TOL {TOL}")
    for s in sorted(S):                      # csv already carries this split?
        ds, role = roles[s]
        if DATASET_NAME in ds.split("+") and role != ("eval" if s in eval_set else "train"):
            fail(f"{s}: csv role {role} disagrees with the regenerated split")


def fmt_table(rows):
    months = sorted(rows[0]["months"])
    head = ("| Slice | Sessions | Frames | Boxes | Boxes/frame | Face-free | "
            "Small (<64px) | " + " | ".join(months) + " |")
    out = [head, "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(months)]
    for r in rows:
        pct = (lambda v: "" if r["name"] == "ALL" else f" ({v:.1%})")
        out.append(
            f"| {r['name']} | {r['sess']:,}{pct(r['sess_pct'])} | "
            f"{r['frames']:,}{pct(r['frames_pct'])} | {r['boxes']:,}{pct(r['boxes_pct'])} | "
            f"{r['bpi']:.3f} | {r['ff']:.1%} | {r['small']:.1%} | "
            + " | ".join(f"{r['months'][m]:.1%}" for m in months) + " |")
    return "\n".join(out)


def write_lists(out_dir, eval_set, train_set, store=None):
    targets = [out_dir] + ([store] if store else [])
    for root in targets:
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / "faceback_eval_sessions_v1.txt").write_text(
                "\n".join(sorted(eval_set)) + "\n")
            (root / "faceback_train_sessions_v1.txt").write_text(
                "\n".join(sorted(train_set)) + "\n")
        except PermissionError:
            print(f"SKIP {root}: not writable")
            continue
        print(f"wrote {root}/faceback_{{train,eval}}_sessions_v1.txt")


def update_csv(csv_path, col, eval_set, train_set):
    """Rewrite only the 334 faceback rows: dataset += faceback_45, role = side.
    Every other byte is preserved (no quoting in this file; checked on read)."""
    side = {s: "eval" for s in eval_set}
    side.update({s: "train" for s in train_set})
    raw = csv_path.read_bytes()
    if b"\r" in raw:
        fail(f"{csv_path}: CR found, line-level edit unsafe")
    lines = raw.decode().split("\n")
    changed = 0
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        r = line.split(",")
        s = r[col["session_id"]]
        if s not in side:
            continue
        ds = r[col["dataset"]]
        if DATASET_NAME in ds.split("+"):
            fail(f"{s}: dataset already contains {DATASET_NAME}; refusing to re-append")
        r[col["dataset"]] = f"{ds}+{DATASET_NAME}" if ds else DATASET_NAME
        r[col["role"]] = side[s]
        new = ",".join(r)
        if new != line:
            changed += 1
        lines[i] = new
    if changed != len(side):
        fail(f"csv: changed {changed} rows, expected {len(side)}")
    csv_path.write_bytes("\n".join(lines).encode())
    print(f"csv: {changed} rows changed, md5 {hashlib.md5(csv_path.read_bytes()).hexdigest()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--eval-list", type=Path, default=SPLITS / "eval_sessions_v1.txt")
    ap.add_argument("--train-list", type=Path, default=SPLITS / "train_sessions_v1.txt")
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--out-dir", type=Path, default=SPLITS)
    ap.add_argument("--no-store", action="store_true",
                    help=f"do not mirror the lists to {DATASET}/splits")
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    S, n_frames, n_boxes = session_features(a.manifest)
    roles, col = csv_roles(a.csv)
    forced = forced_sides(S, read_list(a.eval_list), read_list(a.train_list), roles)
    n_fe = sum(1 for v in forced.values() if v == "eval")
    print(f"manifest: {len(S)} sessions / {n_frames:,} frames / {n_boxes:,} boxes; "
          f"pinned by {FROZEN}: {n_fe} eval, {len(forced) - n_fe} train; "
          f"free: {len(S) - len(forced)}")

    eval_set, scorer, info = search(S, forced)
    train_set = set(S) - eval_set
    verify(S, eval_set, train_set, forced, scorer, roles)
    print(f"seed {SEED}: {info['restarts']} restarts, best from restart "
          f"{info['best_restart']}, {info['evaluated']:,} candidates scored, "
          f"hard max-dev {info['hard']:.4f} (TOL {TOL}), soft {info['soft']:.4f}")
    print(fmt_table(scorer.table(eval_set)))
    if a.dry_run:
        print("dry run: nothing written")
        return
    write_lists(a.out_dir, eval_set, train_set,
                store=None if a.no_store else DATASET / "splits")
    if a.write_csv:
        update_csv(a.csv, col, eval_set, train_set)


if __name__ == "__main__":
    main()

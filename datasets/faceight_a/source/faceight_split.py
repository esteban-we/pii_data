#!/usr/bin/env python3
"""Generate the frozen faceight_a train / eval episode split (WOR-193).

Sibling of faceback_split.py. faceight_a = the round-1 left-eye frames of
data/faceight/frames.csv (annotation_round == 1). Episode-level 80/20 split:
every round-1 frame of an episode, and the right-eye twin of each of them
(column file_right), lands on one side. In this table an episode has exactly
one session and a session exactly one episode; both directions are asserted.

An episode whose session already sits in a frozen split (face_mine_v1 lists
eval_sessions_v1 / train_sessions_v1, faceback_45 lists faceback_*_sessions_v1)
keeps that side, after a two-way consistency check against episode_usage.csv;
any other episode must be dataset '' / role unused there, and no other list in
data/splits/ may contain it. The free episodes are assigned by a seeded
stratified draw (scene_category x face-free bin x armW face-count band)
followed by a deterministic hill-climb (single moves, then a bounded number
of pair swaps), so that episodes, frames and face-bearing frames are each
within TOL of EVAL_FRAC and every scene_category's eval episode share is
within SCENE_TOL of EVAL_FRAC (or within one episode of the target).
Deterministic: SEED, sorted iteration everywhere.

Usage:
  faceight_split.py                 # search, print tables, write the lists
  faceight_split.py --write-csv     # ... and update data/episode_usage.csv
  faceight_split.py --dry-run       # search + checks only, write nothing
  faceight_split.py --check         # regenerate and compare with the lists
                                    # in --out-dir and the csv; write nothing
  --frames/--csv/--splits-dir/--out-dir override the inputs and the output
  dir (used by the sabotage runs).
Exit 1 on any failed check.
"""
import argparse
import collections
import csv
import hashlib
import random
import sys
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import open_index  # noqa: E402  (frames.csv is tracked gzipped, PII-1639)
FRAMES = STORE / "data" / "faceight" / "frames.csv"
CSV = STORE / "data" / "episode_usage.csv"
SPLITS = STORE / "data" / "splits"
DATASET_NAME = "faceight_a"
ROUND = "1"
EVAL_LIST = "faceight_a_eval_episodes_v1.txt"
TRAIN_LIST = "faceight_a_train_episodes_v1.txt"
# Frozen session lists in data/splits/ and the dataset they belong to.
FROZEN_LISTS = {
    "eval_sessions_v1.txt": ("face_mine_v1", "eval"),
    "train_sessions_v1.txt": ("face_mine_v1", "train"),
    "faceback_eval_sessions_v1.txt": ("faceback_45", "eval"),
    "faceback_train_sessions_v1.txt": ("faceback_45", "train"),
}

SEED = 19760703                  # the project's seed (faceight draw, round 1, rounds 2/3)
EVAL_FRAC = 0.20
TOL = 0.01                       # +-1 pt on episodes, frames, face-bearing frames
SCENE_TOL = 0.02                 # +-2 pt on each scene's eval episode share
RESTARTS = 20
MAX_PASSES = 30                  # hill-climb sweeps per restart
SWAP_TRIALS = 20000              # pair-swap trials per sweep (bounded: 19.7k units)
FACE_BANDS = (0, 2, 6, 15)       # total armW faces per episode: 0 / 1-2 / 3-6 / 7-15 / 16+


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- inputs

def episode_features(frames_path):
    """Per-episode frames / face-bearing frames / faces / twins / session from
    the round-1 rows; asserts episode <-> session is 1:1 and one view."""
    E = {}
    ep_sess = collections.defaultdict(set)
    sess_ep = collections.defaultdict(set)
    n_rows = 0
    with open_index(frames_path) as f:
        rd = csv.DictReader(f)
        for need in ("file", "episode_id", "session_id", "n_faces",
                     "annotation_round", "file_right", "view"):
            if need not in rd.fieldnames:
                fail(f"{frames_path}: missing column {need}")
        for r in rd:
            if r["annotation_round"] != ROUND:
                continue
            n_rows += 1
            e, s = r["episode_id"], r["session_id"]
            if not e or not s:
                fail(f"{frames_path}: {r['file']}: empty episode_id/session_id")
            if r["view"] != "vst_left":
                fail(f"{frames_path}: {r['file']}: round-{ROUND} row with view {r['view']!r}")
            ep_sess[e].add(s)
            sess_ep[s].add(e)
            d = E.setdefault(e, dict(session=s, frames=0, fb=0, faces=0, twins=0))
            n = int(r["n_faces"])
            if n < 0:
                fail(f"{frames_path}: {r['file']}: n_faces {n}")
            d["frames"] += 1
            d["fb"] += 1 if n >= 1 else 0
            d["faces"] += n
            d["twins"] += 1 if r["file_right"] else 0
    if not E:
        fail(f"{frames_path}: no rows at annotation_round {ROUND}")
    multi = {e: sorted(v) for e, v in ep_sess.items() if len(v) > 1}
    if multi:
        e, v = sorted(multi.items())[0]
        fail(f"episode {e} spans {len(v)} sessions {v} ({len(multi)} such episodes)")
    multi = {s: sorted(v) for s, v in sess_ep.items() if len(v) > 1}
    if multi:
        s, v = sorted(multi.items())[0]
        fail(f"session {s} holds {len(v)} episodes {v} ({len(multi)} such sessions)")
    return E, n_rows


def read_list(path):
    ids = [l.strip() for l in open(path) if l.strip()]
    if len(set(ids)) != len(ids):
        fail(f"{path}: duplicate ids")
    return set(ids)


def csv_rows(csv_path):
    """episode_id -> (session_id, dataset, role, scene_category); no quoting."""
    rows = {}
    with open(csv_path, newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        if '"' in "".join(header):
            fail(f"{csv_path}: quoted header, line-level edit unsafe")
        col = {c: i for i, c in enumerate(header)}
        for need in ("episode_id", "session_id", "dataset", "role", "scene_category"):
            if need not in col:
                fail(f"{csv_path}: missing column {need}")
        for line in f:
            if '"' in line:
                fail(f"{csv_path}: quoted field, line-level edit unsafe")
            r = line.rstrip("\n").split(",")
            e = r[col["episode_id"]]
            if not e:
                continue
            if e in rows:
                fail(f"{csv_path}: duplicate episode_id {e}")
            rows[e] = (r[col["session_id"]], r[col["dataset"]], r[col["role"]],
                       r[col["scene_category"]])
    return rows, col


def forced_sides(E, splits_dir, rows):
    """Episodes pinned by a frozen split, with two-way csv/list checks; every
    other episode must be unused in the csv and absent from every list."""
    by_session = {d["session"]: e for e, d in E.items()}
    lists = {}
    for p in sorted(splits_dir.glob("*.txt")):
        if p.name in (EVAL_LIST, TRAIN_LIST):
            continue
        lists[p.name] = read_list(p)
    for name, ids in lists.items():
        hit = (ids & set(by_session)) | (ids & set(E))
        if hit and name not in FROZEN_LISTS:
            fail(f"{splits_dir / name}: unknown list holds {len(hit)} faceight_a "
                 f"episodes/sessions, e.g. {sorted(hit)[:3]}")
    forced = {}
    pinned_by = {}
    for name, (ds_name, side) in sorted(FROZEN_LISTS.items()):
        if name not in lists:
            fail(f"{splits_dir / name}: frozen list missing")
        for s in sorted(lists[name] & set(by_session)):
            e = by_session[s]
            if e in forced and forced[e] != side:
                fail(f"{e} ({s}): {pinned_by[e]} says {forced[e]}, {name} says {side}")
            if e in forced and pinned_by[e].split(":")[0] == ds_name:
                fail(f"{e} ({s}): in both {ds_name} lists")
            forced[e] = side
            pinned_by[e] = f"{ds_name}:{name}"
    for e in sorted(E):
        if e not in rows:
            fail(f"{e} missing from episode_usage.csv")
        sess, ds, role, scene = rows[e]
        if sess != E[e]["session"]:
            fail(f"{e}: csv session {sess} != frames.csv session {E[e]['session']}")
        if not scene:
            fail(f"{e}: empty scene_category in episode_usage.csv")
        tokens = [t for t in ds.split("+") if t] if ds else []
        claimed = DATASET_NAME in tokens          # idempotent re-run after --write-csv
        frozen_tokens = [t for t in tokens if t != DATASET_NAME]
        known = {v[0] for v in FROZEN_LISTS.values()}
        for t in frozen_tokens:
            if t not in known:
                fail(f"{e}: csv dataset {ds!r}: already in {t}, cannot claim")
        if frozen_tokens and e not in forced:
            fail(f"{e}: csv says {ds}/{role} but it is in no frozen list")
        if not frozen_tokens and e in forced:
            fail(f"{e}: in {pinned_by[e]} but csv dataset is {ds!r}")
        if e in forced:
            if pinned_by[e].split(":")[0] not in frozen_tokens:
                fail(f"{e}: in {pinned_by[e]} but csv dataset is {ds!r}")
            if role != forced[e]:
                fail(f"{e}: csv role {role} != {pinned_by[e]} side {forced[e]}")
        elif not claimed and (ds or role != "unused"):
            fail(f"{e}: csv dataset {ds!r} role {role}: not unused, cannot claim")
    return forced


# ---------------------------------------------------------------- search

def face_band(faces):
    for i, hi in enumerate(FACE_BANDS):
        if faces <= hi:
            return i
    return len(FACE_BANDS)


def stratum(scene, d):
    return (scene, 0 if d["fb"] == 0 else 1, face_band(d["faces"]))


class Scorer:
    """Per-episode integer feature vectors so a move is a vector add/subtract:
    [episodes, frames, face-bearing frames, faces, episodes per scene..., frames per scene...]."""

    def __init__(self, E, scene_of):
        self.E = E
        self.scene_of = scene_of
        self.scenes = sorted(set(scene_of.values()))
        self.se = {c: 4 + i for i, c in enumerate(self.scenes)}
        self.sf = {c: 4 + len(self.scenes) + i for i, c in enumerate(self.scenes)}
        self.n = 4 + 2 * len(self.scenes)
        self.vec = {e: self._vec(e) for e in E}
        self.tot = self.sum_vec(E)

    def _vec(self, e):
        d = self.E[e]
        v = [0] * self.n
        v[0], v[1], v[2], v[3] = 1, d["frames"], d["fb"], d["faces"]
        v[self.se[self.scene_of[e]]] = 1
        v[self.sf[self.scene_of[e]]] = d["frames"]
        return v

    def sum_vec(self, keys):
        v = [0] * self.n
        for e in keys:
            for i, x in enumerate(self.vec[e]):
                v[i] += x
        return v

    def scene_dev(self, e):
        """Largest per-scene eval episode share deviation, zero when within one
        episode of the target (small categories)."""
        worst = 0.0
        for c in self.scenes:
            i = self.se[c]
            t = self.tot[i]
            if abs(e[i] - EVAL_FRAC * t) <= 1.0:
                continue
            worst = max(worst, abs(e[i] / t - EVAL_FRAC))
        return worst

    def score_vec(self, e):
        t = self.tot
        hard = max(abs(e[k] / t[k] - EVAL_FRAC) for k in (0, 1, 2))
        scene = self.scene_dev(e)
        tr = [t[i] - e[i] for i in range(self.n)]
        soft = abs(e[3] / t[3] - EVAL_FRAC)                      # face mass share
        soft += abs(e[2] / e[1] - tr[2] / tr[1])                 # face-bearing frame rate
        for c in self.scenes:                                    # per-scene frame shares
            i = self.sf[c]
            soft += abs(e[i] / e[1] - tr[i] / tr[1])
        return hard * 100 + scene * 50 + soft, hard, scene, soft

    def score(self, eval_set):
        return self.score_vec(self.sum_vec(eval_set))

    def table(self, eval_set):
        rows = []
        train_set = set(self.E) - eval_set
        for name, keys in (("ALL", set(self.E)), ("TRAIN", train_set), ("EVAL", eval_set)):
            a = self.sum_vec(keys)
            rows.append(dict(
                name=name, eps=a[0], frames=a[1], fb=a[2], faces=a[3],
                twins=sum(self.E[e]["twins"] for e in keys),
                eps_pct=a[0] / self.tot[0], frames_pct=a[1] / self.tot[1],
                fb_pct=a[2] / self.tot[2], faces_pct=a[3] / self.tot[3],
                fb_rate=a[2] / a[1], fpf=a[3] / a[1],
                scene_eps={c: a[self.se[c]] for c in self.scenes},
                scene_frames={c: a[self.sf[c]] for c in self.scenes}))
        return rows


def stratified_start(E, scene_of, free, forced, rng):
    """One seeded stratified draw over the free episodes (face_mine_split style)."""
    strata = collections.defaultdict(list)
    for e in sorted(E):
        strata[stratum(scene_of[e], E[e])].append(e)
    eval_set = {e for e, side in forced.items() if side == "eval"}
    for _, eps in sorted(strata.items()):
        target = EVAL_FRAC * sum(E[e]["frames"] for e in eps)
        acc = sum(E[e]["frames"] for e in eps if forced.get(e) == "eval")
        movable = [e for e in eps if e in free]
        rng.shuffle(movable)
        for e in movable:
            if acc >= target:
                break
            eval_set.add(e)
            acc += E[e]["frames"]
    return eval_set


def hill_climb(scorer, eval_set, free, rng):
    """Single moves then a bounded number of pair swaps over the free episodes,
    first-improvement in a shuffled order; stops when a sweep finds nothing."""
    V = scorer.vec
    free_sorted = sorted(free)
    e = scorer.sum_vec(eval_set)
    cur = scorer.score_vec(e)[0]
    evaluated = 0
    for _ in range(MAX_PASSES):
        improved = False
        order = free_sorted[:]
        rng.shuffle(order)
        for x in order:                              # single moves
            sign = -1 if x in eval_set else 1
            trial = [p + sign * q for p, q in zip(e, V[x])]
            sc = scorer.score_vec(trial)[0]
            evaluated += 1
            if sc < cur - 1e-12:
                e, cur, improved = trial, sc, True
                eval_set.symmetric_difference_update({x})
        ev = [x for x in order if x in eval_set]
        tr = [x for x in order if x not in eval_set]
        for _ in range(SWAP_TRIALS):                 # pair swaps, sampled
            a = ev[rng.randrange(len(ev))]
            b = tr[rng.randrange(len(tr))]
            if a not in eval_set or b in eval_set:
                continue
            trial = [p - q + r for p, q, r in zip(e, V[a], V[b])]
            sc = scorer.score_vec(trial)[0]
            evaluated += 1
            if sc < cur - 1e-12:
                e, cur, improved = trial, sc, True
                eval_set.remove(a)
                eval_set.add(b)
        if not improved:
            break
    return eval_set, cur, evaluated


def search(E, scene_of, forced):
    free = set(E) - set(forced)
    scorer = Scorer(E, scene_of)
    rng = random.Random(SEED)
    best = None
    n_eval = 0
    for r in range(RESTARTS):
        start = stratified_start(E, scene_of, free, forced, rng)
        cand, sc, ne = hill_climb(scorer, set(start), free, rng)
        n_eval += ne
        if best is None or sc < best[1] - 1e-12:
            best = (frozenset(cand), sc, r)
    eval_set, sc, r = best
    _, hard, scene, soft = scorer.score(eval_set)
    return set(eval_set), scorer, dict(score=sc, hard=hard, scene=scene, soft=soft,
                                       best_restart=r, restarts=RESTARTS, evaluated=n_eval)


# ---------------------------------------------------------------- checks / outputs

def verify(E, eval_set, train_set, forced, scorer, rows):
    if eval_set & train_set:
        fail("episode on both sides")
    if eval_set | train_set != set(E):
        fail("split does not cover the round-1 episodes")
    for e, side in forced.items():
        if (e in eval_set) != (side == "eval"):
            fail(f"{e} pinned to {side} by a frozen split but landed on the other side")
    sess_side = {}
    for e in E:
        side = "eval" if e in eval_set else "train"
        if sess_side.setdefault(E[e]["session"], side) != side:
            fail(f"session {E[e]['session']} crosses sides")
    _, hard, scene, _ = scorer.score(eval_set)
    if hard > TOL:
        fail(f"hard balance {hard:.4f} exceeds TOL {TOL}")
    if scene > SCENE_TOL:
        fail(f"scene balance {scene:.4f} exceeds SCENE_TOL {SCENE_TOL}")
    for e in sorted(E):                      # csv already carries this split?
        ds, role = rows[e][1], rows[e][2]
        if DATASET_NAME in ds.split("+") and role != ("eval" if e in eval_set else "train"):
            fail(f"{e}: csv role {role} disagrees with the regenerated split")


def check_lists(out_dir, eval_set, train_set):
    """The lists on disk must be exactly the regenerated split."""
    for name, want in ((EVAL_LIST, eval_set), (TRAIN_LIST, train_set)):
        p = out_dir / name
        if not p.exists():
            fail(f"{p}: missing")
        got = [l.rstrip("\n") for l in open(p)]
        if got != sorted(want):
            have = set(l for l in got if l)
            extra, missing = sorted(have - want), sorted(want - have)
            fail(f"{p}: differs from the regenerated split "
                 f"({len(extra)} extra e.g. {extra[:2]}, {len(missing)} missing e.g. {missing[:2]}, "
                 f"or order/blank-line difference)")
    print(f"check: {out_dir}/{{{EVAL_LIST},{TRAIN_LIST}}} match the regenerated split")


def check_csv(rows, E, eval_set):
    n = 0
    for e in sorted(E):
        ds, role = rows[e][1], rows[e][2]
        if DATASET_NAME not in ds.split("+"):
            fail(f"{e}: csv dataset {ds!r} does not carry {DATASET_NAME}")
        if role != ("eval" if e in eval_set else "train"):
            fail(f"{e}: csv role {role} disagrees with the regenerated split")
        n += 1
    print(f"check: csv carries {DATASET_NAME} on all {n} episodes with the regenerated roles")


def fmt_table(rows):
    out = ["| Slice | Episodes | Frames | Face-bearing frames | armW faces | "
           "Face-bearing rate | Faces/frame | Right twins |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        pct = (lambda v: "" if r["name"] == "ALL" else f" ({v:.1%})")
        out.append(
            f"| {r['name']} | {r['eps']:,}{pct(r['eps_pct'])} | "
            f"{r['frames']:,}{pct(r['frames_pct'])} | {r['fb']:,}{pct(r['fb_pct'])} | "
            f"{r['faces']:,}{pct(r['faces_pct'])} | {r['fb_rate']:.1%} | {r['fpf']:.3f} | "
            f"{r['twins']:,} |")
    return "\n".join(out)


def fmt_scene_table(rows):
    allr, tr, ev = rows
    out = ["| Scene | Episodes | Train ep | Eval ep | Eval ep share | Frames | Eval frames | Eval frame share |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(allr["scene_eps"], key=lambda c: -allr["scene_eps"][c]):
        n, ne, nf, nef = (allr["scene_eps"][c], ev["scene_eps"][c],
                          allr["scene_frames"][c], ev["scene_frames"][c])
        out.append(f"| {c} | {n:,} | {tr['scene_eps'][c]:,} | {ne:,} | {ne / n:.1%} | "
                   f"{nf:,} | {nef:,} | {nef / nf:.1%} |")
    return "\n".join(out)


def write_lists(out_dir, eval_set, train_set):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in ((EVAL_LIST, eval_set), (TRAIN_LIST, train_set)):
        p = out_dir / name
        body = "\n".join(sorted(ids)) + "\n"
        if p.exists() and p.read_text() != body:
            fail(f"{p}: exists and differs from the regenerated split; the split is frozen")
        p.write_text(body)
        print(f"wrote {p} ({len(ids)} episodes, md5 {hashlib.md5(body.encode()).hexdigest()})")


def update_csv(csv_path, col, eval_set, train_set):
    """Rewrite only the faceight_a rows: dataset += faceight_a, role = side.
    Every other byte is preserved (no quoting in this file; checked on read)."""
    side = {e: "eval" for e in eval_set}
    side.update({e: "train" for e in train_set})
    raw = csv_path.read_bytes()
    if b"\r" in raw:
        fail(f"{csv_path}: CR found, line-level edit unsafe")
    lines = raw.decode().split("\n")
    changed = 0
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        r = line.split(",")
        e = r[col["episode_id"]]
        if e not in side:
            continue
        ds = r[col["dataset"]]
        if DATASET_NAME in ds.split("+"):
            fail(f"{e}: dataset already contains {DATASET_NAME}; refusing to re-append")
        r[col["dataset"]] = f"{ds}+{DATASET_NAME}" if ds else DATASET_NAME
        r[col["role"]] = side[e]
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
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS,
                    help="dir of the frozen session lists that pin episodes")
    ap.add_argument("--out-dir", type=Path, default=SPLITS)
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="compare the lists in --out-dir and the csv with the "
                         "regenerated split; write nothing")
    a = ap.parse_args()

    E, n_frames = episode_features(a.frames)
    rows, col = csv_rows(a.csv)
    forced = forced_sides(E, a.splits_dir, rows)
    scene_of = {e: rows[e][3] for e in E}
    n_fe = sum(1 for v in forced.values() if v == "eval")
    print(f"frames.csv round {ROUND}: {n_frames:,} frames / {len(E):,} episodes = "
          f"{len({d['session'] for d in E.values()}):,} sessions; "
          f"pinned by frozen splits: {n_fe} eval, {len(forced) - n_fe} train; "
          f"free: {len(E) - len(forced):,}")

    eval_set, scorer, info = search(E, scene_of, forced)
    train_set = set(E) - eval_set
    verify(E, eval_set, train_set, forced, scorer, rows)
    print(f"seed {SEED}: {info['restarts']} restarts, best from restart "
          f"{info['best_restart']}, {info['evaluated']:,} candidates scored, "
          f"hard max-dev {info['hard']:.4f} (TOL {TOL}), scene max-dev "
          f"{info['scene']:.4f} (SCENE_TOL {SCENE_TOL}), soft {info['soft']:.4f}")
    rows_t = scorer.table(eval_set)
    print(fmt_table(rows_t))
    print(fmt_scene_table(rows_t))
    if a.check:
        check_lists(a.out_dir, eval_set, train_set)
        check_csv(rows, E, eval_set)
        return
    if a.dry_run:
        print("dry run: nothing written")
        return
    write_lists(a.out_dir, eval_set, train_set)
    if a.write_csv:
        update_csv(a.csv, col, eval_set, train_set)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the frozen faceight master train / eval episode split (WOR-1212).

Sibling of faceight_split.py, one level up: instead of splitting a single
annotation round, it splits the whole faceight draw once. Every episode of
data/faceight/episodes.txt (34,667, faceight == 1 in episode_usage.csv) gets a
side, frozen forever; any annotation round is a projection of that split, so a
round's train / eval lists are written by selecting its episodes, never drawn.
Both eyes follow the episode.

Pinned, and reproduced rather than redrawn: the 19,783 faceight_a episodes keep
the side of data/splits/faceight_a_{train,eval}_episodes_v1.txt (models are
already trained on that train part), and the 42 drawn episodes whose session is
in faceback_45 keep the faceback side. The projection of the master split onto
round 1 must equal the faceight_a lists byte for byte. Every other drawn episode
must be dataset '' / role unused in episode_usage.csv and absent from every list
in data/splits/.

The free episodes are assigned by a seeded stratified draw (scene_category x
face-free bin x face-count band, over the episodes that have frames) followed by
the faceight_split.py hill-climb (single moves, then a bounded number of sampled
pair swaps), so that over all 34,667 episodes the eval share is within TOL of
EVAL_FRAC on episodes, on frames (rounds 0 to 3 of frames.csv) and on
face-bearing frames, and every scene_category's eval episode share is within
SCENE_TOL (or within one episode of the target). Face-bearing means an armW face
on round 0 (n_faces in frames.csv) and a human box on rounds 1 to 3 (the
training/manifests/faceight_{a,b,c}_labelv2.txt manifests).

Only the overall split is optimised; the per-round shares are whatever the
projection gives and are reported, not constrained.

Usage:
  faceight_master_split.py                 # search, print tables, write the lists
  faceight_master_split.py --write-csv     # ... and update data/episode_usage.csv
  faceight_master_split.py --dry-run       # search + checks only, write nothing
  faceight_master_split.py --check         # regenerate and compare with the lists
                                           # in --out-dir and the csv; write nothing
  --episodes/--frames/--csv/--manifest-dir/--splits-dir/--out-dir override the
  inputs and the output dir (used by the sabotage runs).
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
from pii_root import CODE_ROOT, open_index  # noqa: E402  (CODE_ROOT: the manifests)

EPISODES = STORE / "data" / "faceight" / "episodes.txt"
FRAMES = STORE / "data" / "faceight" / "frames.csv"
CSV = STORE / "data" / "episode_usage.csv"
SPLITS = STORE / "data" / "splits"
MANIFESTS = Path(CODE_ROOT) / "training" / "manifests"

EVAL_LIST = "faceight_eval_episodes_v1.txt"
TRAIN_LIST = "faceight_train_episodes_v1.txt"
# Labelled rounds: annotation_round -> (dataset token, manifest, list stem).
ROUNDS = {
    "1": ("faceight_a", "faceight_a_labelv2.txt", "faceight_a"),
    "2": ("faceight_b", "faceight_b_labelv2.txt", "faceight_b"),
    "3": ("faceight_c", "faceight_c_labelv2.txt", "faceight_c"),
}
ALL_ROUNDS = ("0", "1", "2", "3")
PIN_ROUND = "1"                  # round 1 is pinned: its lists already exist
FREE_TOKEN = "faceight"          # dataset value of a free episode with no labelled round
# Frozen session lists in data/splits/ and the dataset they belong to.
FROZEN_SESSION_LISTS = {
    "eval_sessions_v1.txt": ("face_mine_v1", "eval"),
    "train_sessions_v1.txt": ("face_mine_v1", "train"),
    "faceback_eval_sessions_v1.txt": ("faceback_45", "eval"),
    "faceback_train_sessions_v1.txt": ("faceback_45", "train"),
}
# Frozen episode lists: the round-1 split, which this split must reproduce.
FROZEN_EPISODE_LISTS = {
    "faceight_a_eval_episodes_v1.txt": ("faceight_a", "eval"),
    "faceight_a_train_episodes_v1.txt": ("faceight_a", "train"),
}
PIN_TOKENS = {"face_mine_v1", "faceback_45", "faceight_a"}
OUR_TOKENS = {FREE_TOKEN, "faceight_b", "faceight_c"}

SEED = 19760703                  # the project's seed (faceight draw, round 1, rounds 2/3)
EVAL_FRAC = 0.20
TOL = 0.01                       # +-1 pt on episodes, frames, face-bearing frames
SCENE_TOL = 0.02                 # +-2 pt on each scene's eval episode share
RESTARTS = 20
MAX_PASSES = 30                  # hill-climb sweeps per restart
SWAP_TRIALS = 20000              # pair-swap trials per sweep
FACE_BANDS = (0, 2, 6, 15)       # faces per episode: 0 / 1-2 / 3-6 / 7-15 / 16+
EXPECT_SHARED_23 = 15069         # episodes annotated in both round 2 and round 3


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- inputs

def read_list(path):
    ids = [l.strip() for l in open(path) if l.strip()]
    if len(set(ids)) != len(ids):
        fail(f"{path}: duplicate ids")
    return set(ids)


def read_manifests(manifest_dir):
    """round -> {frame basename: human box count} from the labelv2 manifests."""
    out = {}
    for rnd, (_, fname, _) in sorted(ROUNDS.items()):
        p = manifest_dir / fname
        if not p.exists():
            fail(f"{p}: manifest missing")
        boxes, cur = {}, None
        for line in open(p):
            if line.startswith("#"):
                parts = line.split()
                if len(parts) < 2:
                    fail(f"{p}: malformed header line {line.rstrip()!r}")
                cur = parts[1].split("/")[-1]
                if cur in boxes:
                    fail(f"{p}: duplicate frame {cur}")
                boxes[cur] = 0
            elif line.strip():
                if cur is None:
                    fail(f"{p}: box line before any frame header")
                boxes[cur] += 1
        out[rnd] = boxes
    return out


def episode_features(episodes_path, frames_path, manifest_dir, rows):
    """Per-episode frames / face-bearing frames / faces per round, over the whole
    drawn set; asserts episode <-> session is 1:1 and that frames.csv and the
    manifests stay inside the draw."""
    drawn = [l.strip() for l in open(episodes_path) if l.strip()]
    if len(set(drawn)) != len(drawn):
        fail(f"{episodes_path}: duplicate episode ids")
    drawn = set(drawn)
    flagged = {e for e, r in rows.items() if r[4] == "1"}
    if drawn != flagged:
        fail(f"{episodes_path} ({len(drawn)}) != faceight == 1 in the csv ({len(flagged)}): "
             f"{len(drawn - flagged)} only in the list, {len(flagged - drawn)} only in the csv")
    manifests = read_manifests(manifest_dir)

    E = {}
    for e in sorted(drawn):
        if e not in rows:
            fail(f"{e} missing from episode_usage.csv")
        if not rows[e][3]:
            fail(f"{e}: empty scene_category in episode_usage.csv")
        E[e] = dict(session=rows[e][0], scene=rows[e][3], frames=0, fb=0, faces=0,
                    per={r: [0, 0, 0] for r in ALL_ROUNDS})
    ep_sess = collections.defaultdict(set)
    sess_ep = collections.defaultdict(set)
    seen = {r: set() for r in ROUNDS}
    with open_index(frames_path) as f:
        rd = csv.DictReader(f)
        for need in ("file", "episode_id", "session_id", "n_faces", "annotation_round", "view"):
            if need not in rd.fieldnames:
                fail(f"{frames_path}: missing column {need}")
        for r in rd:
            e, s, rnd = r["episode_id"], r["session_id"], r["annotation_round"]
            if not e or not s:
                fail(f"{frames_path}: {r['file']}: empty episode_id/session_id")
            if e not in E:
                fail(f"{frames_path}: {r['file']}: episode {e} is not in {episodes_path}")
            if rnd not in ALL_ROUNDS:
                fail(f"{frames_path}: {r['file']}: unknown annotation_round {rnd!r}")
            if r["view"] not in ("vst_left", "vst_right"):
                fail(f"{frames_path}: {r['file']}: unknown view {r['view']!r}")
            ep_sess[e].add(s)
            sess_ep[s].add(e)
            if rnd == "0":
                n = int(r["n_faces"])
                if n < 0:
                    fail(f"{frames_path}: {r['file']}: n_faces {n}")
            else:
                key = r["file"]
                if key not in manifests[rnd]:
                    fail(f"{frames_path}: {key}: round-{rnd} frame absent from "
                         f"{manifest_dir / ROUNDS[rnd][1]}")
                seen[rnd].add(key)
                n = manifests[rnd][key]
            d = E[e]
            d["frames"] += 1
            d["fb"] += 1 if n >= 1 else 0
            d["faces"] += n
            p = d["per"][rnd]
            p[0] += 1
            p[1] += 1 if n >= 1 else 0
            p[2] += n
    for rnd in sorted(ROUNDS):
        extra = set(manifests[rnd]) - seen[rnd]
        if extra:
            fail(f"{manifest_dir / ROUNDS[rnd][1]}: {len(extra)} frames with no round-{rnd} row "
                 f"in {frames_path}, e.g. {sorted(extra)[:2]}")
    multi = {e: sorted(v) for e, v in ep_sess.items() if len(v) > 1}
    if multi:
        e, v = sorted(multi.items())[0]
        fail(f"episode {e} spans {len(v)} sessions {v} ({len(multi)} such episodes)")
    multi = {s: sorted(v) for s, v in sess_ep.items() if len(v) > 1}
    if multi:
        s, v = sorted(multi.items())[0]
        fail(f"session {s} holds {len(v)} episodes {v} ({len(multi)} such sessions)")
    for e, d in E.items():
        got = ep_sess.get(e)
        if got and d["session"] not in got:
            fail(f"{e}: csv session {d['session']} != frames.csv session {sorted(got)}")
    sessions = {d["session"] for d in E.values()}
    if len(sessions) != len(E):
        fail(f"{len(E)} drawn episodes but {len(sessions)} distinct sessions in the csv")
    return E


def csv_rows(csv_path):
    """episode_id -> (session_id, dataset, role, scene_category, faceight); no quoting."""
    rows = {}
    with open(csv_path, newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        if '"' in "".join(header):
            fail(f"{csv_path}: quoted header, line-level edit unsafe")
        col = {c: i for i, c in enumerate(header)}
        for need in ("episode_id", "session_id", "dataset", "role", "scene_category", "faceight"):
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
                       r[col["scene_category"]], r[col["faceight"]])
    return rows, col


def own_lists():
    names = {EVAL_LIST, TRAIN_LIST}
    for rnd, (_, _, stem) in ROUNDS.items():
        if rnd != PIN_ROUND:
            names |= {f"{stem}_eval_episodes_v1.txt", f"{stem}_train_episodes_v1.txt"}
    return names


def forced_sides(E, splits_dir, rows):
    """Episodes pinned by a frozen split, with two-way csv/list checks; every
    other drawn episode must be unused in the csv and absent from every list."""
    by_session = {d["session"]: e for e, d in E.items()}
    mine = own_lists()
    lists = {}
    for p in sorted(splits_dir.glob("*.txt")):
        if p.name in mine:
            continue
        lists[p.name] = read_list(p)
    known = dict(FROZEN_SESSION_LISTS)
    known.update(FROZEN_EPISODE_LISTS)
    for name, ids in lists.items():
        hit = (ids & set(by_session)) | (ids & set(E))
        if hit and name not in known:
            fail(f"{splits_dir / name}: unknown list holds {len(hit)} drawn faceight "
                 f"episodes/sessions, e.g. {sorted(hit)[:3]}")
    forced = {}
    pinned_by = collections.defaultdict(list)
    for name, (ds_name, side) in sorted(known.items()):
        if name not in lists:
            fail(f"{splits_dir / name}: frozen list missing")
        by_episode = name in FROZEN_EPISODE_LISTS
        keys = lists[name] & set(E if by_episode else by_session)
        for k in sorted(keys):
            e = k if by_episode else by_session[k]
            if e in forced and forced[e] != side:
                fail(f"{e}: {pinned_by[e][-1][1]} says {forced[e]}, {name} says {side}")
            if any(ds == ds_name for ds, _ in pinned_by[e]):
                fail(f"{e}: in both {ds_name} lists")
            forced[e] = side
            pinned_by[e].append((ds_name, name))
    for e in sorted(E):
        sess, ds, role, _, _ = rows[e]
        if sess != E[e]["session"]:
            fail(f"{e}: csv session {sess} != feature session {E[e]['session']}")
        tokens = [t for t in ds.split("+") if t] if ds else []
        for t in tokens:
            if t not in PIN_TOKENS | OUR_TOKENS:
                fail(f"{e}: csv dataset {ds!r}: already in {t}, cannot claim")
        claimed = bool(set(tokens) & OUR_TOKENS)   # idempotent re-run after --write-csv
        pin_tokens = [t for t in tokens if t in PIN_TOKENS]
        if pin_tokens and e not in forced:
            fail(f"{e}: csv says {ds}/{role} but it is in no frozen list")
        if not pin_tokens and e in forced:
            fail(f"{e}: in {pinned_by[e][0][1]} but csv dataset is {ds!r}")
        if e in forced:
            for ds_name, name in pinned_by[e]:
                if ds_name not in pin_tokens:
                    fail(f"{e}: in {name} but csv dataset is {ds!r}")
            for t in pin_tokens:
                if t not in {d for d, _ in pinned_by[e]}:
                    fail(f"{e}: csv dataset {ds!r} carries {t} but no {t} list holds it")
            if role != forced[e]:
                fail(f"{e}: csv role {role} != {pinned_by[e][0][0]} side {forced[e]}")
        elif not claimed and (ds or role != "unused"):
            fail(f"{e}: csv dataset {ds!r} role {role}: not unused, cannot claim")
        elif claimed and role not in ("train", "eval"):
            fail(f"{e}: csv dataset {ds!r} but role {role!r}")
    return forced


# ---------------------------------------------------------------- search

def face_band(faces):
    for i, hi in enumerate(FACE_BANDS):
        if faces <= hi:
            return i
    return len(FACE_BANDS)


def stratum(d):
    """Scene x face-free x face band; episodes with no frame get their own band."""
    if d["frames"] == 0:
        return (d["scene"], -1, -1)
    return (d["scene"], 0 if d["fb"] == 0 else 1, face_band(d["faces"]))


class Scorer:
    """Per-episode integer feature vectors so a move is a vector add/subtract:
    [episodes, frames, face-bearing frames, faces, episodes per scene..., frames per scene...]."""

    def __init__(self, E):
        self.E = E
        self.scenes = sorted({d["scene"] for d in E.values()})
        self.se = {c: 4 + i for i, c in enumerate(self.scenes)}
        self.sf = {c: 4 + len(self.scenes) + i for i, c in enumerate(self.scenes)}
        self.n = 4 + 2 * len(self.scenes)
        self.vec = {e: self._vec(e) for e in E}
        self.tot = self.sum_vec(E)

    def _vec(self, e):
        d = self.E[e]
        v = [0] * self.n
        v[0], v[1], v[2], v[3] = 1, d["frames"], d["fb"], d["faces"]
        v[self.se[d["scene"]]] = 1
        v[self.sf[d["scene"]]] = d["frames"]
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


def stratified_start(E, free, forced, rng):
    """One seeded stratified draw over the free episodes, on the stratum's frame
    mass where it has frames and on episode count where it does not."""
    strata = collections.defaultdict(list)
    for e in sorted(E):
        strata[stratum(E[e])].append(e)
    eval_set = {e for e, side in forced.items() if side == "eval"}
    for key, eps in sorted(strata.items()):
        w = (lambda e: E[e]["frames"]) if any(E[e]["frames"] for e in eps) else (lambda e: 1)
        target = EVAL_FRAC * sum(w(e) for e in eps)
        acc = sum(w(e) for e in eps if forced.get(e) == "eval")
        movable = [e for e in eps if e in free]
        rng.shuffle(movable)
        for e in movable:
            if acc >= target:
                break
            eval_set.add(e)
            acc += w(e)
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


def search(E, forced):
    free = set(E) - set(forced)
    scorer = Scorer(E)
    rng = random.Random(SEED)
    best = None
    n_eval = 0
    for r in range(RESTARTS):
        start = stratified_start(E, free, forced, rng)
        cand, sc, ne = hill_climb(scorer, set(start), free, rng)
        n_eval += ne
        if best is None or sc < best[1] - 1e-12:
            best = (frozenset(cand), sc, r)
    eval_set, sc, r = best
    _, hard, scene, soft = scorer.score(eval_set)
    return set(eval_set), scorer, dict(score=sc, hard=hard, scene=scene, soft=soft,
                                       best_restart=r, restarts=RESTARTS, evaluated=n_eval)


# ---------------------------------------------------------------- projections

def round_episodes(E, rnd):
    return {e for e, d in E.items() if d["per"][rnd][0]}


def list_body(ids):
    return "\n".join(sorted(ids)) + "\n"


def projections(E, eval_set):
    """stem -> (train ids, eval ids) for every labelled round."""
    out = {}
    for rnd, (_, _, stem) in sorted(ROUNDS.items()):
        eps = round_episodes(E, rnd)
        out[stem] = (eps - eval_set, eps & eval_set)
    return out


# ---------------------------------------------------------------- checks / outputs

def verify(E, eval_set, train_set, forced, scorer, rows, splits_dir):
    if eval_set & train_set:
        fail("episode on both sides")
    if eval_set | train_set != set(E):
        fail("split does not cover the drawn episodes")
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
    proj = projections(E, eval_set)
    for name, (ds_name, side) in sorted(FROZEN_EPISODE_LISTS.items()):
        stem = ROUNDS[PIN_ROUND][2]
        if ds_name != stem:
            fail(f"{name}: {ds_name} is not the pinned round {PIN_ROUND} dataset {stem}")
        want = proj[stem][0 if side == "train" else 1]
        got = (splits_dir / name).read_bytes()
        if got != list_body(want).encode():
            have = read_list(splits_dir / name)
            fail(f"{splits_dir / name}: the round-{PIN_ROUND} projection does not reproduce it "
                 f"({len(want - have)} missing, {len(have - want)} extra, or an order/byte difference)")
    b, c = round_episodes(E, "2"), round_episodes(E, "3")
    shared = b & c
    if len(shared) != EXPECT_SHARED_23:
        fail(f"rounds 2 and 3 share {len(shared)} episodes, expected {EXPECT_SHARED_23}")
    for e in shared:                             # trivially true by construction; asserted anyway
        if (e in proj["faceight_b"][1]) != (e in proj["faceight_c"][1]):
            fail(f"{e}: on different sides in the round-2 and round-3 projections")
    for e in sorted(E):                          # csv already carries this split?
        ds, role = rows[e][1], rows[e][2]
        tokens = set(ds.split("+"))
        if (tokens & OUR_TOKENS or ROUNDS[PIN_ROUND][0] in tokens) and \
                role != ("eval" if e in eval_set else "train"):
            fail(f"{e}: csv role {role} disagrees with the regenerated split")
    return proj


def wanted_csv(E, eval_set, proj):
    """episode -> (dataset, role) this split wants in episode_usage.csv."""
    want = {}
    for e in sorted(E):
        tokens = []
        for rnd, (tok, _, stem) in sorted(ROUNDS.items()):
            if e in proj[stem][0] or e in proj[stem][1]:
                tokens.append(tok)
        want[e] = (tokens, "eval" if e in eval_set else "train")
    return want


def writable(tokens):
    """The tokens this script appends: the pinned round's token is already in the
    csv (forced_sides checks it against the frozen lists) and is never re-written."""
    return [t for t in tokens if t not in PIN_TOKENS]


def csv_dataset(old, tokens):
    """Existing tokens keep their place and order; ours are appended; a row that
    would end up empty gets the plain faceight token."""
    out = [t for t in old.split("+") if t]
    for t in writable(tokens):
        if t not in out:
            out.append(t)
    if not out:
        out = [FREE_TOKEN]
    return "+".join(out)


def check_lists(out_dir, eval_set, train_set, proj):
    """The lists on disk must be exactly the regenerated split and projections."""
    targets = [(EVAL_LIST, eval_set), (TRAIN_LIST, train_set)]
    for rnd, (_, _, stem) in sorted(ROUNDS.items()):
        if rnd == PIN_ROUND:
            continue
        targets += [(f"{stem}_train_episodes_v1.txt", proj[stem][0]),
                    (f"{stem}_eval_episodes_v1.txt", proj[stem][1])]
    for name, want in targets:
        p = out_dir / name
        if not p.exists():
            fail(f"{p}: missing")
        if p.read_bytes() != list_body(want).encode():
            have = read_list(p)
            extra, missing = sorted(have - want), sorted(want - have)
            fail(f"{p}: differs from the regenerated split "
                 f"({len(extra)} extra e.g. {extra[:2]}, {len(missing)} missing e.g. {missing[:2]}, "
                 f"or order/blank-line difference)")
    print(f"check: {len(targets)} lists in {out_dir} match the regenerated split")


def check_csv(rows, E, want):
    n = 0
    for e in sorted(E):
        ds, role = rows[e][1], rows[e][2]
        tokens, side = want[e]
        have = [t for t in ds.split("+") if t]
        missing = [t for t in tokens if t not in have]
        if missing:
            fail(f"{e}: csv dataset {ds!r} does not carry {'+'.join(missing)}")
        if not tokens and FREE_TOKEN not in have:
            fail(f"{e}: csv dataset {ds!r} does not carry {FREE_TOKEN}")
        if role != side:
            fail(f"{e}: csv role {role} disagrees with the regenerated split ({side})")
        n += 1
    print(f"check: csv carries the split on all {n:,} drawn episodes")


# ---------------------------------------------------------------- report

def slice_stats(E, keys, rounds):
    eps = frames = fb = faces = 0
    for e in keys:
        per = E[e]["per"]
        f = sum(per[r][0] for r in rounds)
        if rounds == ALL_ROUNDS or f:
            eps += 1
        frames += f
        fb += sum(per[r][1] for r in rounds)
        faces += sum(per[r][2] for r in rounds)
    return dict(eps=eps, frames=frames, fb=fb, faces=faces)


def fmt_round_table(E, eval_set, train_set):
    slices = [("master (rounds 0 to 3)", ALL_ROUNDS),
              ("round 0 pool (armW)", ("0",)),
              ("faceight_a (round 1)", ("1",)),
              ("faceight_b (round 2)", ("2",)),
              ("faceight_c (round 3)", ("3",)),
              ("faceight_b + faceight_c", ("2", "3"))]
    out = ["| Slice | Episodes train / eval | Eval ep | Frames train / eval | Eval fr | "
           "Face-bearing train / eval | Eval fb | Faces train / eval | Eval faces |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rounds in slices:
        a = slice_stats(E, set(E), rounds)
        t = slice_stats(E, train_set, rounds)
        v = slice_stats(E, eval_set, rounds)
        cells = []
        for k in ("eps", "frames", "fb", "faces"):
            cells.append(f"{t[k]:,} / {v[k]:,}")
            cells.append(f"{v[k] / a[k]:.1%}" if a[k] else "n/a")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def fmt_scene_table(E, eval_set):
    scenes = sorted({d["scene"] for d in E.values()})
    cols = [("master", ALL_ROUNDS), ("faceight_a", ("1",)), ("faceight_b", ("2",)),
            ("faceight_c", ("3",)), ("b + c", ("2", "3"))]
    out = ["| Scene | Episodes | Train ep | Eval ep | Eval ep share | Eval frame share | "
           + " | ".join(f"Eval ep share {n}" for n, _ in cols[1:]) + " |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = sorted(scenes, key=lambda c: -sum(1 for d in E.values() if d["scene"] == c))
    for c in order:
        keys = {e for e, d in E.items() if d["scene"] == c}
        a = slice_stats(E, keys, ALL_ROUNDS)
        v = slice_stats(E, keys & eval_set, ALL_ROUNDS)
        row = [c, f"{a['eps']:,}", f"{a['eps'] - v['eps']:,}", f"{v['eps']:,}",
               f"{v['eps'] / a['eps']:.1%}",
               f"{v['frames'] / a['frames']:.1%}" if a["frames"] else "n/a"]
        for _, rounds in cols[1:]:
            ra = slice_stats(E, keys, rounds)
            rv = slice_stats(E, keys & eval_set, rounds)
            row.append(f"{rv['eps'] / ra['eps']:.1%}" if ra["eps"] else "n/a")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- writing

def write_lists(out_dir, eval_set, train_set, proj):
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [(TRAIN_LIST, train_set), (EVAL_LIST, eval_set)]
    for rnd, (_, _, stem) in sorted(ROUNDS.items()):
        if rnd == PIN_ROUND:
            continue
        targets += [(f"{stem}_train_episodes_v1.txt", proj[stem][0]),
                    (f"{stem}_eval_episodes_v1.txt", proj[stem][1])]
    for name, ids in targets:
        p = out_dir / name
        body = list_body(ids)
        if p.exists() and p.read_text() != body:
            fail(f"{p}: exists and differs from the regenerated split; the split is frozen")
        p.write_text(body)
        print(f"wrote {p} ({len(ids):,} episodes, md5 {hashlib.md5(body.encode()).hexdigest()})")


def update_csv(csv_path, col, want, rows):
    """Rewrite only the drawn faceight rows: dataset gains our tokens, role is
    the side. Every other byte is preserved (no quoting in this file)."""
    expect = sum(1 for e in want
                 if (csv_dataset(rows[e][1], want[e][0]), want[e][1]) != (rows[e][1], rows[e][2]))
    raw = csv_path.read_bytes()
    if b"\r" in raw:
        fail(f"{csv_path}: CR found, line-level edit unsafe")
    lines = raw.decode().split("\n")
    changed = seen = 0
    counts = collections.Counter()
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        r = line.split(",")
        e = r[col["episode_id"]]
        if e not in want:
            continue
        seen += 1
        tokens, side = want[e]
        ds = r[col["dataset"]]
        have = [t for t in ds.split("+") if t]
        new_tokens = writable(tokens)
        for t in new_tokens:
            if t in have:
                fail(f"{e}: dataset already contains {t}; refusing to re-append")
        if not new_tokens and FREE_TOKEN in have:
            fail(f"{e}: dataset already contains {FREE_TOKEN}; refusing to re-append")
        r[col["dataset"]] = csv_dataset(ds, tokens)
        r[col["role"]] = side
        new = ",".join(r)
        if new != line:
            changed += 1
            counts[(r[col["dataset"]], side)] += 1
        lines[i] = new
    if seen != len(want):
        fail(f"csv: found {seen} of the {len(want)} drawn rows")
    if changed != expect:
        fail(f"csv: changed {changed} rows, expected {expect}")
    csv_path.write_bytes("\n".join(lines).encode())
    print(f"csv: {changed:,} of {len(want):,} drawn rows changed, "
          f"md5 {hashlib.md5(csv_path.read_bytes()).hexdigest()}")
    for (ds, role), n in sorted(counts.items()):
        print(f"  {ds}/{role}: {n:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=Path, default=EPISODES)
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--manifest-dir", type=Path, default=MANIFESTS)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS,
                    help="dir of the frozen lists that pin episodes")
    ap.add_argument("--out-dir", type=Path, default=SPLITS)
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="compare the lists in --out-dir and the csv with the "
                         "regenerated split; write nothing")
    a = ap.parse_args()

    rows, col = csv_rows(a.csv)
    E = episode_features(a.episodes, a.frames, a.manifest_dir, rows)
    forced = forced_sides(E, a.splits_dir, rows)
    n_fe = sum(1 for v in forced.values() if v == "eval")
    n_frames = sum(d["frames"] for d in E.values())
    n_nof = sum(1 for d in E.values() if d["frames"] == 0)
    print(f"faceight draw: {len(E):,} episodes = {len({d['session'] for d in E.values()}):,} "
          f"sessions, {n_frames:,} frames at rounds 0 to 3, {n_nof:,} episodes with no frame "
          f"(chunk_sel -1); pinned by frozen splits: {n_fe:,} eval, {len(forced) - n_fe:,} train; "
          f"free: {len(E) - len(forced):,}")

    eval_set, scorer, info = search(E, forced)
    train_set = set(E) - eval_set
    proj = verify(E, eval_set, train_set, forced, scorer, rows, a.splits_dir)
    want = wanted_csv(E, eval_set, proj)
    print(f"seed {SEED}: {info['restarts']} restarts, best from restart "
          f"{info['best_restart']}, {info['evaluated']:,} candidates scored, "
          f"hard max-dev {info['hard']:.4f} (TOL {TOL}), scene max-dev "
          f"{info['scene']:.4f} (SCENE_TOL {SCENE_TOL}), soft {info['soft']:.4f}")
    print(fmt_round_table(E, eval_set, train_set))
    print(fmt_scene_table(E, eval_set))
    if a.check:
        check_lists(a.out_dir, eval_set, train_set, proj)
        check_csv(rows, E, want)
        return
    if a.dry_run:
        print("dry run: nothing written")
        return
    write_lists(a.out_dir, eval_set, train_set, proj)
    if a.write_csv:
        update_csv(a.csv, col, want, rows)


if __name__ == "__main__":
    main()

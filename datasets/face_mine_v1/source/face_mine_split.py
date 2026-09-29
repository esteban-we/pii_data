#!/usr/bin/env python3
"""Generate the frozen face_mine_v1 train/eval session split (WOR-38).

Session-level 80/20 split: frames within a session are temporal near-duplicates,
so any frame-level split would leak train into eval. Sessions are stratified by
capture month, machine-human disagreement rate, and small-face share so the eval
slice mirrors the hard cases; see the table in
<PII_ROOT>/datasets/face_mine_v1/README.md.

Inputs: the human boxes.json (WOR-37 output) and the backed-up machine dump.
Output: splits/{train,eval}_sessions_v1.txt beside the dataset, plus repo copies
under data/splits/. Deterministic: seed 20260902, sorted iteration everywhere.

PII-1449: the dataset moved to the PII-1315 store, where the frozen split is a
single split.csv (session,role) and the per-dataset boxes are boxes/vN/boxes.csv.
This script still emits the older splits/{train,eval}_sessions_v1.txt pair; the
store's split.csv is what readers use.
"""
import collections
import json
import random
import sys
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import dataset as pii_dataset, pages_media  # noqa: E402

MEDIA = pages_media("face-mine")
DATASET = pii_dataset("face_mine_v1")
SEED = 20260902
EVAL_FRAC = 0.20
SMALL_PX = 64          # box height in the raw 2328x1748 frame

def session_features():
    human = json.load(open(MEDIA / "boxes.json"))
    machine = json.load(open(MEDIA / "boxes_machine.json"))
    mach_n = {i["file"]: len(i.get("boxes", [])) for i in machine["images"]}
    S = collections.defaultdict(
        lambda: dict(imgs=0, boxes=0, new=0, kept=0, mach=0, small=0))
    for img in human["images"]:
        d = S[img["file"].split("_c")[0]]
        d["imgs"] += 1
        d["mach"] += mach_n.get(img["file"], 0)
        for b, m in zip(img["boxes"], img.get("boxes_meta", [])):
            d["boxes"] += 1
            if b[3] - b[1] < SMALL_PX:
                d["small"] += 1
            if m.get("origin") == "human_new":
                d["new"] += 1
            else:
                d["kept"] += 1
    for d in S.values():
        d["del"] = max(0, d["mach"] - d["kept"])
        d["disagree"] = (d["new"] + d["del"]) / max(1, d["mach"] + d["new"])
        d["small_share"] = d["small"] / d["boxes"] if d["boxes"] else 0.0
    return S

def stratum(sess, d):
    month = sess[:6]
    db = 0 if d["disagree"] == 0 else (1 if d["disagree"] <= 0.35 else 2)
    sb = (-1 if d["boxes"] == 0
          else 0 if d["small_share"] <= 0.083
          else 1 if d["small_share"] <= 0.5 else 2)
    return (month, db, sb)

def main():
    S = session_features()
    rng = random.Random(SEED)
    strata = collections.defaultdict(list)
    for s in sorted(S):
        strata[stratum(s, S[s])].append(s)
    eval_set = set()
    for _, sessions in sorted(strata.items()):
        rng.shuffle(sessions)
        target = EVAL_FRAC * sum(S[s]["imgs"] for s in sessions)
        acc = 0
        for s in sessions:
            if acc >= target:
                break
            eval_set.add(s)
            acc += S[s]["imgs"]

    def agg(keys):
        t = collections.Counter()
        for s in keys:
            for k in ("imgs", "boxes", "small", "new", "del"):
                t[k] += S[s][k]
            t["sess"] += 1
        return t

    train_set = set(S) - eval_set
    for name, keys in (("ALL", set(S)), ("TRAIN", train_set), ("EVAL", eval_set)):
        t = agg(keys)
        print(f"{name:5s} sess={t['sess']:5d} imgs={t['imgs']:6d} "
              f"boxes={t['boxes']:6d} boxes/img={t['boxes']/t['imgs']:.3f} "
              f"small={t['small']/max(1,t['boxes']):.1%} "
              f"disagree={(t['new']+t['del'])/max(1,t['boxes']):.1%}")

    for root in (DATASET / "splits", STORE / "data" / "splits"):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"SKIP {root}: not writable (dataset dir owned by the "
                  f"download agent; chown then re-run)")
            continue
        (root / "eval_sessions_v1.txt").write_text(
            "\n".join(sorted(eval_set)) + "\n")
        (root / "train_sessions_v1.txt").write_text(
            "\n".join(sorted(train_set)) + "\n")
        print(f"wrote {root}/{{train,eval}}_sessions_v1.txt")

if __name__ == "__main__":
    main()

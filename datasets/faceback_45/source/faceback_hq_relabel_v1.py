"""faceback_hq re-label prelabels (PII-946): current GT plus unmatched armAE34 boxes.

Same idea as PII-940 (face_mine) but for faceback_hq, and with no inference: the
per-frame armAE34 dump written by PII-213 already holds every detection and every
GT box of the 16,940 faceback_hq eval frames, so this script is a filter over that
one file.

Stages
------
build   read the dump, cut the additions, write additions.jsonl, import.jsonl,
        stats.json and README.md. Deterministic: the same dump gives a
        byte-identical output.
check   re-run the structural checks over the built files against the dump,
        without rebuilding.
render  draw GT (green) and additions (red) on N sampled frames.

Cut (user decision 2026-09-15, PII-946; it differs from PII-940's 0.4)
---
A detection becomes an ADDITION when score >= --score-cut (0.3) AND its max IoU
against every GT box of that frame is < --iou-gate (0.1). "Every GT box" means
`gt` plus `gt_ignored`: faceback_hq has no ignore-flagged boxes, so gt_ignored is
exactly the sub-40 px boxes that the eval drops but a human did draw. No size
floor on the additions: tiny boxes are kept.

Coordinates
-----------
The dump is in original pixels (2328x1748) at one decimal, which is the precision
of the pii-data source table datasets/faceback_hq/boxes/v1.csv, so the GT here is
the labelled value, not a re-rounding. IoU is computed on those pixel corners.
import.jsonl carries normalised top-left x,y,w,h at 4 decimals (the faceback_45
prelabel convention); additions.jsonl keeps the pixel xyxy. Boxes are not clipped
to the frame, following the faceback_45 precedent.

usage:
  faceback_hq_relabel_v1.py build  --out-dir DIR [--dump F] [--score-cut S] [--iou-gate G]
  faceback_hq_relabel_v1.py check  --out-dir DIR [--dump F] [--score-cut S] [--iou-gate G]
  faceback_hq_relabel_v1.py render --out-dir DIR --render-dir DIR [--n 8] [--seed 946]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import images as pii_images, pages_media  # noqa: E402

DUMP = pages_media("face-mine-eval", "faceback_hq", "armAE34.json")
IMAGES = pii_images("faceback_45")
EVAL_PY = "/data/esteban/pii_venv/egoblur/bin/python"   # PII-1378, was runs/eval_venv
MODEL_NAME = "armAE34"
SCORE_BANDS = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
SIDE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 1e9)]


# ------------------------------------------------------------------------ shared

def iou_max(det, gt) -> float:
    """Max IoU of one xyxy box against an (N,4) array of xyxy boxes."""
    if not len(gt):
        return 0.0
    x1 = np.maximum(det[0], gt[:, 0])
    y1 = np.maximum(det[1], gt[:, 1])
    x2 = np.minimum(det[2], gt[:, 2])
    y2 = np.minimum(det[3], gt[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (det[2] - det[0]) * (det[3] - det[1])
    b = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    return float(np.max(inter / np.maximum(a + b - inter, 1e-9)))


def load_dump(path: Path) -> tuple[dict, list]:
    with open(path) as fh:
        obj = json.load(fh)
    frames = obj.pop("frames")
    seen = set()
    for fr in frames:
        if fr["file"] in seen:
            raise SystemExit(f"{path}: duplicate frame {fr['file']}")
        seen.add(fr["file"])
    return obj, frames


def name_parts(name: str) -> tuple[str, str, str, int]:
    """faceback_hq file name -> (session, chunk, view, frame_idx).

    <session>_c<chunk>_<left|right>_f<idx:06d>.jpg, the faceback_45 object name.
    """
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    head, _, frame = stem.rpartition("_f")
    head, _, eye = head.rpartition("_")
    session, _, chunk = head.rpartition("_c")
    if eye not in ("left", "right") or not chunk.isdigit() or not frame.isdigit():
        raise SystemExit(f"cannot parse frame name {name}")
    return session, chunk, {"left": "lview", "right": "rview"}[eye], int(frame)


def import_name(session: str, chunk: str, frame_idx: int) -> str:
    """The name face_view_import derives: no view token (faceback_45 README)."""
    return f"{session}_c{chunk}_f{frame_idx:06d}.jpg"


def select(frames, score_cut: float, iou_gate: float):
    """Yield (frame, gt_all, additions) for frames with at least one addition."""
    for fr in frames:
        gt_all = np.array(fr["gt"] + fr["gt_ignored"], dtype=float).reshape(-1, 4)
        adds = []
        for det in fr["dets"]:
            sc = det[4]
            if sc < score_cut:
                continue
            if iou_max(det[:4], gt_all) >= iou_gate:
                continue
            adds.append((float(det[0]), float(det[1]), float(det[2]), float(det[3]), sc))
        if adds:
            yield fr, gt_all, adds


# -------------------------------------------------------------------------- build

def build(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head, frames = load_dump(Path(args.dump))
    W, H = head["coord_space"]

    st = dict(dump=str(args.dump), model=head["model"], model_path=head["path"],
              det_size=head["det_size"], dump_min_side=head["min_side"],
              dataset=head["dataset"], boxes_version=head["boxes_version"],
              coord_space=[W, H], score_cut=args.score_cut, iou_gate=args.iou_gate,
              dump_frames=len(frames), dump_dets=sum(len(f["dets"]) for f in frames),
              dets_at_cut=sum(1 for f in frames for d in f["dets"] if d[4] >= args.score_cut),
              gt_eval_all=sum(len(f["gt"]) for f in frames),
              gt_ignored_all=sum(len(f["gt_ignored"]) for f in frames),
              frames=0, additions=0, gt_eval=0, gt_ignored=0, gt_boxes=0,
              frames_with_no_gt=0, additions_out_of_frame=0,
              score_bands={f"{a}-{b}": 0 for a, b in SCORE_BANDS},
              side_bands={f"{a}-{b}": 0 for a, b in SIDE_BANDS},
              sessions=0, views={"lview": 0, "rview": 0})

    add_lines, imp_lines, sessions = [], [], set()
    for fr, gt_all, adds in select(frames, args.score_cut, args.iou_gate):
        name = fr["file"]
        session, chunk, view, frame_idx = name_parts(name)
        if session != fr["session"]:
            raise SystemExit(f"{name}: session {fr['session']} != name {session}")
        sessions.add(session)
        st["frames"] += 1
        st["views"][view] += 1
        st["additions"] += len(adds)
        st["gt_eval"] += len(fr["gt"])
        st["gt_ignored"] += len(fr["gt_ignored"])
        st["gt_boxes"] += len(gt_all)
        if not len(gt_all):
            st["frames_with_no_gt"] += 1

        boxes = []
        for src, ign in ((fr["gt"], False), (fr["gt_ignored"], True)):
            for x1, y1, x2, y2 in src:
                b = dict(x=round(x1 / W, 4), y=round(y1 / H, 4),
                         w=round((x2 - x1) / W, 4), h=round((y2 - y1) / H, 4),
                         source="gt")
                if ign:
                    b["gt_ignored"] = True
                boxes.append(b)
        for x1, y1, x2, y2, sc in adds:
            boxes.append(dict(x=round(x1 / W, 4), y=round(y1 / H, 4),
                              w=round((x2 - x1) / W, 4), h=round((y2 - y1) / H, 4),
                              source=MODEL_NAME, score=sc))
            for a, b in SCORE_BANDS:
                if a <= sc < b:
                    st["score_bands"][f"{a}-{b}"] += 1
                    break
            side = max(x2 - x1, y2 - y1)
            for a, b in SIDE_BANDS:
                if a <= side < b:
                    st["side_bands"][f"{a}-{b}"] += 1
                    break
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                st["additions_out_of_frame"] += 1

        add_lines.append(json.dumps({
            "file": name, "session": session, "width": W, "height": H,
            "boxes": [{"xyxy": [x1, y1, x2, y2], "score": sc} for x1, y1, x2, y2, sc in adds],
        }) + "\n")
        imp_lines.append(json.dumps({
            "session": session, "chunk": chunk, "view": view, "frame_idx": frame_idx,
            "width": W, "height": H, "boxes": boxes,
            "image": import_name(session, chunk, frame_idx),
        }) + "\n")

    st["sessions"] = len(sessions)
    with open(out_dir / "additions.jsonl", "w") as fh:
        fh.writelines(add_lines)
    with open(out_dir / "import.jsonl", "w") as fh:
        fh.writelines(imp_lines)
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(st, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out_dir / "README.md", "w") as fh:
        fh.write(render_readme(st))
    print(f"{st['frames']} frames, {st['additions']} additions, "
          f"{st['gt_boxes']} GT boxes carried", flush=True)
    return st


# -------------------------------------------------------------------------- check

def check(args) -> int:
    out_dir = Path(args.out_dir)
    head, frames = load_dump(Path(args.dump))
    W, H = head["coord_space"]
    by_name = {f["file"]: f for f in frames}
    bad = 0

    adds = [json.loads(l) for l in open(out_dir / "additions.jsonl")]
    imps = [json.loads(l) for l in open(out_dir / "import.jsonl")]
    if len(adds) != len(imps):
        print(f"FAIL rows: additions {len(adds)} != import {len(imps)}")
        bad += 1

    # 1. every frame once, and the two files agree frame for frame.
    names = [r["file"] for r in adds]
    if len(set(names)) != len(names):
        print(f"FAIL duplicate frames in additions.jsonl")
        bad += 1
    keys = [(r["session"], r["chunk"], r["view"], r["frame_idx"]) for r in imps]
    if len(set(keys)) != len(keys):
        print("FAIL duplicate (session, chunk, view, frame_idx) in import.jsonl")
        bad += 1
    for a, i in zip(adds, imps):
        if name_parts(a["file"]) != (i["session"], i["chunk"], i["view"], i["frame_idx"]):
            print(f"FAIL row mismatch {a['file']}")
            bad += 1
            break
        if i["image"] != import_name(i["session"], i["chunk"], i["frame_idx"]):
            print(f"FAIL image name {i['image']}")
            bad += 1
            break

    # 2. the expected selection, recomputed from the dump, is exactly what was written.
    want = {f["file"]: a for f, _g, a in select(frames, args.score_cut, args.iou_gate)}
    if set(want) != set(names):
        print(f"FAIL frame set: want {len(want)}, got {len(set(names))}, "
              f"missing {len(set(want) - set(names))}, extra {len(set(names) - set(want))}")
        bad += 1

    # 3. every addition: IoU < gate against gt + gt_ignored, over the whole output.
    worst, n_add, n_viol = 0.0, 0, 0
    for r in adds:
        fr = by_name[r["file"]]
        gt_all = np.array(fr["gt"] + fr["gt_ignored"], dtype=float).reshape(-1, 4)
        for b in r["boxes"]:
            n_add += 1
            v = iou_max(b["xyxy"], gt_all)
            worst = max(worst, v)
            if v >= args.iou_gate or b["score"] < args.score_cut:
                n_viol += 1
    print(f"additions {n_add}, worst IoU vs gt+gt_ignored {worst:.4f}, violations {n_viol}")
    if n_viol:
        bad += 1

    # 4. every GT box of those frames is carried, in order, at the documented rounding.
    n_gt, worst_px = 0, 0.0
    for r in imps:
        fr = by_name[f"{r['session']}_c{r['chunk']}_"
                     f"{'left' if r['view'] == 'lview' else 'right'}_f{r['frame_idx']:06d}.jpg"]
        src = [(b, False) for b in fr["gt"]] + [(b, True) for b in fr["gt_ignored"]]
        got = [b for b in r["boxes"] if b["source"] == "gt"]
        if len(got) != len(src):
            print(f"FAIL GT count {r['image']}: {len(got)} != {len(src)}")
            bad += 1
            break
        for (s, ign), g in zip(src, got):
            n_gt += 1
            if g.get("gt_ignored", False) != ign:
                print(f"FAIL GT ignore flag {r['image']}")
                bad += 1
                break
            exp = dict(x=round(s[0] / W, 4), y=round(s[1] / H, 4),
                       w=round((s[2] - s[0]) / W, 4), h=round((s[3] - s[1]) / H, 4))
            if any(g[k] != exp[k] for k in exp):
                print(f"FAIL GT value {r['image']}: {g} != {exp}")
                bad += 1
                break
            worst_px = max(worst_px, abs(g["x"] * W - s[0]), abs(g["y"] * H - s[1]),
                           abs((g["x"] + g["w"]) * W - s[2]),
                           abs((g["y"] + g["h"]) * H - s[3]))
    exp_gt = sum(len(by_name[n]["gt"]) + len(by_name[n]["gt_ignored"]) for n in names)
    print(f"GT boxes carried {n_gt} of {exp_gt} ({n_gt == exp_gt}), "
          f"worst round-trip {worst_px:.4f} px")
    if n_gt != exp_gt:
        bad += 1

    # 5. the addition boxes in import.jsonl are the additions.jsonl ones, normalised.
    for a, i in zip(adds, imps):
        mb = [b for b in i["boxes"] if b["source"] == MODEL_NAME]
        if len(mb) != len(a["boxes"]):
            print(f"FAIL addition count {a['file']}")
            bad += 1
            break
        for s, g in zip(a["boxes"], mb):
            x1, y1, x2, y2 = s["xyxy"]
            exp = dict(x=round(x1 / W, 4), y=round(y1 / H, 4), w=round((x2 - x1) / W, 4),
                       h=round((y2 - y1) / H, 4), source=MODEL_NAME, score=s["score"])
            if g != exp:
                print(f"FAIL addition value {a['file']}: {g} != {exp}")
                bad += 1
                break

    print("CHECK " + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


# ------------------------------------------------------------------------- render

def render(args) -> None:
    import cv2

    out_dir = Path(args.out_dir)
    rd = Path(args.render_dir)
    rd.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(out_dir / "additions.jsonl")]
    imps = {}
    for l in open(out_dir / "import.jsonl"):
        r = json.loads(l)
        eye = "left" if r["view"] == "lview" else "right"
        imps[f"{r['session']}_c{r['chunk']}_{eye}_f{r['frame_idx']:06d}.jpg"] = r
    random.Random(args.seed).shuffle(rows)
    for r in rows[:args.n]:
        img = cv2.imread(str(IMAGES / r["file"]))
        if img is None:
            print(f"unreadable {r['file']}")
            continue
        H, W = img.shape[:2]
        for b in imps[r["file"]]["boxes"]:
            if b["source"] != "gt":
                continue
            p = (int(b["x"] * W), int(b["y"] * H),
                 int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H))
            cv2.rectangle(img, p[:2], p[2:], (0, 255, 0), 3)
            if b.get("gt_ignored"):
                cv2.putText(img, "ign", (p[0], max(14, p[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for b in r["boxes"]:
            x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"{b['score']:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(rd / r["file"]), img)
        n_gt = sum(1 for b in imps[r["file"]]["boxes"] if b["source"] == "gt")
        scores = ", ".join("%.2f" % b["score"] for b in r["boxes"])
        print(f"{r['file']}: gt {n_gt}, additions {len(r['boxes'])} [{scores}]")


# ------------------------------------------------------------------------- README

def render_readme(st: dict) -> str:
    sb, db = st["score_bands"], st["side_bands"]
    t = []
    t.append("# faceback_hq re-label prelabels v1 (PII-946)\n\n")
    t.append(f"The current faceback_hq GT plus the {st['model']} boxes that GT does not cover, so "
             "the labelers\njudge only the additions. No inference was run: this is a filter over "
             "the existing\nper-frame dump.\n\n")
    t.append("## Source\n\n")
    t.append(f"- dump: `{st['dump']}` ({st['dump_frames']:,} frames, "
             f"{st['dump_dets']:,} detections down to score 0.02)\n")
    t.append(f"- model: {st['model']} (`{st['model_path']}`), det_size {st['det_size']}, "
             f"coords in original {st['coord_space'][0]}x{st['coord_space'][1]} px\n")
    t.append(f"- GT: pii-data dataset `{st['dataset']}` boxes `{st['boxes_version']}` as the dump "
             f"carries it: `gt`\n  ({st['gt_eval_all']:,} boxes, long side >= "
             f"{st['dump_min_side']:g} px, what the eval scores) plus `gt_ignored`\n  "
             f"({st['gt_ignored_all']:,} boxes, under {st['dump_min_side']:g} px; faceback_hq has "
             "no ignore-flagged boxes, so\n  gt_ignored is exactly the sub-floor ones)\n")
    t.append(f"- images: `{IMAGES}/<file>`; faceback_hq is a "
             "relabel of\n  the faceback_45 eval frames (PII-176) and owns no bytes of its own\n\n")
    t.append("## The cut\n\n")
    t.append(f"A detection is an ADDITION when both hold:\n\n"
             f"- `score >= {st['score_cut']}`;\n"
             f"- max IoU against EVERY GT box of that frame, `gt` and `gt_ignored` together, is "
             f"`< {st['iou_gate']}`.\n\n")
    t.append("No size floor: tiny boxes are kept. IoU is plain intersection over union on the "
             "dump's\npixel corners (one decimal, the precision of the pii-data source table). "
             "Boxes are not\nclipped to the frame, following faceback_45; "
             f"{st['additions_out_of_frame']:,} additions reach outside.\n\n")
    t.append("## Counts\n\n")
    t.append("| | |\n|---|---:|\n")
    t.append(f"| Frames in the dump | {st['dump_frames']:,} |\n")
    t.append(f"| Detections at score >= {st['score_cut']} | {st['dets_at_cut']:,} |\n")
    t.append(f"| Additions (after the IoU gate) | {st['additions']:,} |\n")
    t.append(f"| Frames with at least one addition | {st['frames']:,} |\n")
    t.append(f"| Sessions covered | {st['sessions']} |\n")
    t.append(f"| lview / rview frames | {st['views']['lview']:,} / {st['views']['rview']:,} |\n")
    t.append(f"| GT boxes carried | {st['gt_boxes']:,} "
             f"({st['gt_eval']:,} eval + {st['gt_ignored']:,} ignored) |\n")
    t.append(f"| Of those frames, with no GT at all | {st['frames_with_no_gt']:,} |\n\n")
    t.append("Additions by score band:\n\n| band | boxes |\n|---|---:|\n")
    for a, b in SCORE_BANDS:
        lab = f"{a}-{b}" if b <= 1 else f"{a}+"
        t.append(f"| {lab} | {sb[f'{a}-{b}']:,} |\n")
    t.append("\nAdditions by long side (px):\n\n| band | boxes |\n|---|---:|\n")
    for a, b in SIDE_BANDS:
        lab = f"<{b}" if a == 0 else (f"{a}+" if b > 1e8 else f"{a}-{b}")
        t.append(f"| {lab} | {db[f'{a}-{b}']:,} |\n")
    t.append("\n## Files\n\n")
    t.append("`additions.jsonl`: one row per frame that has at least one addition, in the dump's "
             "frame\norder. Original pixels.\n\n")
    t.append('```json\n{"file": "<session>_c<chunk>_<left|right>_f<idx>.jpg", "session": "...",\n'
             ' "width": 2328, "height": 1748,\n'
             ' "boxes": [{"xyxy": [1334.2, 579.9, 1414.6, 681.4], "score": 0.8213}]}\n```\n\n')
    t.append("`import.jsonl`: the same frames in the faceback_45 prelabel shape "
             "(`datasets/faceback_45/prelabels/README.md`), normalised top-left x,y,w,h at 4 "
             "decimals.\nGT first in source order, then the additions in the dump's "
             "score-descending order.\n\n")
    t.append('```json\n{"session": "...", "chunk": "032", "view": "lview", "frame_idx": 0,\n'
             ' "width": 2328, "height": 1748, "image": "<session>_c<chunk>_f<idx:06d>.jpg",\n'
             ' "boxes": [{"x": 0.5731, "y": 0.3317, "w": 0.0175, "h": 0.0295, "source": "gt"},\n'
             '           {"x": 0.7204, "y": 0.5449, "w": 0.0233, "h": 0.0494, "source": "gt",'
             ' "gt_ignored": true},\n'
             '           {"x": 0.2, "y": 0.3, "w": 0.02, "h": 0.03, "source": "armAE34",'
             ' "score": 0.4123}]}\n```\n\n')
    t.append("`source` is `gt` or `armAE34`; machine boxes carry `score`; a GT box that the eval "
             "ignores\n(long side under 40 px) carries `gt_ignored: true`. `image` drops the view "
             "token because\n`face_view_import` derives `<session>_c<chunk>_f<idx:06d>.jpg` and "
             "rejects any other name;\nthe view lives in the `view` field and in the per-view "
             "dataset prefix, as faceback_45 did.\n\n")
    t.append("`stats.json`: every count above, machine readable.\n\n")
    t.append("## Re-cutting at another score or IoU\n\n")
    t.append("Nothing was thrown away: the dump holds every detection down to 0.02 with its "
             "score, plus\nboth GT lists, so another cut is one pass over that file and needs no "
             "GPU.\n\n")
    t.append(f"```sh\n{EVAL_PY} mining/faceback_hq_relabel_v1.py "
             "build \\\n  --out-dir <dir> --score-cut 0.4 --iou-gate 0.1\n"
             f"{EVAL_PY} mining/faceback_hq_relabel_v1.py "
             "check \\\n  --out-dir <dir> --score-cut 0.4 --iou-gate 0.1\n```\n\n")
    t.append("Measured cuts (additions / frames with an addition), same IoU gate 0.1:\n\n")
    t.append("| score | additions | frames |\n|---|---:|---:|\n"
             "| 0.3 | 6,713 | 4,023 |\n| 0.4 | 2,787 | 2,022 |\n| 0.5 | 969 | 854 |\n\n")
    t.append("Raising the gate to 1.0 disables it: every detection at the score becomes an "
             "addition\n(28,788 at 0.3), which is the sabotage check that the gate is not "
             "vacuous.\n\n")
    t.append("## Handling\n\n")
    t.append("These are unblurred faces of real people. PII: this directory stays on this box; "
             "nothing\ngoes into the repo and nothing goes to an external service.\n")
    return "".join(t)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("build", "check", "render"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--dump", default=str(DUMP))
        p.add_argument("--score-cut", type=float, default=0.3)
        p.add_argument("--iou-gate", type=float, default=0.1)
        if name == "render":
            p.add_argument("--render-dir", required=True)
            p.add_argument("--n", type=int, default=8)
            p.add_argument("--seed", type=int, default=946)
    args = ap.parse_args()
    if args.stage == "build":
        build(args)
    elif args.stage == "check":
        sys.exit(check(args))
    else:
        render(args)


if __name__ == "__main__":
    main()

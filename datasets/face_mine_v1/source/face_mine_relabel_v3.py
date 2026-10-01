"""Re-label prelabels (PII-940, PII-990): human boxes plus unmatched armAE34 detections.

Each prelabel frame carries the EXISTING human boxes verbatim plus the armAE34
detections we believe the labelers missed, so the vendor only judges the additions.

--ds picks the dataset: `face_mine` (PII-940, the default, unchanged),
`faceight_c` (PII-980) or `faceight_b` (PII-1015); see DATASETS below.

Stages
------
detect  run the ONNX detector over a frames dir and dump EVERY box down to
        --score-thr with its score, in original pixels, one row per frame.
        Decode is rview_detect.SCRFD (det_size 1024, padded canvas), which is
        eval_onnx.SCRFD's pad path box for box (PII-185, re-checked in PII-940).
        --shard K/N takes files[K::N] of the sorted listing so N GPUs can run at
        once; each shard resumes from its own out file. --list FILE restricts the
        run to the basenames in a text file instead of the whole dir listing
        (PII-982); without it the dir behaviour is unchanged.
merge   concatenate shard dumps into one file sorted by frame name and assert
        the frame set equals the image dir listing (or --list) exactly.
build   join the dump with the human boxes and write import_<view>.jsonl in the
        faceback_45 prelabel shape plus stats.json and README.md. Deterministic:
        the same dump and the same human file give a byte-identical output.
        --prefer human (default): the PII-940 rule below, unchanged.
        --prefer model (PII-1272): a model box at >= --score-cut whose IoU
        against any human box of the frame is >= --iou-gate REPLACES the human
        boxes it overlaps (each human box dropped at most once, each model box
        emitted once); unmatched human boxes stay, unmatched model boxes are
        additions; every box carries source and score (human score null).
        Dedup (PII-1273): when several model boxes overlap one human box at
        IoU >= --iou-gate only the highest-scoring one is emitted; the others
        are DROPPED (not additions). A frame with >= 1 replaced or added box
        is listed in changed_<view>.txt.
check   re-run the structural checks over the built files (frame set, human box
        count, IoU gate) without rebuilding. --prefer model checks the
        replacement rule instead (check_view_model).
render  draw human boxes (green) and model boxes (red) on N sampled frames;
        with --prefer model the replaced human boxes are dashed green and
        render/index.txt lists them per frame.

Human boxes
-----------
PII-1449: the PII-1315 store (PII_ROOT) keeps per-dataset boxes as
boxes/vN/boxes.csv, so the .jsonl human files below and face-mine_labeled.csv
are read from LEGACY_ROOT (/data/esteban/pii_backup). Images come from the live
store where the names are unchanged (face_mine_v1, face_mine_right); the
faceight images_left / images_right trees are gone on this box; their bytes are
the one flat legacy datasets/faceight_<set>/images dir, whose file names are
exactly the human file's basenames (the store re-named its own copies to
<session>_c<chunk>_<eye>_f<idx>.jpg, which these rows do not use).

face_mine
  left  (face_mine_v1): <LEGACY_ROOT>/face-mine_labeled.csv, the HUMAN labels.
        NOT datasets/face_mine_v1/boxes.jsonl, which is the miner's machine output
        (box_src face_mine/two_model@1, 83,536 boxes); see PII-132 / PII-335.
  right (face_mine_right): datasets/face_mine_right/boxes.jsonl, which IS human
        (box_src human, the PII-194 conversion of the 2026-09-11 drop).
faceight_c / faceight_b
  datasets/faceight_c/boxes.jsonl (PII-981) and datasets/faceight_b/boxes.jsonl
  (PII-1016), one file for BOTH views of its round: rows are
  taken by their `view` field, the image name drops its images_<side>/ prefix,
  and the frame index comes from the record's `frame` field, since the faceight
  frame file is named by timestamp (PII-990, PII-1035).
All carry x,y,w,h normalised top-left+size in 2328x1748 at <= 4 decimals; they
are copied into the prelabels verbatim, never modified and never dropped.

Cut
---
A detection becomes an ADDITION when score >= --score-cut (0.4 for face_mine,
0.3 for the faceight rounds per PII-946) AND its max IoU against every human box of that
frame is < --iou-gate (0.1). No size floor.
IoU is computed in pixels: the human box at x*W, y*H, (x+w)*W, (y+h)*H against
the dump's pixel corners, plain intersection over union (eval_onnx.iou_mat).
Boxes are not clipped to the frame, following the faceback_45 precedent (1,817
of its 38,436 import boxes stick out of the frame).

usage:
  face_mine_relabel_v3.py detect --model M.onnx --frames DIR --out D.jsonl [--shard K/N] [--list F]
  face_mine_relabel_v3.py merge --out D.jsonl --shards D.shard*.jsonl --frames DIR [--list F]
  face_mine_relabel_v3.py build --out-dir DIR --dump-left L.jsonl --dump-right R.jsonl
      [--ds faceight_b] [--score-cut 0.3] [--iou-gate 0.1]
  face_mine_relabel_v3.py check --out-dir DIR --dump-left L.jsonl --dump-right R.jsonl
      [--ds faceight_b] [--score-cut 0.3]
  face_mine_relabel_v3.py render --out-dir DIR --render-dir DIR --n 10 [--ds faceight_b]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from pii_root import ROOT, onnx  # noqa: E402
# Provenance (PII-1682): the pre-PII-1315 tree this script read, now retired. The path below
# records where the input came from; it is not a tree to read today.
LEGACY = Path("/data/esteban/pii_backup")

PII = ROOT          # the live store (PII_ROOT), PII-1315 layout
PII_OLD = LEGACY    # the only home the .jsonl human files ever had
# frames_from: "dir" takes the frame set from the image dir listing; "human" takes
# it from the human file, for a dataset whose image dir holds far more than the
# labeled frames (faceight_c points at the whole 372k / 414k frame trees).
# import_name: "file" writes the frame file name into the prelabel row; "verdict"
# writes <session>_c<chunk>_f<frame_idx:06d>.jpg, the name the faceight verdict
# package uses, since the faceight frame file is named by timestamp (PII-990).
DATASETS = {
    "face_mine": {
        "left": dict(
            view="lview",
            images=PII / "datasets/face_mine_v1/images",
            human=PII_OLD / "face-mine_labeled.csv",   # no live equivalent
            human_kind="csv",
            frames_from="dir",
            import_name="file",
        ),
        "right": dict(
            view="rview",
            images=PII / "datasets/face_mine_right/images",
            human=PII_OLD / "datasets/face_mine_right/boxes.jsonl",
            human_kind="jsonl",
            frames_from="dir",
            import_name="file",
        ),
    },
    "faceight_b": {
        "left": dict(
            view="lview",
            images=PII_OLD / "datasets/faceight_b/images",   # flat, both views
            human=PII_OLD / "datasets/faceight_b/boxes.jsonl",
            human_kind="faceight",
            frames_from="human",
            import_name="verdict",
        ),
        "right": dict(
            view="rview",
            images=PII_OLD / "datasets/faceight_b/images",   # flat, both views
            human=PII_OLD / "datasets/faceight_b/boxes.jsonl",
            human_kind="faceight",
            frames_from="human",
            import_name="verdict",
        ),
    },
    "faceight_c": {
        "left": dict(
            view="lview",
            images=PII_OLD / "datasets/faceight_c/images",   # flat, both views
            human=PII_OLD / "datasets/faceight_c/boxes.jsonl",
            human_kind="faceight",
            frames_from="human",
            import_name="verdict",
        ),
        "right": dict(
            view="rview",
            images=PII_OLD / "datasets/faceight_c/images",   # flat, both views
            human=PII_OLD / "datasets/faceight_c/boxes.jsonl",
            human_kind="faceight",
            frames_from="human",
            import_name="verdict",
        ),
    },
}
# The README wording that differs between the faceight rounds; the file shape,
# the cut and the checks are the same, and every count comes from stats.
DS_DOC = {
    "faceight_b": dict(issues="PII-1015 / PII-1035", round="round-2", human_issue="PII-1016"),
    "faceight_c": dict(issues="PII-980 / PII-990", round="round-3", human_issue="PII-981"),
}
MODEL_NAME = "armAE34"
SCORE_BANDS = ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01))
SIDE_BANDS = ((0, 20), (20, 40), (40, 60), (60, 100), (100, 1e9))


def score_bands(cut: float) -> tuple:
    """The score bands a cut can fill: bands entirely below it are dropped.

    At --score-cut 0.4 this is the PII-940 set (0.4-0.5 .. 0.8+) unchanged; at
    0.3 it also reports the 0.3-0.4 band, which is where the noise lives.
    """
    return tuple(b for b in SCORE_BANDS if b[1] > cut)


# ------------------------------------------------------------------ small helpers
def list_frames(d: Path) -> list[str]:
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg")))


def read_list(p: Path) -> list[str]:
    """Frame basenames from a text file, one per line; blanks and # lines skipped.

    Only an explicit --list uses this; with --frames alone the stage still takes
    the whole sorted directory listing exactly as before (PII-982).
    """
    names = []
    with open(p) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(os.path.basename(s))
    if len(set(names)) != len(names):
        raise SystemExit(f"{p}: duplicate names in list")
    if not names:
        raise SystemExit(f"{p}: empty list")
    return sorted(names)


def wanted_frames(args) -> list[str]:
    """The frame set a stage works on: the --list file when given, else the dir."""
    lst = getattr(args, "list", None)
    return read_list(Path(lst)) if lst else list_frames(Path(args.frames))


def parse_name(name: str) -> dict:
    """session / chunk / frame_idx from a frame basename.

    face_mine and faceback name frames {session}_c{chunk}_f{frame_idx}.jpg, which
    is the only shape PII-940 met. faceight names them
    {session}_c{chunk}_{view}_t{t_ms}.jpg instead: that one carries a timestamp,
    not a frame index, so it reports view and t_ms and leaves frame_idx null
    (the faceight frame index lives in data/faceight/frames.csv, PII-982).
    """
    stem = name.rsplit(".", 1)[0]
    session, _, rest = stem.rpartition("_c")
    if not session:
        return {"session": None, "chunk": None, "frame_idx": None}
    if "_f" in rest:
        chunk, frame = rest.split("_f", 1)
        return {"session": session, "chunk": chunk, "frame_idx": int(frame)}
    if "_t" in rest:
        chunk, tail = rest.split("_", 1)
        view, _, t_ms = tail.rpartition("_t")
        return {"session": session, "chunk": chunk, "frame_idx": None,
                "view": view, "t_ms": int(t_ms)}
    return {"session": session, "chunk": rest, "frame_idx": None}


def iou_max(det, hpix) -> float:
    """Max IoU of one xyxy box against an (N,4) array of xyxy boxes."""
    if not len(hpix):
        return 0.0
    x1 = np.maximum(det[0], hpix[:, 0])
    y1 = np.maximum(det[1], hpix[:, 1])
    x2 = np.minimum(det[2], hpix[:, 2])
    y2 = np.minimum(det[3], hpix[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (det[2] - det[0]) * (det[3] - det[1])
    b = (hpix[:, 2] - hpix[:, 0]) * (hpix[:, 3] - hpix[:, 1])
    return float(np.max(inter / np.maximum(a + b - inter, 1e-9)))


def _human_boxes(name: str, raw: list) -> list:
    """x,y,w,h copied verbatim from a source row; more than 4 decimals is fatal."""
    boxes = []
    for b in raw:
        box = {}
        for k in ("x", "y", "w", "h"):
            v = float(b[k])
            if round(v, 4) != v:
                raise SystemExit(f"{name}: {k}={b[k]} has more than 4 decimals")
            box[k] = v
        boxes.append(box)
    return boxes


def load_human(view: str, ds: str = "face_mine") -> dict:
    """frame file name -> dict(session, chunk, frame_idx, width, height, boxes, import_name).

    Values are the source file's own, parsed and re-emitted unchanged (every
    source is <= 4 decimals, asserted here). `import_name` is the name the
    prelabel row will carry (see DATASETS import_name).
    """
    cfg = DATASETS[ds][view]
    out: dict[str, dict] = {}
    if cfg["human_kind"] == "csv":
        for r in csv.DictReader(open(cfg["human"], encoding="utf-8-sig")):
            name = os.path.basename(r["image_uri"])
            want = f"{r['session']}_c{r['chunk']}_f{int(r['frame_idx']):06d}.jpg"
            if name != want:
                raise SystemExit(f"{cfg['human']}: image_uri {name} != {want}")
            rec = out.setdefault(name, dict(
                session=r["session"], chunk=r["chunk"], frame_idx=int(r["frame_idx"]),
                width=int(r["width"]), height=int(r["height"]), boxes=[],
                import_name=name))
            if r["box_i"] == "":
                continue
            rec["boxes"].extend(_human_boxes(name, [r]))
    elif cfg["human_kind"] == "faceight":
        prefix = f"images_{view}/"
        for line in open(cfg["human"]):
            r = json.loads(line)
            if r["view"] != cfg["view"]:
                continue
            if not r["image"].startswith(prefix):
                raise SystemExit(f"{cfg['human']}: {r['image']} does not start with {prefix}")
            name = r["image"][len(prefix):]
            if name in out:
                raise SystemExit(f"{cfg['human']}: duplicate frame {name}")
            chunk, frame_idx = r["chunk"], int(r["frame"])
            if chunk != f"{int(chunk):03d}":
                raise SystemExit(f"{name}: chunk {chunk} is not %03d")
            boxes = _human_boxes(name, r["boxes"])
            if len(boxes) != r["n_boxes"]:
                raise SystemExit(f"{name}: n_boxes {r['n_boxes']} != {len(boxes)}")
            out[name] = dict(session=r["session"], chunk=chunk, frame_idx=frame_idx,
                             width=int(r["width"]), height=int(r["height"]), boxes=boxes,
                             import_name=f"{r['session']}_c{chunk}_f{frame_idx:06d}.jpg")
    else:
        for line in open(cfg["human"]):
            r = json.loads(line)
            name = r["image"].split("/")[-1]
            if name in out:
                raise SystemExit(f"{cfg['human']}: duplicate frame {name}")
            boxes = _human_boxes(name, r["boxes"])
            if len(boxes) != r["n_boxes"]:
                raise SystemExit(f"{name}: n_boxes {r['n_boxes']} != {len(boxes)}")
            out[name] = dict(session=r["session"], chunk=r["chunk"], frame_idx=int(r["frame"]),
                             width=int(r["width"]), height=int(r["height"]), boxes=boxes,
                             import_name=name)
    imp = {}
    for fname, rec in out.items():
        if rec["import_name"] in imp:
            raise SystemExit(f"{cfg['human']}: two frames share the prelabel name "
                             f"{rec['import_name']} ({imp[rec['import_name']]}, {fname})")
        imp[rec["import_name"]] = fname
    return out


def frame_set(view: str, ds: str, humans: dict) -> list[str]:
    """The frame names a build or a check works on, per DATASETS frames_from."""
    cfg = DATASETS[ds][view]
    if cfg["frames_from"] == "human":
        return sorted(humans)
    return list_frames(cfg["images"])


def load_dump(path: Path) -> dict:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["file"] in out:
                raise SystemExit(f"{path}: duplicate frame {r['file']}")
            out[r["file"]] = r
    return out


# ------------------------------------------------------------------------ detect
def stage_detect(args) -> None:
    import cv2
    from rview_detect import SCRFD

    cv2.setNumThreads(2)
    files = wanted_frames(args)
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"bad --shard {args.shard}")
        files = files[k::n]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["file"])
                except (ValueError, KeyError):
                    pass
    todo = [f for f in files if f not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"shard {args.shard or 'all'}: {len(files)} frames, {len(done)} done, {len(todo)} to do; "
          f"det_size {args.det_size} padded, score_thr {args.score_thr}", flush=True)
    if not todo:
        return

    model = SCRFD(args.model, args.det_size, args.score_thr)
    frames_dir = Path(args.frames)
    from concurrent.futures import ThreadPoolExecutor

    def load(name: str):
        img = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            return name, None, None, None
        canvas, ratio = model.preprocess(img)
        return name, canvas, ratio, img.shape[:2]

    n_ok = n_bad = n_boxes = 0
    t0 = time.time()
    with open(args.out, "a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        window = args.workers * 4
        futs = [pool.submit(load, f) for f in todo[:window]]
        nxt, i = window, 0
        while i < len(todo):
            name, canvas, ratio, hw = futs[i].result()
            futs[i] = None
            if nxt < len(todo):
                futs.append(pool.submit(load, todo[nxt]))
                nxt += 1
            i += 1
            if canvas is None:
                n_bad += 1
                print(f"WARN unreadable: {name}", file=sys.stderr, flush=True)
                continue
            dets = model.detect(canvas, ratio)
            rec = {"file": name}
            rec.update(parse_name(name))
            rec.update({
                "w": int(hw[1]), "h": int(hw[0]),
                "boxes": [[round(float(x), 1) for x in d[:4]] + [round(float(d[4]), 4)]
                          for d in dets],
            })
            out.write(json.dumps(rec) + "\n")
            n_ok += 1
            n_boxes += len(dets)
            if n_ok % 2000 == 0:
                out.flush()
                el = time.time() - t0
                print(f"{n_ok}/{len(todo)}  {n_ok / el:.1f} fps  "
                      f"eta {(len(todo) - n_ok) / (n_ok / el) / 60:.1f} min", flush=True)
    el = time.time() - t0
    print(f"done: {n_ok} frames, {n_bad} unreadable, {n_boxes} boxes, {el / 60:.2f} min "
          f"({n_ok / max(el, 1e-9):.1f} fps)", flush=True)


def stage_merge(args) -> None:
    recs: dict[str, str] = {}
    for p in sorted(glob.glob(args.shards)):
        with open(p) as fh:
            for line in fh:
                name = json.loads(line)["file"]
                if name in recs:
                    raise SystemExit(f"duplicate frame {name} across shards")
                recs[name] = line
    want = wanted_frames(args)
    if sorted(recs) != want:
        miss = set(want) - set(recs)
        extra = set(recs) - set(want)
        raise SystemExit(f"frame set mismatch: {len(miss)} missing, {len(extra)} extra")
    with open(args.out, "w") as out:
        for name in want:
            out.write(recs[name])
    print(f"merged {len(want)} frames -> {args.out}", flush=True)


# ------------------------------------------------------------------------- build
def merge_frame_model(h: dict, d: dict, score_cut: float, iou_gate: float,
                      model_name: str) -> tuple[list, dict]:
    """--prefer model: the per-frame merge, model box over human box (PII-1272).

    A model box at score >= score_cut whose IoU against any human box of the
    frame is >= iou_gate REPLACES every human box it overlaps: those human boxes
    are dropped and the model box is emitted once. A model box that overlaps no
    human box is an addition; a human box no model box overlaps stays verbatim.
    Dedup (PII-1273): model boxes are taken in score order (descending, stable);
    a model box that overlaps a human box ALREADY claimed by an emitted
    higher-scoring model box is dropped altogether (not emitted, not an
    addition), so each human box is replaced by at most one model box. A
    dropped box claims nothing: a second human box only it overlapped stays.
    Output order: the surviving human boxes in source order, then the emitted
    model boxes in score order. Human boxes carry score null.
    """
    W, H = h["width"], h["height"]
    hpix = np.array([[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H]
                     for b in h["boxes"]], dtype=float).reshape(-1, 4)
    hits_per_human = [0] * len(h["boxes"])      # candidate model boxes per human box
    claimed = [False] * len(h["boxes"])         # human box replaced by an emitted model box
    model_boxes, f = [], dict(additions=0, replacing=0, multi=0, out_of_frame=0,
                              dedup_dropped=0,
                              add_scores=[], repl_scores=[], add_sides=[], worst_add_iou=0.0)
    cands = [b for b in d["boxes"] if b[4] >= score_cut]
    cands.sort(key=lambda b: -b[4])             # stable: equal scores keep dump order
    for x1, y1, x2, y2, sc in cands:
        hits = []
        if len(hpix):
            ix1 = np.maximum(x1, hpix[:, 0])
            iy1 = np.maximum(y1, hpix[:, 1])
            ix2 = np.minimum(x2, hpix[:, 2])
            iy2 = np.minimum(y2, hpix[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            a = (x2 - x1) * (y2 - y1)
            b = (hpix[:, 2] - hpix[:, 0]) * (hpix[:, 3] - hpix[:, 1])
            ious = inter / np.maximum(a + b - inter, 1e-9)
            hits = [int(i) for i in np.where(ious >= iou_gate)[0]]
            if not hits:
                f["worst_add_iou"] = max(f["worst_add_iou"], float(np.max(ious)))
        if hits:
            for i in hits:
                hits_per_human[i] += 1
            if any(claimed[i] for i in hits):
                f["dedup_dropped"] += 1
                continue
            for i in hits:
                claimed[i] = True
            f["replacing"] += 1
            f["multi"] += 1 if len(hits) > 1 else 0
            f["repl_scores"].append(sc)
        else:
            f["additions"] += 1
            f["add_scores"].append(sc)
            f["add_sides"].append(max(x2 - x1, y2 - y1))
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            f["out_of_frame"] += 1
        model_boxes.append(dict(x=round(x1 / W, 4), y=round(y1 / H, 4),
                                w=round((x2 - x1) / W, 4), h=round((y2 - y1) / H, 4),
                                source=model_name, score=sc))
    boxes = [dict(x=b["x"], y=b["y"], w=b["w"], h=b["h"], source="human", score=None)
             for b, c in zip(h["boxes"], claimed) if not c]
    f["replaced"] = sum(claimed)
    f["human_multi"] = sum(1 for n in hits_per_human if n > 1)
    return boxes + model_boxes, f


def build_view_model(view: str, humans: dict, dump: dict, files: list, score_cut: float,
                     iou_gate: float, ds: str, model_name: str):
    """build for --prefer model; see merge_frame_model for the rule."""
    cfg = DATASETS[ds][view]
    bands = score_bands(score_cut)
    lines, changed_names = [], []
    st = dict(frames=len(files), human_boxes=0, frames_with_human=0, dump_boxes=0,
              dets_at_cut=0, human_replaced=0, additions=0, model_replacing=0,
              model_matched_multi=0, human_hit_by_multi=0, model_dropped_dedup=0,
              boxes_out=0, frames_changed=0, frames_with_addition=0,
              frames_with_replacement=0, frames_with_addition_only=0, frames_with_no_box=0,
              frames_with_no_box_before=0, model_out_of_frame=0, worst_addition_iou=0.0,
              score_bands={f"{a}-{b}": 0 for a, b in bands},
              score_bands_replacing={f"{a}-{b}": 0 for a, b in bands},
              side_bands={f"{a}-{b}": 0 for a, b in SIDE_BANDS})
    for name in files:
        h = humans[name]
        d = dump[name]
        if (d["w"], d["h"]) != (h["width"], h["height"]):
            raise SystemExit(f"{name}: dump size {(d['w'], d['h'])} != human "
                             f"{(h['width'], h['height'])}")
        boxes, f = merge_frame_model(h, d, score_cut, iou_gate, model_name)
        st["human_boxes"] += len(h["boxes"])
        st["frames_with_human"] += 1 if h["boxes"] else 0
        st["frames_with_no_box_before"] += 0 if h["boxes"] else 1
        st["dump_boxes"] += len(d["boxes"])
        st["dets_at_cut"] += f["additions"] + f["replacing"] + f["dedup_dropped"]
        st["model_dropped_dedup"] += f["dedup_dropped"]
        st["human_replaced"] += f["replaced"]
        st["additions"] += f["additions"]
        st["model_replacing"] += f["replacing"]
        st["model_matched_multi"] += f["multi"]
        st["human_hit_by_multi"] += f["human_multi"]
        st["model_out_of_frame"] += f["out_of_frame"]
        st["boxes_out"] += len(boxes)
        st["worst_addition_iou"] = max(st["worst_addition_iou"], f["worst_add_iou"])
        changed = bool(f["additions"] or f["replaced"])
        st["frames_changed"] += 1 if changed else 0
        if changed:
            changed_names.append(h["import_name"])
        st["frames_with_addition"] += 1 if f["additions"] else 0
        st["frames_with_replacement"] += 1 if f["replaced"] else 0
        st["frames_with_addition_only"] += 1 if f["additions"] and not h["boxes"] else 0
        st["frames_with_no_box"] += 0 if boxes else 1
        for key, scores in (("score_bands", f["add_scores"]),
                            ("score_bands_replacing", f["repl_scores"])):
            for sc in scores:
                for a, b in bands:
                    if a <= sc < b:
                        st[key][f"{a}-{b}"] += 1
                        break
        for side in f["add_sides"]:
            for a, b in SIDE_BANDS:
                if a <= side < b:
                    st["side_bands"][f"{a}-{b}"] += 1
                    break
        rec = {"session": h["session"], "chunk": h["chunk"], "view": cfg["view"],
               "frame_idx": h["frame_idx"], "width": h["width"], "height": h["height"],
               "boxes": boxes, "image": h["import_name"]}
        lines.append(json.dumps(rec) + "\n")
    st["changed_names"] = changed_names
    return lines, st


def build_view(view: str, dump_path: Path, score_cut: float, iou_gate: float,
               ds: str = "face_mine", prefer: str = "human", model_name: str = MODEL_NAME):
    cfg = DATASETS[ds][view]
    humans = load_human(view, ds)
    dump = load_dump(dump_path)
    files = frame_set(view, ds, humans)
    if sorted(humans) != files:
        raise SystemExit(f"{view}: human frame set != frame set")
    if sorted(dump) != files:
        miss = sorted(set(files) - set(dump))[:3]
        extra = sorted(set(dump) - set(files))[:3]
        raise SystemExit(f"{view}: dump frame set != frame set "
                         f"({len(files)} wanted, {len(dump)} dumped; "
                         f"missing e.g. {miss}, extra e.g. {extra})")
    if prefer == "model":
        return build_view_model(view, humans, dump, files, score_cut, iou_gate, ds, model_name)

    bands = score_bands(score_cut)
    lines = []
    st = dict(frames=len(files), human_boxes=0, frames_with_human=0, dump_boxes=0,
              dets_at_cut=0, additions=0, frames_with_addition=0, additions_out_of_frame=0,
              frames_with_no_box=0, frames_with_addition_only=0, worst_addition_iou=0.0,
              score_bands={f"{a}-{b}": 0 for a, b in bands},
              side_bands={f"{a}-{b}": 0 for a, b in SIDE_BANDS})
    for name in files:
        h = humans[name]
        d = dump[name]
        if (d["w"], d["h"]) != (h["width"], h["height"]):
            raise SystemExit(f"{name}: dump size {(d['w'], d['h'])} != human "
                             f"{(h['width'], h['height'])}")
        W, H = h["width"], h["height"]
        st["human_boxes"] += len(h["boxes"])
        st["frames_with_human"] += 1 if h["boxes"] else 0
        st["dump_boxes"] += len(d["boxes"])
        hpix = np.array([[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H]
                         for b in h["boxes"]], dtype=float).reshape(-1, 4)
        boxes = [dict(x=b["x"], y=b["y"], w=b["w"], h=b["h"], source="human") for b in h["boxes"]]
        n_add = 0
        for x1, y1, x2, y2, sc in d["boxes"]:
            if sc < score_cut:
                continue
            st["dets_at_cut"] += 1
            v = iou_max((x1, y1, x2, y2), hpix)
            if v >= iou_gate:
                continue
            st["worst_addition_iou"] = max(st["worst_addition_iou"], v)
            n_add += 1
            boxes.append(dict(x=round(x1 / W, 4), y=round(y1 / H, 4),
                              w=round((x2 - x1) / W, 4), h=round((y2 - y1) / H, 4),
                              source=model_name, score=sc))
            for a, b in bands:
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
        st["additions"] += n_add
        st["frames_with_addition"] += 1 if n_add else 0
        st["frames_with_addition_only"] += 1 if n_add and not h["boxes"] else 0
        rec = {"session": h["session"], "chunk": h["chunk"], "view": cfg["view"],
               "frame_idx": h["frame_idx"], "width": W, "height": H,
               "boxes": boxes, "image": h["import_name"]}
        if not boxes:
            st["frames_with_no_box"] += 1
        lines.append(json.dumps(rec) + "\n")
    return lines, st


def stage_build(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ds": args.ds, "model": args.model_name, "model_md5": args.model_md5,
             "det_size": args.det_size, "pad": True, "dump_score_thr": args.score_thr,
             "score_cut": args.score_cut, "iou_gate": args.iou_gate, "views": {}}
    if args.prefer == "model":
        # only the model path adds keys, so the human path's stats.json stays byte-equal
        stats["prefer"] = "model"
        stats["issue"] = args.issue
    for view, dump in (("left", args.dump_left), ("right", args.dump_right)):
        lines, st = build_view(view, Path(dump), args.score_cut, args.iou_gate, args.ds,
                               args.prefer, args.model_name)
        st["human_source"] = str(DATASETS[args.ds][view]["human"])
        st["images"] = str(DATASETS[args.ds][view]["images"])
        st["dump"] = str(dump)
        stats["views"][view] = st
        with open(out_dir / f"import_{view}.jsonl", "w") as fh:
            fh.writelines(lines)
        if args.prefer == "model":
            # frames with >= 1 replaced or added box, the upload set (PII-1273)
            changed = st.pop("changed_names")
            with open(out_dir / f"changed_{view}.txt", "w") as fh:
                fh.write("".join(n + "\n" for n in changed))
            print(f"{view}: {st['frames']} frames, {st['human_boxes']} human in, "
                  f"{st['human_replaced']} replaced by {st['model_replacing']} model boxes, "
                  f"{st['additions']} additions, {st['model_dropped_dedup']} dropped by dedup, "
                  f"{st['boxes_out']} out, {st['frames_changed']} frames changed "
                  f"(changed_{view}.txt {len(changed)})", flush=True)
        else:
            print(f"{view}: {st['frames']} frames, {st['human_boxes']} human, "
                  f"{st['additions']} additions on {st['frames_with_addition']} frames", flush=True)
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(out_dir / "README.md", "w") as fh:
        fh.write(render_readme_model(stats) if args.prefer == "model" else render_readme(stats))


def _row(label, fn, stats):
    L, R = stats["views"]["left"], stats["views"]["right"]
    return f"| {label} | {fn(L):,} | {fn(R):,} |\n"


def render_readme(stats: dict) -> str:
    """The README beside the deliverables: model, cut, counts, how to re-cut."""
    ds = stats.get("ds", "face_mine")
    doc = DS_DOC.get(ds)
    L, R = stats["views"]["left"], stats["views"]["right"]
    t = []
    if doc:
        t.append("# {} re-label prelabels v1 ({})\n\n".format(ds, doc["issues"]))
        t.append("Prelabels for a second vendor pass over the {} {} return. Every frame\n"
                 "carries the human boxes of that return verbatim plus the {} detections that do\n"
                 "not match any of them, so the vendor only has to judge the additions.\n\n"
                 .format(ds, doc["round"], stats["model"]))
    else:
        t.append("# face_mine re-label prelabels v3 (PII-940)\n\n")
        t.append("Prelabels for a second vendor pass over face_mine. Every frame carries the EXISTING\n"
                 "human boxes verbatim plus the {} detections that do not match any of them, so the\n"
                 "vendor only has to judge the additions.\n\n".format(stats["model"]))
    t.append("## Model and cut\n\n")
    t.append(f"- Model: `{onnx('scrfd', stats['model'], stats['model'] + '_det34g.onnx')}`"
             f" (md5 {stats['model_md5']}).\n")
    t.append(f"- Inference: det_size {stats['det_size']}, padded canvas (the historical mode every\n"
             "  eval in this project has used); decode is `mining/rview_detect.py::SCRFD`, verified\n"
             "  box for box against `evaluation/eval_onnx.py::SCRFD` on 20 frames (220 box pairs,\n"
             "  worst |dbox| 0.000000 px, worst |dscore| 0.0; that parity run is PII-942, on this\n"
             "  model, and was not repeated here).\n")
    t.append(f"- A detection is an ADDITION when score >= {stats['score_cut']} AND its max IoU\n"
             f"  against every human box of that frame is < {stats['iou_gate']}. There is no size\n"
             "  floor: tiny boxes are kept.\n")
    t.append("- Human boxes are copied verbatim: never modified, never dropped, even when a\n"
             "  detection overlaps them. A frame with no box at all keeps `\"boxes\": []` (the\n"
             "  faceback_45 rule: an empty list is \"the machine looked and found nothing\", which\n"
             "  the portal shows differently from \"no prelabel at all\", and those frames are\n"
             "  exactly where a genuine miss hides).\n")
    t.append("- Boxes are not clipped to the frame, following the faceback_45 precedent\n"
             f"  (additions reaching outside the frame: left {L['additions_out_of_frame']:,}, "
             f"right {R['additions_out_of_frame']:,}).\n\n")
    t.append("## Human box sources\n\n")
    if doc:
        t.append(f"- Both views: `{L['human_source']}` ({doc['human_issue']}), the {doc['round']} vendor return\n"
                 "  converted to the faceight_a box shape (box_src `human`). One file holds both\n"
                 "  views; rows are taken by their `view` field, and the frame index comes from the\n"
                 "  record's `frame` field.\n")
        t.append(f"- Images: `{L['images']}` and `{R['images']}` (symlinks into the faceight frame\n"
                 f"  trees, which hold far more than these {L['frames'] + R['frames']:,} frames; "
                 "the frame set comes from\n"
                 "  the human file, not from a directory listing).\n\n")
        t.append("## Frame names\n\n")
        t.append("The faceight frame file is named by timestamp, `<session>_c<chunk>_<view>_t<ms>.jpg`,\n"
                 "but the verdict package the portal already ingested names frames\n"
                 "`<session>_c<chunk>_f<frame_idx:06d>.jpg` (chunk `%03d`). `image` in the rows below\n"
                 "is that verdict name, with `frame_idx` taken from the human record, so the two\n"
                 "packages line up; `dets_<view>.jsonl` keeps the timestamp file name instead.\n\n")
    else:
        t.append(f"- left ({L['human_source']}): the HUMAN labels of face_mine_v1. NOT\n"
                 "  `datasets/face_mine_v1/boxes.jsonl`, which holds the miner's machine output\n"
                 "  (box_src `face_mine/two_model@1`, 83,536 boxes); see PII-132 and PII-335.\n")
        t.append(f"- right ({R['human_source']}): human (box_src `human`), the PII-194 conversion of\n"
                 "  the 2026-09-11 drop.\n\n")
    t.append("## Files\n\n")
    t.append("```\n"
             "dets_left.jsonl / dets_right.jsonl    full detection dump, one row per frame\n"
             "import_left.jsonl / import_right.jsonl  the prelabels, faceback_45 import shape\n"
             "stats.json                            every count below, machine readable\n"
             "```\n\n")
    t.append("`import_<view>.jsonl` rows are `{session, chunk, view, frame_idx, width, height,\n"
             "boxes, image}`; coordinates are normalised 0..1 top-left plus size at 4 decimals,\n"
             "the same convention as the faceback_45 import files. Each box carries\n"
             f"`source`: `human` or `{stats['model']}`; machine boxes also carry `score`.\n\n")
    t.append("## Counts\n\n| | left | right |\n|---|---:|---:|\n")
    t.append(_row("Frames", lambda v: v["frames"], stats))
    t.append(_row("Human boxes", lambda v: v["human_boxes"], stats))
    t.append(_row("Frames with a human box", lambda v: v["frames_with_human"], stats))
    t.append(_row("Additions", lambda v: v["additions"], stats))
    t.append(_row("Frames with an addition", lambda v: v["frames_with_addition"], stats))
    t.append(_row("Frames with an addition and no human box",
                  lambda v: v.get("frames_with_addition_only", 0), stats))
    t.append(_row("Frames with no box at all", lambda v: v["frames_with_no_box"], stats))
    t.append(_row(f"Dump boxes (>= {stats['dump_score_thr']})", lambda v: v["dump_boxes"], stats))
    t.append(_row(f"Detections at >= {stats['score_cut']}", lambda v: v["dets_at_cut"], stats))
    t.append("\n### Additions by score\n\n| band | left | right |\n|---|---:|---:|\n")
    for k in L["score_bands"]:
        a, b = (float(v) for v in k.split("-"))
        lbl = f"{a}+" if b > 1 else f"{a}-{b}"
        t.append(f"| {lbl} | {L['score_bands'][k]:,} | {R['score_bands'][k]:,} |\n")
    t.append("\n### Additions by long side (px)\n\n| band | left | right |\n|---|---:|---:|\n")
    for a, b in SIDE_BANDS:
        k = f"{a}-{b}"
        lbl = f"< {b}" if a == 0 else (f"{a}+" if b > 1e8 else f"{a}-{b}")
        t.append(f"| {lbl} | {L['side_bands'][k]:,} | {R['side_bands'][k]:,} |\n")
    t.append("\n## Re-cutting without another inference pass\n\n")
    t.append(f"`dets_<view>.jsonl` holds EVERY box down to score {stats['dump_score_thr']} with its\n"
             "score, in original 2328x1748 pixels, one row per frame:\n\n")
    if doc:
        t.append("```json\n{\"file\": \"<session>_c<chunk>_<view>_t<t_ms>.jpg\", \"session\": \"...\",\n"
                 " \"chunk\": \"000\", \"frame_idx\": null, \"view\": \"lview\", \"t_ms\": 22233,\n"
                 " \"w\": 2328, \"h\": 1748, \"boxes\": [[x1, y1, x2, y2, score], ...]}\n```\n\n")
    else:
        t.append("```json\n{\"file\": \"<session>_c<chunk>_f<frame>.jpg\", \"session\": \"...\",\n"
                 " \"chunk\": \"000\", \"frame_idx\": 2760, \"w\": 2328, \"h\": 1748,\n"
                 " \"boxes\": [[x1, y1, x2, y2, score], ...]}\n```\n\n")
    t.append("Boxes are sorted by score descending, after NMS at 0.4. A different score cut, IoU\n"
             "gate or size floor is a filter over that file, not another inference pass:\n\n")
    ds_flag = f" --ds {ds}" if ds != "face_mine" else ""
    t.append("```sh\nmining/face_mine_relabel_v3.py build --out-dir .{} \\\n"
             "  --dump-left dets_left.jsonl --dump-right dets_right.jsonl \\\n"
             "  --score-cut {} --iou-gate {}\n```\n\n".format(
                 ds_flag, stats["score_cut"], stats["iou_gate"]))
    t.append("`build` is deterministic: the same dump and the same human files give byte-identical\n"
             "outputs. `mining/face_mine_relabel_v3.py check` re-asserts, over the whole output,\n"
             "that every frame appears once, that the frame set equals the source frame set, that\n"
             "the human box count matches the source file, and that no addition reaches the IoU\n"
             "gate against a human box of its frame.\n\n")
    t.append("## PII\n\n")
    t.append("The frames are unblurred faces of real people. Everything here stays on this box.\n")
    return "".join(t)


# ------------------------------------------------------------------------- check
def render_readme_model(stats: dict) -> str:
    """The README for a --prefer model build (PII-1272): rule, sources, counts."""
    L, R = stats["views"]["left"], stats["views"]["right"]
    m = stats["model"]
    t = []
    t.append(f"# {stats['ds']} re-label prelabels, {m} over human ({stats.get('issue', 'PII-1272')})\n\n")
    t.append(f"Prelabels for a vendor pass over both eyes of {stats['ds']}. Every frame carries\n"
             f"the {m} detections at score >= {stats['score_cut']} plus the human boxes no\n"
             "detection overlaps. Where a detection overlaps a human box the MODEL box is kept\n"
             "and the human box dropped (the user's call, PII-1270; PII-940 kept the human box).\n\n")
    t.append("## Model and rule\n\n")
    t.append(f"- Model: `{onnx('scrfd', m, m + '_det34g.onnx')}` (md5 {stats['model_md5']}),\n"
             f"  det_size {stats['det_size']}, padded canvas, decode `mining/rview_detect.py::SCRFD`\n"
             "  (box-for-box parity with `evaluation/eval_onnx.py::SCRFD`, PII-942, not re-run here).\n")
    t.append(f"- Dump: every box at score >= {stats['dump_score_thr']} in original pixels\n"
             "  (`dets_<view>.jsonl`, one row per frame, boxes `[x1, y1, x2, y2, score]` sorted by\n"
             "  score descending after NMS at 0.4).\n")
    t.append(f"- Merge, per frame, `--prefer model --score-cut {stats['score_cut']} "
             f"--iou-gate {stats['iou_gate']}`:\n"
             f"  a model box at score >= {stats['score_cut']} whose IoU (pixels, plain\n"
             f"  intersection over union) against any human box of the frame is >= {stats['iou_gate']}\n"
             "  REPLACES every human box it overlaps: those human boxes are dropped and the model\n"
             "  box is emitted once. A model box overlapping no human box is an ADDITION. A human\n"
             "  box no model box overlaps stays verbatim. No size floor, no clipping to the frame.\n")
    t.append("- Dedup (PII-1273): model boxes are taken in score order; one that overlaps a human\n"
             "  box already claimed by a higher-scoring emitted model box is DROPPED (not emitted,\n"
             "  not an addition), so each human box is replaced by at most one model box. A\n"
             "  dropped box claims nothing: a human box only it overlapped stays verbatim.\n")
    t.append("- Every box carries `source` (`human` or `" + m + "`) and `score` (null for human).\n"
             "  Box order: surviving human boxes in source order, then model boxes by score.\n"
             "  A frame with no box keeps `\"boxes\": []` (faceback_45 rule).\n\n")
    t.append("## Human box sources\n\n")
    t.append(f"- left: `{L['human_source']}`, the HUMAN labels of face_mine_v1 (the only full human\n"
             "  pass over the view). NOT `datasets/face_mine_v1/boxes.jsonl`, which is the miner's\n"
             "  machine output (PII-132, PII-335). NOT pii-data `boxes/v2.csv`: that relabel covers\n"
             "  3,600 eval frames only, is uncommitted in pii-data and PII-142 has not promoted it.\n")
    t.append(f"- right: `{R['human_source']}`, human (box_src `human`, PII-194).\n")
    t.append(f"- Images: `{L['images']}` and `{R['images']}`; the frame set is the image dir\n"
             "  listing, and the dump, the human file and the prelabels all hold it exactly.\n\n")
    t.append("## Files\n\n```\n"
             "dets_left.jsonl / dets_right.jsonl      full detection dump, one row per frame\n"
             "import_left.jsonl / import_right.jsonl  the prelabels, faceback_45 import shape\n"
             "stats.json                              every count below, machine readable\n"
             "render/                                 sampled frames: human green, model red,\n"
             "                                        replaced human dashed; render/index.txt\n"
             "changed_left.txt / changed_right.txt    frames with >= 1 replaced or added box\n"
             "                                        (the upload set), one image name per line\n"
             "*.predup                                the same build before dedup (PII-1272)\n"
             "```\n\n")
    t.append("`import_<view>.jsonl` rows are `{session, chunk, view, frame_idx, width, height,\n"
             "boxes, image}`; coordinates are normalised 0..1 top-left plus size at 4 decimals.\n\n")
    t.append("## Counts\n\n| | left | right |\n|---|---:|---:|\n")
    t.append(_row("Frames", lambda v: v["frames"], stats))
    t.append(_row("Human boxes in", lambda v: v["human_boxes"], stats))
    t.append(_row("Frames with a human box", lambda v: v["frames_with_human"], stats))
    t.append(_row(f"Detections at >= {stats['score_cut']}", lambda v: v["dets_at_cut"], stats))
    t.append(_row("Human boxes replaced (dropped)", lambda v: v["human_replaced"], stats))
    t.append(_row("Model boxes replacing a human box", lambda v: v["model_replacing"], stats))
    t.append(_row("Model boxes matched to more than one human box",
                  lambda v: v["model_matched_multi"], stats))
    t.append(_row("Human boxes hit by more than one candidate model box (before dedup)",
                  lambda v: v["human_hit_by_multi"], stats))
    t.append(_row("Model boxes dropped by dedup", lambda v: v["model_dropped_dedup"], stats))
    t.append(_row("Additions (model, no human overlap)", lambda v: v["additions"], stats))
    t.append(_row("Boxes out", lambda v: v["boxes_out"], stats))
    t.append(_row("Frames changed", lambda v: v["frames_changed"], stats))
    t.append(_row("Frames with an addition", lambda v: v["frames_with_addition"], stats))
    t.append(_row("Frames with a replacement", lambda v: v["frames_with_replacement"], stats))
    t.append(_row("Frames with no box (before)", lambda v: v["frames_with_no_box_before"], stats))
    t.append(_row("Frames with no box (after)", lambda v: v["frames_with_no_box"], stats))
    t.append(_row("Model boxes reaching outside the frame", lambda v: v["model_out_of_frame"], stats))
    t.append("\n### Model boxes by score\n\n| band | additions L | additions R | replacing L | replacing R |\n"
             "|---|---:|---:|---:|---:|\n")
    for k in L["score_bands"]:
        a, b = (float(v) for v in k.split("-"))
        lbl = f"{a}+" if b > 1 else f"{a}-{b}"
        t.append(f"| {lbl} | {L['score_bands'][k]:,} | {R['score_bands'][k]:,} | "
                 f"{L['score_bands_replacing'][k]:,} | {R['score_bands_replacing'][k]:,} |\n")
    t.append("\n### Additions by long side (px)\n\n| band | left | right |\n|---|---:|---:|\n")
    for a, b in SIDE_BANDS:
        k = f"{a}-{b}"
        lbl = f"< {b}" if a == 0 else (f"{a}+" if b > 1e8 else f"{a}-{b}")
        t.append(f"| {lbl} | {L['side_bands'][k]:,} | {R['side_bands'][k]:,} |\n")
    t.append("\n## Re-cutting\n\n")
    t.append("A different score cut or IoU gate is one `build --prefer model` call over the same\n"
             "dumps (no inference); `build` without `--prefer` gives the PII-940 shape (human\n"
             "boxes verbatim plus unmatched additions). `check --prefer model` re-verifies the\n"
             "output against the sources; `render --prefer model` draws sampled frames.\n")
    return "".join(t)


def check_view_model(view: str, args) -> int:
    """--prefer model checks, against the sources and independent of the builder's loop.

    Per frame: human boxes in the output are a verbatim, in-order subset of the
    source; every model box is a dump box at >= the cut, emitted at most once;
    every dump box at >= the cut that overlaps no human box is emitted (an
    addition); every dropped human box overlaps some emitted model box at
    IoU >= the gate; every kept human box overlaps no emitted model box at
    IoU >= the gate; no human box is overlapped by two emitted model boxes
    (dedup); every non-emitted dump box at >= the cut overlaps a human box
    that an emitted model box of >= its score also overlaps (its dedup
    justification); human boxes carry score null.
    """
    out_dir = Path(args.out_dir)
    cfg = DATASETS[args.ds][view]
    humans = load_human(view, args.ds)
    files = frame_set(view, args.ds, humans)
    by_import = {rec["import_name"]: f for f, rec in humans.items()}
    dump = load_dump(Path(getattr(args, f"dump_{view}")))
    rc = 0
    seen, n_hum_in, n_hum_out, n_model, n_add, n_repl_boxes = [], 0, 0, 0, 0, 0
    n_dedup = 0
    bad = dict(unknown=0, human_not_subset=0, human_score=0, model_unmatched=0,
               model_dup=0, addition_missing=0, dedup_unjustified=0, human_two_model=0,
               dropped_unjustified=0, kept_overlapped=0, below_cut=0, bad_source=0)
    for line in open(out_dir / f"import_{view}.jsonl"):
        r = json.loads(line)
        name = by_import.get(r["image"])
        if name is None:
            bad["unknown"] += 1
            rc = 1
            continue
        seen.append(name)
        W, H = r["width"], r["height"]
        hb = [b for b in r["boxes"] if b["source"] == "human"]
        mb = [b for b in r["boxes"] if b["source"] == args.model_name]
        if len(hb) + len(mb) != len(r["boxes"]):
            bad["bad_source"] += 1
        if any(b.get("score", 0) is not None for b in hb):
            bad["human_score"] += 1
        hsrc = humans[name]["boxes"]
        n_hum_in += len(hsrc)
        n_hum_out += len(hb)
        n_model += len(mb)
        # in-order subset: walk the source, match kept boxes greedily
        keys_out = [(b["x"], b["y"], b["w"], b["h"]) for b in hb]
        kept, i = [False] * len(hsrc), 0
        for j, b in enumerate(hsrc):
            if i < len(keys_out) and keys_out[i] == (b["x"], b["y"], b["w"], b["h"]):
                kept[j], i = True, i + 1
        if i != len(keys_out):
            bad["human_not_subset"] += 1
        # every dump box at the cut exactly once, keyed by its 4-decimal echo
        cand = {}
        for x1, y1, x2, y2, sc in dump[name]["boxes"]:
            if sc < args.score_cut:
                continue
            cand[(round(x1 / W, 4), round(y1 / H, 4), round((x2 - x1) / W, 4),
                  round((y2 - y1) / H, 4), sc)] = (x1, y1, x2, y2)
        got = [(b["x"], b["y"], b["w"], b["h"], b["score"]) for b in mb]
        if any(b["score"] < args.score_cut for b in mb):
            bad["below_cut"] += 1
        if any(k not in cand for k in got):
            bad["model_unmatched"] += 1
        if len(set(got)) != len(got):
            bad["model_dup"] += 1
        mpix = np.array([cand[k] for k in got if k in cand], dtype=float).reshape(-1, 4)
        hp_all = np.array([[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W,
                            (b["y"] + b["h"]) * H] for b in hsrc], dtype=float).reshape(-1, 4)
        # IoU of every emitted model box against every human box, (M, N)
        if len(mpix) and len(hp_all):
            ix1 = np.maximum(mpix[:, None, 0], hp_all[None, :, 0])
            iy1 = np.maximum(mpix[:, None, 1], hp_all[None, :, 1])
            ix2 = np.minimum(mpix[:, None, 2], hp_all[None, :, 2])
            iy2 = np.minimum(mpix[:, None, 3], hp_all[None, :, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            am = (mpix[:, 2] - mpix[:, 0]) * (mpix[:, 3] - mpix[:, 1])
            ah = (hp_all[:, 2] - hp_all[:, 0]) * (hp_all[:, 3] - hp_all[:, 1])
            mh = inter / np.maximum(am[:, None] + ah[None, :] - inter, 1e-9) >= args.iou_gate
        else:
            mh = np.zeros((len(mpix), len(hp_all)), dtype=bool)
        if len(hp_all) and np.any(mh.sum(axis=0) > 1):
            bad["human_two_model"] += 1
        for j, b in enumerate(hsrc):
            v = bool(mh[:, j].any()) if len(mpix) else False
            if kept[j] and v:
                bad["kept_overlapped"] += 1
            if not kept[j] and not v:
                bad["dropped_unjustified"] += 1
        # non-emitted dump boxes: must be dedup drops, never additions
        got_set = set(got)
        emitted_score_per_human = [max([k[4] for k, row in zip(got, mh) if row[j]] or [-1.0])
                                   for j in range(len(hsrc))]
        for k, src in cand.items():
            if k in got_set:
                continue
            n_dedup += 1
            hits = []
            if len(hp_all):
                ix1 = np.maximum(src[0], hp_all[:, 0]); iy1 = np.maximum(src[1], hp_all[:, 1])
                ix2 = np.minimum(src[2], hp_all[:, 2]); iy2 = np.minimum(src[3], hp_all[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                a = (src[2] - src[0]) * (src[3] - src[1])
                ious = inter / np.maximum(a + ah - inter, 1e-9)
                hits = [int(i) for i in np.where(ious >= args.iou_gate)[0]]
            if not hits:
                bad["addition_missing"] += 1
            elif not any(emitted_score_per_human[i] >= k[4] for i in hits):
                bad["dedup_unjustified"] += 1
        # additions vs replacements, recounted here for the report
        for k in got:
            if k in cand:
                src = cand[k]
                hp_all = np.array([[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W,
                                    (b["y"] + b["h"]) * H] for b in hsrc],
                                  dtype=float).reshape(-1, 4)
                if iou_max(src, hp_all) >= args.iou_gate:
                    n_repl_boxes += 1
                else:
                    n_add += 1
    src_human = sum(len(v["boxes"]) for v in humans.values())
    ok_frames = sorted(seen) == files and len(seen) == len(files)
    print(f"{view}: frames {len(seen)} (set==source, each once: {ok_frames}); "
          f"human in {src_human} (source {n_hum_in}), out {n_hum_out}, "
          f"dropped {n_hum_in - n_hum_out}; model {n_model} = {n_repl_boxes} replacing + "
          f"{n_add} additions; dump boxes at cut not emitted (dedup) {n_dedup}; "
          f"violations {bad}")
    if not ok_frames or src_human != n_hum_in or any(bad.values()):
        rc = 1
    return rc


def stage_check(args) -> None:
    """Structural checks over the built import files, against the sources.

    The IoU assertion is made on the DUMP's unrounded pixel corners (the values
    the gate was applied to), not on the 4-decimal normalised echo in the import
    file: each addition is matched back to its dump box first.
    """
    out_dir = Path(args.out_dir)
    rc = 0
    if args.prefer == "model":
        for view in ("left", "right"):
            rc |= check_view_model(view, args)
        print("CHECKS PASS" if rc == 0 else "CHECKS FAIL")
        sys.exit(rc)
    for view, dump_path in (("left", args.dump_left), ("right", args.dump_right)):
        cfg = DATASETS[args.ds][view]
        humans = load_human(view, args.ds)
        files = frame_set(view, args.ds, humans)
        by_import = {rec["import_name"]: f for f, rec in humans.items()}
        dump = load_dump(Path(dump_path))
        seen, n_human, n_add, bad_iou, unmatched = [], 0, 0, 0, 0
        worst_iou = 0.0
        unknown = 0
        for line in open(out_dir / f"import_{view}.jsonl"):
            r = json.loads(line)
            name = by_import.get(r["image"])
            if name is None:
                unknown += 1
                print(f"FAIL {view}: prelabel name {r['image']} has no human record")
                rc = 1
                continue
            seen.append(name)
            W, H = r["width"], r["height"]
            hb = [b for b in r["boxes"] if b["source"] == "human"]
            ab = [b for b in r["boxes"] if b["source"] == args.model_name]
            n_human += len(hb)
            n_add += len(ab)
            hsrc = humans[name]["boxes"]
            if [(b["x"], b["y"], b["w"], b["h"]) for b in hb] != \
               [(b["x"], b["y"], b["w"], b["h"]) for b in hsrc]:
                print(f"FAIL {view} {name}: human boxes not verbatim")
                rc = 1
            hpix = np.array([[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W,
                              (b["y"] + b["h"]) * H] for b in hb], dtype=float).reshape(-1, 4)
            # dump boxes at the cut, keyed by their normalised 4-decimal echo
            cand = {}
            for x1, y1, x2, y2, sc in dump[name]["boxes"]:
                if sc < args.score_cut:
                    continue
                cand[(round(x1 / W, 4), round(y1 / H, 4), round((x2 - x1) / W, 4),
                      round((y2 - y1) / H, 4), sc)] = (x1, y1, x2, y2)
            for b in ab:
                if b["score"] < args.score_cut:
                    print(f"FAIL {view} {name}: addition below the score cut")
                    rc = 1
                key = (b["x"], b["y"], b["w"], b["h"], b["score"])
                src = cand.get(key)
                if src is None:
                    unmatched += 1
                    continue
                v = iou_max(src, hpix)
                worst_iou = max(worst_iou, v)
                if v >= args.iou_gate:
                    bad_iou += 1
        src_human = sum(len(v["boxes"]) for v in humans.values())
        ok_frames = sorted(seen) == files and len(seen) == len(files)
        print(f"{view}: frames {len(seen)} (set==source, each once: {ok_frames}, "
              f"unknown names {unknown}); "
              f"human {n_human} vs source {src_human} ({n_human == src_human}); "
              f"additions {n_add}; unmatched-to-dump {unmatched}; "
              f"IoU >= {args.iou_gate}: {bad_iou}; worst IoU {worst_iou:.4f}")
        if not ok_frames or n_human != src_human or bad_iou or unmatched or unknown:
            rc = 1
    print("CHECKS PASS" if rc == 0 else "CHECKS FAIL")
    sys.exit(rc)


# ------------------------------------------------------------------------ render
def stage_render(args) -> None:
    import cv2
    out_dir = Path(args.out_dir)
    dst = Path(args.render_dir)
    dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    model_name = args.model_name
    index = []

    def dashed(img, p, col, thick=3, dash=18):
        """A dashed rectangle: the human boxes a model box replaced (--prefer model)."""
        x1, y1, x2, y2 = p
        for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
            n = max(1, int(np.hypot(bx - ax, by - ay) // dash))
            for i in range(0, n, 2):
                sx, sy = ax + (bx - ax) * i / n, ay + (by - ay) * i / n
                ex, ey = ax + (bx - ax) * (i + 1) / n, ay + (by - ay) * (i + 1) / n
                cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), col, thick)

    for view in ("left", "right"):
        cfg = DATASETS[args.ds][view]
        humans = load_human(view, args.ds)
        by_import = {rec["import_name"]: f for f, rec in humans.items()}
        cand = []
        for line in open(out_dir / f"import_{view}.jsonl"):
            r = json.loads(line)
            if any(b["source"] == model_name for b in r["boxes"]):
                cand.append(r)
        picks = rng.sample(cand, args.n)
        for r in picks:
            # the prelabel name is not always the frame file name (faceight_c)
            name = by_import[r["image"]]
            img = cv2.imread(str(cfg["images"] / name), cv2.IMREAD_COLOR)
            W, H = r["width"], r["height"]
            kept = {(b["x"], b["y"], b["w"], b["h"]) for b in r["boxes"] if b["source"] == "human"}
            replaced = [b for b in humans[name]["boxes"]
                        if (b["x"], b["y"], b["w"], b["h"]) not in kept]
            for b in replaced:
                p = (int(b["x"] * W), int(b["y"] * H),
                     int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H))
                dashed(img, p, (0, 255, 0))
            for b in r["boxes"]:
                p = (int(b["x"] * W), int(b["y"] * H),
                     int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H))
                col = (0, 255, 0) if b["source"] == "human" else (0, 0, 255)
                cv2.rectangle(img, p[:2], p[2:], col, 3)
                if b["source"] != "human":
                    cv2.putText(img, f"{b['score']:.2f}", (p[0], max(0, p[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            cv2.imwrite(str(dst / f"{view}_{r['image']}"), img)
            if args.prefer == "model":
                mb = [b for b in r["boxes"] if b["source"] != "human"]
                index.append(f"{view}_{r['image']}: human kept {len(kept)}, human replaced "
                             f"{len(replaced)} {[(b['x'], b['y'], b['w'], b['h']) for b in replaced]}, "
                             f"model {len(mb)} scores {[b['score'] for b in mb]}\n")
        print(f"{view}: rendered {len(picks)} of {len(cand)} frames with a model box", flush=True)
    if args.prefer == "model":
        with open(dst / "index.txt", "w") as fh:
            fh.write("solid green: human box kept; dashed green: human box replaced (dropped); "
                     "red: model box with score\n")
            fh.writelines(index)


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    d = sub.add_parser("detect")
    d.add_argument("--model", required=True)
    d.add_argument("--frames", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--list", help="text file of frame basenames; default is the whole dir")
    d.add_argument("--shard")
    d.add_argument("--det-size", type=int, default=1024)
    d.add_argument("--score-thr", type=float, default=0.1)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--limit", type=int, default=0)
    d.set_defaults(fn=stage_detect)

    m = sub.add_parser("merge")
    m.add_argument("--shards", required=True, help="glob, quoted")
    m.add_argument("--frames", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--list", help="text file of frame basenames; default is the whole dir")
    m.set_defaults(fn=stage_merge)

    for name, fn in (("build", stage_build), ("check", stage_check)):
        b = sub.add_parser(name)
        b.add_argument("--out-dir", required=True)
        b.add_argument("--dump-left", required=True)
        b.add_argument("--dump-right", required=True)
        b.add_argument("--score-cut", type=float, default=0.4)
        b.add_argument("--iou-gate", type=float, default=0.1)
        b.add_argument("--det-size", type=int, default=1024)
        b.add_argument("--score-thr", type=float, default=0.1)
        b.add_argument("--ds", default="face_mine", choices=sorted(DATASETS))
        b.add_argument("--model-name", default=MODEL_NAME)
        b.add_argument("--model-md5", default="3fa016c5dc7241e448ec101f4234e738")
        b.add_argument("--prefer", default="human", choices=("human", "model"),
                       help="human: PII-940 rule, human boxes verbatim plus unmatched "
                            "additions (default); model: a model box at the cut replaces "
                            "the human boxes it overlaps at IoU >= the gate (PII-1272)")
        b.add_argument("--issue", default="PII-1272", help="README heading for --prefer model")
        b.set_defaults(fn=fn)

    r = sub.add_parser("render")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--render-dir", required=True)
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--seed", type=int, default=940)
    r.add_argument("--ds", default="face_mine", choices=sorted(DATASETS))
    r.add_argument("--model-name", default=MODEL_NAME)
    r.add_argument("--prefer", default="human", choices=("human", "model"))
    r.set_defaults(fn=stage_render)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

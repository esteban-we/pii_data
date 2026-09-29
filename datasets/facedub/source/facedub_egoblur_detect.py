#!/usr/bin/env python
"""facedub prelabels: egoblurC_n8_fluence over a list of frame paths, one JSONL record per frame.

PII-1490. This is .knuth/tmp/pii1473/egoblur/eval_pii2_d2.py (PII-1475) with the ground-truth
half removed: the frame list comes from a text file instead of face_mine_match_dump.load_pii2,
there are no annotations, no eval accumulators and no results JSON, and every detection is
written to a JSONL record in the faceight schema instead of a det cache.

Everything that touches the model is imported unchanged from evaluation/egoblur_d2/train_egoblur:
EvalFrames (cv2 resize so the longest side is RESIZE_LONGEST, no padding of our own), score_shard
(model.inference(do_postprocess=False), boxes divided back to frame pixels, torchvision NMS 0.3,
cap MAX_DETS 100, score descending), and the cfg/build_model/DetectionCheckpointer sequence of
eval_d2.main(). --eval-longest sets train_egoblur.RESIZE_LONGEST before score_shard, as PII-1291's
eval_native.py and PII-1475 did (2328 for egoblurC_n8_fluence); the repo file is not edited.

Records (one per frame, boxes in stored-image pixels, the 1024x1280 rotated wrist frames):
  episode_id (= session_id: a wrist session has no episode id), session_id, chunk, view
  (uvc_left / uvc_right), s, t_ms, frame_idx (from fetch.jsonl), w, h, rotation_deg, file,
  model, n_faces (boxes >= --face-thr, 0.4 = the facedub prelabel threshold), boxes
  [[x1, y1, x2, y2, score]] with score >= --score-thr (0.1), so the dump can be re-thresholded.

Records are written as the frames are scored (the recording hook fires once per frame, in list
order), so the run is resumable: files already in --out are skipped. --shard K/N takes the frames
with index % N == K of the full list, for one process per GPU; concatenate the shards afterwards.

usage:
  CUDA_VISIBLE_DEVICES=2 /data/esteban/pii_venv/egoblur/bin/python \
      mining/facedub_egoblur_detect.py --frames-list frames.txt --fetch-jsonl fetch.jsonl \
      --config-file CFG.yaml --weights CKPT.pth --model-name egoblurC_n8_fluence \
      --eval-longest 2328 --out facedub_egoblurC.jsonl [--shard 0/2] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

STORE = Path(__file__).resolve().parents[3]   # the pii_data checkout (PII-1639)
sys.path.insert(0, str(STORE / "data"))
from pii_root import CODE  # noqa: E402  (the d2 eval code stayed in the pii repo)
sys.path.insert(0, str(CODE / "evaluation" / "egoblur_d2"))
import train_egoblur as te  # noqa: E402  (imports eval_onnx, d2_dataset; no work-dir side effects)

import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.utils import comm  # noqa: E402

NAME_RE = re.compile(r"^(?P<session>\d{8}_\d{6}_[A-Z]+)_c(?P<chunk>\d{3})_w(?P<hand>left|right)"
                     r"_t(?P<t_ms>\d+)\.jpg$")
PHASE_MS = 233
ROT_CW = 270


def file_md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_shard(text: str) -> tuple[int, int]:
    k, n = (int(x) for x in text.split("/"))
    if n < 1 or not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= K < N")
    return k, n


def meta_of(path: str) -> dict:
    """session_id, chunk, hand, view, s, t_ms from the file name written by facedub_pull.py."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        raise SystemExit(f"unexpected frame name: {path}")
    t_ms = int(m.group("t_ms"))
    return {"session_id": m.group("session"), "chunk": int(m.group("chunk")),
            "view": f"uvc_{m.group('hand')}", "s": (t_ms - PHASE_MS) // 1000, "t_ms": t_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-list", required=True, help="text file, one absolute frame path per line")
    ap.add_argument("--fetch-jsonl", default=None, help="facedub fetch.jsonl, for frame_idx")
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-md5", default=None, help="refuse to run unless the checkpoint matches")
    ap.add_argument("--model-name", default="egoblurC_n8_fluence")
    ap.add_argument("--out", required=True, help="JSONL, appended (resumable)")
    ap.add_argument("--meta-out", default=None, help="default <out>.meta.json")
    ap.add_argument("--eval-longest", type=int, required=True)
    ap.add_argument("--score-thr", type=float, default=0.1, help="boxes kept in the record")
    ap.add_argument("--face-thr", type=float, default=0.4, help="threshold for n_faces (prelabels)")
    ap.add_argument("--expect-size", default="1024x1280", help="asserted on every frame; '' to skip")
    ap.add_argument("--shard", default=None, metavar="K/N", type=parse_shard)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert comm.get_world_size() == 1 and comm.get_rank() == 0

    if a.weights_md5:
        got = file_md5(a.weights)
        if not got.startswith(a.weights_md5):
            raise SystemExit(f"{a.weights}: md5 {got}, expected {a.weights_md5}")
        print(f"checkpoint md5 {got} (expected {a.weights_md5})", flush=True)

    files = [l.strip() for l in open(a.frames_list) if l.strip()]
    n_all = len(files)
    if a.shard:
        k, n = a.shard
        files = files[k::n]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["file"])
                except (ValueError, KeyError):
                    pass
    todo = [f for f in files if os.path.basename(f) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{n_all} frames in {a.frames_list}"
          + (f"; shard {a.shard[0]}/{a.shard[1]}: {len(files)}" if a.shard else "")
          + f"; {len(done)} already in {a.out}; {len(todo)} to do", flush=True)
    if not todo:
        return

    frame_idx = {}
    if a.fetch_jsonl and os.path.exists(a.fetch_jsonl):
        with open(a.fetch_jsonl) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                frame_idx[r["file"]] = r.get("frame_idx")
        print(f"frame_idx from {a.fetch_jsonl}: {len(frame_idx)} frames", flush=True)

    exp = tuple(int(x) for x in a.expect_size.split("x")) if a.expect_size else None
    if exp:
        t0 = time.time()
        bad = []
        for p in todo:
            with Image.open(p) as im:
                if im.size != exp:
                    bad.append((p, im.size))
        if bad:
            raise SystemExit(f"{len(bad)} frames are not {exp}, e.g. {bad[:3]}")
        print(f"size check: all {len(todo)} frames {exp[0]}x{exp[1]} ({time.time() - t0:.0f}s)", flush=True)

    recs = [{"file_name": p, "width": exp[0] if exp else 0, "height": exp[1] if exp else 0,
             "image_id": i, "annotations": []} for i, p in enumerate(todo)]

    # ---- the model, exactly as eval_d2.main() / eval_pii2_d2.py build it --------------------
    assert te.RESIZE_LONGEST == 1200, te.RESIZE_LONGEST
    te.RESIZE_LONGEST = a.eval_longest
    print(f"RESIZE_LONGEST 1200 -> {te.RESIZE_LONGEST}, NMS {te.NMS_IOU}, max_dets {te.MAX_DETS}", flush=True)
    cfg = get_cfg()
    cfg.merge_from_file(a.config_file)
    cfg.MODEL.WEIGHTS = a.weights
    cfg.OUTPUT_DIR = str(Path(a.out).resolve().parent)   # never the training work dir
    cfg.freeze()
    model = build_model(cfg)
    rest = DetectionCheckpointer(model).load(a.weights)
    print("checkpoint load:", a.weights, "extra fields", sorted(rest), flush=True)
    model.eval()

    # ---- inference; the accumulate hook fires once per frame, in list order ------------------
    out = open(a.out, "a")
    real_accumulate = te.accumulate
    state = {"i": 0, "boxes": 0, "kept": 0, "hi": 0, "frames_hi": 0, "none": 0}
    t0 = time.time()

    def recording_accumulate(acc, det, gtb, ignb):
        i = state["i"]
        state["i"] += 1
        r = recs[i]
        name = os.path.basename(r["file_name"])
        if det is None:
            state["none"] += 1
        else:
            state["boxes"] += len(det)
            m = meta_of(name)
            boxes = [[round(float(x), 1) for x in d[:4]] + [round(float(d[4]), 4)]
                     for d in det if float(d[4]) >= a.score_thr]
            n_faces = sum(1 for b in boxes if b[4] >= a.face_thr)
            state["kept"] += len(boxes)
            state["hi"] += n_faces
            state["frames_hi"] += int(n_faces > 0)
            out.write(json.dumps({
                "episode_id": m["session_id"], "session_id": m["session_id"], "chunk": m["chunk"],
                "view": m["view"], "s": m["s"], "t_ms": m["t_ms"], "frame_idx": frame_idx.get(name),
                "w": r["width"], "h": r["height"], "rotation_deg": ROT_CW, "file": name,
                "model": a.model_name, "n_faces": n_faces, "boxes": boxes}) + "\n")
        if state["i"] % 200 == 0:
            out.flush()
            el = time.time() - t0
            print(f"detect {state['i']}/{len(recs)}  {state['i'] / el:.2f} fps  "
                  f"boxes/frame {state['kept'] / max(state['i'], 1):.2f} "
                  f"(>= {a.face_thr}: {state['hi'] / max(state['i'], 1):.2f})  "
                  f"eta {(len(recs) - state['i']) / (state['i'] / el) / 60:.1f} min", flush=True)
        return real_accumulate(acc, det, gtb, ignb)

    te.accumulate = recording_accumulate
    acc, st, n_items = te.score_shard(model, recs, batch_size=a.batch, num_workers=a.workers)
    dt = time.time() - t0
    out.flush()
    out.close()
    assert n_items == len(recs) == state["i"], (n_items, len(recs), state["i"])
    print(f"detect done: {n_items} frames, {state['none']} unreadable, {dt:.0f}s "
          f"({n_items / dt:.2f} fps), {st['n_dets_raw']} raw dets, {st['n_dets_after_nms']} after NMS, "
          f"{state['kept']} kept at >= {a.score_thr}, {state['hi']} at >= {a.face_thr} on "
          f"{state['frames_hi']} frames", flush=True)

    meta = {"model": a.model_name, "checkpoint": a.weights, "checkpoint_md5": file_md5(a.weights),
            "config_file": a.config_file, "config_md5": file_md5(a.config_file),
            "resize_longest": te.RESIZE_LONGEST, "nms_iou": te.NMS_IOU, "max_dets": te.MAX_DETS,
            "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, "score_thr": a.score_thr,
            "face_thr": a.face_thr, "expect_size": a.expect_size, "n_frames": n_items,
            "n_unreadable": acc["n_unreadable"], "n_dets_raw": st["n_dets_raw"],
            "n_dets_after_nms": st["n_dets_after_nms"], "n_boxes_kept": state["kept"],
            "seconds": round(dt, 1), "fps": round(n_items / dt, 3), "shard": a.shard,
            "frames_list": a.frames_list, "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "eval_path": "train_egoblur.score_shard (single process), frames from --frames-list",
            "script": str(Path(__file__).resolve()), "ts": time.strftime("%F %T")}
    json.dump(meta, open(a.meta_out or a.out + ".meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()

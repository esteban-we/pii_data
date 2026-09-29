"""Run an SCRFD-10G ONNX export over a directory of raw fisheye frames and dump boxes.

One JSONL line per frame: {"file", "session", "chunk", "frame_idx", "w", "h",
"boxes": [[x1, y1, x2, y2, score], ...]} in original pixel coordinates, all
detections with score >= --score-thr after NMS, sorted by score desc.
Resumable: frames already present in --out are skipped.

Decode is the bottleneck, not the GPU: JPEG decode + letterbox runs in a thread
pool (cv2 releases the GIL); the main thread owns the ORT session.

usage: rview_detect.py --model det_10g_armW.onnx --frames DIR --out boxes.jsonl
                       [--det-size 1024] [--score-thr 0.1] [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# pip-installed nvidia-* libs (cuDNN/cuBLAS) are not on the loader path; ORT >= 1.21 can find them.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

NMS_THR = 0.4
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
NAME_RE = re.compile(r"^(?P<session>\d{8}_\d{6}_[A-Z]+)_c(?P<chunk>\d+)_f(?P<frame>\d+)\.jpe?g$")


def nms(dets, thr):
    """dets: (N,5) x1,y1,x2,y2,score sorted desc. Standard InsightFace NMS."""
    x1, y1, x2, y2, s = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= thr]
    return keep


class SCRFD:
    def __init__(self, path: str, det_size: int, score_thr: float):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("CUDAExecutionProvider not active; refusing to run on CPU")
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        outs = self.sess.get_outputs()
        if len(outs) != 9:
            raise RuntimeError(f"{path}: expected 9 outputs, got {len(outs)}")
        self.out_names = [o.name for o in outs]
        # Layout: outputs 0-2 = score_8/16/32, 3-5 = bbox_8/16/32, 6-8 = kps_8/16/32.
        # Verify with names when they carry it, otherwise with static last dims.
        names = [n.lower() for n in self.out_names]
        if all(n.startswith(("score", "bbox", "kps")) for n in names):
            want = [f"{k}_{s}" for k in ("score", "bbox", "kps") for s in STRIDES]
            if names != want:
                raise RuntimeError(f"{path}: unexpected output order {names}")
        else:
            dims = [o.shape[-1] for o in outs]
            if dims != [1, 1, 1, 4, 4, 4, 10, 10, 10]:
                raise RuntimeError(f"{path}: unexpected output layout {list(zip(names, dims))}")
        self.det_size = det_size
        self.score_thr = score_thr
        self._anchor_cache: dict[tuple, np.ndarray] = {}

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._anchor_cache:
            ac = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32) * stride
            ac = ac.reshape(-1, 2)
            ac = np.stack([ac] * NUM_ANCHORS, axis=1).reshape(-1, 2)
            self._anchor_cache[key] = ac
        return self._anchor_cache[key]

    def preprocess(self, img_bgr: np.ndarray):
        """Letterbox to det_size square. Returns (canvas uint8, ratio). Runs in worker threads."""
        ih, iw = img_bgr.shape[:2]
        s = self.det_size
        ratio = s / max(ih, iw)
        nw, nh = int(round(iw * ratio)), int(round(ih * ratio))
        ratio = nw / iw
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, ratio

    def detect(self, canvas: np.ndarray, ratio: float) -> np.ndarray:
        """Returns (N,5) x1,y1,x2,y2,score in original image pixels."""
        s = self.det_size
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (s, s), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(self.out_names, {self.input_name: blob})
        boxes_l, scores_l = [], []
        for k, stride in enumerate(STRIDES):
            scores = np.asarray(outs[k]).reshape(-1)
            bbox = np.asarray(outs[k + 3]).reshape(-1, 4) * stride
            fh = fw = -(-s // stride)
            ac = self._anchors(fh, fw, stride)
            if ac.shape[0] != scores.shape[0]:
                raise RuntimeError(f"anchor/score count mismatch at stride {stride}")
            pos = np.where(scores >= self.score_thr)[0]
            if not len(pos):
                continue
            a, d = ac[pos], bbox[pos]
            boxes_l.append(np.stack([a[:, 0] - d[:, 0], a[:, 1] - d[:, 1],
                                     a[:, 0] + d[:, 2], a[:, 1] + d[:, 3]], axis=1))
            scores_l.append(scores[pos])
        if not boxes_l:
            return np.zeros((0, 5), np.float32)
        boxes = np.concatenate(boxes_l) / ratio
        scores = np.concatenate(scores_l)
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        dets = dets[dets[:, 4].argsort()[::-1]]
        return dets[nms(dets, NMS_THR)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-size", type=int, default=1024)
    ap.add_argument("--score-thr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cv2.setNumThreads(2)
    files = sorted(f for f in os.listdir(args.frames) if f.lower().endswith((".jpg", ".jpeg")))
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
    print(f"{len(files)} frames in {args.frames}; {len(done)} already done; {len(todo)} to do; "
          f"det_size {args.det_size}, score_thr {args.score_thr}", flush=True)
    if not todo:
        return

    model = SCRFD(args.model, args.det_size, args.score_thr)
    frames_dir = Path(args.frames)

    def load(name: str):
        img = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            return name, None, None, None
        canvas, ratio = model.preprocess(img)
        return name, canvas, ratio, img.shape[:2]

    n_ok = n_bad = n_boxes = 0
    t0 = time.time()
    with open(args.out, "a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        # bounded prefetch: keep ~4x workers in flight so RAM stays flat
        window = args.workers * 4
        futs = [pool.submit(load, f) for f in todo[:window]]
        nxt = window
        i = 0
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
            m = NAME_RE.match(name)
            rec = {
                "file": name,
                "session": m.group("session") if m else None,
                "chunk": m.group("chunk") if m else None,
                "frame_idx": int(m.group("frame")) if m else None,
                "w": int(hw[1]), "h": int(hw[0]),
                "boxes": [[round(float(x), 1) for x in d[:4]] + [round(float(d[4]), 4)] for d in dets],
            }
            out.write(json.dumps(rec) + "\n")
            n_ok += 1
            n_boxes += len(dets)
            if n_ok % 500 == 0:
                out.flush()
                el = time.time() - t0
                print(f"{n_ok}/{len(todo)}  {n_ok / el:.1f} fps  boxes/frame {n_boxes / n_ok:.2f}  "
                      f"eta {(len(todo) - n_ok) / (n_ok / el) / 60:.1f} min", flush=True)
    el = time.time() - t0
    print(f"done: {n_ok} frames, {n_bad} unreadable, {n_boxes} boxes, {el / 60:.1f} min "
          f"({n_ok / max(el, 1e-9):.1f} fps)", flush=True)


if __name__ == "__main__":
    main()

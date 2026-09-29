"""Dump the GT frames to jpg so the evaluation needs no video decoder.

torchcodec lives in the face_pii venv, not the training one, and adding it there could
drag torch with it -- the training environment took long enough to get right. Extracting
once under the venv that already has the decoder keeps the two apart.

Emits gt_eval.json: [{path, boxes}] with boxes in raw 2328x1748 coordinates, which is
the space the model outputs in, so the evaluation maps nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torchcodec.decoders import VideoDecoder

GT_BUNDLE = "/data/liangzhenghao/tmp/gt_bundle.json"
GT_MAP = "/data/liangzhenghao/tmp/nb44_chunks.json"
FPV = Path("/data/liangzhenghao/face_pii/pii_test/r2_base/fpv")
OUT = Path("/data/liangzhenghao/face_pii/gt_eval_frames")
PER_CHUNK = 120
MIN_SIDE = 40.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = json.load(open(GT_BUNDLE))
    gmap = json.load(open(GT_MAP))
    items = []
    for uuid, (sess, cc) in sorted(gmap.items()):
        ann = gt.get(uuid)
        if not ann or not ann["frames"]:
            print(f"[skip] {sess}: no annotations", flush=True)
            continue
        vid = FPV / sess / f"chunk_{cc:03d}" / "vst_left" / "vst_left_video.mp4"
        if not vid.exists():
            print(f"[skip] {sess}: no video", flush=True)
            continue
        frames = sorted(ann["frames"], key=lambda x: int(x))
        step = max(1, len(frames) // PER_CHUNK)
        pick = frames[::step][:PER_CHUNK]
        try:
            dec = VideoDecoder(str(vid))
            batch = dec.get_frames_at([int(f) for f in pick])
        except Exception as exc:                                  # noqa: BLE001
            print(f"[skip] {sess}: {type(exc).__name__}: {exc}", flush=True)
            continue
        n_kept = 0
        for k, f in enumerate(pick):
            arr = batch.data[k].permute(1, 2, 0).cpu().numpy()
            img = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)
            name = f"{sess}_{cc:03d}_f{int(f):06d}.jpg"
            cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            boxes = [b for b in ann["frames"][f]
                     if max(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE]
            items.append({"path": str(OUT / name), "boxes": boxes,
                          "session": sess, "chunk": cc})
            n_kept += len(boxes)
        print(f"[ok] {sess[-6:]} {len(pick)} frames, {n_kept} boxes >= {MIN_SIDE}px",
              flush=True)

    json.dump(items, open("/data/liangzhenghao/train/gt_eval.json", "w"))
    tot = sum(len(i["boxes"]) for i in items)
    print(f"\n{len(items)} frames, {tot} GT boxes -> gt_eval.json")


if __name__ == "__main__":
    main()

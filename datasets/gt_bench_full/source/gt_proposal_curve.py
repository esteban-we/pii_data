"""How many proposals does it take before a re-labeller stops having to hunt for faces by eye?

The GT bench was labelled in a hurry by one person and needs correcting. Three kinds of error exist
and they cost very different amounts to fix:

  a wrong box            -- cheap: it is on screen, judge it
  a missed face the model DOES find -- cheap: same, once it is proposed
  a missed face the model does NOT find -- expensive: nobody can fix it without scanning the frame

Only the third is real work, and the way to shrink it is to propose more, not to review more frames.
Judging a proposed box is seconds; finding an unboxed face in a fisheye factory frame is not. So the
question is where to put the proposal threshold: low enough that few real faces go unproposed, not so
low that the reviewer drowns in texture.

Measured against the 1,541 boxes already in the bench. Two caveats stated rather than buried: those
boxes are the very labels being questioned, so coverage of them is a proxy, not truth; and a proposal
that lands on no GT box is ambiguous -- it is either junk to reject or exactly the missed face we are
looking for. Both numbers are reported, never netted.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data/liangzhenghao/train")

T = Path("/data/liangzhenghao/train")
IOF_HIT = 0.5
BATCH = 8
KEEP_FLOOR = 0.02          # lowest armW score fetched; configs filter up from here


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return 0.0 if inter <= 0 else inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _iof(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return 0.0 if a <= 0 else (iw * ih) / a


CONFIGS = [
    ("现在的包", 0.25, True, False),
    ("armW≥0.25 全部", 0.25, False, False),
    ("armW≥0.15 全部", 0.15, False, False),
    ("armW≥0.10 全部", 0.10, False, False),
    ("armW≥0.05 全部", 0.05, False, False),
    ("armW≥0.10 + 头框", 0.10, False, True),
    ("armW≥0.05 + 头框", 0.05, False, True),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import cv2
    import torch
    from torchcodec.decoders import VideoDecoder
    from face_pii.detect import FaceDetector
    from head_trt import HeadDetector
    from we_io import R2Bucket

    armw = FaceDetector(str(T / "trt_out/scrfd_1024_fp16_dynb16_5090_armW.engine"),
                        det_thresh=0.02, nms_thresh=0.4, size=1024)
    head = HeadDetector(T / "trt_out/crowdhuman_640_fp16_dynb16_5090.engine")
    bucket = R2Bucket(str(T / "gt_curve_cache"), bucket="we-fpv-sh-ns",
                      storage_profile="aliyun_oss")

    gt = defaultdict(dict)
    for e in json.load(open(T / "gt_eval.json")):
        f = int(re.search(r"_f(\d+)\.jpg", e["path"]).group(1))
        gt[(e["session"], int(e["chunk"]))][f] = e["boxes"]

    frames_data = []            # (gt_boxes, armw_dets, head_boxes)
    for (sid, ck), frames in sorted(gt.items()):
        want = sorted(frames)
        key = f"{sid}/chunk_{ck:03d}/vst_left/vst_left_video.mp4"
        path = None
        dec = None
        try:
            path = Path(str(bucket.get(key)))
            dec = VideoDecoder(str(path), device="cuda:0")
            for i in range(0, len(want), BATCH):
                blk = want[i:i + BATCH]
                fr = dec.get_frames_at(blk)
                data = fr.data if hasattr(fr, "data") else fr
                face = armw.detect_batch(data)
                hds = head.heads(data)
                for j, f in enumerate(blk):
                    dets = [([float(x) for x in bb], float(ss))
                            for bb, ss in face[j] if ss >= KEEP_FLOOR]
                    frames_data.append((frames[f], dets, hds[j]))
                del data, fr
        finally:
            if dec is not None:
                del dec
            if path is not None:
                path.unlink(missing_ok=True)
            torch.cuda.empty_cache()
        print(f"  {sid[-6:]} c{ck:03d} {len(want)} 帧", flush=True)

    n_gt = sum(len(g) for g, _d, _h in frames_data)
    print(f"\n左目 {len(frames_data)} 帧，人工框 {n_gt}\n")
    print(f"{'方案':20}{'提议框':>9}{'每帧':>7}{'覆盖人工框':>12}{'落在人工框外':>14}")
    for name, thr, need_head, add_head in CONFIGS:
        total = hit = extra = 0
        for gtb, dets, heads in frames_data:
            head_of = {}
            for hb in heads:
                m = [(b, s) for b, s in dets if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5]
                if m:
                    ab, _ = max(m, key=lambda x: x[1])
                    if long_side(hb) <= 3.0 * long_side(ab):
                        head_of[id(ab)] = hb
            props = []
            for b, s in dets:
                if s < thr:
                    continue
                if need_head and s < 0.75 and id(b) not in head_of:
                    continue
                props.append(b)
            if add_head:
                for hb in heads:
                    if not any(_iou(hb, p) >= 0.3 for p in props):
                        props.append(hb)
            total += len(props)
            for g in gtb:
                if any(_iof(g, p) >= IOF_HIT for p in props):
                    hit += 1
            for p in props:
                if not any(_iof(g, p) >= IOF_HIT for g in gtb):
                    extra += 1
        print(f"{name:20}{total:>9,}{total/len(frames_data):>7.1f}"
              f"{hit:>7,}/{n_gt} = {hit/n_gt*100:>4.1f}%{extra:>14,}")
    print("\n  「覆盖人工框」= 已有的人工框里有多少被提议框盖住 —— 越高，需要人眼找的越少")
    print("  「落在人工框外」= 需要人判断的新框，里面既有误报，也有当初漏标的真脸")


if __name__ == "__main__":
    main()

"""Steady-state marginal cost of a second detector, measured properly this time.

The first pass produced an impossible number: model one cost +0.250s and model two cost -0.002s.
With only 100 sampled frames (13 batches) that difference is one-time warmup -- CUDA graph capture,
allocator growth, kernel autotuning -- not steady state, so "the second model is free" was not
something the measurement could support.

Fixed three ways: warm every path before timing, use ~10x the frames, and use two genuinely
different engines (armW and the shipped one) rather than calling one engine twice, which could share
cached state and understate the cost.

The question this settles: if decode dominates, the two-model T1 criterion (head box AND armW in the
score band) can run in one pass over the corpus instead of two, which is the difference between one
sweep and two.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

T = Path("/data/liangzhenghao/train")
VID = Path("/data/liangzhenghao/face_pii/gt_videos")
ATLAS = Path("/data/liangzhenghao/code/atlas")
ENG_A = T / "trt_out/scrfd_1024_fp16_dynb16_5090_armW.engine"
ENG_B = ATLAS / "res/weights/face_pii/tensorrt/10_16_1_11/scrfd_1024_fp16_dynb16_5090.engine"
STRIDE, BATCH, REPS = 30, 8, 4
N_TOTAL = 11000          # a whole chunk's worth of traversal


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from torchcodec.decoders import VideoDecoder
    from face_pii.detect import FaceDetector

    clip = sorted(VID.rglob("vst_left_video.mp4"))[0]
    d0 = VideoDecoder(str(clip), device="cuda:0")
    n_total = min(N_TOTAL, d0.metadata.num_frames or N_TOTAL)
    idx = list(range(0, n_total, STRIDE))
    print(f"{clip.parts[-4]}  总帧 {n_total}  抽样 {len(idx)}（1/{STRIDE}）", flush=True)

    a = FaceDetector(str(ENG_A), det_thresh=0.1, nms_thresh=0.4, size=1024)
    b = FaceDetector(str(ENG_B), det_thresh=0.1, nms_thresh=0.4, size=1024)
    print("两个引擎已载入（armW + 生产）", flush=True)

    def batches(dec):
        for i in range(0, len(idx), BATCH):
            fr = dec.get_frames_at(idx[i:i + BATCH])
            yield fr.data if hasattr(fr, "data") else fr

    def run(mode):
        dec = VideoDecoder(str(clip), device="cuda:0")
        for fr in batches(dec):
            if mode >= 1:
                a.detect_batch(fr)
            if mode >= 2:
                b.detect_batch(fr)

    # warm every path before any timing
    for m in (0, 1, 2):
        run(m)
    torch.cuda.synchronize()
    print("预热完成\n", flush=True)

    res = {}
    for m, name in ((0, "只解抽样帧"), (1, "+armW"), (2, "+armW+生产引擎")):
        ts = []
        for _ in range(REPS):
            torch.cuda.synchronize()
            t0 = time.time()
            run(m)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        res[name] = ts
        print(f"{name:18} 中位 {st.median(ts):6.3f}s  "
              f"（{[round(x,3) for x in ts]}）  {n_total/st.median(ts):6.0f} 帧/s", flush=True)

    d, e, f = (st.median(res[k]) for k in ("只解抽样帧", "+armW", "+armW+生产引擎"))
    print(f"\n=== 稳态边际成本 ===")
    print(f"  解码      {d:.3f}s  ({d/f*100:.0f}% of 两模型总时)")
    print(f"  第 1 个模型 +{e-d:.3f}s  (+{(e/d-1)*100:.0f}%)")
    print(f"  第 2 个模型 +{f-e:.3f}s  (+{(f/e-1)*100:.0f}%)")
    per_chunk = f / n_total * 10113
    print(f"\n  两模型粗筛：{n_total/f:.0f} 帧/s → 每 chunk {per_chunk:.1f}s")
    print(f"  全库 1,927,457 个未检 chunk ≈ {1927457*per_chunk/3600:,.0f} shard·小时"
          f"（7 卡 {1927457*per_chunk/3600/7/24:.1f} 天）")
    json.dump({k: v for k, v in res.items()}, open(T / "nvdec_bench2.json", "w"))


if __name__ == "__main__":
    main()

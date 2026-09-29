"""Read the pilot's partial results: 1fps miss rate, per-scene density, and the real throughput.

Part A is the one that decides whether the coarse-to-fine plan is sound at all. Production has
already detected these chunks over every frame, so "did 1fps find a face where production did" is a
straight comparison against truth, not an estimate. A chunk where production found faces but 1fps
found none is a chunk the triage would have thrown away.

Part B gives density for the scene types the delivered-chunk prior could not cover (Home, Retail,
Storage are 5-11% covered), and the wall-clock per chunk, which the 611 shard-hour cost model assumed
rather than measured.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T = Path("/data/liangzhenghao/train")
OUT = T / "pilot_out"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    B = [json.load(open(p)) for p in OUT.glob("res_B_*.json")]
    A = [json.load(open(p)) for p in OUT.glob("res_A_*.json")]
    print(f"已完成 B {len(B)}/300   A {len(A)}/40\n")

    # ---------- C. throughput ----------
    all_r = B + A
    fps = [r["fps_effective"] for r in all_r if r.get("fps_effective")]
    secs = [r["t_decode_infer"] for r in all_r]
    frames = sum(r["frames"] for r in all_r)
    sampled = sum(r["sampled"] for r in all_r)
    print("=== C. 稀疏采样的真实成本 ===")
    print(f"  每 chunk 解码 {st.mean([r['frames'] for r in all_r]):.0f} 帧，"
          f"只推理 {st.mean([r['sampled'] for r in all_r]):.0f} 帧（1/{round(frames/sampled)}）")
    print(f"  实测 {st.median(fps):.0f} 帧/s（中位）  每 chunk {st.median(secs):.0f}s")
    print(f"  对照：生产 1,181 帧/s（nvdec 硬解 + 每帧都推）")
    print(f"  ⇒ 跳帧省掉 97% 的推理，但整体反而慢 {1181/st.median(fps):.0f}× —— 瓶颈全在软解")

    # ---------- B. density by scene ----------
    print("\n=== B. 各场景人脸密度（未检 chunk，1fps 口径）===")
    by = defaultdict(list)
    for r in B:
        by[r["scene"]].append(r)
    print(f"{'场景':14}{'chunk':>7}{'零人脸':>9}{'占比':>8}{'有脸帧/抽样帧':>14}{'高分框':>9}")
    tot_z = 0
    for s, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        z = sum(1 for r in rs if r["face_frames_1fps"] == 0)
        tot_z += z
        hit = sum(r["face_frames_1fps"] for r in rs)
        smp = sum(r["sampled"] for r in rs)
        print(f"{s:14}{len(rs):>7}{z:>9}{z/len(rs)*100:>7.0f}%"
              f"{hit}/{smp} = {hit/max(1,smp)*100:>5.1f}%{sum(r['boxes_hi'] for r in rs):>9}")
    print(f"{'合计':14}{len(B):>7}{tot_z:>9}{tot_z/max(1,len(B))*100:>7.0f}%")
    print(f"  对照：已交付 chunk 的先验是 52.8% 整段零人脸")

    # ---------- A. miss rate of 1fps ----------
    print("\n=== A. 1fps 漏判率（拿现网全帧结果当真值）===")
    try:
        idx = json.load(open(T / "pii_cache_index.json"))
    except Exception as e:
        print(f"  读不到缓存索引 {type(e).__name__}"); return
    dens = {}
    try:
        for r in json.load(open(T / "pii_density.json")):
            if "n_rows" in r:
                dens[r["did"]] = r
    except Exception:
        pass

    tp = fn = tn = fp_ = 0
    misses = []
    for r in A:
        did = r.get("did")
        truth = dens.get(did)
        if truth is None:
            continue
        has_truth = truth["face_frames"] > 0
        has_1fps = r["face_frames_1fps"] > 0
        if has_truth and has_1fps:
            tp += 1
        elif has_truth and not has_1fps:
            fn += 1
            misses.append((did, truth["face_frames"], truth["n_rows"]))
        elif not has_truth and not has_1fps:
            tn += 1
        else:
            fp_ += 1
    n = tp + fn + tn + fp_
    if not n:
        print("  A 组还没有能和密度样本对上的 chunk（两次抽样的 stride 不同）")
        print("  → 需要按 A 组的 data_id 重新读一次缓存才能算，见下一步")
        return
    print(f"  可比对 {n} 个 chunk")
    print(f"  现网有脸 & 1fps 也找到 : {tp}")
    print(f"  现网有脸 & 1fps 漏掉   : {fn}   ← 漏判率 {fn/max(1,tp+fn)*100:.1f}%")
    print(f"  现网无脸 & 1fps 也无   : {tn}")
    print(f"  现网无脸 & 1fps 有     : {fp_}")
    for did, ff, nr in misses[:8]:
        print(f"    漏: {did[-22:]}  现网有脸帧 {ff}/{nr} = {ff/nr*100:.2f}%")


if __name__ == "__main__":
    main()

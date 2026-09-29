"""Where the 14.6s per chunk goes, and what each lever would actually buy.

A full sweep costs whatever the continuation check costs, because that check is most of the run: it
re-decodes a +/-150 frame window at FULL rate around every frame that produced a candidate, while
the 1fps pass touches one frame in thirty. Dropping or narrowing it is the only large lever, and it
is not free -- it is what turns "armW scored this 0.6" into "production would not have blurred this",
which is the difference between a candidate and a known miss.

Reported as measured splits, plus what the corpus-wide cost becomes under each option, so the
trade is visible rather than asserted.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict

T = "/data/liangzhenghao/train"
GPUS = 7
REMAIN_EP = 174_459          # eligible episodes not yet mined (one chunk each)
REMAIN_CH = 1_899_347        # eligible chunks not yet mined (exhaustive)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parts = defaultdict(list)
    win_frac = []
    for p in glob.glob(f"{T}/mine_run/shard*/chunks/*.json"):
        r = json.load(open(p))
        if r.get("error") or not r.get("t_total"):
            continue
        parts["下载"].append(r.get("t_dl", 0))
        parts["双模型 1fps"].append(r.get("t_infer", 0))
        parts["续轨检验"].append(r.get("t_track", 0))
        parts["总"].append(r["t_total"])
        if r.get("frames"):
            win_frac.append(r.get("window_frames", 0) / r["frames"])

    tot = st.mean(parts["总"])
    print("=== 单 chunk 耗时构成（15,852 个实测均值）===")
    for k in ("下载", "双模型 1fps", "续轨检验"):
        m = st.mean(parts[k])
        print(f"  {k:12}{m:>7.2f}s  {m/tot*100:>5.1f}%")
    other = tot - sum(st.mean(parts[k]) for k in ("下载", "双模型 1fps", "续轨检验"))
    print(f"  {'其他/裁图':12}{other:>7.2f}s  {other/tot*100:>5.1f}%")
    print(f"  {'合计':12}{tot:>7.2f}s")
    print(f"\n  续轨窗口平均覆盖整段的 {st.mean(win_frac)*100:.0f}%"
          f"（中位 {st.median(win_frac)*100:.0f}%）—— 有脸的 chunk 里它几乎等于全片重解一遍")

    track = st.mean(parts["续轨检验"])
    print("\n=== 三个方案的全库成本（7 卡，按 14.6s/chunk 的场景加权）===")
    base = 14.6
    for name, per, note in (
            ("保持现状（±150 帧续轨检验）", base,
             "候选带「生产确实会漏」的判定"),
            ("窗口收窄到 ±45 帧", base - track * (1 - 45 / 150),
             "省掉约 2/3 续轨；远处高分帧连续续轨救回的脸会被误判成遗漏"),
            ("完全不做续轨检验", base - track,
             "只知道 armW 分数落在带内，不知道生产是否已打码 —— 昨晚 T1 有 47% 属于这种")):
        h_ep = REMAIN_EP * per / 3600 / GPUS
        h_ch = REMAIN_CH * per / 3600 / GPUS
        print(f"\n  {name}   {per:.1f}s/chunk")
        print(f"    A 每 session 一个 chunk（{REMAIN_EP:,}）: "
              f"{h_ep:,.0f} GPU-小时 = {h_ep/24:.1f} 天")
        print(f"    B 全部 chunk（{REMAIN_CH:,}）:            "
              f"{h_ch:,.0f} GPU-小时 = {h_ch/24:.1f} 天")
        print(f"    代价：{note}")


if __name__ == "__main__":
    main()

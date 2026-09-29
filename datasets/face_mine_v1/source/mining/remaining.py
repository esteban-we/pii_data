"""How much is left to mine, and what a full sweep would cost -- from measured rates, not estimates.

Two corrections the obvious answer gets wrong:

The pool that ran last night was a SAMPLE. episodes_sample.json holds 20,000 episodes drawn from a
corpus of 229,125, and pilot_pool.json filtered those down to the eligible ones. So "16,612 minus
15,852 = 760 left" is wrong by two orders of magnitude -- the eligible count has to be scaled back up
to the corpus before subtracting what was done.

And per-chunk cost is not one number. Factory chunks run ~4x longer than home chunks because the
continuation window only opens around frames that produced a candidate, and empty chunks skip it
almost entirely. Last night's mix was 48% factory; a sweep over ALL undetected chunks is ~64% factory
by chunk count, so projecting last night's average would understate it. Per-scene means are measured
here and re-weighted to whichever mix the question implies.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict

T = "/data/liangzhenghao/train"
GPUS = 7


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # ---- measured cost per chunk, by scene -------------------------------
    times = defaultdict(list)
    nch_by_scene = defaultdict(list)
    for p in glob.glob(f"{T}/mine_run/shard*/chunks/*.json"):
        r = json.load(open(p))
        if r.get("error") or not r.get("t_total"):
            continue
        times[r["scene"]].append(r["t_total"])
    print("=== 实测单 chunk 耗时（昨晚 15,852 个）===")
    print(f"{'场景':12}{'chunk':>8}{'中位s':>8}{'均值s':>8}")
    for k in sorted(times, key=lambda x: -len(times[x])):
        v = times[k]
        print(f"{k:12}{len(v):>8,}{st.median(v):>8.1f}{st.mean(v):>8.1f}")
    all_t = [x for v in times.values() for x in v]
    print(f"{'全体':12}{len(all_t):>8,}{st.median(all_t):>8.1f}{st.mean(all_t):>8.1f}")
    thr = GPUS * 3600 / st.mean(all_t)
    print(f"  昨晚配比下 {GPUS} 卡吞吐 ≈ {thr:,.0f} chunk/小时")

    # ---- how much is left ------------------------------------------------
    d = json.load(open(f"{T}/episodes_sample.json"))
    sampled, total_ep = d["sampled"], d["total_episodes"]
    pool = json.load(open(f"{T}/pilot_pool.json"))["pool"]
    frac_eligible = len(pool) / len(sampled)
    ep_eligible = frac_eligible * total_ep
    chunks_in_pool = sum(e.get("nch", 0) for e in pool)
    chunks_eligible = chunks_in_pool / len(sampled) * total_ep
    done = sum(1 for _ in glob.glob(f"{T}/mine_run/shard*/chunks/*.json"))

    print("\n=== 还剩多少 ===")
    print(f"  全库 episode                {total_ep:>12,}")
    print(f"  抽样 episode                {len(sampled):>12,}  （池子就是从这里筛的）")
    print(f"  抽样中合格（未检测/非评测台）  {len(pool):>12,} = {frac_eligible*100:.1f}%")
    print(f"  → 外推全库合格 episode       {ep_eligible:>12,.0f}")
    print(f"  → 外推全库合格 chunk         {chunks_eligible:>12,.0f}"
          f"  （每 episode 中位 {st.median([e['nch'] for e in pool]):.0f} 个 chunk）")
    print(f"  昨晚已挖 episode             {done:>12,}"
          f" = 合格 episode 的 {done/ep_eligible*100:.1f}%")
    print(f"  剩余 episode                {ep_eligible-done:>12,.0f}")
    print(f"  剩余 chunk                  {chunks_eligible-done:>12,.0f}")

    # ---- projections -----------------------------------------------------
    # scene shares differ between the two questions: one-chunk-per-episode follows the EPISODE mix,
    # an exhaustive sweep follows the CHUNK mix (factory sessions are longer, so factory weighs more)
    ep_mix, ch_mix = defaultdict(int), defaultdict(int)
    import re

    def norm_scene(s):
        raw = re.sub(r"[^a-z]", "", ((s or "").split("_")[3:4] or [""])[0].lower())
        for pat, name in [(r"factor|facatory|manufactur|electronicsfact|toyfact", "工厂"),
                          (r"home|livingroom", "家庭"),
                          (r"supermarket|fruitshop|petstore|retail|store", "零售/超市"),
                          (r"workbench|craft|draft|marquetry|mosaic|handcraft|studio", "手作/工位"),
                          (r"logistic|storage", "仓储物流"),
                          (r"health|hospital|beauty", "医疗/健康"),
                          (r"office|offic|school", "办公/学校"),
                          (r"restaurant|coffee|hotel|hotal|resort", "餐饮/酒店"),
                          (r"repair|mobilerepair|applianc", "维修"),
                          (r"outdoor", "户外")]:
            if re.search(pat, raw):
                return name
        return "其他/未标注"

    for e in pool:
        s = norm_scene(e["scene"])
        ep_mix[s] += 1
        ch_mix[s] += e.get("nch", 0)

    fallback = st.mean(all_t)

    def cost(mix, n_total):
        tot_w = sum(mix.values())
        secs = sum(mix[s] / tot_w * st.mean(times.get(s, [fallback])) for s in mix)
        return secs, n_total * secs / 3600 / GPUS

    print("\n=== 全量粗挖要多久（按实测速率）===")
    for label, mix, n in (
            (f"A 每 episode 取 1 个 chunk（剩 {ep_eligible-done:,.0f} 个）",
             ep_mix, ep_eligible - done),
            (f"B 全部 chunk 都扫（剩 {chunks_eligible-done:,.0f} 个）",
             ch_mix, chunks_eligible - done)):
        secs, hours = cost(mix, n)
        fac = mix.get("工厂", 0) / max(1, sum(mix.values())) * 100
        print(f"  {label}")
        print(f"     场景配比工厂 {fac:.0f}%   加权单 chunk {secs:.1f}s")
        for g in (7, 8):
            h = n * secs / 3600 / g
            print(f"     {g} 卡: {h:,.0f} GPU-小时 → {h/24:,.1f} 天"
                  f"（墙钟 {h*g/g:,.0f}h ≈ {h/24:.1f} 天）")


if __name__ == "__main__":
    main()

"""How many distinct faces is the candidate queue really, and how much does that depend on the gate?

The in-run clustering used continuation()'s own gate -- centre distance below the larger box's long
side -- and collapsed almost nothing (1.6 candidate frames per "face"). That is not credible for
head-mounted video sampled once a second: two people in view for a whole chunk cannot be 400 faces.
The gate is the problem, not the data. It was designed for frames 1/30s apart, where a face barely
moves; across a 1s gap on an FPV camera it moves several times its own width.

There is no principled value to substitute, so instead of picking one and quoting it as fact, sweep
it and show how the answer moves. What survives the sweep is the shape: which tier dominates, and
whether the queue is thousands or tens per chunk. Reads only the saved boxes -- no GPU, no re-run.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

T = Path("/data/liangzhenghao/train")
GATES = [1.0, 2.0, 3.0, 5.0, 8.0]        # multiples of the box long side
GAPS = [3, 10]                            # samples (x30 frames)


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def cdist(a, b):
    return (((a[0] + a[2]) - (b[0] + b[2])) ** 2
            + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2


def clusters(items, gate, gap_frames):
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for k in range(i + 1, len(items)):
            a, b = items[i], items[k]
            if abs(a["frame"] - b["frame"]) > gap_frames:
                continue
            if cdist(a["box"], b["box"]) < gate * max(long_side(a["box"]),
                                                      long_side(b["box"])):
                ra, rb = find(i), find(k)
                if ra != rb:
                    parent[ra] = rb
    g = defaultdict(list)
    for i in range(len(items)):
        g[find(i)].append(i)
    return list(g.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.load(open(T / "cand_mine/index.json"))
    rows = [r for r in d["rows"] if r.get("box")]
    n_chunk = len(d["per_chunk"])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["tier"]][(r["sid"], r["chunk"])].append(r)

    print(f"渲染出的遗漏候选 {len(rows)}（{n_chunk} 个 chunk）")
    print("\n=== 去重后的「面孔」数，随空间门限变化（只算 ≥40px 的）===")
    for gap in GAPS:
        print(f"\n  时间门限 ±{gap} 个抽样（{gap*30} 帧）")
        print(f"    {'空间门限':12}" + "".join(f"{t:>10}" for t in ("T1", "T2", "T3"))
              + f"{'T1+T2/chunk':>14}")
        for gate in GATES:
            cnt = {}
            for tier in ("T1", "T2", "T3"):
                tot = 0
                for items in by[tier].values():
                    for cl in clusters(items, gate, gap * 30):
                        if any(not items[i].get("small") for i in cl):
                            tot += 1
                cnt[tier] = tot
            print(f"    {gate:>4.0f}× 长边   " + "".join(f"{cnt[t]:>10}" for t in
                                                       ("T1", "T2", "T3"))
                  + f"{(cnt['T1']+cnt['T2'])/n_chunk:>14.1f}")
    print("\n  注：门限越松，同一张脸被并在一起的越多，数字越接近真实人数；")
    print("      但过松会把画面里不同的人并成一个，所以真值在中间某处，这里不硬取一个。")


if __name__ == "__main__":
    main()

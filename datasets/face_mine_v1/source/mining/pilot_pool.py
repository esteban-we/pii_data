"""Build the pilot's candidate pool, with every evaluation bench excluded, and price the I/O.

Two things must not leak into a mining pool that will become training data:

  the GT bench -- 10 hand-labelled sequences, the only honest measure we have of this detector;
  the complaint sequences -- the 71-frame bench, and evidence for an open customer complaint.

Excluded at SESSION level, not chunk level: a different chunk of the same session is the same scene,
the same people and the same lighting, so training on it contaminates the bench nearly as much.
face10k's source sessions are excluded too -- they are already training data, so re-mining them
spends annotation budget on frames the model has seen.

Also probes what actually exists in OSS for a sampled undetected chunk. The earlier cost model
counted inference only; at ~1GB of MCAP per chunk the download is what decides how big a pilot can
be, and that has not been measured.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/data/liangzhenghao/train")

T = "/data/liangzhenghao/train"


def norm_scene(s):
    p = (s or "").split("_")
    raw = re.sub(r"[^a-z]", "", (p[3] if len(p) > 3 else "").lower())
    rules = [(r"factor|facatory|manufactur|electronicsfact|toyfact", "工厂"),
             (r"home|livingroom", "家庭"),
             (r"supermarket|fruitshop|petstore|retail|store", "零售/超市"),
             (r"workbench|craft|draft|marquetry|mosaic|handcraft|studio", "手作/工位"),
             (r"logistic|storage", "仓储物流"),
             (r"health|hospital|beauty", "医疗/健康"),
             (r"office|offic|school", "办公/学校"),
             (r"restaurant|coffee|hotel|hotal|resort", "餐饮/酒店"),
             (r"repair|mobilerepair|applianc", "维修"),
             (r"outdoor", "户外")]
    for pat, name in rules:
        if re.search(pat, raw):
            return name
    return "其他/未标注" if raw else "(无 scene_id)"


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # ---- exclusions -------------------------------------------------------
    excl = {}
    gt_sess = set()
    for e in json.load(open(f"{T}/gt_eval.json")):
        gt_sess.add(e["session"])
    excl["GT 评测台"] = gt_sess

    try:
        comp = json.load(open(f"{T}/complaints.json"))
        excl["投诉序列"] = {c["session"] for c in comp}
    except Exception as e:
        print(f"  ⚠️ 读不到 complaints.json: {type(e).__name__}")
        excl["投诉序列"] = set()

    # the doll sequences are an eval bench too -- they are how over-blur is measured
    import os
    gtv = "/data/liangzhenghao/face_pii/gt_videos"
    if os.path.isdir(gtv):
        excl["GT 评测台"] |= set(os.listdir(gtv))

    f10k = set()
    try:
        for r in json.load(open("/data/liangzhenghao/face_pii/face10k_export/diff_index.json")):
            if r.get("session"):
                f10k.add(r["session"])
    except Exception as e:
        print(f"  ⚠️ face10k 索引读不到: {type(e).__name__}")
    for cand in ("train_W.txt", "train_V.txt"):
        try:
            for ln in open(f"/data/liangzhenghao/face_pii/mix_ds/{cand}"):
                m = re.search(r"(20\d{6}_\d{6}_[A-Z]{6})", ln)
                if m:
                    f10k.add(m.group(1))
        except FileNotFoundError:
            continue
    excl["face10k 训练来源"] = f10k

    all_excl = set().union(*excl.values())
    print("=== 排除名单（按 session）===")
    for k, v in excl.items():
        print(f"  {k:16} {len(v):>6} 个 session")
    print(f"  合并去重后        {len(all_excl):>6} 个 session")

    # ---- pool -------------------------------------------------------------
    d = json.load(open(f"{T}/episodes_sample.json"))
    eps, TOTAL = d["sampled"], d["total_episodes"]
    det = set(json.load(open(f"{T}/pii_sessions.json"))["sessions"])

    pool = [e for e in eps
            if e["sid"] not in det and e["sid"] not in all_excl]
    dropped_det = sum(1 for e in eps if e["sid"] in det)
    dropped_ex = sum(1 for e in eps if e["sid"] not in det and e["sid"] in all_excl)
    print(f"\n=== 候选池（来自 {len(eps):,} 个抽样 episode）===")
    print(f"  已检测，排除            {dropped_det:>6}")
    print(f"  命中评测台/训练集，排除  {dropped_ex:>6}")
    print(f"  可用                    {len(pool):>6}"
          f"   → 外推全库约 {len(pool)/len(eps)*TOTAL:,.0f} episode")

    by = Counter(norm_scene(e["scene"]) for e in pool)
    print("\n  按场景：")
    for k, v in by.most_common():
        print(f"    {k:14}{v:>6}")

    json.dump({"exclude_sessions": sorted(all_excl),
               "pool": pool,
               "excl_counts": {k: len(v) for k, v in excl.items()}},
              open(f"{T}/pilot_pool.json", "w"), ensure_ascii=False)
    print(f"\n写出 {T}/pilot_pool.json")

    # ---- what can we actually download for an undetected chunk? -----------
    print("\n=== 未检 chunk 在 OSS 里有什么（探 3 个）===")
    from we_io import R2Bucket
    b = R2Bucket("/tmp/probe_pool", bucket="atlas-fpv-data-25",
                 storage_profile="aliyun_oss")
    scan = json.load(open(f"{T}/eval100_scan.json"))
    any_run = next(iter(scan.values()))
    run_id = any_run[0].split("/")[1] if any_run else None
    print(f"  已知 run 的键形如: runs/<run_id>/<data_id>/...")
    print(f"  未检 chunk 不一定跑过 pipeline —— 若没有 run，就只能从采集桶 we-fpv-sh-ns 取原始件")
    hit = 0
    for e in pool[:3]:
        sid = e["sid"]
        found = []
        for pre in (f"{sid}/", f"{sid}_000/"):
            try:
                ks = list(b.list(pre))[:6]
                if ks:
                    found += ks
            except Exception as exc:
                found.append(f"[{type(exc).__name__}]")
        print(f"  {sid}: {found[:3] if found else '在 atlas-fpv-data-25 顶层查不到'}")
        hit += bool(found)


if __name__ == "__main__":
    main()

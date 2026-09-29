"""Phase 1 pilot, fetch half: build the job list and stage vst_left MP4s.

  A. Does 1fps triage actually find chunks that contain faces? Measured against already-detected
     chunks, where production's full-frame result is the truth. If sparse sampling misses chunks,
     the whole coarse-to-fine plan is unsound and no amount of throughput saves it.
  B. What is the real face density per scene type? The 52.8%-empty prior comes only from delivered
     chunks, and Home / Retail / Storage -- the biggest untouched buckets -- are 5-11% covered, so
     their density is currently a guess.
  C. What does sparse sampling actually cost? Skipping to every 30th frame saves inference but a
     seek still has to decode from the previous keyframe, so the saving may be near zero. Both
     strategies are timed on the same chunks.

Frames come from the capture bucket (we-fpv-sh-ns), which is where production's face_detect stages
them too: <session>/chunk_NNN/vst_left/vst_left_video.mp4, 360MB, plain MP4.

Every GT, complaint and face10k session is already excluded from pilot_pool.json -- those are the
evaluation benches, and a mined frame that becomes training data would destroy them.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zlib

sys.path.insert(0, "/data/liangzhenghao/train")

T = Path("/data/liangzhenghao/train")
CFG = "configs/scrfd/scrfd_10g_bnkps_ft2.py"
CKPT = T / "wd_armW/epoch_20.pth"
SCRATCH = Path("/data/liangzhenghao/train/pilot_cache")
OUT = T / "pilot_out"
STRIDE = 30                 # 1 fps at 30 fps capture
KEEP, DET_SIZE = 0.25, 1024
T_HIGH, MIN_LONG = 0.75, 40.0
N_PER_SCENE = {"工厂": 90, "家庭": 60, "零售/超市": 45, "手作/工位": 40,
               "仓储物流": 20, "医疗/健康": 18, "办公/学校": 12,
               "餐饮/酒店": 6, "维修": 5, "户外": 4}
N_TRUTH = 40                # already-detected chunks for the miss-rate check


def norm_scene(s):
    p = (s or "").split("_")
    raw = re.sub(r"[^a-z]", "", (p[3] if len(p) > 3 else "").lower())
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
    return "其他/未标注" if raw else "(无 scene_id)"


def long_side(b):
    return max(b[2] - b[0], b[3] - b[1])


def set_scale(m, s):
    aug = m.cfg.data.test.pipeline[1]
    aug["img_scale"] = (s, s)
    for t in aug["transforms"]:
        if t["type"] == "Pad":
            t["size"] = (s, s)


def pick_chunks():
    """Stratified chunk list, deterministic, benches already excluded upstream."""
    pool = json.load(open(T / "pilot_pool.json"))["pool"]
    by = defaultdict(list)
    for e in pool:
        if e.get("nch"):
            by[norm_scene(e["scene"])].append(e)
    out = []
    for scene, quota in N_PER_SCENE.items():
        eps = sorted(by.get(scene, []), key=lambda x: x["sid"])
        if not eps:
            continue
        step = max(1, len(eps) // quota)
        for e in eps[::step][:quota]:
            # spread across the session rather than always chunk_000
            ci = zlib.crc32(e["sid"].encode()) % max(1, e["nch"])
            out.append({"sid": e["sid"], "chunk": ci, "scene": scene,
                        "prov": e.get("prov"), "hw": e.get("hw")})
    return out


def truth_chunks():
    """Already-detected chunks: production's full-frame boxes are the ground truth for part A."""
    idx = json.load(open(T / "pii_cache_index.json"))
    ids = sorted(idx)
    step = max(1, len(ids) // N_TRUTH)
    out = []
    for did in ids[::step][:N_TRUTH]:
        m = re.match(r"(20\d{6}_\d{6}_[A-Z]{6})_(\d{3})_", did)
        if m:
            out.append({"did": did, "sid": m.group(1), "chunk": int(m.group(2)),
                        "fp": idx[did][0]})
    return out




def main() -> None:
    """Download half. Runs in fetch_env (scrfd_env has no requests, so no oss2 there).

    Writes <tag>.mp4 then <tag>.ready, so an inference worker never opens a partial file.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    from we_io import R2Bucket
    mode = sys.argv[1] if len(sys.argv) > 1 else "B"
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    SCRATCH.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

    jobs = pick_chunks() if mode == "B" else truth_chunks()
    json.dump(jobs, open(OUT / f"jobs_{mode}.json", "w"), ensure_ascii=False)
    mine = [j for i, j in enumerate(jobs) if i % nshard == shard]
    print(f"fetch {mode} shard {shard}/{nshard}: {len(mine)} / 共 {len(jobs)}", flush=True)

    b = R2Bucket(str(SCRATCH), bucket="we-fpv-sh-ns", storage_profile="aliyun_oss")
    for k, j in enumerate(mine):
        tag = f"{mode}_{j['sid']}_c{j['chunk']:03d}"
        mp4, ready = SCRATCH / f"{tag}.mp4", SCRATCH / f"{tag}.ready"
        if ready.exists() or (OUT / f"res_{tag}.json").exists():
            continue
        key = f"{j['sid']}/chunk_{j['chunk']:03d}/vst_left/vst_left_video.mp4"
        t0 = time.time()
        try:
            p = Path(str(b.get(key)))
        except Exception as e:
            print(f"  {tag} 下载失败 {type(e).__name__}", flush=True)
            (SCRATCH / f"{tag}.fail").write_text(type(e).__name__)
            continue
        p.replace(mp4)
        ready.write_text(f"{mp4.stat().st_size} {time.time()-t0:.1f}")
        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(mine)}] {tag} {mp4.stat().st_size/1e6:.0f}MB "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"fetch {mode} shard {shard} 完成", flush=True)


if __name__ == "__main__":
    main()

"""Phase 1 pilot, inference half: 1fps triage over the staged MP4s.

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




WAIT_S, GIVE_UP_S = 15, 3600


def main() -> None:
    """Inference half. Consumes <tag>.ready markers written by pilot_fetch."""
    sys.stdout.reconfigure(encoding="utf-8")
    import cv2
    import scrfd_shim  # noqa: F401
    from mmdet.apis import inference_detector, init_detector

    mode = sys.argv[1] if len(sys.argv) > 1 else "B"
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    OUT.mkdir(exist_ok=True)

    jobs = json.load(open(OUT / f"jobs_{mode}.json"))
    mine = [j for i, j in enumerate(jobs) if i % nshard == shard]
    print(f"infer {mode} shard {shard}/{nshard}: {len(mine)} 个", flush=True)

    m = init_detector(CFG, str(CKPT), device="cuda:0")
    m.test_cfg.score_thr = 0.02
    set_scale(m, DET_SIZE)

    for i, j in enumerate(mine):
        tag = f"{mode}_{j['sid']}_c{j['chunk']:03d}"
        res_p = OUT / f"res_{tag}.json"
        if res_p.exists():
            continue
        mp4, ready = SCRATCH / f"{tag}.mp4", SCRATCH / f"{tag}.ready"
        waited = 0
        while not ready.exists():
            if (SCRATCH / f"{tag}.fail").exists():
                break
            time.sleep(WAIT_S); waited += WAIT_S
            if waited >= GIVE_UP_S:
                break
        if not ready.exists():
            continue

        t1 = time.time()
        cap = cv2.VideoCapture(str(mp4))
        n = fi = hits = boxes_hi = 0
        best = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if n % STRIDE == 0:
                d = np.asarray(inference_detector(m, img)[0]).reshape(-1, 5)
                d = d[d[:, 4] >= KEEP]
                fi += 1
                hi = [b for b in d if b[4] >= T_HIGH and long_side(b[:4]) >= MIN_LONG]
                if hi:
                    hits += 1; boxes_hi += len(hi)
                if len(d):
                    best.append(float(d[:, 4].max()))
            n += 1
        cap.release()
        dt = time.time() - t1
        row = {**j, "frames": n, "sampled": fi, "face_frames_1fps": hits,
               "boxes_hi": boxes_hi, "t_decode_infer": round(dt, 1),
               "fps_effective": round(n / max(1e-6, dt), 1),
               "p90_score": round(float(np.quantile(best, 0.9)), 3) if best else 0.0,
               "n_scored": len(best)}
        json.dump(row, open(res_p, "w"), ensure_ascii=False)
        mp4.unlink(missing_ok=True); ready.unlink(missing_ok=True)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(mine)}] {tag[-12:]} {n}帧 抽{fi} 有脸{hits} "
                  f"| {dt:.0f}s = {n/max(1e-6,dt):.0f} 帧/s", flush=True)
    print(f"infer {mode} shard {shard} 完成", flush=True)


if __name__ == "__main__":
    main()

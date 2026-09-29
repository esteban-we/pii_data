"""Recover what the sweep should have saved: the original frame and the box coordinates in it.

The sweep persisted a 260x260 crop with the boxes burned into the pixels, plus session / chunk /
frame. That is reviewable by eye and useless as a pre-label -- an annotation platform needs the
original image and xyxy in that image's coordinate space. The coordinates were computed and thrown
away; this recovers them.

The continuation check does NOT need re-running, which is what makes this an hour instead of five.
Whether production would already blur a candidate depends on a +/-150 frame window and costs 66% of
the sweep -- but that verdict is already recorded for every row in candidates.json. What is missing
is only the box, and a box depends on nothing but its own frame: armW and the head model are both
per-frame, and the tier follows from the armW score on that frame alone. So the sparse pass is
replayed on just the frames that produced a candidate.

Reproducibility is not assumed. Every re-derived candidate is matched back to its recorded (tier,
score, long side); a row that fails to match is written to unmatched.jsonl rather than silently
dropped or silently paired with the wrong face. The engines are deterministic -- swapping the head
model from CPU-ONNX to TensorRT moved tier counts by <=0.13% over 24 chunks, and the control chunk
reproduced exactly -- so a non-trivial unmatched count means something is wrong and should stop the
handoff.

Coordinate space is sidestepped rather than documented: the image shipped IS the array the boxes were
computed on (torchcodec's decode of vst_left_video.mp4, 2328x1748, no resize, nothing drawn on it),
so xyxy needs no scaling, no letterbox undo, and no agreement about conventions. The OSS key and
frame index travel with each record so the platform can pull the frame itself instead if it prefers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data/liangzhenghao/train")

T = Path("/data/liangzhenghao/train")
CAND = T / "handoff/candidates.json"
OUT = T / "handoff_pkg"
SCRATCH = T / "recover_cache"
BUCKET = "we-fpv-sh-ns"
JPEG_Q = 92
BATCH = 8
KEEP = 0.25
MAX_HEAD_RATIO = 3.0
TIERS = [("T0", 0.75, 1.01), ("T1", 0.45, 0.75), ("T2", 0.25, 0.45), ("T3", -1.0, 0.25)]
SCORE_TOL, SIZE_TOL = 0.004, 1.5      # matching slack against the recorded values


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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    import cv2
    import torch
    from torchcodec.decoders import VideoDecoder
    from face_pii.detect import FaceDetector
    from head_trt import HeadDetector
    from we_io import R2Bucket

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "parts").mkdir(exist_ok=True)
    scratch = SCRATCH / f"shard{args.shard}"
    scratch.mkdir(parents=True, exist_ok=True)

    armw = FaceDetector(str(T / "trt_out/scrfd_1024_fp16_dynb16_5090_armW.engine"),
                        det_thresh=0.1, nms_thresh=0.4, size=1024)
    head = HeadDetector(T / "trt_out/crowdhuman_640_fp16_dynb16_5090.engine")
    bucket = R2Bucket(str(scratch), bucket=BUCKET, storage_profile="aliyun_oss")

    rows = json.load(open(CAND))
    want = defaultdict(lambda: defaultdict(list))          # (sid, chunk) -> frame -> [rows]
    for r in rows:
        want[(r["session"], r["chunk"])][r["frame"]].append(r)
    keys = sorted(want)
    keys = [k for i, k in enumerate(keys) if i % args.nshard == args.shard]
    print(f"分片 {args.shard}/{args.nshard}  {len(keys):,} 个 chunk", flush=True)

    ann_path = OUT / "parts" / f"annotations_{args.shard}.jsonl"
    unm_path = OUT / "parts" / f"unmatched_{args.shard}.jsonl"
    done = set()
    if ann_path.exists():
        for ln in open(ann_path, encoding="utf-8"):
            try:
                done.add(json.loads(ln)["_chunk_key"])
            except Exception:
                pass
    ann = open(ann_path, "a", encoding="utf-8")
    unm = open(unm_path, "a", encoding="utf-8")

    n_img = n_box = n_target = n_unmatched = n_err = 0
    t_start = time.time()
    for ci, (sid, ck) in enumerate(keys):
        ckey = f"{sid}_c{ck:03d}"
        if ckey in done:
            continue
        oss_key = f"{sid}/chunk_{ck:03d}/vst_left/vst_left_video.mp4"
        path = None
        dec = None
        try:
            path = Path(str(bucket.get(oss_key)))
            dec = VideoDecoder(str(path), device="cuda:0")
            frames = sorted(want[(sid, ck)])
            for i in range(0, len(frames), BATCH):
                blk = frames[i:i + BATCH]
                fr = dec.get_frames_at(blk)
                data = fr.data if hasattr(fr, "data") else fr
                face_res = armw.detect_batch(data)
                head_res = head.heads(data)
                for j, f in enumerate(blk):
                    dets = [([float(x) for x in bb], float(ss))
                            for bb, ss in face_res[j] if ss >= KEEP]
                    # re-derive this frame's candidates exactly as the sweep did
                    cands = []
                    for hb in head_res[j]:
                        m = [(b, s) for b, s in dets
                             if _iou(hb, b) >= 0.1 or _iof(b, hb) >= 0.5]
                        ab, best = max(m, key=lambda x: x[1]) if m else (None, 0.0)
                        if ab is not None and long_side(hb) > MAX_HEAD_RATIO * long_side(ab):
                            continue
                        tier = next((nm for nm, lo, hi in TIERS if lo <= best < hi), "T0")
                        cands.append({"tier": tier, "armw": best, "face": ab, "head": hb})

                    # A pre-label for the WHOLE frame, not just the mined box. A frame with three
                    # faces and one box tells the annotator the other two are not faces; they would
                    # either skip them or redraw them by hand. So every armW detection on the frame
                    # is emitted, each tagged with what it is and whether it is the thing we are
                    # asking about. Head-only boxes (no armW response at all -- the old T3) are left
                    # out: by eye they were ~90% cloth, hands and dark patches, and flooding each
                    # frame with them would cost more review time than the few faces they add.
                    tgt = {}
                    used = set()
                    for rec in want[(sid, ck)][f]:
                        pool = [(k, c) for k, c in enumerate(cands)
                                if k not in used and c["tier"] == rec["tier"]
                                and c["face"] is not None
                                and abs(c["armw"] - rec["armw_score"]) <= SCORE_TOL
                                and abs(long_side(c["face"]) - rec["box_long_px"]) <= SIZE_TOL]
                        if not pool:
                            n_unmatched += 1
                            unm.write(json.dumps({**rec, "reason": "重跑未复现"},
                                                 ensure_ascii=False) + "\n")
                            continue
                        k, c = min(pool, key=lambda kc: abs(kc[1]["armw"] - rec["armw_score"]))
                        used.add(k)
                        tgt[id(c["face"])] = rec           # this armW box is a mining target
                    if not tgt:
                        continue

                    head_of = {}
                    for c in cands:
                        if c["face"] is not None:
                            head_of[id(c["face"])] = c["head"]
                    boxes = []
                    for b, s in sorted(dets, key=lambda x: -x[1]):
                        rec = tgt.get(id(b))
                        if rec is not None:
                            role, need = "候选", True
                        elif s >= 0.75:
                            role, need = "已打码", False
                        else:
                            role, need = "参考", False
                        boxes.append({
                            "xyxy": [round(v, 1) for v in b],
                            "role": role,
                            "need_review": need,
                            "tier": next((nm for nm, lo, hi in TIERS if lo <= s < hi), "T0"),
                            "armw_score": round(s, 4),
                            "long_side_px": round(long_side(b), 1),
                            "head_xyxy": ([round(v, 1) for v in head_of[id(b)]]
                                          if id(b) in head_of else None),
                            "preview_crop": rec["file"] if rec else None,
                        })
                    img = data[j].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]   # RGB->BGR
                    name = f"{sid}_c{ck:03d}_f{f:06d}.jpg"
                    cv2.imwrite(str(img_dir / name), img,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    h, w = img.shape[:2]
                    ann.write(json.dumps({
                        "_chunk_key": ckey,
                        "image": f"images/{name}",
                        "width": w, "height": h,
                        "session": sid, "chunk": ck, "frame": f,
                        "scene": want[(sid, ck)][f][0]["scene"],
                        "source": {"bucket": BUCKET, "key": oss_key, "frame_index": f},
                        "n_need_review": sum(1 for b in boxes if b["need_review"]),
                        "boxes": boxes,
                    }, ensure_ascii=False) + "\n")
                    n_img += 1
                    n_box += len(boxes)
                    n_target += sum(1 for b in boxes if b["need_review"])
                del data, fr
            ann.flush()
            unm.flush()
        except Exception as exc:
            n_err += 1
            print(f"  {ckey} 失败 {type(exc).__name__}: {exc}", flush=True)
        finally:
            if dec is not None:
                del dec
            if path is not None:
                path.unlink(missing_ok=True)
            torch.cuda.empty_cache()
        if (ci + 1) % 200 == 0:
            rate = (ci + 1) / max(1e-9, time.time() - t_start) * 3600
            print(f"  [{ci+1}/{len(keys)}] {rate:.0f} chunk/h  图 {n_img:,}  框 {n_box:,}(待判 {n_target:,})  "
                  f"未复现 {n_unmatched}  失败 {n_err}", flush=True)
    ann.close()
    unm.close()
    print(f"\n分片 {args.shard} 完成：图 {n_img:,}  框 {n_box:,}  待判 {n_target:,}  "
          f"未复现 {n_unmatched}  失败 {n_err}", flush=True)


if __name__ == "__main__":
    main()

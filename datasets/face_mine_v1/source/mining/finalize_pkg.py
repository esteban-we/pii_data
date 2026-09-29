"""Close out the handoff package: merge the shard outputs, write the spec beside the data, open it up.

Seven shards each append to their own annotations_<k>.jsonl so they never contend for a file. That
is right for the run and wrong for the consumer, who wants one file. Merged here, sorted by image
path so the order is stable across re-runs.

The format spec goes into README.md inside the package rather than living only in a chat message --
whoever pulls this in a week will have the directory and not the conversation. Counts in the README
are computed from the merged file, not copied from anywhere, so they cannot drift from what shipped.

Also chmods the tree readable: the annotation team pulls this from gpu1 under their own accounts, and
a package nobody can read is the most boring possible way to lose a day.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path

PKG = Path("/data/liangzhenghao/train/handoff_pkg")

README = """# 人脸 PII 困难样本 —— 人工复核数据包

由两个独立模型交叉筛选得到：生产在用的人脸检测模型（SCRFD/armW）+ 一个独立的头部检测模型
（CrowdHuman）。素材来自 {n_sess:,} 个**从未做过人脸检测**的拍摄场次。评测集、客诉序列、
已有训练集涉及的场次已按「场次」整体排除。

## 内容

```
images/<场次>_c<片段>_f<帧号>.jpg    {n_img:,} 张原图，{w}×{h}，未做任何绘制
annotations.jsonl                    {n_img:,} 行，每行一张图，含该帧全部预标框
unmatched.jsonl                      {n_unm} 行（重跑未能复现的候选，正常应为 0）
manifest.json                        统计与口径
README.md                            本文件
```

总计 **{n_box:,} 个预标框**，其中 **{n_need:,} 个需要人工判断**。

## annotations.jsonl 格式

每行一个 JSON 对象：

```json
{{"image": "images/20260712_075725_ACCPKU_c002_f010440.jpg",
  "width": 2328, "height": 1748,
  "session": "20260712_075725_ACCPKU", "chunk": 2, "frame": 10440,
  "scene": "工厂",
  "source": {{"bucket": "we-fpv-sh-ns",
             "key": "20260712_075725_ACCPKU/chunk_002/vst_left/vst_left_video.mp4",
             "frame_index": 10440}},
  "n_need_review": 1,
  "boxes": [
    {{"xyxy": [969.9, 798.3, 1117.0, 918.8],
      "role": "候选", "need_review": true,
      "tier": "T1", "armw_score": 0.6016, "long_side_px": 147.1,
      "head_xyxy": [939.2, 671.5, 1134.2, 890.8],
      "preview_crop": "T1_工厂_20260712_075725_ACCPKU_c002_f010440.jpg"}}
  ]}}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `image` | 相对本目录的图片路径 |
| `session` / `chunk` / `frame` | 场次 / 片段号 / 帧号（帧号是该片段内的序号） |
| `source` | 原始视频在 OSS 的位置和帧序号，需要自行取帧时用 |
| `boxes[].xyxy` | **像素坐标，直接对应 `image` 那张图**，无需任何缩放或换算 |
| `boxes[].role` | `候选` / `已打码` / `参考`，见下 |
| `boxes[].need_review` | 是否需要人工判断，等价于 `role == "候选"` |
| `boxes[].tier` | `T1` / `T2`，见下 |
| `boxes[].armw_score` | 人脸检测模型给的分数（0~1） |
| `boxes[].head_xyxy` | 头部检测模型的框，作参考；可能为 `null` |
| `boxes[].preview_crop` | 对应的缩略图文件名（在 `../mine_run/shard*/crops/` 下），可选 |

## 预标框怎么读

给的是**整帧的预标**，不是只给一个框。这样做是为了避免"一张图里三张脸只框了一个"，
让人误以为其余的不是脸。

| role | 含义 | 是否需要判断 |
|---|---|---|
| **`候选`** | **本次要人工确认的目标**：两个模型都指向同一处，且经过续轨模拟确认现网**不会**打码 | **需要** |
| `已打码` | 人脸模型分数 ≥0.75，现网本来就会打码 | 不需要，仅作参考 |
| `参考` | 人脸模型分数 0.25~0.75 但判定为非目标 | 不需要，仅作参考 |

直接用 `need_review == true` 过滤即可。

## 需要判断什么

对每个 `候选` 框：**框住的是不是一张需要打码的真人脸？**

这批是**刻意挑出来的困难样本**，大量是小脸、侧脸、背光、运动模糊、戴口罩——这些是目标，
不是误检。同时确实混有误检，典型的是**娃娃/玩偶的脸、包装或海报上的印刷人像、深色布料和手部**，
判为「不是」同样有价值。

如果发现**图里有人脸但完全没有任何框**，能顺手补一个框最好（这类是两个模型都漏的，价值最高），
不作硬性要求。

## 两层的质量不一样，建议分开排期

| 层 | 待判框数 | 说明 |
|---|---:|---|
| **T1** | {n_t1:,} | 两个模型都指向同一处。抽样目测约 **85–90% 是真人脸**，建议优先。 |
| **T2** | {n_t2:,} | 只有头部模型比较有把握。抽样目测约 **五到六成**，更脏更难，误检集中在这一层。 |

按场景（待判框数）：

{scene_table}

## 已知限制

- 头部模型单独报出、人脸模型完全无响应的那类框（内部叫 T3）**没有放进来**：抽样目测约九成是
  黑布、手部和暗区，铺进每一帧会拖慢复核。
- `参考` 框也要求两个模型都有响应才保留，且每帧最多 8 个。只有人脸模型低分报出的框
  （多为冲孔板、模具盘、货架标签一类纹理误检）已剔除，否则个别帧会有上百个框把待判框淹掉。
  未剔除的完整版另存为 `annotations_full_context.jsonl`，一般用不到。
- `候选` 的「现网会漏」是用 ±150 帧窗口做续轨模拟得到的。窗口是有限的，被更远处高分帧
  连续续轨救回的脸会被误判成漏检，因此这一判定**偏保守地多报**。
- 图片是鱼眼镜头原始画面，未做去畸变。
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # Re-runnable: once the merged file exists it is the source of truth, so a later pass (e.g.
    # after pruning context boxes) refreshes the README and manifest from what actually ships
    # instead of rebuilding from the shard parts and silently reverting the edit.
    merged = PKG / "annotations.jsonl"
    recs = []
    if merged.exists() and merged.stat().st_size:
        for ln in open(merged, encoding="utf-8"):
            if ln.strip():
                recs.append(json.loads(ln))
        print(f"从已合并的 annotations.jsonl 读入 {len(recs):,} 行")
    else:
        for p in sorted((PKG / "parts").glob("annotations_*.jsonl")):
            for ln in open(p, encoding="utf-8"):
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    r.pop("_chunk_key", None)
                    recs.append(r)
    recs.sort(key=lambda r: r["image"])
    seen = set()
    dedup = []
    for r in recs:                      # a resumed shard could re-emit a chunk
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        dedup.append(r)
    with open(PKG / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for r in dedup:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    unm = []
    for p in sorted((PKG / "parts").glob("unmatched_*.jsonl")):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                unm.append(json.loads(ln))
    with open(PKG / "unmatched.jsonl", "w", encoding="utf-8") as fh:
        for r in unm:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_box = sum(len(r["boxes"]) for r in dedup)
    n_need = sum(r["n_need_review"] for r in dedup)
    tier = Counter(b["tier"] for r in dedup for b in r["boxes"] if b["need_review"])
    scene = defaultdict(Counter)
    for r in dedup:
        for b in r["boxes"]:
            if b["need_review"]:
                scene[r["scene"]][b["tier"]] += 1
    w = dedup[0]["width"] if dedup else 0
    h = dedup[0]["height"] if dedup else 0

    rows = ["| 场景 | T1 | T2 | 合计 |", "|---|---:|---:|---:|"]
    for k in sorted(scene, key=lambda x: -sum(scene[x].values())):
        v = scene[k]
        rows.append(f"| {k} | {v['T1']:,} | {v['T2']:,} | {sum(v.values()):,} |")
    manifest = {
        "图片数": len(dedup), "预标框总数": n_box, "待判框数": n_need,
        "待判分层": dict(tier), "未复现": len(unm),
        "分辨率": [w, h], "场次数": len({r["session"] for r in dedup}),
        "片段数": len({(r["session"], r["chunk"]) for r in dedup}),
        "坐标口径": "xyxy 像素坐标，直接对应 images/ 下同名 jpg，无需缩放",
        "按场景待判框数": {k: dict(v) for k, v in scene.items()},
    }
    json.dump(manifest, open(PKG / "manifest.json", "w"), ensure_ascii=False, indent=1)
    (PKG / "README.md").write_text(README.format(
        n_img=len(dedup), n_box=n_box, n_need=n_need, n_unm=len(unm),
        n_sess=len({r["session"] for r in dedup}), w=w, h=h,
        n_t1=tier["T1"], n_t2=tier["T2"], scene_table="\n".join(rows)),
        encoding="utf-8")

    # readable by whoever pulls it, without touching ownership
    for root, dirs, files in os.walk(PKG):
        os.chmod(root, os.stat(root).st_mode | stat.S_IRGRP | stat.S_IXGRP
                 | stat.S_IROTH | stat.S_IXOTH)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, os.stat(p).st_mode | stat.S_IRGRP | stat.S_IROTH)

    size = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file())
    print(f"图片 {len(dedup):,}  预标框 {n_box:,}  待判 {n_need:,} "
          f"(T1 {tier['T1']:,} / T2 {tier['T2']:,})  未复现 {len(unm)}")
    print(f"包大小 {size/1e9:.1f} GB   -> {PKG}")


if __name__ == "__main__":
    main()

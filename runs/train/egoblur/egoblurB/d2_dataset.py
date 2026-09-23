"""PII-1009 d4: pii face data -> detectron2 dataset (standard dict list json + register).

Two sources:
  --tables <dataset> --boxes-version vN --role eval|train --min-side S
      pii-data tables via evaluation/face_mine_match_dump.load_tables (the WOR-137 loader):
      frames.csv rows of that dataset/role, boxes/vN.csv, images via data/oss_pii_keys.jsonl.
      Eval GT (ignore 0, long side >= S) -> annotations; ignore rows and sub-floor boxes ->
      iscrowd=1 (detectron2 drops iscrowd boxes from training targets; they do not become
      negatives, and its COCO evaluator treats them as ignore regions).
  --manifest <scrfd txt> --img-root <dir>
      the mmdet/scrfd manifest the arms train on ("# <path> <w> <h>" then one box per line:
      x1 y1 x2 y2 [15 kps values | ignore flag]). Path resolves ONLY to <img-root>/<path>;
      sources whose images are not on this host are reported missing and dropped (counts
      in the summary), never aliased.
Boxes are XYXY_ABS in the frame's own pixel space; category_id 0 = face (the jit's class 0,
PII-1012). Output: <out>/<name>.json (list of dataset dicts) and <name>.summary.json.
register(name, json) puts it in DatasetCatalog with thing_classes ["face", "egoblur_class1"]."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = Path("/data/esteban/pii/datasets/d2")
THING_CLASSES = ["face", "egoblur_class1"]   # keep NUM_CLASSES=2 so the jit head loads exactly


def file_md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def from_tables(dataset: str, boxes_version: str, role: str, min_side: float):
    sys.path.insert(0, str(REPO / "evaluation"))
    import face_mine_match_dump as fmm
    items, _session_of, file_of, n = fmm.load_tables(dataset, min_side, boxes_version=boxes_version, role=role)
    size = {r["image"]: (int(r["width"]), int(r["height"])) for r in fmm.read_table_frames(fmm.PII_DATA, dataset)}
    recs, c = [], Counter()
    for i, (lp, gt, ign) in enumerate(items):
        w, h = size[file_of[lp]]
        ann = [{"bbox": [float(x) for x in b], "bbox_mode": 0, "category_id": 0, "iscrowd": 0} for b in gt]
        ann += [{"bbox": [float(x) for x in b], "bbox_mode": 0, "category_id": 0, "iscrowd": 1} for b in ign]
        recs.append({"file_name": lp, "width": w, "height": h, "image_id": i, "annotations": ann})
        c["boxes"] += len(gt); c["ignore"] += len(ign); c["empty_frames"] += int(len(gt) == 0)
    src = fmm.PII_DATA / "datasets" / dataset / "boxes" / f"{boxes_version}.csv"
    meta = {"source": "tables", "dataset": dataset, "boxes_version": boxes_version, "role": role,
            "min_side": min_side, "boxes_csv": str(src), "boxes_md5": file_md5(src),
            "frames_csv_md5": file_md5(fmm.PII_DATA / "frames.csv"), "n_frames_loader": n}
    return recs, c, meta


def from_manifest(manifest: str, img_root: str, min_side: float):
    recs, c, cur = [], Counter(), None
    per_src = {}
    def flush():
        if cur is None: return
        src = cur["file_name"].split("/")[0]
        ps = per_src.setdefault(src, Counter()); ps["frames"] += 1
        p = os.path.join(img_root, cur["file_name"])
        if not os.path.exists(p):
            ps["missing"] += 1; c["missing"] += 1; return
        ann = cur["annotations"]; ps["boxes"] += sum(a["iscrowd"] == 0 for a in ann)
        ps["ignore"] += sum(a["iscrowd"] == 1 for a in ann)
        c["boxes"] += ps and sum(a["iscrowd"] == 0 for a in ann); c["ignore"] += sum(a["iscrowd"] == 1 for a in ann)
        c["empty_frames"] += int(all(a["iscrowd"] for a in ann))
        recs.append({"file_name": p, "width": cur["width"], "height": cur["height"],
                     "image_id": len(recs), "annotations": ann})
    with open(manifest) as f:
        for line in f:
            if line.startswith("#"):
                flush()
                parts = line[1:].split()
                cur = {"file_name": parts[0], "width": int(parts[1]), "height": int(parts[2]), "annotations": []}
                continue
            v = [float(x) for x in line.split()]
            if len(v) < 4: continue
            x1, y1, x2, y2 = v[:4]
            ignore = (len(v) == 5 and v[4] == 1)
            if min_side > 0 and max(x2 - x1, y2 - y1) < min_side: ignore = True
            cur["annotations"].append({"bbox": [x1, y1, x2, y2], "bbox_mode": 0, "category_id": 0, "iscrowd": int(ignore)})
    flush()
    meta = {"source": "manifest", "manifest": manifest, "manifest_md5": file_md5(manifest), "img_root": img_root,
            "min_side": min_side, "per_source": {k: dict(v) for k, v in per_src.items()}}
    return recs, c, meta


def register(name: str, json_path: str):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if name in DatasetCatalog.list(): DatasetCatalog.remove(name); MetadataCatalog.remove(name)
    DatasetCatalog.register(name, lambda: json.load(open(json_path)))
    MetadataCatalog.get(name).set(thing_classes=THING_CLASSES, json_file=json_path, evaluator_type=None)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tables"); ap.add_argument("--boxes-version", default="v1"); ap.add_argument("--role", default="eval")
    ap.add_argument("--manifest"); ap.add_argument("--img-root", default="/data/esteban/pii/datasets/views/mix_ds/images")
    ap.add_argument("--min-side", type=float, default=0.0)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    t0 = time.time()
    if a.tables: recs, c, meta = from_tables(a.tables, a.boxes_version, a.role, a.min_side)
    elif a.manifest: recs, c, meta = from_manifest(a.manifest, a.img_root, a.min_side)
    else: sys.exit("give --tables or --manifest")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jp = Path(a.out) / f"{a.name}.json"; json.dump(recs, open(jp, "w"))
    summ = {"name": a.name, "n_images": len(recs), **{k: int(v) for k, v in c.items()}, "thing_classes": THING_CLASSES,
            "json": str(jp), "json_md5": file_md5(jp), "seconds": round(time.time() - t0, 1), **meta}
    json.dump(summ, open(Path(a.out) / f"{a.name}.summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

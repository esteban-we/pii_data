"""PII-1028: fine-tune EgoBlur Gen2 face (detectron2 GeneralizedRCNN) on train_Z3.

DefaultTrainer subclass; datasets come from d2_dataset.register (PII-1013 json files). Every
SOLVER.CHECKPOINT_PERIOD iterations (and after the last one) the in-memory model is scored on
faceback_hq_v2_eval exactly like egoblur2_1200 in PII-1006: cv2 resize so the longest side is 1200,
model.inference(do_postprocess=False), boxes back to frame pixels, NMS 0.3, then OUR metric
(eval_onnx.prf_counts / iou_mat: greedy 1-1 match at IoU 0.2/0.4/0.6, ignore regions, the SCORES
grid, min_side 40 already applied by the dataset loader). Each rank scores its slice; rank 0
merges (same arithmetic as eval_onnx --merge), writes
<work_dir>/eval/results_faceback_hq_v2_<name>_iter<N>.json in the PII-1006 results shape, and logs
faceback_hq_v2_ft/{R@0.5,P@0.5,R@0.1,...} plus the training losses to W&B (project pii-face-eval,
entity as wandb_log_model.py, run name == model name so the final wandb_log_model.py call resumes it).

Launch (6 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,3,5,6,7 python train_egoblur.py --config-file egoblurA.yaml --num-gpus 6 \
      --dist-url tcp://127.0.0.1:29517 --name egoblurA --resume
Smoke: add --work-dir <scratch> --eval-limit 200 --eval-period 50 --wandb-name egoblurA_smoke
       SOLVER.MAX_ITER 50 (with WANDB_MODE=offline).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
# appended, never inserted: evaluation/wandb/ and the repo-root wandb/ are W&B run-log dirs that
# would shadow the wandb package if they came first on sys.path.
sys.path.append(str(REPO / "evaluation"))

import cv2  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import wandb  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultTrainer, HookBase, default_argument_parser, default_setup, hooks, launch  # noqa: E402
from detectron2.utils import comm  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, EventWriter, JSONWriter, get_event_storage  # noqa: E402

from d2_dataset import register  # noqa: E402

# eval_onnx re-execs the interpreter at import time unless runs/cudnn_lib is on LD_LIBRARY_PATH (its
# onnxruntime-gpu cuDNN shim; with `python -` that re-exec exits 0 silently). Satisfy the check for
# the import only, then restore the env so the spawned ranks and DataLoader workers inherit nothing
# (the dynamic loader of this process read LD_LIBRARY_PATH at start, so the value is inert here).
_LDLP = os.environ.get("LD_LIBRARY_PATH")
os.environ["LD_LIBRARY_PATH"] = "/data/esteban/pii/runs/cudnn_lib" + os.pathsep + (_LDLP or "")
from eval_onnx import IOU_MATCHES, SCORES, accumulate, finalize, new_acc  # noqa: E402
if _LDLP is None:
    del os.environ["LD_LIBRARY_PATH"]
else:
    os.environ["LD_LIBRARY_PATH"] = _LDLP

WANDB_PROJECT = "alex-qiu-worldengineai/pii-face-eval"   # == wandb_log_model.PROJECT
D2_DIR = Path("/data/esteban/pii/datasets/d2")
TRAIN_SET, EVAL_SET = "train_Z3", "faceback_hq_v2_eval"
RESIZE_LONGEST = 1200     # egoblur2_1200 geometry (PII-1006)
NMS_IOU = 0.3             # eval_egoblur_tables.NMS_IOU
MAX_DETS = 100            # TEST.DETECTIONS_PER_IMAGE, the jit's cap
NS = "faceback_hq_v2_ft"  # W&B namespace of the per-checkpoint numbers (wandb_log_model.py owns faceback_hq_v2/)

ARGS = None
log = logging.getLogger("detectron2.egoblurA")


def file_md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- eval
class EvalFrames(torch.utils.data.Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        bgr = cv2.imread(r["file_name"], cv2.IMREAD_COLOR)
        if bgr is None:
            return {"idx": i, "image": None, "scale": 1.0}
        h, w = bgr.shape[:2]
        s = RESIZE_LONGEST / max(h, w)
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)))
        return {"idx": i, "image": torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1))), "scale": s}


def _list_collate(x):   # module-level: DataLoader workers under a spawned rank must pickle it
    return x


def eval_records(limit: int):
    recs = json.load(open(D2_DIR / f"{EVAL_SET}.json"))
    return recs[:limit] if limit and limit > 0 else recs


def score_shard(model, recs, batch_size=4, num_workers=4):
    """Rank-local: (acc, stats) over this rank's slice of recs, PII-1006 geometry and NMS."""
    rank, world = comm.get_rank(), comm.get_world_size()
    mine = recs[rank::world]
    loader = torch.utils.data.DataLoader(EvalFrames(mine), batch_size=batch_size, num_workers=num_workers,
                                         shuffle=False, collate_fn=_list_collate)
    dev = next(model.parameters()).device
    acc, st = new_acc(), {"n_frames_at_cap": 0, "n_dets_raw": 0, "n_dets_after_nms": 0, "score_min_raw": None}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ok = [b for b in batch if b["image"] is not None]
            insts = model.inference([{"image": b["image"].to(dev).float()} for b in ok], do_postprocess=False) if ok else []
            outs = {b["idx"]: (b["scale"], inst) for b, inst in zip(ok, insts)}
            for b in batch:
                r = mine[b["idx"]]
                gtb = np.array([a["bbox"] for a in r["annotations"] if a["iscrowd"] == 0], float).reshape(-1, 4)
                ignb = np.array([a["bbox"] for a in r["annotations"] if a["iscrowd"] == 1], float).reshape(-1, 4)
                if b["idx"] not in outs:
                    accumulate(acc, None, gtb, ignb)
                    continue
                s, inst = outs[b["idx"]]
                boxes = inst.pred_boxes.tensor.float().cpu().numpy().reshape(-1, 4) / s
                sc = inst.scores.float().cpu().numpy().reshape(-1)
                st["n_dets_raw"] += len(sc)
                st["n_frames_at_cap"] += int(len(sc) >= MAX_DETS)
                if len(sc):
                    m = float(sc.min())
                    st["score_min_raw"] = m if st["score_min_raw"] is None else min(st["score_min_raw"], m)
                    keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(sc), NMS_IOU).numpy()
                    boxes, sc = boxes[keep], sc[keep]
                st["n_dets_after_nms"] += len(sc)
                det = np.concatenate([boxes.reshape(-1, 4), sc.reshape(-1, 1)], axis=1).astype(np.float32)
                det = det[np.argsort(-det[:, 4], kind="stable")]
                accumulate(acc, det, gtb, ignb)
    if was_training:
        model.train()
    return acc, st, len(mine)


def merge(parts):
    acc, st, n = new_acc(), {"n_frames_at_cap": 0, "n_dets_raw": 0, "n_dets_after_nms": 0, "score_min_raw": None}, 0
    for a, s, k in parts:
        n += k
        acc["n_unreadable"] += a["n_unreadable"]
        for t in IOU_MATCHES:
            acc["matched"][t].extend(a["matched"][t])
            for sc in SCORES:
                acc["tp"][t][sc] += a["tp"][t][sc]
                acc["fp"][t][sc] += a["fp"][t][sc]
        for k2 in ("n_frames_at_cap", "n_dets_raw", "n_dets_after_nms"):
            st[k2] += s[k2]
        if s["score_min_raw"] is not None:
            st["score_min_raw"] = s["score_min_raw"] if st["score_min_raw"] is None else min(st["score_min_raw"], s["score_min_raw"])
    return acc, st, n


def run_eval(trainer, n_iter: int, run):
    """All ranks: score the current weights; rank 0 writes JSON, logs to W&B, returns the entry."""
    cfg, args = trainer.cfg, ARGS
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    recs = eval_records(args.eval_limit)
    t0 = time.time()
    part = score_shard(model, recs)
    parts = comm.gather(part, dst=0)
    comm.synchronize()
    if not comm.is_main_process():
        return None
    acc, st, n_items = merge(parts)
    dt = time.time() - t0
    ckpt = "model_final.pth" if n_iter >= trainer.max_iter else f"model_{n_iter - 1:07d}.pth"
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, ckpt)
    entry, line = finalize(acc, args.name, ckpt_path, dt, n_items)
    summ = json.load(open(D2_DIR / f"{EVAL_SET}.summary.json"))
    n_gt = sum(sum(a["iscrowd"] == 0 for a in r["annotations"]) for r in recs)
    out = {"set": "tables", "det_size": RESIZE_LONGEST, "min_side": summ["min_side"], "pad": False,
           "n_frames": len(recs), "n_gt": n_gt, "models": {args.name: entry},
           "datasets": [summ["dataset"]], "role": summ["role"], "boxes_version": summ["boxes_version"],
           "pii_data": str(Path(summ["boxes_csv"]).parents[3]), "boxes_md5": {summ["dataset"]: summ["boxes_md5"]},
           "n_frames_by_dataset": {summ["dataset"]: len(recs)}, "n_gt_by_dataset": {summ["dataset"]: n_gt},
           "egoblur": {"checkpoint": ckpt_path, "iter": n_iter, "base_weights": cfg.MODEL.WEIGHTS,
                       "resize_longest": RESIZE_LONGEST, "nms_iou": NMS_IOU,
                       "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, "max_dets": MAX_DETS,
                       "torch": torch.__version__, "torchvision": torchvision.__version__, **st},
           "n_shards": comm.get_world_size(), "eval_limit": args.eval_limit or None,
           "d2_json_md5": summ["json_md5"], "n_unreadable": acc["n_unreadable"]}
    ed = Path(cfg.OUTPUT_DIR) / "eval"
    ed.mkdir(exist_ok=True)
    jp = ed / f"results_faceback_hq_v2_{args.name}_iter{n_iter:05d}.json"
    json.dump(out, open(jp, "w"), indent=1)
    r05, p05, r01 = entry["recall"]["0.5"], entry["precision_by_iou"]["0.4"]["0.5"], entry["recall"]["0.1"]
    log.info(f"EVAL iter {n_iter} {EVAL_SET} n={len(recs)} R@0.5 {r05:.4f} P@0.5 {p05:.4f} R@0.1 {r01:.4f} "
             f"p50 {entry['p50_score']:.4f} n_hit {entry['n_hit']} {dt:.0f}s -> {jp}")
    log.info("table " + line)
    with open(ed / "eval_table.tsv", "a") as f:
        f.write(f"{n_iter}\t{r05:.4f}\t{p05:.4f}\t{r01:.4f}\t{entry['p50_score']:.4f}\t{entry['n_hit']}\t{dt:.0f}\t{jp.name}\n")
    row = {f"{NS}/R@0.5": r05, f"{NS}/P@0.5": p05, f"{NS}/R@0.1": r01, f"{NS}/p50_score": entry["p50_score"],
           f"{NS}/n_hit": entry["n_hit"], f"{NS}/eval_seconds": dt, f"{NS}/n_frames": len(recs), "iter": n_iter}
    for s in ("0.3", "0.7"):
        row[f"{NS}/R@{s}"] = entry["recall"][s]
        row[f"{NS}/P@{s}"] = entry["precision_by_iou"]["0.4"][s]
    trainer.storage.put_scalars(**{k: v for k, v in row.items() if k != "iter"}, smoothing_hint=False)
    if run is not None:
        run.log(row)
    return entry


class OurEvalHook(HookBase):
    def __init__(self, period, fn):
        self._period, self._fn = period, fn

    def after_step(self):
        n = self.trainer.iter + 1
        if self._period > 0 and n % self._period == 0 and n != self.trainer.max_iter:
            self._fn(n)
            comm.synchronize()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._fn(self.trainer.max_iter)   # trainer.iter == max_iter after the loop
            comm.synchronize()


class PeakMemHook(HookBase):
    """Every rank logs its own torch peak memory at the end (and every 500 iters) to the log."""
    def after_step(self):
        if (self.trainer.iter + 1) % 500 == 0 or self.trainer.iter + 1 == self.trainer.max_iter:
            log.info(f"rank {comm.get_rank()} iter {self.trainer.iter + 1} peak alloc "
                     f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB reserved "
                     f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


class WandbWriter(EventWriter):
    def __init__(self, run, window=20):
        self._run, self._window = run, window

    def write(self):
        st = get_event_storage()
        it = st.iter
        row = {"iter": it}
        for k, (v, i) in st.latest_with_smoothing_hint(self._window).items():
            if i >= it - self._window:
                row[k] = v
        row["memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
        self._run.log(row)

    def close(self):
        pass


# ----------------------------------------------------------------------------- trainer
class Trainer(DefaultTrainer):
    wandb_run = None

    def build_hooks(self):
        ret = [h for h in super().build_hooks() if not isinstance(h, hooks.EvalHook)]
        period = self.cfg.SOLVER.CHECKPOINT_PERIOD if ARGS.eval_period < 0 else ARGS.eval_period
        ev = OurEvalHook(period, lambda n: run_eval(self, n, Trainer.wandb_run))
        mem = PeakMemHook()
        if comm.is_main_process():
            ret.insert(len(ret) - 1, ev)   # before PeriodicWriter, like detectron2's EvalHook
            ret.insert(len(ret) - 1, mem)
        else:
            ret += [ev, mem]
        return ret

    def build_writers(self):
        w = [CommonMetricPrinter(self.max_iter), JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"))]
        if Trainer.wandb_run is not None:
            w.append(WandbWriter(Trainer.wandb_run))
        return w


def wandb_start(cfg, args):
    if not comm.is_main_process() or args.no_wandb:
        return None
    os.environ.setdefault("WANDB_DIR", cfg.OUTPUT_DIR)
    idf = Path(cfg.OUTPUT_DIR) / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else "".join(
        random.choices("0123456789abcdefghijklmnopqrstuvwxyz", k=8))   # wandb 0.30 dropped util.generate_id
    idf.write_text(run_id + "\n")
    entity, project = WANDB_PROJECT.split("/")
    conf = {"model": args.wandb_name, "config_file": os.path.basename(args.config_file),
            "config_md5": file_md5(args.config_file), "work_dir": cfg.OUTPUT_DIR, "num_gpus": args.num_gpus,
            "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH, "base_lr": cfg.SOLVER.BASE_LR,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS, "steps": list(cfg.SOLVER.STEPS), "max_iter": cfg.SOLVER.MAX_ITER,
            "checkpoint_period": cfg.SOLVER.CHECKPOINT_PERIOD, "amp": cfg.SOLVER.AMP.ENABLED,
            "min_size_train": list(cfg.INPUT.MIN_SIZE_TRAIN), "max_size_train": cfg.INPUT.MAX_SIZE_TRAIN,
            "base_weights": cfg.MODEL.WEIGHTS, "train_dataset": TRAIN_SET, "ft_eval_dataset": EVAL_SET,
            f"{NS}_det_size": RESIZE_LONGEST, f"{NS}_nms_iou": NMS_IOU, f"{NS}_eval_limit": args.eval_limit or None,
            "torch": torch.__version__, "hostname": os.uname().nodename}
    run = wandb.init(project=project, entity=entity, name=args.wandb_name, id=run_id, resume="allow",
                     config=conf, dir=cfg.OUTPUT_DIR)
    run.define_metric("iter")
    run.define_metric("*", step_metric="iter")
    log.info(f"wandb run {run.id} {run.url}")
    return run


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.work_dir:
        cfg.OUTPUT_DIR = args.work_dir
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    global ARGS
    ARGS = args
    cfg = setup(args)
    register(TRAIN_SET, str(D2_DIR / f"{TRAIN_SET}.json"))
    register(EVAL_SET, str(D2_DIR / f"{EVAL_SET}.json"))
    log.info(f"config {args.config_file} md5 {file_md5(args.config_file)}; name {args.name}; "
             f"world {comm.get_world_size()}; eval every {args.eval_period if args.eval_period >= 0 else cfg.SOLVER.CHECKPOINT_PERIOD} "
             f"iters on {EVAL_SET} (limit {args.eval_limit or 'none'})")
    Trainer.wandb_run = wandb_start(cfg, args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    t0 = time.time()
    trainer.train()
    if comm.is_main_process():
        wall = time.time() - t0
        log.info(f"train done: {trainer.iter} iters, wall {datetime.timedelta(seconds=int(wall))}, "
                 f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        if Trainer.wandb_run is not None:
            Trainer.wandb_run.summary.update({"wall_clock_h": round(wall / 3600, 3),
                                              "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20})
            Trainer.wandb_run.finish()


if __name__ == "__main__":
    ap = default_argument_parser()
    ap.add_argument("--name", default="egoblurA", help="model name: results key, checkpoint eval entries")
    ap.add_argument("--wandb-name", default=None, help="W&B run name (default: --name)")
    ap.add_argument("--work-dir", default=None, help="override OUTPUT_DIR")
    ap.add_argument("--eval-period", type=int, default=-1, help="iters between evals (-1: CHECKPOINT_PERIOD)")
    ap.add_argument("--eval-limit", type=int, default=0, help="score only the first N eval frames (smoke)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    args.wandb_name = args.wandb_name or args.name
    args.config_file = str(Path(args.config_file).resolve())
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))

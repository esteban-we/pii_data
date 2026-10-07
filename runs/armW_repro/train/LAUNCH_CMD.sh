# Launched 2026-08-30 00:50 from SCRFD_DIR. --no-validate because raw_ds/val is not staged on this machine
# (evaluation.interval=10000 never fires in 20 epochs, so this does not change training).
cd /data/esteban/pii/train/insightface/detection/scrfd && export PYTHONPATH=/data/esteban/pii/train/insightface/detection/scrfd && \
CUDA_VISIBLE_DEVICES=1,7 nohup /data/esteban/pii/train/venv/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29521 \
  tools/train.py configs/scrfd/scrfd_armW.py --launcher pytorch --work-dir /data/esteban/pii/train/wd_armW_repro --no-validate \
  > /data/esteban/pii/train/wd_armW_repro/train.log 2>&1 &

# 00:57 relaunched with workers_per_gpu=16 (was 4; input-bound at 4.8 s/iter). Dataloader-only change, model hyperparameters untouched. First attempt log: train_attempt1_w4.log

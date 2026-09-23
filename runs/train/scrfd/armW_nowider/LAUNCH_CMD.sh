# Ablation: armW recipe with WIDER FACE removed from the training manifest (train_W_nowider.txt:
# 21,740 images = 10,249 pii + 11,491 face10k; the 12,879 wider/ blocks dropped). Config differs from
# scrfd_armW.py only in ann_file. total_epochs=20 kept, so ~340 iters/epoch instead of 542 (not compensated).
# Launched 2026-08-30 09:4x from SCRFD_DIR, mirroring wd_armW_repro/LAUNCH_CMD.sh (workers_per_gpu=16 variant),
# different --master_port only.
cd /data/esteban/pii/train/insightface/detection/scrfd && export PYTHONPATH=/data/esteban/pii/train/insightface/detection/scrfd && \
CUDA_VISIBLE_DEVICES=1,7 nohup /data/esteban/pii/train/venv/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29531 \
  tools/train.py configs/scrfd/scrfd_armW_nowider.py --launcher pytorch --work-dir /data/esteban/pii/train/wd_armW_nowider --no-validate \
  > /data/esteban/pii/train/wd_armW_nowider/train.log 2>&1 &
echo $! > /data/esteban/pii/train/wd_armW_nowider/launcher.pid

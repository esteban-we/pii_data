# Launched 2026-08-30 ~09:55 from SCRFD_DIR. Same form as wd_armW_repro/LAUNCH_CMD.sh (workers_per_gpu=16 from the
# start); config scrfd_armW_fe2.py = scrfd_armW.py with ann_file -> data/mix_ds/train_W_fe2.txt (WIDER blocks replaced
# by the wider_fe2/ fisheye-v2 warp, "fill" zoom policy). --no-validate as in the repro run. GPUs 4,5.
cd /data/esteban/pii/train/insightface/detection/scrfd && export PYTHONPATH=/data/esteban/pii/train/insightface/detection/scrfd && \
CUDA_VISIBLE_DEVICES=4,5 nohup /data/esteban/pii/train/venv/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  tools/train.py configs/scrfd/scrfd_armW_fe2.py --launcher pytorch --work-dir /data/esteban/pii/train/wd_armW_fe2 --no-validate \
  > /data/esteban/pii/train/wd_armW_fe2/train.log 2>&1 &

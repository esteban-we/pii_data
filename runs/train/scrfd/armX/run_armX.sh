#!/bin/bash
# Arm X full run with auto-resume (WOR-44). GPU 5 is shared and neighbours fluctuate by tens of
# GB (an OOM from a neighbour burst killed a smoke run), so a plain nohup launch is not enough:
# on any non-zero exit this relaunches with --resume-from latest.pth until epoch_20.pth exists.
# Every attempt and outcome is appended to babysit.log.
#
# Deviations from wd_armW_fe2/LAUNCH_CMD.sh, all launch-side, none touching training math:
#   CUDA_VISIBLE_DEVICES=3, nproc_per_node=1   pinned to GPU 5 only (box shared)
#   samples_per_gpu=64 (in scrfd_armX.py)      keeps effective batch 64 == armW's 32 x 2 GPUs
#   checkpoint_config.interval=1               34 MB/epoch, 7.9 TB free; max 1 epoch lost on crash
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   allocator resilience on a contended GPU
T=/data/esteban/pii/runs/train
S=$T/insightface/detection/scrfd
W=$T/wd_armX
PY=$T/venv/bin/python
cd $S || exit 1
export PYTHONPATH=$S
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "$(date '+%F %T') $*" >> $W/babysit.log; }

log "=== babysitter start (pid $$) ==="
for attempt in $(seq 1 60); do
  if [ -f $W/epoch_20.pth ]; then log "epoch_20.pth exists; done after $((attempt-1)) attempts"; exit 0; fi
  RESUME=""
  [ -f $W/latest.pth ] && RESUME="--resume-from $W/latest.pth"
  log "attempt $attempt launching (resume: ${RESUME:-none})"
  CUDA_VISIBLE_DEVICES=3 $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29563 \
    tools/train.py configs/scrfd/scrfd_armX.py --launcher pytorch --work-dir $W --no-validate \
    --cfg-options checkpoint_config.interval=1 $RESUME >> $W/train.log 2>&1
  rc=$?
  log "attempt $attempt exited rc=$rc"
  [ -f $W/epoch_20.pth ] && { log "epoch_20.pth exists; success"; exit 0; }
  sleep 180
done
log "attempts exhausted without epoch_20.pth"
exit 1

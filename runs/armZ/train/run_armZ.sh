#!/bin/bash
# Arm Z full run (WOR-105): port of wd_armX/run_armX.sh (WOR-44 babysitter) plus the free-memory
# gate of wd_armX/watch_and_launch.sh. Config configs/scrfd/scrfd_armZ.py (train_Z = train_X minus
# wider; load_from armW_epoch_20.pth; 20 epochs x 1,119 iters at batch 64 on one GPU).
#
# Behaviour:
#   - before every attempt, wait until GPU $GPU has >= $THRESH_MIB free (poll every 60 s; the box
#     is shared and neighbours fluctuate by tens of GB; shang measured 17,250 MiB at batch 64)
#   - launch tools/train.py on GPU $GPU, port $PORT, work dir $W, checkpoint every epoch
#     (checkpoint_config.interval=1 overrides the config's 5: 34 MB/epoch, max 1 epoch lost)
#   - on non-zero exit relaunch with --resume-from latest.pth, up to $MAX_ATTEMPTS times, 180 s apart
#   - exit 0 as soon as $W/epoch_20.pth exists; every attempt and outcome goes to $W/babysit.log
#   - stopping: `kill -TERM <babysitter pid>` (this script). The trap TERMs the torch.distributed.run
#     group and the process group of each of its children: train.py runs in its own session (pgid =
#     its pid, measured in wd_armZ_dryrun/run3), and its 16 fork DataLoader workers inherit that
#     pgid, so they go with it (WOR-97: killing train.py alone with -9 orphans them and nvidia-smi
#     keeps ~17 GB attributed to the dead pid). Manual equivalent: pkill -TERM -u $USER -f
#     "tools/train.py configs/scrfd/scrfd_armZ.py" (torchrun, train.py and the workers all match).
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: allocator resilience on a contended GPU
#
# Launch (detached):  setsid nohup bash run_armZ.sh > /dev/null 2>&1 &
# Every setting below can be overridden from the environment (dry runs use W and EXTRA_CFG).
GPU=${GPU:-7}                                  # physical GPU index (nvidia-smi numbering)
PORT=${PORT:-29597}                            # distinct from armX (29563) and 29641
THRESH_MIB=${THRESH_MIB:-21000}                # free-memory gate (watch_and_launch.sh value)
MAX_ATTEMPTS=${MAX_ATTEMPTS:-60}
T=/data/esteban/pii/runs/train
S=$T/insightface/detection/scrfd
W=${W:-$T/wd_armZ}
PY=$T/venv/bin/python
CONFIG=${CONFIG:-configs/scrfd/scrfd_armZ.py}  # relative to $S
EXTRA_CFG=${EXTRA_CFG:-}                       # extra --cfg-options tokens (dry run: log_config.interval=1)

mkdir -p "$W" || exit 1
cd "$S" || exit 1
export PYTHONPATH=$S
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "$(date '+%F %T') $*" >> "$W/babysit.log"; }

TPID=""
cleanup() {
  log "signal received; killing trainer ${TPID:-none} and its children's process groups"
  if [ -n "$TPID" ]; then
    for c in $(pgrep -P "$TPID"); do kill -TERM -- -"$c" 2>/dev/null || kill -TERM "$c" 2>/dev/null; done
    kill -TERM -- -"$TPID" 2>/dev/null
  fi
  exit 130
}
trap cleanup TERM INT

wait_for_gpu() {
  local n=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    if [ -n "$free" ] && [ "$free" -ge "$THRESH_MIB" ]; then
      log "gpu $GPU window open: free=${free}MiB >= ${THRESH_MIB}MiB"; return 0
    fi
    [ -z "$free" ] && { log "nvidia-smi gave no reading for gpu $GPU; aborting"; exit 2; }
    n=$((n+1)); [ $((n % 30)) -eq 0 ] && log "gpu $GPU waiting, free=${free}MiB < ${THRESH_MIB}MiB"
    sleep 60
  done
}

log "=== babysitter start (pid $$) gpu=$GPU port=$PORT config=$CONFIG workdir=$W extra_cfg='${EXTRA_CFG}' ==="
[ -f "$S/$CONFIG" ] || { log "config $S/$CONFIG does not exist; aborting"; exit 3; }
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -f "$W/epoch_20.pth" ]; then log "epoch_20.pth exists; done after $((attempt-1)) attempts"; exit 0; fi
  wait_for_gpu
  RESUME=""
  [ -f "$W/latest.pth" ] && RESUME="--resume-from $W/latest.pth"
  log "attempt $attempt launching (resume: ${RESUME:-none})"
  CUDA_VISIBLE_DEVICES=$GPU setsid "$PY" -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
    tools/train.py "$CONFIG" --launcher pytorch --work-dir "$W" --no-validate \
    --cfg-options checkpoint_config.interval=1 $EXTRA_CFG $RESUME >> "$W/train.log" 2>&1 &
  TPID=$!
  log "attempt $attempt trainer pid $TPID (stop the run: kill -TERM $$)"
  wait "$TPID"; rc=$?
  TPID=""
  log "attempt $attempt exited rc=$rc"
  [ -f "$W/epoch_20.pth" ] && { log "epoch_20.pth exists; success"; exit 0; }
  sleep 180
done
log "attempts exhausted without epoch_20.pth"
exit 1

#!/bin/bash
# Generic arm babysitter (WOR-108..). Port of wd_armZ/run_armZ.sh (WOR-105) generalised to any arm and
# to 1 or 2 GPUs. Behaviour: wait until EVERY GPU in $GPU has >= $THRESH_MIB free; launch tools/train.py
# via torch.distributed.run with nproc = number of GPUs listed; checkpoint every epoch; on non-zero exit
# relaunch with --resume-from latest.pth (up to $MAX_ATTEMPTS, 180 s apart); exit 0 once epoch_${EPOCHS}.pth
# exists. Stop: kill -TERM <babysitter pid> (see babysit.log). All settings come from the environment.
#   ARM=AA GPU=1,2 PORT=29601 setsid nohup bash run_arm.sh > /dev/null 2>&1 &
ARM=${ARM:?set ARM (e.g. AA, AA34, Z, Z34, ZZ, ZZ34)}
GPU=${GPU:?set GPU as comma-separated physical indices, e.g. 1,2}
PORT=${PORT:?set a distinct master port per run}
THRESH_MIB=${THRESH_MIB:-21000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-60}
EPOCHS=${EPOCHS:-20}
T=/data/esteban/pii/runs/train
S=$T/insightface/detection/scrfd
W=${W:-$T/wd_arm$ARM}
PY=$T/venv/bin/python
CONFIG=${CONFIG:-configs/scrfd/scrfd_arm$ARM.py}
EXTRA_CFG=${EXTRA_CFG:-}
NPROC=$(echo "$GPU" | tr ',' '\n' | grep -c .)

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

wait_for_gpus() {
  local n=0 ok free g
  while true; do
    ok=1
    for g in $(echo "$GPU" | tr ',' ' '); do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ -z "$free" ] && { log "nvidia-smi gave no reading for gpu $g; aborting"; exit 2; }
      [ "$free" -ge "$THRESH_MIB" ] || { ok=0; last="gpu $g free=${free}MiB"; }
    done
    [ $ok -eq 1 ] && { log "gpu window open on $GPU (>= ${THRESH_MIB}MiB each)"; return 0; }
    n=$((n+1)); [ $((n % 30)) -eq 0 ] && log "waiting: $last < ${THRESH_MIB}MiB"
    sleep 60
  done
}

log "=== babysitter start (pid $$) arm=$ARM gpu=$GPU nproc=$NPROC port=$PORT config=$CONFIG workdir=$W extra_cfg='${EXTRA_CFG}' ==="
[ -f "$S/$CONFIG" ] || { log "config $S/$CONFIG does not exist; aborting"; exit 3; }
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -f "$W/epoch_${EPOCHS}.pth" ]; then log "epoch_${EPOCHS}.pth exists; done after $((attempt-1)) attempts"; exit 0; fi
  wait_for_gpus
  RESUME=""
  # WOR-196 (armAB only): EMAHook must re-load the checkpoint itself after registering its ema_* buffers,
  # otherwise --resume-from alone drops the ema_* keys as unexpected. Same token goes into --cfg-options.
  [ -f "$W/latest.pth" ] && RESUME="custom_hooks.0.resume_from=$W/latest.pth --resume-from $W/latest.pth"
  log "attempt $attempt launching (resume: ${RESUME:-none})"
  CUDA_VISIBLE_DEVICES=$GPU setsid "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT" \
    tools/train.py "$CONFIG" --launcher pytorch --work-dir "$W" --no-validate \
    --cfg-options checkpoint_config.interval=1 $EXTRA_CFG $RESUME >> "$W/train.log" 2>&1 &
  TPID=$!
  log "attempt $attempt trainer pid $TPID (stop the run: kill -TERM $$)"
  wait "$TPID"; rc=$?
  TPID=""
  log "attempt $attempt exited rc=$rc"
  [ -f "$W/epoch_${EPOCHS}.pth" ] && { log "epoch_${EPOCHS}.pth exists; success"; exit 0; }
  sleep 180
done
log "attempts exhausted without epoch_${EPOCHS}.pth"
exit 1

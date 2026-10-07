#!/bin/bash
# PII-1028 babysitter for the egoblurA fine-tune (port of training/run_arm.sh to detectron2).
# Launches train_egoblur.py on $GPU (6 ranks), always with --resume (DefaultTrainer resumes from
# <W>/last_checkpoint when present, else loads MODEL.WEIGHTS); relaunches on non-zero exit up to
# $MAX_ATTEMPTS, 180 s apart; exits 0 once <W>/model_final.pth exists. Stop: kill -TERM <babysitter pid>.
#   GPU=0,1,3,5,6,7 PORT=29517 setsid nohup bash run_egoblurA.sh > /dev/null 2>&1 &
NAME=${NAME:-egoblurA}
GPU=${GPU:?set GPU as comma-separated physical indices}
PORT=${PORT:?set a distinct master port}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
W=${W:-/data/esteban/pii/runs/train/wd_$NAME}
PY=/data/esteban/pii/runs/egoblur_venv/bin/python
SCRIPT=${SCRIPT:-/home/esteban/repos/pii/evaluation/egoblur_d2/train_egoblur.py}
CONFIG=${CONFIG:-/home/esteban/repos/pii/evaluation/egoblur_d2/$NAME.yaml}
EXTRA=${EXTRA:-}
NPROC=$(echo "$GPU" | tr ',' '\n' | grep -c .)
mkdir -p "$W" || exit 1
cd "$(mktemp -d)" || exit 1      # never the repo: its wandb/ run dirs shadow the wandb package
export WANDB_DIR=$W
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" >> "$W/babysit.log"; }
TPID=""
cleanup() {
  log "signal received; killing trainer ${TPID:-none} and its process group"
  [ -n "$TPID" ] && { kill -TERM -- -"$TPID" 2>/dev/null || kill -TERM "$TPID" 2>/dev/null; }
  exit 130
}
trap cleanup TERM INT
log "=== babysitter start (pid $$) name=$NAME gpu=$GPU nproc=$NPROC port=$PORT config=$CONFIG workdir=$W extra='$EXTRA' ==="
[ -f "$CONFIG" ] || { log "config $CONFIG missing; aborting"; exit 3; }
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  [ -f "$W/model_final.pth" ] && { log "model_final.pth exists; done after $((attempt-1)) attempts"; exit 0; }
  log "attempt $attempt launching (resume from $(cat "$W/last_checkpoint" 2>/dev/null || echo none))"
  CUDA_VISIBLE_DEVICES=$GPU setsid "$PY" "$SCRIPT" --config-file "$CONFIG" --num-gpus "$NPROC" \
    --dist-url "tcp://127.0.0.1:$PORT" --name "$NAME" --work-dir "$W" --resume $EXTRA >> "$W/train.log" 2>&1 &
  TPID=$!
  log "attempt $attempt trainer pid $TPID (stop the run: kill -TERM $$)"
  wait "$TPID"; rc=$?
  TPID=""
  log "attempt $attempt exited rc=$rc"
  [ -f "$W/model_final.pth" ] && { log "model_final.pth exists; success"; exit 0; }
  sleep 180
done
log "attempts exhausted without model_final.pth"
exit 1

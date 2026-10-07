#!/bin/bash
# WOR-44: fire-and-forget launcher for the arm X full run.
# Polls GPU 5 free memory every 60 s; when at least THRESH_MIB is free (default 21000:
# fe2 measured 8.9 GB at batch 32 on this GPU model -> ~18 GB at batch 64, +3 GB margin),
# writes a timestamped marker to LAUNCHED_AT and execs run_armX.sh (the auto-resume
# babysitter). Idempotent: exits immediately if LAUNCHED_AT already exists.
# Env overrides for testing: THRESH_MIB, LAUNCH_CMD.
W=/data/esteban/pii/runs/train/wd_armX
THRESH_MIB=${THRESH_MIB:-21000}
LAUNCH_CMD=${LAUNCH_CMD:-"bash $W/run_armX.sh"}
MARKER=${MARKER:-$W/LAUNCHED_AT}
LOG=${WATCHLOG:-$W/watcher.log}

echo "$(date '+%F %T') watcher start pid=$$ thresh=${THRESH_MIB}MiB cmd='$LAUNCH_CMD'" >> $LOG
[ -f "$MARKER" ] && { echo "$(date '+%F %T') marker exists, exiting" >> $LOG; exit 0; }
n=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5 2>/dev/null)
  if [ -n "$free" ] && [ "$free" -ge "$THRESH_MIB" ]; then
    echo "$(date '+%F %T') window open: free=${free}MiB >= ${THRESH_MIB}MiB, launching" >> $LOG
    echo "$(date '+%F %T') free=${free}MiB" > "$MARKER"
    exec $LAUNCH_CMD >> $LOG 2>&1
  fi
  n=$((n+1)); [ $((n % 30)) -eq 0 ] && echo "$(date '+%F %T') waiting, free=${free}MiB" >> $LOG
  sleep 60
done

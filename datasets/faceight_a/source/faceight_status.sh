#!/usr/bin/env bash
# One-ssh status of the faceight pull / detect runs on shang (WOR-186). Run locally:
#   mining/faceight_status.sh            one snapshot
#   mining/faceight_status.sh --watch N  repeat every N seconds (Ctrl-C to stop)
# Prints: every faceight_pull.py process (pid, elapsed, CUDA_VISIBLE_DEVICES, --view/--stage/
# --model-name/--shard), the last line of every /data/esteban/faceight/*.log modified in the last
# 24 h, line counts of faceight_*.jsonl (shards included) and fetch/failed jsonls, jpg counts of
# frames/ and frames_right/, one nvidia-smi line per GPU, df -h /data. Bash only on shang.
WATCH=0
if [ "$1" = "--watch" ]; then WATCH=${2:-60}; fi
snapshot() {
ssh shang '
D=/data/esteban/faceight
echo "== $(date "+%F %T") shang"
echo "== faceight_pull.py processes"
FOUND=0
for P in $(pgrep -f "faceight_pul[l].py" 2>/dev/null); do
  [ "$P" != "$$" ] && [ -r /proc/$P/cmdline ] || continue
  case "$(tr "\0" "\n" < /proc/$P/cmdline | head -n 1)" in *python*) ;; *) continue ;; esac
  FOUND=1
  ARGS=$(tr "\0" " " < /proc/$P/cmdline)
  GPU=$(tr "\0" "\n" < /proc/$P/environ 2>/dev/null | sed -n "s/^CUDA_VISIBLE_DEVICES=//p")
  SEL=$(echo "$ARGS" | grep -oE -- "--(view|stage|model-name|shard|batch|round) [^ ]+" | tr "\n" " ")
  echo "pid $P  up $(ps -o etime= -p $P | tr -d " ")  gpu ${GPU:-?}  $SEL"
done
[ $FOUND = 1 ] || echo "(none)"
echo "== logs modified in the last 24 h (last line)"
for L in $(find $D -maxdepth 1 -name "*.log" -mmin -1440 | sort); do
  printf "%-28s %s\n" "$(basename $L)" "$(tail -n 1 $L)"
done
echo "== records"
for J in $(ls $D/faceight_*.jsonl $D/fetch*.jsonl $D/failed*.jsonl 2>/dev/null | sort); do
  printf "%10d  %s\n" "$(wc -l < $J)" "$(basename $J)"
done
echo "== jpg: frames/ $(ls $D/frames 2>/dev/null | grep -c "\.jpg$")  frames_right/ $(ls $D/frames_right 2>/dev/null | grep -c "\.jpg$")"
echo "== gpus (index, mem used, util)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== disk"; df -h /data | tail -n 1
'
}
if [ "$WATCH" -gt 0 ] 2>/dev/null; then
  while true; do snapshot; echo; sleep "$WATCH"; done
else
  snapshot
fi

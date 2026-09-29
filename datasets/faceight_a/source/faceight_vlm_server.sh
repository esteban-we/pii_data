#!/usr/bin/env bash
# vLLM server for mining/faceight_vlm.py (WOR-70). Runs ON shang.
#   faceight_vlm_server.sh start [GPUS]   default GPUS=2,3 ; tensor-parallel 2, port 8100
#   faceight_vlm_server.sh stop
#   faceight_vlm_server.sh status
# Log: /data/esteban/vlm/server.log. Model: /data/esteban/models/Qwen2.5-VL-72B-Instruct-AWQ
set -u
VLM=/data/esteban/vlm
MODEL=/data/esteban/models/Qwen2.5-VL-72B-Instruct-AWQ
NAME=Qwen2.5-VL-72B-Instruct-AWQ
PORT=8100
PIDFILE=$VLM/server.pid

case "${1:-status}" in
  start)
    GPUS=${2:-2,3}
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running pid $(cat "$PIDFILE")"; exit 0; fi
    cd "$VLM"
    # flashinfer sampler JIT-compiles with ninja/nvcc at first request; not needed for greedy decoding
    export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
    CUDA_VISIBLE_DEVICES=$GPUS nohup "$VLM/venv/bin/vllm" serve "$MODEL" \
      --served-model-name "$NAME" --host 127.0.0.1 --port $PORT \
      --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.88 \
      --limit-mm-per-prompt '{"image":1}' --max-num-seqs 48 \
      > "$VLM/server.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started pid $! on GPUs $GPUS, log $VLM/server.log (model load takes a few minutes)"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null && echo "sent TERM to $(cat "$PIDFILE")"; fi
    pkill -f "vllm serve $MODEL" 2>/dev/null
    rm -f "$PIDFILE"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "server pid $(cat "$PIDFILE") alive"; else echo "server: not running"; fi
    curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && echo "health: ok" || echo "health: not ready"
    ;;
  *) echo "usage: $0 start [GPUS] | stop | status"; exit 1 ;;
esac

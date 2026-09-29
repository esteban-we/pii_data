#!/usr/bin/env bash
# One-ssh status of the faceight VLM head-count run (WOR-70). Run locally: mining/faceight_vlm_status.sh
# Prints vlm_progress.json, the last log line, and whether the vLLM server and the client are alive.
#   stop client :  ssh shang 'kill $(cat /data/esteban/faceight/vlm_client.pid)'   (restart resumes)
#   start client:  ssh shang 'cd /root/repos/pii && nohup /data/esteban/vlm/venv/bin/python mining/faceight_vlm.py
#                             --concurrency 24 >> /data/esteban/faceight/vlm.client.log 2>&1 < /dev/null &'
#   server      :  ssh shang '/root/repos/pii/mining/faceight_vlm_server.sh start|stop|status'
ssh shang '
D=/data/esteban/faceight
echo "== $D/vlm_progress.json"; cat $D/vlm_progress.json 2>/dev/null || echo "(no progress file yet)"; echo
echo "== last log line"; tail -n 1 $D/vlm.log 2>/dev/null || echo "(no log yet)"
echo "== records: $(wc -l < $D/faceight_vlm.jsonl 2>/dev/null || echo 0) in faceight_vlm.jsonl"
CP=$(cat $D/vlm_client.pid 2>/dev/null)
if [ -n "$CP" ] && kill -0 "$CP" 2>/dev/null && grep -q faceight_vlm /proc/$CP/cmdline 2>/dev/null; then echo "client : alive (pid $CP, up $(ps -o etime= -p $CP | tr -d " "))";
elif [ -n "$CP" ]; then echo "client : NOT running (stale pidfile, pid $CP)"; else echo "client : NOT running"; fi
SP=$(cat /data/esteban/vlm/server.pid 2>/dev/null)
if [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null; then echo "server : alive (pid $SP, up $(ps -o etime= -p $SP | tr -d " "))"; else echo "server : NOT running"; fi
curl -s -m 3 http://127.0.0.1:8100/health >/dev/null && echo "health : ok" || echo "health : not ready"
'

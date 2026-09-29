#!/usr/bin/env bash
# Progress of a faceight detect run: frames scored out of frames on disk, per eye.
# usage: mining/faceight_progress.sh [model_tag] [--watch N]   (default tag armAA34)
tag=armAA34; watch=0
for a in "$@"; do case "$a" in --watch) watch=-1;; -*) ;; *) [ "$watch" = -1 ] && watch=$a || tag=$a;; esac; done
show() {
  ssh shang "cd /data/esteban/faceight && \
    L=\$(cat faceight_${tag}.jsonl faceight_${tag}.shard*.jsonl 2>/dev/null | wc -l); \
    R=\$(cat faceight_${tag}_right.jsonl faceight_${tag}_right.shard*.jsonl 2>/dev/null | wc -l); \
    TL=\$(ls frames | wc -l); TR=\$(ls frames_right | wc -l); \
    printf 'left  %7d / %7d  (%5.1f%%)\nright %7d / %7d  (%5.1f%%)\n' \$L \$TL \$(echo \"100*\$L/\$TL\" | bc -l) \$R \$TR \$(echo \"100*\$R/\$TR\" | bc -l)"
}
if [ "$watch" -gt 0 ] 2>/dev/null; then while true; do clear; date; show; sleep "$watch"; done; else show; fi

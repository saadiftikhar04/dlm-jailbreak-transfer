#!/bin/bash
# R2 Step 13 — wait for a free local GPU (>=18GB free) then run Falcon fullpool.
# Falcon needs raven_rag's working mamba (HPC raven mamba is ABI-broken); local
# GPUs are currently busy with the user's other work. When ≥18GB frees up on a
# card, launch falcon fullpool with CUDA_VISIBLE_DEVICES pinned to that card.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/falcon_wait.log"
touch "$LOG"
echo "[$(date)] Falcon GPU waiter started" >> "$LOG"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -1)
  IDX=$(echo "$FREE" | cut -d, -f1)
  FREEMEM=$(echo "$FREE" | tr -d ' ' | cut -d, -f2)
  # also require <80% util so we don't starve the user's job
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$IDX" 2>/dev/null | tr -d ' ')
  if [ -n "$FREEMEM" ] && [ "$FREEMEM" -ge 18000 ] && [ "$UTIL" -lt 80 ]; then
    echo "[$(date)] GPU $IDX free ($FREEMEM MiB free, ${UTIL}% util) — launching Falcon fullpool" >> "$LOG"
    export CUDA_VISIBLE_DEVICES="$IDX"
    export HF_HOME=/home/bc3194/Desktop/huggingface_cache
    export HF_HUB_DISABLE_XET=1
    export TARGET=falcon
    export DEEPSEEK_API_KEY="$(bash -ic 'echo $DEEPSEEK_API_KEY' 2>/dev/null || true)"
    cd "$HERE/falcon_env" || exit 1
    /home/bc3194/miniconda3/envs/raven_rag/bin/python stage5_fullpool.py
    echo "[$(date)] Falcon fullpool finished rc=$?" >> "$LOG"
    exit 0
  fi
  sleep 900   # check every 15 min
done
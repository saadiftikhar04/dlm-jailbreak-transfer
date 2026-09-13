#!/bin/bash
# R2 Step 13 — run Falcon full-pool ArrAttack locally (raven_rag).
# Falcon must run locally (HPC raven mamba_ssm ABI-broken under torch 2.13).
cd "$(dirname "$0")/falcon_env" || exit 1
export HF_HOME=/home/bc3194/Desktop/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TARGET=falcon
export DEEPSEEK_API_KEY="$(bash -ic 'echo $DEEPSEEK_API_KEY' 2>/dev/null || true)"
PY=/home/bc3194/miniconda3/envs/raven_rag/bin/python
echo "[$(date)] Falcon fullpool local start"
"$PY" stage5_fullpool.py
echo "[$(date)] Falcon fullpool rc=$?"
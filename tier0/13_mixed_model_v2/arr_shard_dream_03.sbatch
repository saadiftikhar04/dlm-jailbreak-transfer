#!/bin/bash
# R2 Step 13 — ArrAttack full-pool SHARD for one victim (dream/diffucoder).
# One shard = ~92 prompts; 10 shards per victim run in parallel.
#SBATCH --job-name=arr_s03_dream
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=100G
#SBATCH --output=arr_s03_dream.out
#SBATCH --error=arr_s03_dream.err
set -euo pipefail

VKEY=dream
SHARD=03
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
cd "$ROOT" || exit 1

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export TARGET="$VKEY"
export ARR_DATASET_BASE="$ROOT/step13_dataset_shard/$SHARD"
export ARR_RESULTS_DIR="$ROOT/results_shard/${VKEY}_${SHARD}"
mkdir -p "$ARR_RESULTS_DIR"
export DEEPSEEK_API_KEY="$(bash -ic 'echo $DEEPSEEK_API_KEY' 2>/dev/null || true)"
PY=/scratch/bc3194/conda_envs/envs/raven/bin/python

echo "[$(date)] START ArrAttack shard victim=$VKEY shard=$SHARD"
echo "dataset=$ARR_DATASET_BASE  results=$ARR_RESULTS_DIR"
# show how many prompts this shard has (ascertainable from malicious_instruct line count etc.)
"$PY" stage5_fullpool.py || { echo "FAILED v=$VKEY s=$SHARD rc=$?"; exit 1; }
echo "[$(date)] DONE shard victim=$VKEY shard=$SHARD"
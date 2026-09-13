#!/bin/bash
# R2 Step 13 — full-pool ArrAttack for one victim (stage5_fullpool.py).
# One GPU per job. Resume-safe via stage5 PROGRESS_CSV (job rerun continues).
#SBATCH --job-name=arr_full_dream
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=100G
#SBATCH --output=arr_full_dream.out
#SBATCH --error=arr_full_dream.err
set -euo pipefail

HPC=1
VKEY=dream          # stage5 key: qwen2.5 / llama / falcon / llada / dream / diffucoder
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
cd "$ROOT" || exit 1

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export TARGET="$VKEY"
export DEEPSEEK_API_KEY="$(bash -ic 'echo $DEEPSEEK_API_KEY' 2>/dev/null || true)"
PY=/scratch/bc3194/conda_envs/envs/raven/bin/python

echo "[$(date)] START ArrAttack-fullpool victim=$VKEY"
echo "HF_HOME=$HF_HOME  DEEPSEEK set=$([ -n \"$DEEPSEEK_API_KEY\" ] && echo yes || echo NO)"

"$PY" stage5_fullpool.py || { echo "stage5_fullpool FAILED victim=$VKEY rc=$?" ; exit 1; }

echo "[$(date)] DONE ArrAttack-fullpool victim=$VKEY"
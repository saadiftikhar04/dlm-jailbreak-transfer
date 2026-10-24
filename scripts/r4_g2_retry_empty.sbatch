#!/bin/bash
# R4 G2: regenerate only the empty-response rows of a plaintext-control shard.
#SBATCH --job-name=r4_g2_retry
#SBATCH --partition=condo
#SBATCH --qos=ebrainccs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=03:00:00
#SBATCH --mem=56G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G2_plaintext_control/logs/g2_retry_%A_%a.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G2_plaintext_control/logs/g2_retry_%A_%a.err

set -euo pipefail
umask 077
cd /scratch/bc3194/dlm-jailbreak-transfer

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_CACHE=/scratch/bc3194/huggingface_cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=/scratch/bc3194/conda_envs/envs/raven/bin/python

case "$SLURM_ARRAY_TASK_ID" in
  0) MODEL_ALIAS=diffucoder ;;
  1) MODEL_ALIAS=dream ;;
  *) echo "Retry mode is limited to indices 0 (diffucoder) and 1 (dream)" >&2; exit 2 ;;
esac

ARGS=(--model-alias "$MODEL_ALIAS" --tag "${G2_RETRY_TAG:-retry_empty}")
if [[ -n "${G2_RETRY_LIMIT:-}" ]]; then
  ARGS+=(--limit "$G2_RETRY_LIMIT")
fi

echo "model=$MODEL_ALIAS tag=${G2_RETRY_TAG:-retry_empty} limit=${G2_RETRY_LIMIT:-none}"
"$PY" scripts/r4_g2_retry_empty.py "${ARGS[@]}"

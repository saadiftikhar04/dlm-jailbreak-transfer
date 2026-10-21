#!/bin/bash
# R4 G3: regenerate only missing PiF first-victim-query rows.
#SBATCH --job-name=r4_g3_pif_first
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --array=0-1,3-5%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=56G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G3_first_response/logs/g3_pif_%A_%a.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G3_first_response/logs/g3_pif_%A_%a.err

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
G3_ARGS=()
if [[ "${G3_RETRY:-0}" == "1" ]]; then
  case "$SLURM_ARRAY_TASK_ID" in
    3) MODEL_ALIAS=llada ;;
    4) MODEL_ALIAS=dream ;;
    5) MODEL_ALIAS=diffucoder ;;
    *) echo "G3 retry mode is limited to failed DLM indices 3, 4, 5" >&2; exit 2 ;;
  esac
  RETRY_TAG="${G3_RETRY_OUTPUT_TAG:-retry1}"
  if [[ ! "$RETRY_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid G3_RETRY_OUTPUT_TAG; use letters, digits, underscore, or hyphen" >&2
    exit 2
  fi
  MODEL_DIR="/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G3_first_response/pif/$MODEL_ALIAS"
  G3_ARGS+=(--retry-failures-from "$MODEL_DIR/g3_pif_first_response.csv")
  G3_ARGS+=(--output-path "$MODEL_DIR/g3_pif_first_response_${RETRY_TAG}.csv")
  if [[ -n "${G3_ONLY_KEY:-}" ]]; then
    G3_ARGS+=(--only-key "$G3_ONLY_KEY")
  fi
  if [[ -n "${G3_RETRY_LIMIT:-}" ]]; then
    G3_ARGS+=(--limit "$G3_RETRY_LIMIT")
  fi
fi

"$PY" scripts/r4_g3_pif_first_query.py --model-index "$SLURM_ARRAY_TASK_ID" "${G3_ARGS[@]}"

#!/bin/bash
# R4 G4: LLaDA MetaCipher, full canonical pool, locked at 128 steps/block.
#SBATCH --job-name=r4_g4_metallada
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=56G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/logs/g4_metacipher_%j.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/logs/g4_metacipher_%j.err

source ~/.bashrc >/dev/null 2>&1
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
export R4_TASK=G4
export R4_MODEL_ALIAS=llada
export R4_OUT_DIR=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/metacipher_openrouter_v4_0731
export METACIPHER_API_BASE_URL=https://openrouter.ai/api/v1
export METACIPHER_API_MODEL=deepseek/deepseek-v4-flash-0731

OPENROUTER_ENV_FILE="$HOME/.config/dlm-jailbreak-transfer/openrouter.env"
if [[ ! -r "$OPENROUTER_ENV_FILE" ]]; then
    echo "Missing private OpenRouter credential file: $OPENROUTER_ENV_FILE" >&2
    exit 2
fi
if [[ "$(stat -c '%a' "$OPENROUTER_ENV_FILE")" != "600" ]]; then
    echo "OpenRouter credential file must have mode 600" >&2
    exit 2
fi
source "$OPENROUTER_ENV_FILE"
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is missing from the private credential file" >&2
    exit 2
fi

/scratch/bc3194/conda_envs/envs/raven/bin/python scripts/r4_g1_generate.py --attack metacipher

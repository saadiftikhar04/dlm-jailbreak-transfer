#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=t02_tokenlen
#SBATCH --output=t02_tokenlen_%j.out
#SBATCH --error=t02_tokenlen_%j.err

# T02.2: true-token response-length statistics (raven env, tokenizer-only).
# HF_HUB_DISABLE_XET=1 avoids the xet thread-pool panic on constrained nodes.
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== starting $(date) ==="
/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/02_dedup_and_config_audit/build_true_token_stats_hpc.py
echo "=== RC=$? $(date) ==="
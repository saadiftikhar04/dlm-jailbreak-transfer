#!/bin/bash
#SBATCH --job-name=t07_gen
#SBATCH --output=t07_gen_%j.out
#SBATCH --error=t07_gen_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# T07 plaintext baseline generation for ONE victim (model key passed on cmdline).
# Downloads model on first run (offline=False). Adapt partition below if needed.
# Usage: sbatch --export=ALL,MODEL_KEY=qwen2.5 t07_generate.sbatch
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/07_plaintext_harmful_baseline/t07_generate_hpc.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
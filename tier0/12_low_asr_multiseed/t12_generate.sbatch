#!/bin/bash
#SBATCH --job-name=t12_gen
#SBATCH --output=t12_gen_%j.out
#SBATCH --error=t12_gen_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

# T12 multi-seed generation for ONE victim (MODEL_KEY passed).
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/12_low_asr_multiseed/t12_generate.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
#!/bin/bash
#SBATCH --job-name=t09_gen
#SBATCH --output=t09_gen_%j.out
#SBATCH --error=t09_gen_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# T09 benign PiF-intelligibility generation for ONE victim (MODEL_KEY passed).
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/09_benign_pif_intelligibility/t09_generate_hpc.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
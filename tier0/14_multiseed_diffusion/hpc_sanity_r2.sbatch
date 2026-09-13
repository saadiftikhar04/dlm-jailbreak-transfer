#!/bin/bash
#SBATCH --job-name=ms_sanity
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ms_sanity_%j.out
#SBATCH --error=ms_sanity_%j.err
#SBATCH -C 80g

set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/14_multiseed_diffusion/hpc_sanity_r2.py
echo "=== RC=$? $(date) ==="
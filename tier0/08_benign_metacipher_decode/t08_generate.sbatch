#!/bin/bash
#SBATCH --job-name=t08_gen
#SBATCH --output=t08_gen_%j.out
#SBATCH --error=t08_gen_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# T08.3 benign MetaCipher generation for ONE victim (MODEL_KEY passed).
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/08_benign_metacipher_decode/t08_generate_hpc.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
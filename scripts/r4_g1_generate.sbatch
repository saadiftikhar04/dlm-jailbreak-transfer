#!/bin/bash
# R4 G1 one-victim generation. Run pif_control, pif and metacipher sequentially.
#SBATCH --job-name=r4_g1
#SBATCH --partition=condo
#SBATCH --qos=ebrainccs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:00
#SBATCH --mem=56G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G1_llama31_8b/logs/g1_%j.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G1_llama31_8b/logs/g1_%j.err
source ~/.bashrc >/dev/null 2>&1
set -euo pipefail
umask 077
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_CACHE=/scratch/bc3194/huggingface_cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
/scratch/bc3194/conda_envs/envs/raven/bin/python scripts/r4_g1_generate.py --attack "$1"

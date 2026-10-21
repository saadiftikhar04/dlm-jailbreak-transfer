#!/bin/bash
# G2 paired plaintext controls: six model groups, one A100-80GB each.
# Submit as an array of model indices 0..5 from the project root.
# Deliberately omit --qos: this uses the user's normal condo allocation.
#SBATCH --job-name=r4_g2_plaintext
#SBATCH --partition=condo
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=30:00:00
#SBATCH --mem=56G
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G2_plaintext_control/logs/g2_%A_%a.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G2_plaintext_control/logs/g2_%A_%a.err
set -euo pipefail
umask 077
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_CACHE=/scratch/bc3194/huggingface_cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/scratch/bc3194/conda_envs/envs/raven/bin/python scripts/r4_g2_plaintext_control.py --model-index "$SLURM_ARRAY_TASK_ID"

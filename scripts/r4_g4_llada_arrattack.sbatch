#!/bin/bash
# R4 G4: LLaDA ArrAttack, D4 ASR@1, locked at 128 steps per 128-token block.
#SBATCH --job-name=r4_g4_arrllada
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=100G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/logs/g4_arrattack_%j.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/logs/g4_arrattack_%j.err

set -euo pipefail
umask 077
cd /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_CACHE=/scratch/bc3194/huggingface_cache/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TARGET=llada
export R4_G7_ASR1=1
export R4_TASK_ID=G4
export R4_ARR_LLaDA_STEPS=512
export ARR_RESULTS_DIR=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G4_llada_locked/arrattack
export TOKENIZERS_PARALLELISM=false

/scratch/bc3194/conda_envs/envs/raven/bin/python stage5_fullpool.py

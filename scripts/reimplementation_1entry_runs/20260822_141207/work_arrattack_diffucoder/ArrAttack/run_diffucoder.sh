#!/bin/bash
#SBATCH --job-name=arr_llada
#SBATCH --output=logs/stage5_llada_%j.out
#SBATCH --error=logs/stage5_llada_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

export TARGET=diffucoder
export DATASET_SPLIT=combined
export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/si2356/dlm-jailbreak-transfer/ArrAttack/logs
cd /scratch/si2356/dlm-jailbreak-transfer/ArrAttack
$PYTHON stage5_attack.py

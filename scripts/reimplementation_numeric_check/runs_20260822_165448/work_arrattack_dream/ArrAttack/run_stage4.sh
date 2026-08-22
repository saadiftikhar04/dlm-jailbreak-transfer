#!/bin/bash
#SBATCH --job-name=arr_stage4
#SBATCH --output=logs/stage4_%j.out
#SBATCH --error=logs/stage4_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/si2356/dlm-jailbreak-transfer/ArrAttack/logs
cd /scratch/si2356/dlm-jailbreak-transfer/ArrAttack
$PYTHON sft/sft_GenerationModel.py

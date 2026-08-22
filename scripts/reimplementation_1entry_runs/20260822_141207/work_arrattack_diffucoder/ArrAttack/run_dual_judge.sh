#!/bin/bash
#SBATCH --job-name=dual_judge
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/dual_judge_%j.out
#SBATCH --error=logs/dual_judge_%j.err

mkdir -p logs

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

export HF_HOME=/scratch/si2356/.cache/huggingface

/scratch/si2356/conda-envs/dlm_attack/bin/python dual_judge.py

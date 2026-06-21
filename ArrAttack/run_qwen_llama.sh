#!/bin/bash
# =============================================================================
# run_qwen_llama.sh — Stage 5: Qwen2.5 then LLaMA, sequential, 2x A100
# =============================================================================
#SBATCH --job-name=arr_qwen_llama
#SBATCH --output=logs/stage5_qwen_llama_%j.out
#SBATCH --error=logs/stage5_qwen_llama_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"
export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DATASET_SPLIT=combined

mkdir -p /scratch/si2356/dlm-jailbreak-transfer/ArrAttack/logs
cd /scratch/si2356/dlm-jailbreak-transfer/ArrAttack

# ── Run 1: Qwen2.5 on GPU 0 ──────────────────────────────────────────────────
echo "============================================================"
echo "START: Qwen2.5-7B-Instruct on GPU 0"
echo "============================================================"
TARGET=qwen2.5 CUDA_VISIBLE_DEVICES=0 $PYTHON stage5_attack.py
echo "============================================================"
echo "DONE: Qwen2.5 finished at $(date)"
echo "============================================================"

# ── Run 2: LLaMA on GPU 1 ────────────────────────────────────────────────────
echo "============================================================"
echo "START: Meta-Llama-3.1-8B-Instruct on GPU 1"
echo "============================================================"
TARGET=llama CUDA_VISIBLE_DEVICES=0 $PYTHON stage5_attack.py
echo "============================================================"
echo "DONE: LLaMA finished at $(date)"
echo "============================================================"

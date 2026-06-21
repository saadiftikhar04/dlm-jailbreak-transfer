#!/bin/bash
#SBATCH --job-name=arrattack_3prompt
#SBATCH --output=arrattack_3prompt_%j.out
#SBATCH --error=arrattack_3prompt_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

set -uo pipefail

echo "============================================================"
echo "ArrAttack 3-Prompt Full Pipeline"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "============================================================"

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/si2356/.cache/huggingface/datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

echo "Python: $($PYTHON --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

REPO=/scratch/si2356/dlm-jailbreak-transfer/ArrAttack
cd "$REPO"

# Ensure dataset symlink exists
[ ! -L dataset ] && ln -s /scratch/si2356/dlm-jailbreak-transfer/dataset dataset

$PYTHON arrattack_3prompt_full.py
EXIT=$?

echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "SUCCESS — End: $(date)"
    echo ""
    echo "Output files:"
    ls -lh /scratch/si2356/dlm-jailbreak-transfer/results/3prompt/ 2>/dev/null
else
    echo "FAILED (exit $EXIT) — End: $(date)"
fi
echo "====
========================================================"
exit $EXIT

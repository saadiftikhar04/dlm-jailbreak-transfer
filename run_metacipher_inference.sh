#!/bin/bash
#SBATCH --job-name=metacipher_inference
#SBATCH --output=metacipher_inference_%j.out
#SBATCH --error=metacipher_inference_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

set -euo pipefail

echo "============================================================"
echo "MetaCipher Inference Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "============================================================"

# Activate conda environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

# Verify python exists
if [ ! -f "$PYTHON" ]; then
    echo "✗ ERROR: Python not found at $PYTHON"
    exit 1
fi

# Set API key

# Point HF cache explicitly to scratch (belt-and-suspenders alongside local_files_only)
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/si2356/.cache/huggingface/datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "Python: $($PYTHON --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /scratch/si2356/dlm-jailbreak-transfer

echo "------------------------------------------------------------"
echo "Starting inference..."
echo "------------------------------------------------------------"

$PYTHON qwen_metacipher_inference.py
PYTHON_EXIT=$?

if [ $PYTHON_EXIT -ne 0 ]; then
    echo "✗ ERROR: Inference script failed with exit code $PYTHON_EXIT"
    exit $PYTHON_EXIT
fi

if [ ! -f metacipher_inference_results.csv ]; then
    echo "✗ ERROR: Output CSV was not produced"
    exit 1
fi

RESULT_LINES=$(wc -l < metacipher_inference_results.csv)
echo "------------------------------------------------------------"
echo "✓ SUCCESS: Inference complete"
echo "Output file: metacipher_inference_results.csv ($RESULT_LINES lines)"
echo "End: $(date)"
echo "------------------------------------------------------------"

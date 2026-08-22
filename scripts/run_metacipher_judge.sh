#!/bin/bash
#SBATCH --job-name=metacipher_judge
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=metacipher_judge_%j.out
#SBATCH --error=metacipher_judge_%j.err

# Setup environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

echo "=================================================="
echo "MetaCipher Judging - Step 2 of 2"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Python path: $(which python)"
python --version

# CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export HF_HOME=/scratch/si2356/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/si2356/.cache/huggingface/hub
export FLASHINFER_WORKSPACE_DIR=/scratch/si2356/.cache/flashinfer
export VLLM_CONFIG_ROOT=/scratch/si2356/.cache/vllm
export XDG_CACHE_HOME=/scratch/si2356/.cache

# Change to working directory
cd /scratch/si2356/dlm-jailbreak-transfer

# Check if inference results exist
if [ ! -f "metacipher_inference_results.csv" ]; then
    echo ""
    echo "=================================================="
    echo "ERROR: metacipher_inference_results.csv not found!"
    echo "Please run scripts/qwen_metacipher_inference.py first"
    echo "Submit with: sbatch scripts/run_metacipher_inference.sh"
    echo "=================================================="
    exit 1
fi

echo "Found inference results:"
echo "  File: metacipher_inference_results.csv"
echo "  Size: $(du -h metacipher_inference_results.csv | cut -f1)"
echo "  Number of results: $(tail -n +2 metacipher_inference_results.csv | wc -l)"

# Clear CUDA cache
echo ""
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Run judging
echo ""
echo "Starting HarmBench judging..."
echo "=================================================="
/scratch/si2356/conda-envs/dlm_attack/bin/python scripts/qwen_metacipher_judge.py \
    --behaviors_path /scratch/si2356/dlm-jailbreak-transfer/dataset/harmbench/text_all.csv
status=$?
if [ $status -ne 0 ]; then
  echo "✗ ERROR: Judge failed (exit code $status)"
  exit $status
fi

# Check if output file was created
if [ -f "metacipher_evaluation_results.json" ]; then
    echo ""
    echo "=================================================="
    echo "SUCCESS: Judging completed!"
    echo "Output file: metacipher_evaluation_results.json"
    echo "File size: $(du -h metacipher_evaluation_results.json | cut -f1)"
    echo ""
    echo "Quick Statistics:"
    /scratch/si2356/conda-envs/dlm_attack/bin/python -c "
import json
with open('metacipher_evaluation_results.json') as f:
    data = json.load(f)
print(f'  Total behaviors: {data[\"num_behaviors\"]}')
print(f'  Overall ASR: {data[\"overall_asr\"]:.1%}')
print(f'  Avg attempts to jailbreak: {data[\"avg_attempts_to_jailbreak\"]:.2f}')
" 2>/dev/null || echo "  (Could not calculate statistics)"
else
    echo ""
    echo "=================================================="
    echo "ERROR: Output file not found!"
    echo "Check the error log: metacipher_judge_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "=================================================="

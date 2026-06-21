#!/bin/bash
#SBATCH --job-name=metacipher
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=metacipher_%j.out
#SBATCH --error=metacipher_%j.err

# Use full path to python in dlm_attack environment directly
export PATH="/scratch/si2356/conda-envs/dlm_attack/bin:$PATH"

# Verify we're using the right python
echo "Python path: $(which python)"
python --version

# Set DeepSeek API key
export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"

# Set CUDA environment
export CUDA_LAUNCH_BLOCKING=1

# Navigate to working directory
cd /scratch/si2356/dlm-jailbreak-transfer

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Run the script (using correct filename)
/scratch/si2356/conda-envs/dlm_attack/bin/python metacipher_qwen2.5.py

echo "MetaCipher experiment completed!"

#!/bin/bash
# T1 plaintext control generation - ebrainccs condo, one GPU per victim.
# __VKEY__ = qwen2.5 | falcon | llama | llada | dream | diffucoder
#SBATCH --job-name=t1_gen___VKEY__
#SBATCH --partition=condo
#SBATCH --qos=ebrainccs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=56G
#SBATCH --output=t1___VKEY__.%j.out
#SBATCH --error=t1___VKEY__.%j.err
set -euo pipefail

ROOT=/scratch/bc3194/dlm-jailbreak-transfer
cd "$ROOT/tier0/18_t1_plaintext_control"

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=0
export CC=/share/apps/NYUAD5/gcc/9.2.0/bin/gcc
export CXX=/share/apps/NYUAD5/gcc/9.2.0/bin/g++
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] T1 gen victim=__VKEY__"
/scratch/bc3194/conda_envs/envs/raven/bin/python t1_generate.py __VKEY__ || { echo "FAILED __VKEY__"; exit 1; }
echo "[$(date)] done __VKEY__"
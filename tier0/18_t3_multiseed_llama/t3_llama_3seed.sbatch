#!/bin/bash
# T3 - Llama MetaCipher 3-seed generation. ebrainccs condo.
#SBATCH --job-name=t3_llama_3seed
#SBATCH --partition=condo
#SBATCH --qos=ebrainccs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=56G
#SBATCH --output=t3_llama3seed.%j.out
#SBATCH --error=t3_llama3seed.%j.err
set -ex
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export CC=/share/apps/NYUAD5/gcc/9.2.0/bin/gcc
export CXX=/share/apps/NYUAD5/gcc/9.2.0/bin/g++
export MODEL_KEY=llama
/scratch/bc3194/conda_envs/envs/raven/bin/python tier0/12_low_asr_multiseed/t12_generate.py "$MODEL_KEY"
echo "=== RC=$? ==="
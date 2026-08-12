#!/bin/bash
#SBATCH --job-name=pif_llama
#SBATCH --output=logs/pif_llama_%j.out
#SBATCH --error=logs/pif_llama_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

export TARGET=llama
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DEEPSEEK_API_KEY=sk-28b099947e1d4690b93914ad9ccafcdf

PIF_DIR=/scratch/si2356/dlm-jailbreak-transfer/PiF
mkdir -p $PIF_DIR/logs

for DATASET in strongreject; do
    echo "=== PiF: target=llama dataset=$DATASET ==="
    cd $PIF_DIR
    $PYTHON run_pif.py \
        --target llama \
        --dataset $DATASET \
        --bert_path /scratch/si2356/.cache/huggingface/hub/models--google-bert--bert-large-uncased/snapshots/6da4b6a26a1877e173fca3225479512db81a5e5b \
        --opt_objective ASR+GPT
done

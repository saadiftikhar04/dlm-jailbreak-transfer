#!/bin/bash
#SBATCH --job-name=arrattack_dlms
#SBATCH --output=/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/logs/arrattack_dlms_%j.out
#SBATCH --error=/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/logs/arrattack_dlms_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=120:00:00

# ─────────────────────────────────────────────────────────────────────────────
# ArrAttack — Stage 5 on the 3 diffusion language models
#   1. LLaDA-1.5
#   2. Dream-v0-Instruct-7B
#   3. DiffuCoder-7B-Instruct
#
# Runs sequentially on a single GPU. Each model writes to:
#   /scratch/si2356/dlm-jailbreak-transfer/results/<model>/arrattack_results.csv
#   /scratch/si2356/dlm-jailbreak-transfer/results/<model>/arrattack_progress.csv
#
# Dataset (165 prompts total, in order):
#   harmbench (last 75) + strongreject (last 50)
#   + jailbreakbench (last 20) + malicious_instruct (last 20)
#
# DLM hyperparameters are 100% MetaCipher-faithful (inlined in stage5_attack.py):
#   LLaDA : steps=128, gen_length=512, block_length=128, temperature=0.,
#           cfg_scale=0., remasking='low_confidence', mask_id=126336
#   Dream / DiffuCoder : steps=128, temperature=0.2, top_p=0.95,
#                        alg="entropy", alg_temp=0.
# ─────────────────────────────────────────────────────────────────────────────

set -u   # error on undefined variables; do NOT set -e (we want to continue
         # to the next DLM even if one fails)

# ── Environment ──────────────────────────────────────────────────────────────
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

cd /scratch/si2356/dlm-jailbreak-transfer/ArrAttack
mkdir -p logs

export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DeepSeek API key (secondary judge)
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${DEEPSEEK_API_KEY}}"

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "=========================================================="
echo "ArrAttack Stage 5 — DLM sweep"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================================="

# ── Run each DLM ─────────────────────────────────────────────────────────────
DLMS=(llada dream diffucoder)

for TARGET in "${DLMS[@]}"; do
    echo ""
    echo "##########################################################"
    echo "##  TARGET = $TARGET"
    echo "##  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "##########################################################"

    TARGET=$TARGET python3 stage5_attack.py
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "##  $TARGET FINISHED OK at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "##  $TARGET FAILED (exit $EXIT_CODE) at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "##  Continuing to next DLM..."
    fi

    # Free GPU memory between runs
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo ""
echo "=========================================================="
echo "All DLM runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dirs:"
for TARGET in "${DLMS[@]}"; do
    DIR=/scratch/si2356/dlm-jailbreak-transfer/results/$TARGET
    if [ -d "$DIR" ]; then
        echo "  $DIR"
        ls -lh "$DIR"/*.csv 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
    fi
done
echo "=========================================================="

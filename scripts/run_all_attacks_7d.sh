#!/bin/bash
#SBATCH --job-name=all_attacks_7d
#SBATCH --output=logs/all_attacks_7d_%j.out
#SBATCH --error=logs/all_attacks_7d_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00

# 7-day comprehensive attack sweep
# All 3 attacks × Dream & DiffuCoder × 4 datasets each

set -u
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR="/scratch/si2356/dlm-jailbreak-transfer"
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python
mkdir -p logs

echo "=================================================================="
echo "7-Day Comprehensive Attack Sweep"
echo "Started: $(date)"
echo "=================================================================="

# ── METACIPHER ──────────────────────────────────────────────────────
echo ""
echo ">>> [$(date +'%F %T')] METACIPHER: Dream (4 datasets)"
cd "${PROJECT_DIR}"

${PYTHON} -u scripts/metacipher_multi.py \
    --model dream \
    --max-attempts 5 \
    --checkpoint-every 10 \
    --datasets harmbench,strongreject,malicious_instruct,jailbreakbench \
    --run-suffix _final 2>&1 | tee logs/metacipher_dream_final.log

echo ">>> [$(date +'%F %T')] METACIPHER: DiffuCoder (4 datasets)"

${PYTHON} -u scripts/metacipher_multi.py \
    --model diffucoder \
    --max-attempts 5 \
    --checkpoint-every 10 \
    --datasets harmbench,strongreject,malicious_instruct,jailbreakbench \
    --run-suffix _final 2>&1 | tee logs/metacipher_diffucoder_final.log

# ── PiF ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [$(date +'%F %T')] PiF: Dream (3 datasets)"
cd "${PROJECT_DIR}/PiF"

TARGET=dream ${PYTHON} -u run_pif.py 2>&1 | tee ../logs/pif_dream_final.log

echo ">>> [$(date +'%F %T')] PiF: DiffuCoder (3 datasets)"

TARGET=diffucoder ${PYTHON} -u run_pif.py 2>&1 | tee ../logs/pif_diffucoder_final.log

# ── ARRATTACK ────────────────────────────────────────────────────────
echo ""
echo ">>> [$(date +'%F %T')] ArrAttack: Dream (4 datasets)"
cd "${PROJECT_DIR}/ArrAttack"

TARGET=dream ${PYTHON} -u stage5_attack.py 2>&1 | tee ../logs/arrattack_dream_final.log

echo ">>> [$(date +'%F %T')] ArrAttack: DiffuCoder (4 datasets)"

TARGET=diffucoder ${PYTHON} -u stage5_attack.py 2>&1 | tee ../logs/arrattack_diffucoder_final.log

# ── SUMMARY ──────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "7-Day Comprehensive Attack Sweep COMPLETE"
echo "Finished: $(date)"
echo "=================================================================="
echo ""
echo "Results locations:"
echo "  MetaCipher  → ${PROJECT_DIR}/metacipher_*_final_results_FINAL.csv"
echo "  PiF         → ${PROJECT_DIR}/results/pif/{dream,diffucoder}/*"
echo "  ArrAttack   → ${PROJECT_DIR}/results/{dream,diffucoder}/"
echo ""

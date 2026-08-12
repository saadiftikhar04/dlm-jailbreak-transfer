#!/bin/bash
#SBATCH --job-name=mc_all6
#SBATCH --output=logs/metacipher_all_%j.out
#SBATCH --error=logs/metacipher_all_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00

# ──────────────────────────────────────────────────────────────
# MetaCipher full sweep: 6 models, sequential
#   AR  : qwen2.5  → falcon  → llama
#   DLM : llada    → dream   → diffucoder
# ──────────────────────────────────────────────────────────────
set -u

# ── Conda env (matches PiF / ArrAttack scripts) ──────────────
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

# ── Environment variables ────────────────────────────────────
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ── Working directory ────────────────────────────────────────
METACIPHER_DIR=/scratch/si2356/dlm-jailbreak-transfer
cd "${METACIPHER_DIR}" || { echo "Cannot cd to ${METACIPHER_DIR}"; exit 1; }
mkdir -p logs

# ── Banner ───────────────────────────────────────────────────
echo "=================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Started        : $(date)"
echo "Working dir    : $(pwd)"
echo "Python         : ${PYTHON}"
echo "Conda env      : ${CONDA_PREFIX}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=================================================================="

# ── Models, in order ─────────────────────────────────────────
MODELS=(
    "qwen2.5"      # AR  (Qwen/Qwen2.5-7B-Instruct)
    "falcon"       # AR  (tiiuae/Falcon-H1R-7B)
    "llama"        # AR  (meta-llama/Llama-3.2-3B-Instruct)
    "llada"        # DLM (GSAI-ML/LLaDA-1.5)
    "dream"        # DLM (Dream-org/Dream-v0-Instruct-7B)
    "diffucoder"   # DLM (apple/DiffuCoder-7B-Instruct)
)

OVERALL_START=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=================================================================="
    echo ">>> [$(date +'%F %T')]  STARTING model: ${MODEL}"
    echo "=================================================================="

    MODEL_START=$(date +%s)
    LOG_FILE="logs/metacipher_${MODEL}_${SLURM_JOB_ID}.log"

    ${PYTHON} -u metacipher_multi.py \
        --model "${MODEL}" \
        --max-attempts 5 \
        --checkpoint-every 10 \
        2>&1 | tee "${LOG_FILE}"

    RC=${PIPESTATUS[0]}
    MODEL_END=$(date +%s)
    ELAPSED=$(( MODEL_END - MODEL_START ))
    H=$(( ELAPSED / 3600 ))
    M=$(( (ELAPSED % 3600) / 60 ))
    S=$(( ELAPSED % 60 ))

    if [ "${RC}" -eq 0 ]; then
        echo ">>> [$(date +'%F %T')]  FINISHED ${MODEL}  (rc=0, elapsed ${H}h${M}m${S}s)"
    else
        echo ">>> [$(date +'%F %T')]  FAILED   ${MODEL}  (rc=${RC}, elapsed ${H}h${M}m${S}s)"
        echo ">>> Per-model log: ${LOG_FILE}"
    fi

    # Free GPU memory between runs
    ${PYTHON} -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

# ── Wrap-up ──────────────────────────────────────────────────
OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))
TH=$(( TOTAL / 3600 ))
TM=$(( (TOTAL % 3600) / 60 ))
TS=$(( TOTAL % 60 ))

echo ""
echo "=================================================================="
echo "ALL MODELS DONE"
echo "Total wall time : ${TH}h${TM}m${TS}s"
echo "Finished        : $(date)"
echo "=================================================================="

echo ""
echo "Final CSVs in $(pwd):"
ls -lh metacipher_*_results_FINAL.csv 2>/dev/null || echo "  (none found)"

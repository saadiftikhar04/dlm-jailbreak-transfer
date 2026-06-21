#!/bin/bash
#SBATCH --job-name=pif_dd
#SBATCH --output=logs/pif_dd_%j.out
#SBATCH --error=logs/pif_dd_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00

set -uo pipefail

# ── env (mirrors run_pif_llada.sh exactly) ───────────────────────────────────
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python
export PYTHONNOUSERSITE=1

export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ── paths ────────────────────────────────────────────────────────────────────
PIF_DIR=/scratch/si2356/dlm-jailbreak-transfer/PiF
BERT_PATH=/scratch/si2356/.cache/huggingface/hub/models--google-bert--bert-large-uncased/snapshots/6da4b6a26a1877e173fca3225479512db81a5e5b
LOGDIR=${PIF_DIR}/logs

mkdir -p "$LOGDIR"
cd "$PIF_DIR" || { echo "ERROR: cannot cd to $PIF_DIR"; exit 1; }

echo "============================================================"
echo " PiF diffu+dream runner"
echo " host   : $(hostname)"
echo " gpu    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " time   : $(date)"
echo "============================================================"

"$PYTHON" -c "import yaml,torch,transformers,openai; print('deps OK')" \
    || { echo "ERROR: deps missing"; exit 1; }

# ── NO cleanup block — llada is running in parallel, don't touch anything ────

MODELS=(diffucoder dream)
TS=$(date +%Y%m%d_%H%M%S)
declare -A STATUS

for m in "${MODELS[@]}"; do
    log="${LOGDIR}/pif_${m}_${TS}.log"
    echo
    echo "============================================================"
    echo ">> RUNNING: TARGET=${m}"
    echo "   log -> ${log}"
    echo "   start: $(date)"
    echo "============================================================"

    export TARGET="$m"
    "$PYTHON" run_pif.py \
        --target "$m" \
        --bert_path "$BERT_PATH" \
        --opt_objective ASR 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    STATUS["$m"]=$rc

    if [ "$rc" -eq 0 ]; then
        echo ">> ${m} FINISHED OK at $(date)"
    else
        echo ">> ${m} FAILED (exit ${rc}) at $(date) — continuing"
    fi
done

echo
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
overall=0
for m in "${MODELS[@]}"; do
    rc=${STATUS["$m"]}
    [ "$rc" -eq 0 ] && echo "   ${m}: OK" || { echo "   ${m}: FAILED (exit ${rc})"; overall=1; }
done
echo " finished: $(date)"
echo "============================================================"
exit "$overall"

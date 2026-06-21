#!/bin/bash
# =============================================================================
# run_pif_dlm.sh
# -----------------------------------------------------------------------------
# Cleans all previous PiF results/checkpoints for the three diffusion models,
# then runs them sequentially in the required order:
#       1) llada   2) diffucoder   3) dream
#
# Env / account / qos / gres all mirror the known-good run_pif_llada.sh.
#
# Submit with:   sbatch run_pif_dlm.sh
# =============================================================================
#SBATCH --job-name=pif_dlm
#SBATCH --output=logs/pif_dlm_%j.out
#SBATCH --error=logs/pif_dlm_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00

set -uo pipefail   # NOT -e: one model failing must not abort the others

# ---- conda env (exactly as in run_pif_llada.sh) -----------------------------
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack
PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python
export PYTHONNOUSERSITE=1   # block ~/.local shadowing (the yaml-import failure)

# ---- runtime env vars (exactly as in run_pif_llada.sh) ----------------------
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- paths ------------------------------------------------------------------
PROJECT="/scratch/si2356/dlm-jailbreak-transfer"
PIF_DIR="${PROJECT}/PiF"
RESULTS="${PROJECT}/results/pif"
LOGDIR="${PIF_DIR}/logs"
BERT_PATH="/scratch/si2356/.cache/huggingface/hub/models--google-bert--bert-large-uncased/snapshots/6da4b6a26a1877e173fca3225479512db81a5e5b"

# ---- run order (DO NOT REORDER) ---------------------------------------------
MODELS=(llada diffucoder dream)

# ---- sanity checks ----------------------------------------------------------
cd "$PIF_DIR" || { echo "ERROR: cannot cd to $PIF_DIR"; exit 1; }
[ -f run_pif.py ] || { echo "ERROR: run_pif.py not found in $PIF_DIR"; exit 1; }
mkdir -p "$LOGDIR"

echo "============================================================"
echo " PiF DLM batch runner"
echo " host    : $(hostname)"
echo " gpu     : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " python  : ${PYTHON}"
echo " models  : ${MODELS[*]}"
echo " time    : $(date)"
echo "============================================================"

# quick import guard — fail loudly here rather than mid-run
"$PYTHON" -c "import yaml, torch, transformers, openai; print('  deps OK:', __import__('sys').executable)" \
    || { echo "ERROR: dependency import failed in dlm_attack env"; exit 1; }

# ---- 1. CLEAN previous results + checkpoints --------------------------------
# Removes results.csv, steps.csv, checkpoint.json, summary.json for ALL
# datasets under each of the three DLM targets. llama/falcon/qwen untouched.
echo
echo ">> Removing previous results/checkpoints for: ${MODELS[*]}"
for m in "${MODELS[@]}"; do
    tgt="${RESULTS}/${m}"
    if [ -d "$tgt" ]; then
        echo "   rm -rf ${tgt}"
        rm -rf "$tgt"
    else
        echo "   (nothing to remove at ${tgt})"
    fi
done
echo ">> Clean complete."

# ---- 2. RUN each model sequentially -----------------------------------------
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
        --opt_objective ASR+GPT 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    STATUS["$m"]=$rc

    if [ "$rc" -eq 0 ]; then
        echo ">> ${m} FINISHED OK (exit ${rc}) at $(date)"
    else
        echo ">> ${m} FAILED (exit ${rc}) at $(date) — continuing to next model"
    fi
done

# ---- 3. summary -------------------------------------------------------------
echo
echo "============================================================"
echo " RUN SUMMARY"
echo "============================================================"
overall=0
for m in "${MODELS[@]}"; do
    rc=${STATUS["$m"]}
    if [ "$rc" -eq 0 ]; then
        echo "   ${m}: OK"
    else
        echo "   ${m}: FAILED (exit ${rc})"
        overall=1
    fi
done
echo " finished: $(date)"
echo "============================================================"

exit "$overall"

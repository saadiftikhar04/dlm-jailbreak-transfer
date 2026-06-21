#!/bin/bash
#SBATCH --job-name=metacipher_dlm_jbb_rerun
#SBATCH --output=/scratch/si2356/dlm-jailbreak-transfer/logs/dlm_jbb_rerun_%j.out
#SBATCH --error=/scratch/si2356/dlm-jailbreak-transfer/logs/dlm_jbb_rerun_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nvidia
#SBATCH --mail-type=END,FAIL

echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

source ~/.bashrc
conda activate base

export HF_HOME=/scratch/si2356/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/si2356/.cache/huggingface/hub
export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"

WORKDIR=/scratch/si2356/dlm-jailbreak-transfer
cd "${WORKDIR}" || { echo "ERROR: cannot cd to ${WORKDIR}"; exit 1; }

echo ""
echo "=== Applying DLM fix patch ==="
python3 /scratch/si2356/dlm-jailbreak-transfer/apply_dlm_fix.py
PATCH_EXIT=$?
if [ ${PATCH_EXIT} -ne 0 ]; then
    echo "ERROR: patch failed (exit ${PATCH_EXIT}). Aborting."
    exit ${PATCH_EXIT}
fi
echo "Patch applied successfully."

echo ""
echo "=== Sanity check: DLM branch in patched file ==="
grep -n 'if model_type == "dlm"' metacipher_multi.py \
    && echo "✓ DLM branch confirmed" \
    || { echo "ERROR: DLM branch missing after patch"; exit 1; }

echo ""
echo "=== Running Dream-v0 on jailbreakbench ==="
echo "Start: $(date)"
python3 metacipher_multi.py \
    --model dream \
    --datasets jailbreakbench \
    --run-suffix _jbb_v2 \
    --max-attempts 5 \
    --checkpoint-every 10
DREAM_EXIT=$?
echo "Dream finished at $(date) — exit code: ${DREAM_EXIT}"

echo ""
echo "=== Running DiffuCoder on jailbreakbench ==="
echo "Start: $(date)"
python3 metacipher_multi.py \
    --model diffucoder \
    --datasets jailbreakbench \
    --run-suffix _jbb_v2 \
    --max-attempts 5 \
    --checkpoint-every 10
DIFFUCODER_EXIT=$?
echo "DiffuCoder finished at $(date) — exit code: ${DIFFUCODER_EXIT}"

echo ""
echo "=== Results summary ==="
for model in dream diffucoder; do
    RESULTS="${WORKDIR}/metacipher_${model}_jbb_v2_results_FINAL.csv"
    if [ -f "${RESULTS}" ]; then
        python3 -c "
import pandas as pd
df = pd.read_csv('${RESULTS}')
n_total = len(df)
n_success = df['success'].sum()
asr = n_success / n_total if n_total > 0 else 0
print(f'${model}: {n_success}/{n_total} = {asr:.3f} ASR')
fm = df[df[\"success\"]==False][\"failure_mode\"].value_counts().to_dict()
print(f'  failure modes: {fm}')
"
    else
        echo "${model}: no results file found"
    fi
done

echo ""
echo "=== Job done at $(date) ==="
exit $(( DREAM_EXIT | DIFFUCODER_EXIT ))

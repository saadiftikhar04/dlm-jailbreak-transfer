#!/bin/bash
#SBATCH --job-name=arrattack_pipeline
#SBATCH --output=arrattack_pipeline_%j.out
#SBATCH --error=arrattack_pipeline_%j.err
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00

set -uo pipefail

echo "============================================================"
echo "ArrAttack Pipeline (forked repo)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "============================================================"

# ── Activate conda environment ────────────────────────────────────────────────
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /scratch/si2356/conda-envs/dlm_attack

PYTHON=/scratch/si2356/conda-envs/dlm_attack/bin/python

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

# ── API key ───────────────────────────────────────────────────────────────────
export DEEPSEEK_API_KEY="sk-80b9c3e36a374e7489c5ac4438139fdb"

# ── HuggingFace cache (offline — all models pre-downloaded) ──────────────────
export HF_HOME=/scratch/si2356/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/si2356/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/si2356/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/si2356/.cache/huggingface/datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── CUDA tuning ───────────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

echo "Python:               $($PYTHON --version)"
echo "GPU:                  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"

# ── Change into the forked repo directory ─────────────────────────────────────
REPO=/scratch/si2356/dlm-jailbreak-transfer/ArrAttack
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
for f in run_harmbench.py smoothllm.py utils/qwen_utils.py \
          build_jailbreak_samples.py generate_robustPrompts.py \
          sft_RobustnessModel.py sft/sft_GenerationModel.py sft/generate.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file missing: $REPO/$f"
        exit 1
    fi
done

HARMBENCH="dataset/harmbench/text_all.csv"
# dataset lives one level up — symlink so scripts find it via relative path
if [ ! -f "$HARMBENCH" ] && [ ! -L "dataset" ]; then
    ln -s /scratch/si2356/dlm-jailbreak-transfer/dataset dataset
    echo "Symlinked dataset -> /scratch/si2356/dlm-jailbreak-transfer/dataset"
fi

echo "------------------------------------------------------------"
echo "Stage 1-2: BRJ + SmoothLLM -> judgment dataset + SFT"
echo "Stage 3-4: BRJwr -> generation dataset + SFT"
echo "Stage 5:   Attack inference on test prompts"
echo "------------------------------------------------------------"

# Pass --smoke for a quick 5-prompt sanity check:
#   sbatch run_arrattack.sh --smoke
EXTRA="${1:-}"

$PYTHON run_harmbench.py $EXTRA
EXIT=$?

if [ $EXIT -ne 0 ]; then
    echo "ERROR: Pipeline failed with exit code $EXIT"
    exit $EXIT
fi

# ── Output summary ────────────────────────────────────────────────────────────
RESULTS=/scratch/si2356/dlm-jailbreak-transfer/results
PRIMARY="$RESULTS/arrattack_results.csv"
EVAL="$RESULTS/arrattack_evaluation_results.csv"

echo "------------------------------------------------------------"
echo "SUCCESS: ArrAttack pipeline complete"
echo ""
echo "Results:"
[ -f "$PRIMARY" ] && echo "  $PRIMARY  ($(wc -l < "$PRIMARY") lines)"
[ -f "$EVAL"    ] && echo "  $EVAL"

echo ""
echo "SFT checkpoints:"
PROJ=/scratch/si2356/dlm-jailbreak-transfer
[ -d "$PROJ/robustness_judgment_model" ] && echo "  robustness_judgment_model/"
[ -d "$PROJ/generation_model"          ] && echo "  generation_model/"

echo ""
echo "End: $(date)"
echo "------------------------------------------------------------"

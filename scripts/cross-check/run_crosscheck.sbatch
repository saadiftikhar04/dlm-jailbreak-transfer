#!/bin/bash
#SBATCH --job-name=dlm-crosscheck
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:3
#SBATCH --time=71:59:59
#SBATCH --mem=100G
#SBATCH --output=crosscheck_%j.out
#SBATCH --error=crosscheck_%j.err

###############################################################################
# run_crosscheck.sbatch — Faithfully reproduce the sampled DLM-jailbreak rows
#                          on the NYUAD Jubail HPC (ebrainccs condo, 3x A100-80G).
#
# Reproduces ONLY the sampled rows (from scripts/cross-check/outputs/manifest.json),
# regenerating each victim response and diffing against the recorded CSV value
# (see 06_compare.py). Writes under scripts/cross-check/outputs/reproduced/.
#
# HARD REQUIREMENTS — verify BEFORE submitting (run 04_preflight.py):
#   The results/ tree (the thing being checked) ships in git — just git-pull the
#   repo on the HPC and this script compares against <repo>/results directly; no
#   extra results path is needed.
#
#   The STUDENT WEIGHTS + DATASET SOURCE for faithful reproduction must be reachable
#   by bc3194. The student's real data is at /scratch/si2356/dlm-jailbreak-transfer
#   BUT that dir is mode 700 owned by si2356 -> bc3194 CANNOT read it. Copy it under
#   bc3194's scratch (or get it chmod'd), then record the path in <repo>/paths.txt:
#
#       /scratch/bc3194/dlm-jailbreak-transfer     # line 1 = repo root
#       DATA_DIR=/scratch/bc3194/dlm-jailbreak-data
#       HF_HOME=/scratch/bc3194/huggingface_cache
#
#   paths.txt is git-ignored (one copy per machine). Every runner in cross-check/
#   reads it; nothing is hard-coded. DEEPSEEK_API_KEY (metacipher judge) is sourced
#   from ~/.bashrc below.
#
# NOTE on PiF/ArrAttack: run_pif.py imports repo-local modules deepseek_patch.py +
# attack_mlm.py and stage5_attack.py imports the ArrAttack/ tree, none of which are
# in this checkout. If they are absent on this host the reproducer reports
# REPRO-BLOCKER (exit 3) and those samples are marked MISSING — NOT a silent bogus
# result. Supply those files in the data dir / scripts to unblock them.
###############################################################################

set -euo pipefail

module purge

# /etc/bashrc (sourced via ~/.bashrc) reads $BASHRCSOURCED on line 12; under
# `set -u` that unbound var aborts the batch shell instantly (job 17159190.
# failed 0s with "BASHRCSOURCED: unbound variable"). Guard the source.
set +u
source ~/.bashrc
set -u

# --- select the three GPUs (A100-80G each) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

# --- repo root on the HPC: from paths.txt (per-machine), else submit dir ---
PT="$SLURM_SUBMIT_DIR/paths.txt"
if [[ -f "$PT" ]]; then
  export DLM_REPO_ROOT="$(grep -m1 -vE '^\s*#|^\s*$' "$PT" | tr -d '[:space:]')"
fi
REPO="${DLM_REPO_ROOT:-${REPO:-$SLURM_SUBMIT_DIR}}"
CHK="$REPO/scripts/cross-check"
cd "$REPO"

# --- HF cache: from paths.txt override, else env, else HPC default ---
HF_HOME_FROM_PT="$(grep -iE '^HF_HOME=' "$PT" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
export HF_HOME="${HF_HOME:-${HF_HOME_FROM_PT:-/scratch/bc3194/huggingface_cache}}"
export CROSSCHECK_HF_CACHE="${CROSSCHECK_HF_CACHE:-$HF_HOME/hub}"

# --- student data dir: from paths.txt override, else env ---
DATA_FROM_PT="$(grep -iE '^DATA_DIR=' "$PT" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
export CROSSCHECK_DATA_DIR="${CROSSCHECK_DATA_DIR:-${DATA_FROM_PT:-}}"

# --- DeepSeek key for the metacipher judge (from ~/.bashrc, non-interactive) ---
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc | tail -1 | cut -d= -f2- | tr -d '"' || true)}"

echo "=== SLURM ======================="
echo "job:   ${SLURM_JOB_ID:-?}  node: ${SLURM_JOB_NODELIST:-?}"
echo "gpus:  $CUDA_VISIBLE_DEVICES"
echo "repo:  $REPO"
echo "paths: $(cat "$PT" 2>/dev/null | tr '\n' ';')"
echo "data:  ${CROSSCHECK_DATA_DIR:-<unset — reproduction will be MISSING>}"
echo "cache: $CROSSCHECK_HF_CACHE"
echo "key:   ${DEEPSEEK_API_KEY:+set}"
echo "================================="
nvidia-smi

# 1. Inventory + sample must match the local manifest (seed fixed). Regenerate on
#    the HPC so the manifest exists here; identical seed -> identical manifest.
python3 "$CHK/01_inventory.py"
python3 "$CHK/02_sample.py"

# 2. Zero-GPU sanity (fast; catches contradictions before spending GPU)
python3 "$CHK/03_deterministic_check.py"
python3 "$CHK/04_preflight.py"

# 3. Faithful reproduction of every sampled row (the expensive part)
bash "$CHK/05_reproduce.sh"

# 4. Diff recorded vs reproduced
python3 "$CHK/06_compare.py"

echo "=== DONE === cross-check finished. See $CHK/outputs/compare.txt"
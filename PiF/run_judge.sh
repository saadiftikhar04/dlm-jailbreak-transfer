#!/usr/bin/env bash
# run_judge.sh — set up, validate, and run the semantic PiF judge on the cluster.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash run_judge.sh                      # judge qwen llama diffucoder llada
#   ROOT=results OUT=pif_judged bash run_judge.sh
#   MODELS="all" bash run_judge.sh         # all six
#
# It will: self-test -> dry-run (shows API calls, no spend) -> ask -> judge.
#
# (Login-node friendly: judging is API/IO bound, no GPU needed. To submit as a
#  Slurm job instead, see the commented #SBATCH header at the bottom.)

set -euo pipefail

# ---- config (override via env) -------------------------------------------- #
ROOT="${ROOT:-pif_runs}"
OUT="${OUT:-pif_judged}"
MODELS="${MODELS:-qwen llama diffucoder llada}"
BACKEND="${BACKEND:-deepseek}"
WORKERS="${WORKERS:-8}"
CACHE="${CACHE:-judge_cache.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
JUDGE="$SCRIPT_DIR/judge_pif_llm.py"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

bold "=== PiF semantic judge runner ==="
echo "ROOT=$ROOT  OUT=$OUT  BACKEND=$BACKEND  WORKERS=$WORKERS"
echo "MODELS=$MODELS"
echo

# ---- 1. sanity: files & deps ---------------------------------------------- #
[ -f "$JUDGE" ] || fail "judge_pif_llm.py not found next to this script ($JUDGE)"
[ -d "$ROOT" ]  || fail "results root '$ROOT' not found (set ROOT=... to your pif_runs dir)"

command -v "$PY" >/dev/null 2>&1 || fail "python not found (set PYTHON=...)"
"$PY" - <<'PYDEPS' || fail "missing python deps. Run: pip install pandas"
import importlib, sys
for mod in ("pandas",):
    importlib.import_module(mod)
print("deps ok:", sys.version.split()[0])
PYDEPS

# ---- 2. key present? (backend-specific) ----------------------------------- #
if [ "$BACKEND" = "deepseek" ]; then
  [ -n "${DEEPSEEK_API_KEY:-}" ] || fail "DEEPSEEK_API_KEY is not set. 'export DEEPSEEK_API_KEY=sk-...'"
else
  [ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY is not set."
fi

# ---- 3. self-test (no API) ------------------------------------------------ #
bold "[1/3] self-test (no API)"
"$PY" "$JUDGE" --self-test

# ---- 4. dry-run (no API spend) -------------------------------------------- #
bold "[2/3] dry-run — how many API calls are needed?"
"$PY" "$JUDGE" --root "$ROOT" --out "$OUT" --models $MODELS --dry-run

# ---- 5. confirm then run -------------------------------------------------- #
if [ "${ASSUME_YES:-0}" != "1" ] && [ -t 0 ]; then
  read -r -p "Proceed with semantic judging (this spends API budget)? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0;; esac
fi

bold "[3/3] semantic judging"
"$PY" "$JUDGE" --root "$ROOT" --out "$OUT" --models $MODELS \
    --backend "$BACKEND" --workers "$WORKERS" --cache "$CACHE"

bold "=== done ==="
echo "Per-model CSVs + pif_summary.csv written to: $OUT"
echo "Cache (resumable): $CACHE"

# --------------------------------------------------------------------------- #
# Optional Slurm submission (judging needs no GPU). To use:
#   sbatch run_judge.sh
# and uncomment:
# #SBATCH --job-name=pif_judge
# #SBATCH --partition=compute
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=8G
# #SBATCH --time=02:00:00
# #SBATCH --output=pif_judge_%j.log
# (Slurm reads #SBATCH lines only if they are at the very top of the file, so
#  move them above 'set -euo pipefail' if you submit with sbatch.)
# --------------------------------------------------------------------------- #

#!/usr/bin/env bash
#
# 05_reproduce.sh — Reproduction runner for the sampled rows.
#
# WHAT THIS DOES
# --------------
# Each sampled row carries its ORIGINAL PROMPT + the recorded victim response
# (see outputs/manifest.json). To trust the student's numbers we must re-run the
# actual attack on a machine that HAS the model weights + dataset, regenerate the
# victim response for the sampled prompts, and diff it against the recorded one.
#
# This script is the faithful-reproduction driver. Because the three attack
# scripts hardcode the student's HPC paths as module-level constants
# (PROJECT="/scratch/si2356/dlm-jailbreak-transfer", HF_CACHE, DATASETS,
# DATASET_PLAN, PROJECT_DIR), we DO NOT edit the repo. Instead we tell the
# underlying attack which snapshot dir to build slim datasets from, re-run ONLY
# the sampled (model,dataset,prompt_idx) rows, and write the new responses under
# outputs/reproduced/ so they can be diffed by 06_compare.py.
#
# REQUIRED ENVIRONMENT (set before running):
#   CROSSCHECK_DATA_DIR  : dir that mirrors the student's repo layout:
#                          <dir>/dataset/harmbench/text_all.csv  (+ strongreject,
#                          malicious_instruct, jailbreakbench) and, for arrattack,
#                          <dir>/ArrAttack + <dir>/generation_model + model weights
#                          reachable in HF_CACHE. If you run on the student HPC this
#                          is exactly /scratch/si2356/dlm-jailbreak-transfer.
#   CROSSCHECK_HF_CACHE  : HF hub cache dir containing the model weights
#                          (default: $HF_HOME/hub or $HOME/.cache/huggingface/hub).
#   DEEPSEEK_API_KEY     : required for metacipher judge.
#   GPU (optional restrict): CUDA_VISIBLE_DEVICES
#
# MODES:
#   ./05_reproduce.sh --plan                 print every command that WOULD run, no execution
#   ./05_reproduce.sh --attack pif           reproduce only pif samples
#   ./05_reproduce.sh --model falcon         reproduce only one model
#   ./05_reproduce.sh --attack pif --limit 2 only first two pif samples (sanity)
#   (default)                                reproduce every sample in the manifest
#
# DESIGN NOTE — we do not literally re-run the full 913-prompt pipeline. Each
# sampled row is reproduced by injecting the sampled prompts as a SLIM dataset
# into a COPY of the attack script whose hardcoded paths point at this machine.
# The script's own checkpoint/resume semantics are preserved so a partial re-run
# is safe. New victim responses land in outputs/reproduced/.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/outputs"
REPRO="$OUT/reproduced"
MANIFEST="$OUT/manifest.json"

# per-machine config from <repo>/paths.txt (git-ignored): line1 = repo root,
# then optional KEY=VALUE lines (DATA_DIR=, HF_HOME=). Nothing hard-coded.
PT="$(dirname "$HERE")/paths.txt"
[[ -f "$PT" ]] && export DLM_REPO_ROOT="$(grep -m1 -vE '^\s*#|^\s*$' "$PT" 2>/dev/null | tr -d '[:space:]' || true)" || true
DATA_DIR="${CROSSCHECK_DATA_DIR:-$(grep -iE '^DATA_DIR=' "$PT" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)}"
HF_HOME_PT="$(grep -iE '^HF_HOME=' "$PT" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
HF_CACHE="${CROSSCHECK_HF_CACHE:-${HF_HOME:-${HF_HOME_PT:-$HOME/.cache/huggingface}}/hub}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(grep -E '^export DEEPSEEK_API_KEY=' "$HOME/.bashrc" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' || true)}"

PLAN=0; ATTACK=""; MODEL=""; LIMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN=1 ;;
    --attack) ATTACK="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
  shift
done

if [[ -z "$DATA_DIR" ]]; then
  echo "ERROR: CROSSCHECK_DATA_DIR must be set (dir mirroring the student repo layout,"
  echo "       with dataset/ + ArrAttack/ + generation models; on student HPC this is"
  echo "       /scratch/si2356/dlm-jailbreak-transfer)." >&2
  exit 1
fi
[[ -f "$DATA_DIR/dataset/harmbench/text_all.csv" ]] || {
  echo "ERROR: $DATA_DIR/dataset/harmbench/text_all.csv not found — not a valid data dir." >&2; exit 1; }
[[ -d "$MANIFEST" ]] || [[ -f "$MANIFEST" ]] || { echo "run 01+02 first (no manifest.json)"; exit 1; }

mkdir -p "$REPRO"

# ---- select samples ----
python3 "$HERE/05b_select_samples.py" "$ATTACK" "$MODEL" "$LIMIT" > "$OUT/.sample_list.tsv"

N=$(wc -l < "$OUT/.sample_list.tsv")
echo "Selected $N sample row(s) for reproduction."

mkdir -p "$REPRO"

# For each selected (attack, model, dataset, prompt): run the faithful reproducer.
planline(){ [[ $PLAN -eq 1 ]] && echo "[PLAN] $*" || eval "$*"; }

# columns: attack \t role \t model \t dataset \t prompt_idx \t prompt
while IFS=$'\t' read -r attack role model dataset prompt_idx prompt; do
  [[ -n "$attack" ]] || continue
  case "$attack" in
    pif)
      # PIF reproduces one (model, dataset) file; results.csv is per (model,dataset).
      outdir="$REPRO/pif/$model/$dataset"
      planline bash "$HERE/reproducers/repro_pif.py" "$DATA_DIR" "$HF_CACHE" "$model" "$dataset" "$outdir" "${prompt_idx}" "${prompt}"
      ;;
    metacipher)
      # MetaCipher is per-model over all datasets; we build a slim dataset dir of
      # ONLY the sampled prompts for that model so it runs a handful of rows.
      outdir="$REPRO/metacipher/$model"
      planline bash "$HERE/reproducers/repro_metacipher.py" "$DATA_DIR" "$HF_CACHE" "$model" "$outdir" "${prompt_idx}" "${prompt}"
      ;;
    arrattack)
      outdir="$REPRO/arrattack/$model"
      planline bash "$HERE/reproducers/repro_arrattack.sh" "$DATA_DIR" "$HF_CACHE" "$model" "$outdir" "${prompt_idx}" "${prompt}"
      ;;
    *) echo "skip attack=$attack" ;;
  esac
done < "$OUT/.sample_list.tsv"

echo
echo "New victim responses (when present) are under $REPRO/"
echo "Run 06_compare.py afterwards to diff them against the recorded CSV responses."
echo "NOTE: this runner only reproduces responses the environment can faithfully re-run."
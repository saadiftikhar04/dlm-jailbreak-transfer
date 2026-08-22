#!/usr/bin/env bash
#
# 00_run_all.sh — End-to-end driver for the cross-check / reproduction audit.
#
# Runs the deterministic (zero-GPU) passes and, optionally, the faithful
# reproduction pass against a data dir that has the model weights + dataset.
#
# Usage:
#   ./00_run_all.sh                     # inventory + sample + deterministic check + compare
#   CROSSCHECK_DATA_DIR=/scratch/si2356/dlm-jailbreak-transfer \
#       ./00_run_all.sh --reproduce     # + faithfully reproduce sampled rows on that host
#   ./00_run_all.sh --attack pif        # only reproduce pif samples
#   ./00_run_all.sh --step 02           # run a single step by number
#
# The zero-GPU steps (01..03) run anywhere. --reproduce needs a GPU host with the
# weights + dataset (see 04_preflight.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"
REPRO=0; STEP=""; ATTACK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reproduce) REPRO=1 ;;
    --attack) ATTACK="$2"; shift ;;
    --step) STEP="$2"; shift ;;
    *) echo "Unknown: $1"; exit 2 ;;
  esac
  shift
done

run() { echo; echo "════════ $* ════════"; "$@"; }

if [[ -z "$STEP" || "$STEP" == "01" ]]; then run "$PY" 01_inventory.py; fi
if [[ -z "$STEP" || "$STEP" == "02" ]]; then run "$PY" 02_sample.py; fi
if [[ -z "$STEP" || "$STEP" == "03" ]]; then run "$PY" 03_deterministic_check.py; fi
if [[ -z "$STEP" || "$STEP" == "04" ]]; then run "$PY" 04_preflight.py; fi
if [[ -z "$STEP" || "$STEP" == "05" ]]; then
  run "$PY" 05b_select_samples.py "" "" 0 | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5}' | head -1 >/dev/null
  # quick sanity: emit plan (no execution) for the requested filter
  ARGS=(--plan)
  [[ -n "$ATTACK" ]] && ARGS+=(--attack "$ATTACK")
  run bash 05_reproduce.sh "${ARGS[@]}"
fi
if [[ -z "$STEP" || "$STEP" == "06" ]]; then run "$PY" 06_compare.py; fi

if [[ "$REPRO" -eq 1 ]]; then
  ARGS=()
  [[ -n "$ATTACK" ]] && ARGS+=(--attack "$ATTACK")
  run bash 05_reproduce.sh "${ARGS[@]}"
  run "$PY" 06_compare.py
fi

echo
echo "Done. Zero-GPU reports are in outputs/ (inventory.json, manifest.json,"
echo "deterministic_check.txt, preflight.txt). Faithful-repro results + diff under outputs/reproduced/ + compare.txt."
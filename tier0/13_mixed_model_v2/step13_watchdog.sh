#!/bin/bash
# R2 Step13 — HPC ArrAttack progress watchdog for cron.
# Prints nothing (silent) while jobs are still running; prints a completion
# summary ONLY when all 5 HPC victims have >=913 rows in their progress CSV.
# Set ARR_FULLPOOL_RESULTS if results live elsewhere.
set -uo pipefail
REMOTE="bc3194@jubail.abudhabi.nyu.edu"
BASE=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
VICTIMS="qwen2.5 llama llada dream diffucoder"
DONE_ALL=1
STATUS=""
for v in $VICTIMS; do
  csv="$BASE/results/$v/arrattack_progress.csv"
  n=$(timeout 40 ssh -o ConnectTimeout=25 -o BatchMode=yes "$REMOTE" \
      "wc -l < '$csv' 2>/dev/null || echo 0" 2>/dev/null | tr -d ' ')
  n=${n:-0}
  # wc -l counts header too, so done rows = n-1
  done_rows=$(( n > 0 ? n-1 : 0 ))
  STATUS="$STATUS| $v: $done_rows/913"
  if [ "$done_rows" -lt 913 ]; then
    DONE_ALL=0
  fi
done
if [ "$DONE_ALL" -eq 1 ]; then
  echo "STEP13 HPC ALL 5 VICTIMS DONE:"
  echo "$STATUS"
  echo "Run step13_aggregate.py to produce summary; falcon may still be pending locally."
fi
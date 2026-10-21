#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/bc3194/dlm-jailbreak-transfer
ENV=/home/bc3194/.local/uv-envs/dlm-g6
OUT="$ROOT/revision_experiments/r4/G6_stochasticity"
LOG="$OUT/g6_generation.log"
EXIT_FILE="$OUT/g6_generation.exit"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4" ]]; then
    echo "Refusing G6 generation: CUDA_VISIBLE_DEVICES must be exactly 4" >&2
    exit 2
fi
if [[ -e "$EXIT_FILE" ]]; then
    echo "Refusing to overwrite existing G6 generation exit marker" >&2
    exit 2
fi
if [[ ! -d "$OUT" || ! -x "$ENV/bin/python" ]]; then
    echo "G6 output directory or uv environment is missing" >&2
    exit 2
fi

cd "$ROOT" || exit 2
umask 077
if [[ ! -e "$LOG" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=4\nHF_HOME=/data/bc3194/huggingface_cache\n' > "$LOG"
else
    printf '\n=== G6 generation resume %s ===\n' "$(date -Is)" >> "$LOG"
fi

set +e
HF_HOME=/data/bc3194/huggingface_cache "$ENV/bin/python" -u scripts/r4_g6_stochasticity.py generate >> "$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" > "$EXIT_FILE"
chmod 600 "$EXIT_FILE"
printf 'generation_exit=%s\n' "$rc" >> "$LOG"
nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$LOG" 2>&1
exit "$rc"

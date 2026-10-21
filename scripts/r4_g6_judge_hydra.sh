#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/bc3194/dlm-jailbreak-transfer
ENV=/home/bc3194/.local/uv-envs/dlm-g6
OUT="$ROOT/revision_experiments/r4/G6_stochasticity"
LOG="$OUT/g6_hydra_judge.log"
EXIT_FILE="$OUT/g6_hydra_judge.exit"
LOCK_FILE="$OUT/g6_hydra_judge.lock"
KEY_FILE=/tmp/g6_openrouter_key

umask 077
cd "$ROOT"
exec >> "$LOG" 2>&1
echo "G6 Hydra D3 judge launcher started at $(date -Is)"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another G6 Hydra judge holds the lock; this duplicate launch is a no-op."
    exit 0
fi

finish() {
    local rc=$?
    if [[ ! -e "$EXIT_FILE" ]]; then
        printf '%s\n' "$rc" > "$EXIT_FILE"
        chmod 600 "$EXIT_FILE"
    fi
    echo "G6 Hydra D3 judge launcher exit=$rc at $(date -Is)"
}
trap finish EXIT

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4" ]]; then
    echo "Refusing: CUDA_VISIBLE_DEVICES must be exactly 4" >&2
    exit 2
fi
if [[ "${HTTPS_PROXY:-}" != "http://127.0.0.1:18080" ]]; then
    echo "Refusing: expected the restricted local OpenRouter proxy tunnel" >&2
    exit 2
fi
if [[ ! -x "$ENV/bin/python" || ! -s "$OUT/g6_generation_ledger.csv" || ! -s "$OUT/g6_d3_judge_ledger.csv" ]]; then
    echo "Refusing: environment or resumable G6 ledgers are missing" >&2
    exit 2
fi
if [[ -e "$EXIT_FILE" ]]; then
    echo "Refusing: existing G6 Hydra judge exit marker must be reviewed first" >&2
    exit 2
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    if [[ ! -s "$KEY_FILE" ]]; then
        echo "Refusing: no OpenRouter key was provided through the protected staging file" >&2
        exit 2
    fi
    OPENROUTER_API_KEY="$(<"$KEY_FILE")"
    rm -f "$KEY_FILE"
    export OPENROUTER_API_KEY
fi
if [[ "$OPENROUTER_API_KEY" != sk-or-v1-* ]]; then
    echo "Refusing: unexpected OpenRouter key format" >&2
    exit 2
fi

export HF_HOME=/data/bc3194/huggingface_cache
export HTTP_PROXY=http://127.0.0.1:18080
export ALL_PROXY=http://127.0.0.1:18080
export NO_PROXY=127.0.0.1,localhost,::1

rc=0
"$ENV/bin/python" -u scripts/r4_g6_stochasticity.py judge || rc=$?
if [[ "$rc" -eq 0 ]]; then
    "$ENV/bin/python" -u scripts/r4_g6_stochasticity.py summary || rc=$?
fi
printf '%s\n' "$rc" > "$EXIT_FILE"
chmod 600 "$EXIT_FILE"
echo "G6 Hydra D3 judge exit=$rc at $(date -Is)"
nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
exit "$rc"

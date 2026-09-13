#!/bin/bash
# R2 Step 13 — HPC: make a clean empty mamba_ssm stub so
# transformers.is_mamba_2_ssm_available() -> False (broken CUDA mamba shadows).
set -euo pipefail
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
rm -rf "$ROOT/mamba_stub"
mkdir -p "$ROOT/mamba_stub/mamba_ssm"
printf '' > "$ROOT/mamba_stub/mamba_ssm/__init__.py"
echo "stub created:"
find "$ROOT/mamba_stub" -type f
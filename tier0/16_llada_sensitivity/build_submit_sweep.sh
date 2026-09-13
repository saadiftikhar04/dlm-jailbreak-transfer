#!/bin/bash
# R2 Step 16: submit 6 LLaDA sweep jobs (one per config). PRUNE=0 to submit.
set -e
REPO=/scratch/bc3194/dlm-jailbreak-transfer
SBATCH=$REPO/tier0/16_llada_sensitivity/sweep_llada_r2.sbatch
PRUNE=${PRUNE:-1}
mkdir -p "$REPO/tier0/16_llada_sensitivity/jobs"

configs=(base128s base32s steps256 steps64 temp05 temp10)
for c in "${configs[@]}"; do
  cmd="CONFIG_NAME=$c sbatch --job-name=sw_$c --output=$REPO/tier0/16_llada_sensitivity/jobs/sw_${c}_%j.out --error=$REPO/tier0/16_llada_sensitivity/jobs/sw_${c}_%j.err --partition=nvidia --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=71:59:59 $SBATCH"
  echo "$cmd"
  if [ "$PRUNE" = "0" ]; then eval "$cmd"; fi
done
echo "=== done (PRUNE=$PRUNE) ==="
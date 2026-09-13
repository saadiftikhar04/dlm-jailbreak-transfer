#!/bin/bash
# R2 Step 15: submit 6 benign-decode generation jobs (one per victim).
# Run ON the HPC. PRUNE=0 to submit, else dry-run.
set -e
REPO=/scratch/bc3194/dlm-jailbreak-transfer
SBATCH=$REPO/tier0/08_benign_metacipher_decode/generate_t15_r2.sbatch
mkdir -p "$REPO/tier0/08_benign_metacipher_decode/jobs"
PRUNE=${PRUNE:-1}

victims=(qwen2.5 llama falcon llada dream diffucoder)
for m in "${victims[@]}"; do
  cmd="MODEL_KEY=$m sbatch --job-name=t15_$m --output=$REPO/tier0/08_benign_metacipher_decode/jobs/t15_${m}_%j.out --error=$REPO/tier0/08_benign_metacipher_decode/jobs/t15_${m}_%j.err --partition=nvidia --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=71:59:59 $SBATCH"
  echo "$cmd"
  if [ "$PRUNE" = "0" ]; then eval "$cmd"; fi
done
echo "=== done (PRUNE=$PRUNE) ==="
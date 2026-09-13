#!/bin/bash
# R2 Step 14: submit 9 (model x seed) multi-seed generation jobs to the nvidia
# A100 pool using env-prefix on the shared generic sbatch (T09-verified pattern).
# Run ON the HPC (submit dir = repo root). SLURM exports the submission env.
#
# Usage:
#   bash build_and_submit_r2.sh          # dry-run: print the 9 commands
#   PRUNE=0 bash build_and_submit_r2.sh  # actually submit (sbatch --parsable)
set -e
REPO=/scratch/bc3194/dlm-jailbreak-transfer
DIR=$REPO/tier0/14_multiseed_diffusion
SBATCH=$DIR/gen_multiseed_r2.sbatch

models=(dream diffucoder llada)
seeds=(1 2 3)
PRUNE=${PRUNE:-1}  # 1 = dry-run (echo only); 0 = submit

for m in "${models[@]}"; do
  for s in "${seeds[@]}"; do
    cmd="MODEL_KEY=$m SEED_IDX=$s sbatch --job-name=ms_${m}_s${s} --output=$DIR/jobs/ms_${m}_s${s}_%j.out --error=$DIR/jobs/ms_${m}_s${s}_%j.err --partition=nvidia --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=71:59:59 $SBATCH"
    echo "$cmd"
    if [ "$PRUNE" = "0" ]; then
      eval "$cmd"
    fi
  done
done
echo "=== done (PRUNE=$PRUNE) ==="
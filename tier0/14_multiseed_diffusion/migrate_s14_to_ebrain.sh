#!/bin/bash
# Migrate the still-pending Step-14 multiseed generation jobs from nvidia to
# ebrainccs condo (r7cuminf still holds nvidia's MaxJobs=12; ebrainccs is idle).
# Idempotent: scancels the nvidia job then re-sbaths to condo -q ebrainccs.
set -e
REPO=/scratch/bc3194/dlm-jailbreak-transfer
SBATCH=$REPO/tier0/14_multiseed_diffusion/gen_multiseed_r2.sbatch
LOG=/tmp/migrate_s14.log

# (old_jobid, model, seed)
MIGRATE=(
  "17934720 dream 1"
  "17934721 dream 2"
  "17934722 dream 3"
  "17934723 diffucoder 1"
  "17934724 diffucoder 2"
  "17934725 diffucoder 3"
  "17934726 llada 1"
  "17934727 llada 2"
  "17934728 llada 3"
)
for row in "${MIGRATE[@]}"; do
  set -- $row
  old=$1; m=$2; s=$3
  scancel "$old" 2>/dev/null
  sleep 1
  new=$(MODEL_KEY=$m SEED_IDX=$s sbatch --parsable --job-name=ms_${m}_s${s} \
    --partition=condo -q ebrainccs -C 80g --gres=gpu:a100:1 --cpus-per-task=8 \
    --mem=64G --time=71:59:59 \
    --output=$REPO/tier0/14_multiseed_diffusion/jobs/ms_${m}_s${s}_%j.out \
    --error=$REPO/tier0/14_multiseed_diffusion/jobs/ms_${m}_s${s}_%j.err \
    $SBATCH 2>&1)
  echo "migrated $old -> $m s$s -> $new" | tee -a "$LOG"
  sleep 1
done
echo "=== done ===" | tee -a "$LOG"
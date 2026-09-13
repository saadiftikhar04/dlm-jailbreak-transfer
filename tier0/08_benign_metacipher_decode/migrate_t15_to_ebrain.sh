#!/bin/bash
# Migrate pending Step-15 (benign decode) + Step-14 sanity jobs from nvidia to
# ebrainccs condo. Idempotent: scancel old, re-sbath with same MODEL_KEY.
set -e
REPO=/scratch/bc3194/dlm-jailbreak-transfer

# Step 14 sanity (ms_sanity, model dream 1-row)
scancel 17934712 2>/dev/null; sleep 1
SB14=$REPO/tier0/14_multiseed_diffusion/hpc_sanity_r2.sbatch
S14NEW=$(MODEL_KEY=dream SEED_IDX=1 sbatch --parsable --job-name=ms_sanity \
  --partition=condo -q ebrainccs -C 80g --gres=gpu:a100:1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00 \
  --output=$REPO/tier0/14_multiseed_diffusion/jobs/ms_sanity_%j.out \
  --error=$REPO/tier0/14_multiseed_diffusion/jobs/ms_sanity_%j.err \
  $SB14 2>&1)
echo "ms_sanity -> $S14NEW"

# Step 15
SB15=$REPO/tier0/08_benign_metacipher_decode/generate_t15_r2.sbatch
declare -A T15=( [17934891]=qwen2.5 [17934892]=llama [17934893]=falcon [17934894]=llada [17934895]=dream [17934896]=diffucoder )
for old in "${!T15[@]}"; do
  m=${T15[$old]}
  scancel "$old" 2>/dev/null
  sleep 1
  new=$(MODEL_KEY=$m sbatch --parsable --job-name=t15_$m \
    --partition=condo -q ebrainccs -C 80g --gres=gpu:a100:1 --cpus-per-task=8 \
    --mem=64G --time=71:59:59 \
    --output=$REPO/tier0/08_benign_metacipher_decode/jobs/t15_${m}_%j.out \
    --error=$REPO/tier0/08_benign_metacipher_decode/jobs/t15_${m}_%j.err \
    $SB15 2>&1)
  echo "t15_$m ($old) -> $new"
  sleep 1
done
echo "=== done ==="
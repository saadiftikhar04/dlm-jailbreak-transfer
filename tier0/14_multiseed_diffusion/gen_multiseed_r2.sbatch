#!/bin/bash
# R2 Step 14 multi-seed generation for ONE (model, seed) pair.
# MODEL_KEY and SEED_IDX passed; gen_multiseed_r2.py reads SEED_IDX env to
# pick the seed to run (single-seed mode), writing model_outputs/{m}_seed{N}.
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL_KEY="$MODEL_KEY"
export R2_SEED_IDX="$SEED_IDX"

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/14_multiseed_diffusion/gen_multiseed_r2.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
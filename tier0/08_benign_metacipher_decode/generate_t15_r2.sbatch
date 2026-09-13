#!/bin/bash
# R2 Step 15: benign decode generation for ONE victim (env MODEL_KEY).
# falcon needs the mamba stub stack (HPC raven mamba ABI broken).
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ "$MODEL_KEY" = "falcon" ]; then
  export PYTHONPATH="/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/mamba_stub${PYTHONPATH:+:$PYTHONPATH}"
fi

/scratch/bc3194/conda_envs/envs/raven/bin/python \
  tier0/08_benign_metacipher_decode/generate_t15_r2.py "$MODEL_KEY"
echo "=== RC=$? $(date) ==="
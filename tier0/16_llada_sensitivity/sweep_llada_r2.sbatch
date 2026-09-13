#!/bin/bash
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/tier0/16_llada_sensitivity/logs/sweep_%x_%j.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/tier0/16_llada_sensitivity/logs/sweep_%x_%j.err
# R2 Step 16: LLaDA sweep for ONE config (env CONFIG_NAME). Reads sweep50_dataset.json.
set -e
cd /scratch/bc3194/dlm-jailbreak-transfer
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== env ==="
which python
/scratch/bc3194/conda_envs/envs/raven/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== config: $CONFIG_NAME ==="

DIR=tier0/16_llada_sensitivity
/scratch/bc3194/conda_envs/envs/raven/bin/python \
  $DIR/sweep_llada_r2.py \
  --dataset $DIR/sweep50_dataset.json \
  --out $DIR/sweep_out_${CONFIG_NAME}.jsonl \
  --config "$CONFIG_NAME"
echo "=== RC=$? $(date) ==="
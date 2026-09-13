#!/bin/bash
# R2 Step 13 — place llada_generate.py at repo root (model_utils needs it as a
# top-level module because stage5 puts arrattack_fullpool on sys.path, not .../utils)
cd /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool || exit 1
if [ ! -f ./llada_generate.py ] && [ -f ./utils/llada_generate.py ]; then
  cp ./utils/llada_generate.py ./llada_generate.py
  echo "copied llada_generate.py to root"
fi
ls -la ./llada_generate.py
# re-run sanity
/scratch/bc3194/conda_envs/envs/raven/bin/python hpc_sanity_fullpool.py 2>&1 | grep -vE "FutureWarning|warnings.warn" | tail -12
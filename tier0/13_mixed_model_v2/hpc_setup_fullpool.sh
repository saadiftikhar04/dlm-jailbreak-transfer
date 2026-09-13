#!/bin/bash
# R2 Step 13 — HPC: place bc3194 utils + generate stage5_fullpool.py
cd /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool || exit 1
# temp dir where we scp'd the bc3194 versions
TMP=utils_import_temp
if [ -f "$TMP/qwen_utils_bc3194.py" ]; then
  cp "$TMP/qwen_utils_bc3194.py" utils/qwen_utils.py
  rm -rf utils/__pycache__
  echo "utils/qwen_utils.py <- bc3194 version"
fi
if [ -f "$TMP/model_utils_bc3194.py" ]; then
  cp "$TMP/model_utils_bc3194.py" ./model_utils.py
  rm -rf __pycache__
  echo "model_utils.py <- bc3194 version"
fi
rm -rf "$TMP"
# generate stage5_fullpool.py
/scratch/bc3194/conda_envs/envs/raven/bin/python make_stage5_fullpool.py
echo "--- sanity: full counts in stage5_fullpool ---"
grep -nE "400\)|313\)|100\)|step13_dataset|arrattack_fullpool')" stage5_fullpool.py | head
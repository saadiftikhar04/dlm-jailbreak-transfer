#!/bin/bash
# R2 Step 13 — HPC: test replacing broken mamba_ssm with an empty stub so
# transformers.is_mamba_2_ssm_available() returns False and Falcon-H1R uses the
# naive (non-CUDA) path. Stub shadows the broken package via PYTHONPATH.
set -uo pipefail
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
STUB=${ROOT}/mamba_stub/mamba_ssm
mkdir -p "$STUB"
touch "$STUB/__init__.py" "${STUB}/ops" 2>/dev/null
mkdir -p "$STUB/ops"
printf 'def foo(): pass\n' > "$STUB/ops/__init__.py"
export PYTHONPATH="$ROOT/mamba_stub:${PYTHONPATH:-}"
export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
echo "=== does transformers see mamba as available? ==="
/scratch/bc3194/conda_envs/envs/raven/bin/python - <<'PY' 2>&1 | tail -6
import sys
try:
    from transformers.utils import is_mamba_2_ssm_available
    print("is_mamba_2_ssm_available:", is_mamba_2_ssm_available())
except Exception as e:
    print("is_avail err:", type(e).__name__, e)
try:
    import mamba_ssm
    print("import mamba_ssm ->", mamba_ssm, getattr(mamba_ssm,'__file__','?'))
except Exception as e:
    print("mamba_ssm import err:", type(e).__name__, str(e)[:120])
PY
echo "=== try loading Falcon-H1R (silent) ==="
/scratch/bc3194/conda_envs/envs/raven/bin/python - <<'PY' 2>&1 | tail -8
import os, torch
os.environ["TRANSFORMERS_OFFLINE"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    m = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon-H1R-7B", local_files_only=True)
    print("FALCON_LOAD_OK", sum(p.numel() for p in m.parameters())/1e9, "B")
except Exception as e:
    import traceback; traceback.print_exc(); print("FALCON_LOAD_FAIL", type(e).__name__, str(e)[:150])
PY
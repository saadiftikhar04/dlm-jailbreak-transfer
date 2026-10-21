#!/bin/bash
# R2 Step 13 — full-pool ArrAttack for one victim (stage5_fullpool.py).
# One GPU per job. Resume-safe via stage5 PROGRESS_CSV (job rerun continues).
#SBATCH --job-name=arr_full_falcon
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=100G
#SBATCH --output=arr_full_falcon.out
#SBATCH --error=arr_full_falcon.err
set -euo pipefail

HPC=1
VKEY=falcon          # stage5 key: qwen2.5 / llama / falcon / llada / dream / diffucoder
ROOT=/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool
cd "$ROOT" || exit 1

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export TARGET="$VKEY"
export DEEPSEEK_API_KEY="$(bash -ic 'echo $DEEPSEEK_API_KEY' 2>/dev/null || true)"
PY=/scratch/bc3194/conda_envs/envs/raven/bin/python

# Raven's installed Mamba CUDA extensions have a stale PyTorch ABI. Rebuild
# into job-local storage against the active Python 3.11 / torch / CUDA stack;
# do not use the incompatible site-packages binaries or Python 3.10 mamba_ext.
export CUDA_HOME=/share/apps/NYUAD/cuda/13.0.0
GCC9=/share/apps/NYUAD5/gcc/9.2.0
export CC="$GCC9/bin/gcc"
export CXX="$GCC9/bin/g++"
export PATH="$GCC9/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GCC9/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=1
MAMBA_RUNTIME="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/arr_falcon_mamba_${SLURM_JOB_ID:-$$}"
mkdir -p "$MAMBA_RUNTIME"

"$PY" - <<'PY'
import torch

print(f"Raven torch={torch.__version__}, torch CUDA={torch.version.cuda}")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit("Unexpected Raven torch/CUDA build; refusing an unverified Mamba compile")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the allocated Falcon job")
PY

"$PY" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation --no-deps --no-cache-dir --no-binary=:all: \
  --target "$MAMBA_RUNTIME" \
  causal-conv1d==1.6.2.post1 mamba-ssm==2.3.1
export PYTHONPATH="$MAMBA_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

"$PY" - <<'PY'
import torch
import causal_conv1d
import mamba_ssm
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

x = torch.randn(1, 8, 16, device="cuda", dtype=torch.float16)
w = torch.randn(8, 3, device="cuda", dtype=torch.float16)
y = causal_conv1d_fn(x, w, activation=None)
assert tuple(y.shape) == tuple(x.shape), tuple(y.shape)

u = torch.randn(1, 8, 16, device="cuda", dtype=torch.float32)
delta = torch.rand_like(u)
a = -torch.ones(8, 4, device="cuda", dtype=torch.float32)
b = torch.randn(1, 4, 16, device="cuda", dtype=torch.float32)
c = torch.randn_like(b)
z = selective_scan_fn(u, delta, a, b, c, delta_softplus=True)
assert tuple(z.shape) == tuple(u.shape), tuple(z.shape)
print("MAMBA_CUDA_OK", mamba_ssm.__file__, causal_conv1d.__file__, FalconH1ForCausalLM.__name__)
PY

echo "[$(date)] START ArrAttack-fullpool victim=$VKEY"
echo "HF_HOME=$HF_HOME  DEEPSEEK set=$([ -n \"$DEEPSEEK_API_KEY\" ] && echo yes || echo NO)"

"$PY" stage5_fullpool.py || { echo "stage5_fullpool FAILED victim=$VKEY rc=$?" ; exit 1; }

echo "[$(date)] DONE ArrAttack-fullpool victim=$VKEY"

#!/bin/bash
# R4 G5: disjoint AdvBench Falcon-H1R generation-budget pilot.
#SBATCH --job-name=r4_g5_falcon_pilot
#SBATCH --partition=nvidia
#SBATCH --qos=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -C 80g
#SBATCH --time=71:59:59
#SBATCH --mem=56G
#SBATCH --output=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G5_falcon_budget/logs/g5_budget_%j.out
#SBATCH --error=/scratch/bc3194/dlm-jailbreak-transfer/revision_experiments/r4/G5_falcon_budget/logs/g5_budget_%j.err

source ~/.bashrc >/dev/null 2>&1
set -euo pipefail
umask 077
cd /scratch/bc3194/dlm-jailbreak-transfer

export HF_HOME=/scratch/bc3194/huggingface_cache
export HF_HUB_CACHE=/scratch/bc3194/huggingface_cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=/scratch/bc3194/conda_envs/envs/raven/bin/python

# The preinstalled Raven Mamba CUDA extensions were built against an older
# PyTorch ABI. Build job-local extensions against the active Raven torch/CUDA
# instead of importing the incompatible site-packages binaries or the old
# Python 3.10 mamba_ext tree.
export CUDA_HOME=/share/apps/NYUAD/cuda/13.0.0
GCC11=/share/apps/NYUAD5/gcc/11.5.0
export CC="$GCC11/bin/gcc"
export CXX="$GCC11/bin/g++"
export PATH="$GCC11/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GCC11/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=1
MAMBA_RUNTIME="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/r4_mamba_${SLURM_JOB_ID:-$$}"
mkdir -p "$MAMBA_RUNTIME"

"$PY" - <<'PY'
import torch

print(f"Raven torch={torch.__version__}, torch CUDA={torch.version.cuda}")
if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
    raise SystemExit("Unexpected Raven torch/CUDA build; refusing an unverified Mamba compile")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the allocated Falcon job")
PY

# Use the already-deployed package versions, but compile their CUDA extensions
# against this job's active Python 3.11 / PyTorch 2.13 / CUDA 13.0 stack.
"$PY" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation --no-deps --no-cache-dir --no-binary=:all: \
  --target "$MAMBA_RUNTIME" \
  causal-conv1d==1.6.2.post1 mamba-ssm==2.3.1
export PYTHONPATH="$MAMBA_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

# Fail before loading the model if either compiled Mamba CUDA operator is not
# importable/executable for the allocated A100 and active torch ABI.
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

"$PY" scripts/r4_g5_falcon_budget_pilot.py

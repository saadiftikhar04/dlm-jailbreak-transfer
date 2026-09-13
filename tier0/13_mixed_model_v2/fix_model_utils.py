"""R2 Step 13 — fix model_utils_bc3194.py registry to use bc3194 HF cache IDs.

The naive si2356->bc3194 sed left wrong paths (.cache vs huggingface_cache) and
si2356-specific model dirs. Point the registry at real cached HF model IDs so
load_model works offline from /scratch/bc3194/huggingface_cache.
"""
import os

SRC = os.path.expanduser("~/Desktop/dlm-jailbreak-transfer/ArrAttack/utils/model_utils.py")
s = open(SRC, encoding="utf-8").read()

# HF cache that actually holds the victims
s = s.replace('HF_CACHE = "/scratch/si2356/.cache/huggingface/hub"',
              'HF_CACHE = "/scratch/bc3194/huggingface_cache/hub"')

# Replace the MODEL_REGISTRY block wholesale with bc3194 HF IDs (offline-resolved
# from the cache that exists; paper-faithful IDs).
START = 'MODEL_REGISTRY = {'
i0 = s.index(START)
i1 = s.index("DLM_KEYS", i0)
REG = '''MODEL_REGISTRY = {
    # LLMs
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct",
    "falcon":    "tiiuae/Falcon-H1R-7B",
    "llama":     "meta-llama/Llama-3.1-8B-Instruct",
    # DLMs
    "llada":     "GSAI-ML/LLaDA-1.5",
    "dream":     "Dream-org/Dream-v0-Instruct-7B",
    "diffucoder":"apple/DiffuCoder-7B-Instruct",
}

'''
s = s[:i0] + REG + s[i1:]
out = os.path.expanduser("~/Desktop/dlm-jailbreak-transfer/ArrAttack/utils/model_utils_bc3194.py")
open(out, "w", encoding="utf-8").write(s)
print("wrote model_utils_bc3194.py")
for l in s.splitlines():
    if 'HF_CACHE' in l or '"\""' in l or 'llama' in l.lower() and '"' in l and 'MODEL' not in l:
        pass
print("--- registry ---")
import re
print("\n".join(re.findall(r'\s+"[a-z0-9.]+":\s+"[^"]+"', s)))
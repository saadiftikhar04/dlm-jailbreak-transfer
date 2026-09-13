"""R2 Step 13 — build a LOCAL self-contained ArrAttack env for Falcon (full pool).

fc is the only victim that must run locally (HPC raven mamba_ssm is ABI-broken).
We build <repo>/tier0/13_mixed_model_v2/falcon_env/ with:
  - qwen_utils.py  (local HF cache paths)
  - model_utils.py (local HF-cache model IDs)
  - stage5_fullpool.py  (full-pool DATASET_CONFIG -> local step13_dataset)
from the local ArrAttack sources. Output CSV lands in falcon_env/results/falcon.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = os.path.join(HERE, "..", "..", "ArrAttack")
HF = "/home/bc3194/Desktop/huggingface_cache"
ENV = os.path.join(HERE, "falcon_env")
os.makedirs(ENV, exist_ok=True)
os.makedirs(os.path.join(ENV, "utils"), exist_ok=True)

# ---- qwen_utils.py --------------------------------------------------------
q = open(os.path.join(ARR, "utils", "qwen_utils.py"), encoding="utf-8").read()
q = q.replace('SNAPSHOT_BASE = "/scratch/si2356/.cache/huggingface/hub"',
              f'SNAPSHOT_BASE = "{HF}/hub"')
q = q.replace('PROJECT_DIR = "/scratch/si2356/dlm-jailbreak-transfer"',
              f'PROJECT_DIR = "{ENV}"')
open(os.path.join(ENV, "utils", "qwen_utils.py"), "w", encoding="utf-8").write(q)

# ---- llada_generate + others (copy unchanged) ------------------------------
import shutil
for f in ["llada_generate.py", "opt_utils.py", "string_utils.py", "__init__.py"]:
    shutil.copy(os.path.join(ARR, "utils", f), os.path.join(ENV, "utils", f))

# ---- model_utils.py --------------------------------------------------------
m = open(os.path.join(ARR, "utils", "model_utils.py"), encoding="utf-8").read()
m = m.replace('HF_CACHE = "/scratch/si2356/.cache/huggingface/hub"',
              f'HF_CACHE = "{HF}/hub"')
REG = '''MODEL_REGISTRY = {
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct",
    "falcon":    "tiiuae/Falcon-H1R-7B",
    "llama":     "meta-llama/Llama-3.1-8B-Instruct",
    "llada":     "GSAI-ML/LLaDA-1.5",
    "dream":     "Dream-org/Dream-v0-Instruct-7B",
    "diffucoder":"apple/DiffuCoder-7B-Instruct",
}
'''
i0 = m.index("MODEL_REGISTRY = {")
i1 = m.index("DLM_KEYS", i0)
m = m[:i0] + REG + m[i1:]
open(os.path.join(ENV, "model_utils.py"), "w", encoding="utf-8").write(m)

# ---- stage5_fullpool.py ------------------------------------------------------
s = open(os.path.join(ARR, "stage5_attack.py"), encoding="utf-8").read()
s = s.replace("sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack')",
              f"sys.path.insert(0, '{ENV}')")
s = s.replace("sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/utils')",
              f"sys.path.insert(0, '{ENV}')")
DATASET = os.path.join(HERE, "step13_dataset")
i0 = s.index("DATASET_BASE = PROJECT_DIR")
i1 = s.index("]", i0) + 1
CONFIG = f'''DATASET_BASE = "{DATASET}"

DATASET_CONFIG = [
    ("harmbench", DATASET_BASE + "/harmbench/text_all.csv", "Behavior", 400),
    ("strongreject", DATASET_BASE + "/strongreject/strongreject.csv", "forbidden_prompt", 313),
    ("jailbreakbench", DATASET_BASE + "/jailbreakbench/jailbreakbench.csv", "Goal", 100),
    ("malicious_instruct", DATASET_BASE + "/malicious_instruct/malicious_instruct.txt", None, 100),
]'''
s = s[:i0] + CONFIG + s[i1:]
# local results dir: PROJECT_DIR(=ENV)/results/falcon
open(os.path.join(ENV, "stage5_fullpool.py"), "w", encoding="utf-8").write(s)

print("built falcon_env/")
for fp in os.listdir(ENV):
    print(" ", fp)
print(" utils:", os.listdir(os.path.join(ENV, "utils")))
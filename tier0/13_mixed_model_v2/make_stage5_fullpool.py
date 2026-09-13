"""R2 Step 13 — generate HPC stage5_fullpool.py (block-level rewrite).

On HPC, from the synced arrattack_fullpool/stage5_attack.py:
  - fix the two sys.path inserts to point at arrattack_fullpool
  - replace the DATASET_BASE + DATASET_CONFIG block with full-pool (913)
    config pointing at step13_dataset absolute paths
  - RESULTS_DIR is PROJECT_DIR + /results/{TARGET_KEY}; PROJECT_DIR now resolves
    to arrattack_fullpool (via the synced qwen_utils_bc3194), so output lands in
    arrattack_fullpool/results/<key> (independent of the original 165 results).
Output: arrattack_fullpool/stage5_fullpool.py
"""
import os, re

SRC = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/stage5_attack.py"
OUT = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/stage5_fullpool.py"

s = open(SRC, encoding="utf-8").read()

# --- 1. sys.path fixes ------------------------------------------------
s = s.replace(
    "sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack')",
    "sys.path.insert(0, '/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool')")
s = s.replace(
    "sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/utils')",
    "sys.path.insert(0, '/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool')")

# --- 2. DATASET_BASE + DATASET_CONFIG block -> full pool ------------------
START = "DATASET_BASE = PROJECT_DIR"
END = "]"
i0 = s.index(START)
# find the block end: the DATASET_CONFIG list closes with ']' on its own line
i1 = s.find("\n]\n", i0)
if i1 == -1:
    i1 = s.index("]", i0)
else:
    i1 = s.index("]", i0)
DATASET_BASE = 'DATASET_BASE = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/step13_dataset"'
CONFIG = '''DATASET_CONFIG = [
    # full-pool: all 913 prompts (harmbench 400, strongreject 313,
    # jailbreakbench 100, malicious_instruct 100)
    ("harmbench",
     DATASET_BASE + "/harmbench/text_all.csv",
     "Behavior",
     400),
    ("strongreject",
     DATASET_BASE + "/strongreject/strongreject.csv",
     "forbidden_prompt",
     313),
    ("jailbreakbench",
     DATASET_BASE + "/jailbreakbench/jailbreakbench.csv",
     "Goal",
     100),
    ("malicious_instruct",
     DATASET_BASE + "/malicious_instruct/malicious_instruct.txt",
     None,
     100),
]'''
s = s[:i0] + DATASET_BASE + "\n\n" + CONFIG + s[i1 + 1:]

open(OUT, "w", encoding="utf-8").write(s)
print("wrote", OUT)
# sanity
print("sys.path fixed:", all(
    "arrattack_fullpool" in l for l in s.splitlines()
    if l.strip().startswith("sys.path.insert(0, '/scratch/bc3194")))
print("DATASET_CONFIG full counts present:",
      "400)," in s and "313)," in s and "100)," in s)
print("dataset_base:", 'step13_dataset' in s)
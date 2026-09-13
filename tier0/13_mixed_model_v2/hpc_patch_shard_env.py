"""R2 Step 13 — patch HPC stage5_fullpool.py so DATASET_BASE and RESULTS_DIR can be
overridden per-shard via env (ARR_DATASET_BASE / ARR_RESULTS_DIR), defaulting to
the full 913-pool run. This lets the 10 dream/diffucoder shards run in parallel
each pointing at its own 92-row dataset + results dir.
"""
import os

P = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/stage5_fullpool.py"
s = open(P, encoding="utf-8").read()

# 1. RESULTS_DIR env override (insert just before line 90)
s = s.replace(
    'RESULTS_DIR   = PROJECT_DIR + f"/results/{TARGET_KEY}"',
    'RESULTS_DIR = os.environ.get("ARR_RESULTS_DIR", '
    'PROJECT_DIR + f"/results/{TARGET_KEY}")')

# 2. DATASET_BASE env override
s = s.replace(
    'DATASET_BASE = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/step13_dataset"',
    'DATASET_BASE = os.environ.get("ARR_DATASET_BASE", '
    '"/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/step13_dataset")')

# ensure os is importable (line 43: "import csv, logging, os, re, sys, time, gc")
assert any("import" in l and "os" in l for l in s.splitlines()[:90]), "os import missing"
open(P, "w", encoding="utf-8").write(s)
print("patched stage5_fullpool.py for shard env overrides")
# verify
for line in s.splitlines():
    if "ARR_DATASET_BASE" in line or "ARR_RESULTS_DIR" in line:
        print("  *", line.strip()[:90])
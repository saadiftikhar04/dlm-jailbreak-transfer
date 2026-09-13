"""R2 Step 13 — self-contained qwen_utils for bc3194 full-pool ArrAttack.

Copies of the student's ArrAttack utils with the hard-coded si2356 paths pointed
at bc3194's HF cache. Produced for the full-pool run only; never overwrites the
original student files (which live in the read-only /scratch/si2356 tree).
"""
import re, os, shutil

SRC = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/utils/qwen_utils.py"
SRC = os.path.expanduser("~/Desktop/dlm-jailbreak-transfer/ArrAttack/utils/qwen_utils.py")
print("reading local:", SRC, os.path.exists(SRC))
s = open(SRC, encoding="utf-8").read()

s = s.replace('SNAPSHOT_BASE = "/scratch/si2356/.cache/huggingface/hub"',
              'SNAPSHOT_BASE = "/scratch/bc3194/huggingface_cache/hub"')
s = s.replace('PROJECT_DIR = "/scratch/si2356/dlm-jailbreak-transfer"',
              'PROJECT_DIR = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"')

out_local = os.path.expanduser("~/Desktop/dlm-jailbreak-transfer/ArrAttack/utils/qwen_utils_bc3194.py")
open(out_local, "w", encoding="utf-8").write(s)
print("wrote local qwen_utils_bc3194.py ->", out_local)
print("\n--- changed lines ---")
for line in s.splitlines():
    if "bc3194" in line and ("SNAPSHOT_BASE" in line or "PROJECT_DIR" in line):
        print(" ", line)
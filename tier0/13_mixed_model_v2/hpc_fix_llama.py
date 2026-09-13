"""R2 Step 13 — HPC: point ONLY the 'llama' registry entry at its local snapshot
dir so transformers skips gated-online revalidation (the other running victims
are untouched). Output still land in results/llama.
"""
import os, re, glob

SRC = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/model_utils.py"
s = open(SRC, encoding="utf-8").read()

g = glob.glob("/scratch/bc3194/huggingface_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")
assert g, "no llama snapshot"
g.sort(key=os.path.getmtime, reverse=True)
snap = g[0]
print("llama snapshot:", snap)

# replace the llama line value with the absolute snapshot path
s2 = re.sub(r'("llama"\s*:\s*)"[^"]*"', r'\1"%s"' % snap, s)
assert s2 != s, "llama line not found"
open(SRC, "w", encoding="utf-8").write(s2)
print("patched llama ->", snap)
for line in s2.splitlines():
    if '"llama"' in line:
        print("  *", line.strip()[:110])
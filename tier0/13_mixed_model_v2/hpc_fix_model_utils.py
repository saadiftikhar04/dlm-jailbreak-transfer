"""R2 Step 13 — HPC: point model_utils llama (and gated repos) at local snapshot so
transformers skips online token revalidation (offline + gated => OSError).

Uses the actual snapshot dirs present in /scratch/bc3194/huggingface_cache/hub.
"""
import os, re, glob

SRC = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/model_utils.py"
s = open(SRC, encoding="utf-8").read()

HUB = "/scratch/bc3194/huggingface_cache/hub"

def snap(repo):
    g = glob.glob(f"{HUB}/models--{repo.replace('/','--')}/snapshots/*")
    if not g:
        return None
    # latest by mtime
    g.sort(key=os.path.getmtime, reverse=True)
    return g[0]

need = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llada": "GSAI-ML/LLaDA-1.5",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "diffucoder": "apple/DiffuCoder-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "falcon": "tiiuae/Falcon-H1R-7B",
}
for key, repo in need.items():
    p = snap(repo)
    if p:
        # replace this key's quoted path with the absolute snapshot dir
        s = re.sub(r'("{key}"[ ]*:[ ]*)"[^"]*"'.format(key=key),
                   r'\1"%s"' % p, s)
        print(f"{key} -> {p}")
    else:
        print(f"{key}: NO snapshot for {repo}")

open(SRC, "w", encoding="utf-8").write(s)

# sanity
for line in s.splitlines():
    if '"llama"' in line or '"qwen2.5"' in line or '"diffucoder"' in line:
        print("  *", line.strip()[:100])
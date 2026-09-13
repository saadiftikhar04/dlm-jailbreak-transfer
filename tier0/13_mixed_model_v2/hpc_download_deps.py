"""R2 Step 13 — HPC: download full-pool ArrAttack dependencies to bc3194 HF cache."""
import os
os.environ["HF_HOME"] = "/scratch/bc3194/huggingface_cache"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

targets = [
    "humarin/chatgpt_paraphraser_on_T5_base",          # rewriter
    "hubert233/GPTFuzz",                                # robustness judge (RoBERTa)
    "sentence-transformers/all-mpnet-base-v2",          # similarity encoder
    "meta-llama/Llama-3.1-8B-Instruct",                 # paper llama (only 3B cached)
]
for t in targets:
    print(f"=== downloading {t}", flush=True)
    try:
        p = snapshot_download(t, token=None)  # token handled by cache/env
        print(f"  OK -> {p}", flush=True)
    except Exception as e:
        print(f"  FAIL {t}: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("DONE", flush=True)
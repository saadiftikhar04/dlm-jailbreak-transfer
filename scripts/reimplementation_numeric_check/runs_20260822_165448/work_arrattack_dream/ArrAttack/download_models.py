#!/usr/bin/env python3
"""
download_models.py
==================
Run this ONCE on a login node with internet access to cache all models.
Submit as a short SLURM job or run interactively.

Models to download:
  LLMs: Falcon-H1R-7B, Llama-3.2-3B-Instruct
  DLMs: LLaDA-1.5, Dream-v0-Instruct-7B, DiffuCoder-7B-Instruct

Qwen2.5-7B-Instruct and Qwen2.5-7B are already cached.
"""

import os
os.environ["HF_HOME"] = "/scratch/si2356/.cache/huggingface"

from huggingface_hub import snapshot_download

models = [
    # LLMs
    ("tiiuae/Falcon-H1R-7B",           "falcon"),
    ("meta-llama/Llama-3.2-3B-Instruct","llama"),
    # DLMs
    ("GSAI-ML/LLaDA-1.5",              "llada"),
    ("Dream-org/Dream-v0-Instruct-7B",  "dream"),
    ("apple/DiffuCoder-7B-Instruct",    "diffucoder"),
]

for hf_id, name in models:
    print(f"\n{'='*50}")
    print(f"Downloading: {hf_id}")
    print(f"{'='*50}")
    try:
        path = snapshot_download(
            repo_id=hf_id,
            cache_dir="/scratch/si2356/.cache/huggingface/hub",
            ignore_patterns=["*.gguf", "*.ggml", "flax_*", "tf_*"],
        )
        print(f"✓ {name} saved to: {path}")
    except Exception as e:
        print(f"✗ {name} FAILED: {e}")

print("\n=== All downloads complete ===")
print("Now set TRANSFORMERS_OFFLINE=1 in your SLURM scripts.")

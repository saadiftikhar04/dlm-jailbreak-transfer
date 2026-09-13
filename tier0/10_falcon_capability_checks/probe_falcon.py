"""Probe Falcon-H1R raw output shape for one benign prompt (T10.1 prep).
Runs locally (HPC mamba_ssm ABI broken, same as T07/T08/T09).
Prints the FULL raw generation so we can see how reasoning vs answer are
delimited, to decide how to split raw/stripped.
"""
import os, sys
os.environ["HF_HOME"] = "/home/bc3194/Desktop/huggingface_cache"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = "falcon"
BENIGN = "Explain how to make a simple cup of tea from choosing the water temperature to steeping the leaves for the right amount of time."

model, tokenizer = load_target(MODEL_KEY, offline=True)
model.eval()
resp = target_generate(model, tokenizer, MODEL_KEY, BENIGN, max_new_tokens=512)
print("=== FULL RAW OUTPUT (%d chars) ===" % len(resp))
print(repr(resp))
print("\n=== printed ===")
print(resp)
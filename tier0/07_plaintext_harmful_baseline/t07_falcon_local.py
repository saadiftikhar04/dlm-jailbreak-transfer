"""Run Falcon-H1R plaintext baseline generation on the LOCAL desktop (HPC raven
env has a broken mamba_ssm/selective_scan_cuda ABI; local raven_rag works).
Reuses the same t07_generate_hpc logic: 100 sampled harmful prompts, unmodified,
target_generate with main-experiment decoding. Writes model_outputs/falcon_outputs.json.
"""
import os, sys, json, time
os.environ["HF_HOME"] = "/home/bc3194/Desktop/huggingface_cache"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate, is_model_key

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_KEY = "falcon"
with open(os.path.join(BASE, "baseline_sample.json")) as f:
    sample = json.load(f)

model, tokenizer = load_target(MODEL_KEY, offline=True)  # local HF cache
model.eval()

records = []
t0 = time.time()
for i, q in enumerate(sample):
    try:
        resp = target_generate(model, tokenizer, MODEL_KEY, q["original_prompt"],
                               max_new_tokens=512)
        status = "ok"
    except Exception as e:
        resp = ""
        status = f"error:{type(e).__name__}:{str(e)[:120]}"
    records.append({"prompt_idx": q["prompt_idx"], "dataset": q["dataset"],
                    "model": "falcon", "original_prompt": q["original_prompt"],
                    "response": resp, "status": status,
                    "generation_seconds": round(time.time() - t0, 2)})
    if (i + 1) % 10 == 0:
        print(f"[falcon] {i+1}/{len(sample)}", flush=True)
    t0 = time.time()

out = os.path.join(OUTDIR, "falcon_outputs.json")
with open(out, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print(f"wrote {out}: {len(records)} records, ok="
      f"{sum(1 for r in records if r['status']=='ok')}")
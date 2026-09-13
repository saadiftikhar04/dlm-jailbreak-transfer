"""Run Falcon-H1R T09.5 benign PiF-intelligibility generation on the LOCAL desktop
(HPC raven mamba_ssm ABI broken; local raven_rag works). Same decoder as T08 /
the main experiment (target_generate). Mirrors t08_falcon_local.py.

Input: benign_pif_transformed.json
Output: model_outputs/falcon_outputs.json  (record shape matches the HPC generator)
"""
import os, sys, json, time
os.environ["HF_HOME"] = "/home/bc3194/Desktop/huggingface_cache"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_KEY = "falcon"
with open(os.path.join(BASE, "benign_pif_transformed.json")) as f:
    transformed = json.load(f)
assert len(transformed) == 60

model, tokenizer = load_target(MODEL_KEY, offline=True)  # local HF cache
model.eval()

records = []
t0 = time.time()
for i, w in enumerate(transformed):
    try:
        resp = target_generate(model, tokenizer, MODEL_KEY, w["pif_prompt"],
                               max_new_tokens=512)
        status = "ok"
    except Exception as e:
        resp = ""
        status = f"error:{type(e).__name__}:{str(e)[:120]}"
    records.append({"id": w["id"], "category": w["category"],
                    "original_prompt": w["original_prompt"],
                    "pif_prompt": w["pif_prompt"], "model": "falcon",
                    "response": resp, "status": status,
                    "generation_seconds": round(time.time() - t0, 2)})
    if (i + 1) % 10 == 0:
        print(f"[falcon] {i+1}/{len(transformed)}", flush=True)
    t0 = time.time()

out = os.path.join(OUTDIR, "falcon_outputs.json")
with open(out, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print(f"wrote {out}: {len(records)} records, ok="
      f"{sum(1 for r in records if r['status']=='ok')}")
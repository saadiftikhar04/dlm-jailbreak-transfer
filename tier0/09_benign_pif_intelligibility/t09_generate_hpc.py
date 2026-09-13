"""T09.5 (todo): generate the benign PiF-transformed prompts on all six victims.
Identical decoder as T08 / the main experiment (target_generate, unmodified
pass-through of the pif_prompt). Models already on HPC (HF cache). One process
per victim, one GPU per job.

CLI: python t09_generate_hpc.py <model_key>   e.g. qwen2.5 (uniform key: qwen2.5/llama/falcon/llada/dream/diffucoder)

Output: 09_benign_pif_intelligibility/model_outputs/{qwen,llama,falcon,llada,dream,diffucoder}_outputs.json
record: {id, category, original_prompt, pif_prompt, response, generation_seconds, status}
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = sys.argv[1].strip().lower()
BASE = "/scratch/bc3194/dlm-jailbreak-transfer/tier0/09_benign_pif_intelligibility"
OUTDIR = os.path.join(BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)

FILENAME = {"qwen2.5": "qwen", "llama": "llama", "falcon": "falcon",
            "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}
assert MODEL_KEY in FILENAME, MODEL_KEY

with open(os.path.join(BASE, "benign_pif_transformed.json")) as f:
    transformed = json.load(f)
assert len(transformed) == 60, len(transformed)

model, tokenizer = load_target(MODEL_KEY, offline=False)
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
                    "pif_prompt": w["pif_prompt"], "model": FILENAME[MODEL_KEY],
                    "response": resp, "status": status,
                    "generation_seconds": round(time.time() - t0, 2)})
    if (i + 1) % 10 == 0:
        print(f"  [{FILENAME[MODEL_KEY]}] {i+1}/{len(transformed)}", flush=True)
    t0 = time.time()

out = os.path.join(OUTDIR, f"{FILENAME[MODEL_KEY]}_outputs.json")
with open(out, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print(f"\nwrote {out}: {len(records)} records, ok="
      f"{sum(1 for r in records if r['status']=='ok')}")
"""T07 generation: submit each of 100 sampled harmful prompts UNMODIFIED to all
six victims using the main experiment's decoding settings (pif_target_models.py
target_generate). Runs on the HPC raven env; downloads models if absent.
Output: model_outputs/{model}_outputs.json  (one per victim)

CLI: python t07_generate_hpc.py <model_key> [shard]  e.g. qwen2.5
One process per victim so each model is loaded once (diffusion victims are slow).
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate, is_model_key, resolve_hf_id

MODEL_KEY = sys.argv[1].strip().lower()
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0

BASE = "/scratch/bc3194/dlm-jailbreak-transfer/tier0/07_plaintext_harmful_baseline"
SAMPLE = os.path.join(BASE, "baseline_sample.json")
OUTDIR = os.path.join(BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)

# unified key -> file name (matching todo outputs)
FILENAME = {"qwen2.5": "qwen", "falcon": "falcon", "llama": "llama",
            "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}
assert MODEL_KEY in FILENAME, MODEL_KEY

with open(SAMPLE) as f:
    sample = json.load(f)

model, tokenizer = load_target(MODEL_KEY, offline=False)  # allow download
model.eval()

records = []
t0 = time.time()
for i, q in enumerate(sample):
    prompt = q["original_prompt"]
    try:
        resp = target_generate(model, tokenizer, MODEL_KEY, prompt, max_new_tokens=512)
        status = "ok"
    except Exception as e:
        resp = ""
        status = f"error:{type(e).__name__}:{str(e)[:120]}"
    records.append({
        "prompt_idx": q["prompt_idx"], "dataset": q["dataset"],
        "model": FILENAME[MODEL_KEY],
        "original_prompt": prompt,
        "response": resp, "status": status,
        "generation_seconds": round(time.time() - t0, 2),
    })
    if (i + 1) % 10 == 0:
        print(f"  [{FILENAME[MODEL_KEY]}] {i+1}/{len(sample)}")
    t0 = time.time()

out = os.path.join(OUTDIR, f"{FILENAME[MODEL_KEY]}_outputs.json")
with open(out, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print(f"\nwrote {out}: {len(records)} records, status ok = "
      f"{sum(1 for r in records if r['status']=='ok')}")
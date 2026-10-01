"""T1 (C1) generation: submit each sampled harmful prompt UNMODIFIED to one
victim using the validated plaintext chat-template-only path
(pif_target_models.py target_generate, same path T07 validated). One process
per victim; model loaded once. Output: model_outputs/{file}_outputs.json.

CLI: python t1_generate.py <model_key> [shard0]
  model_key in qwen2.5|falcon|llama|llada|dream|diffucoder
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = sys.argv[1].strip().lower()
BASE = "/scratch/bc3194/dlm-jailbreak-transfer/tier0/18_t1_plaintext_control"
SAMPLE = os.path.join(BASE, "t1_sample.json")
OUTDIR = os.path.join(BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)

FILENAME = {"qwen2.5": "qwen", "falcon": "falcon", "llama": "llama",
            "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}
assert MODEL_KEY in FILENAME, MODEL_KEY

with open(SAMPLE) as f:
    sample = json.load(f)

model, tokenizer = load_target(MODEL_KEY, offline=False)
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
    if (i + 1) % 25 == 0:
        print(f"  [{FILENAME[MODEL_KEY]}] {i+1}/{len(sample)}", flush=True)
    t0 = time.time()

out = os.path.join(OUTDIR, f"{FILENAME[MODEL_KEY]}_outputs.json")
with open(out, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)
print(f"\nwrote {out}: {len(records)} records, ok={sum(1 for r in records if r['status']=='ok')}", flush=True)
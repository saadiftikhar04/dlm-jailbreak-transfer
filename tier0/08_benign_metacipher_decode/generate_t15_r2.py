"""R2 Step 15: generate responses for the 150 benign MetaCipher-wrapped prompts
on one victim. Same decoder as the main experiment (target_generate). Resumable
per-(id) via JSONL append.

CLI: python generate_t15_r2.py <model_key>   (qwen2.5/llama/falcon/llada/dream/diffucoder)
Output: 08_benign_metacipher_decode/model_outputs_r2/{model}_outputs.jsonl
record: {id, category, original_prompt, wrapped_prompt, n_masks, response, status}
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = sys.argv[1].strip().lower()
BASE = "/scratch/bc3194/dlm-jailbreak-transfer/tier0/08_benign_metacipher_decode"
OUTDIR = os.path.join(BASE, "model_outputs_r2")
os.makedirs(OUTDIR, exist_ok=True)

FILENAME = {"qwen2.5": "qwen", "llama": "llama", "falcon": "falcon",
            "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}
assert MODEL_KEY in FILENAME, MODEL_KEY

with open(os.path.join(BASE, "benign_metacipher_wrapped_r2.json")) as f:
    wrapped = json.load(f)
assert len(wrapped) == 150

out = os.path.join(OUTDIR, f"{FILENAME[MODEL_KEY]}_outputs.jsonl")
done = set()
if os.path.exists(out):
    with open(out) as fh:
        for line in fh:
            if line.strip():
                done.add(json.loads(line)["id"])
todo = [w for w in wrapped if w["id"] not in done]
print(f"{FILENAME[MODEL_KEY]}: {len(todo)} to run (done {len(done)}/150)", flush=True)
if not todo:
    print("already complete"); sys.exit(0)

model, tokenizer = load_target(MODEL_KEY, offline=False)
model.eval()
t0 = time.time()
open_f = open(out, "a")
ok = 0
try:
    for i, w in enumerate(todo):
        try:
            resp = target_generate(model, tokenizer, MODEL_KEY, w["wrapped_prompt"],
                                   max_new_tokens=512)
            status = "ok"
            ok += 1
        except Exception as e:
            resp = ""
            status = f"error:{type(e).__name__}:{str(e)[:120]}"
        rec = {"id": w["id"], "category": w["category"],
               "original_prompt": w["original_prompt"],
               "wrapped_prompt": w["wrapped_prompt"],
               "n_masks": w["n_masks"], "model": FILENAME[MODEL_KEY],
               "response": resp, "status": status,
               "generation_seconds": round(time.time() - t0, 2)}
        open_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); open_f.flush()
        t0 = time.time()
        if (i + 1) % 10 == 0:
            print(f"  [{FILENAME[MODEL_KEY]}] {len(done)+i+1}/150 (ok={ok})", flush=True)
finally:
    open_f.close()
print(f"wrote {out}: new={len(todo)} ok={ok}", flush=True)
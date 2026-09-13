"""R2 Step 14: multi-seed regeneration for the 6 diffusion cells.

Reuses the recorded attacked_prompt (PiF pif_prompt / ArrAttack jailbreak_prompt)
and regenerates the victim under 3 torch seeds via target_generate. Does NOT
re-run the attack. Resumable per-(attack,model,dataset,prompt_idx,seed).

CLI: python gen_multiseed_r2.py <model_key>  (dream|diffucoder|llada)
Reads sample filter rows for this model (all its attacks), writes:
  model_outputs/{model}_seed{N}_outputs.json (JSONL append, resumable)
Per-generation wall-clock timeout (R2_MS_TIMEOUT, default 150s) so a pathological
diffusion row is recorded error:timeout instead of stalling the whole job.

Diffusion victims need the diffusion_generate path; target_generate already routes
the key correctly (dream/diffucoder/llada via load_target).
"""
import os, sys, json, signal
import torch

HF = os.environ.get("HF_HOME", "/scratch/bc3194/huggingface_cache")
os.environ.setdefault("HF_HOME", HF)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = sys.argv[1].strip().lower()
assert MODEL_KEY in ("dream", "diffucoder", "llada"), MODEL_KEY
_BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.environ.get("R2_SAMPLE_OVERRIDE",
                        os.path.join(_BASE, "multiseed_sample_r2.json"))
OUTDIR = os.path.join(_BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)
SEEDS = [1, 2, 3]
_seed_env = os.environ.get("R2_SEED_IDX")
if _seed_env:
    SEEDS = [int(_seed_env)]
TIMEOUT = int(os.environ.get("R2_MS_TIMEOUT", "150"))


class _Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

with open(SAMPLE) as f:
    sample = json.load(f)
rows = [r for r in sample if r["model"] == MODEL_KEY]
print(f"{MODEL_KEY}: {len(rows)} sample rows across attacks "
      f"{sorted(set(r['attack'] for r in rows))}", flush=True)

for seed in SEEDS:
    out = os.path.join(OUTDIR, f"{MODEL_KEY}_seed{seed}_outputs.json")
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["attack"], r["dataset"], r["prompt_idx"]))
    todo = [r for r in rows if (r["attack"], r["dataset"], r["prompt_idx"]) not in done]
    print(f"  seed {seed}: {len(todo)} to run (done {len(done)})", flush=True)
    if not todo:
        continue
    model = tokenizer = None
    open_f = open(out, "a")
    try:
        n = 0
        for r in todo:
            if model is None:
                model, tokenizer = load_target(MODEL_KEY, offline=False)
                model.eval()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            signal.alarm(TIMEOUT)
            try:
                resp = target_generate(model, tokenizer, MODEL_KEY,
                                       r["attacked_prompt"], max_new_tokens=512)
                status = "ok"
            except _Timeout:
                resp = ""
                status = f"error:timeout({TIMEOUT}s)"
            except Exception as e:
                resp = ""
                status = f"error:{type(e).__name__}:{str(e)[:120]}"
            finally:
                signal.alarm(0)
            rec = {**{k: r[k] for k in ["attack", "model", "model_family",
                                        "dataset", "prompt_idx", "arm",
                                        "original_prompt", "attacked_prompt"]},
                   "seed": seed, "response": resp, "status": status}
            open_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            open_f.flush()
            n += 1
            if n % 10 == 0:
                print(f"    [{n}/{len(todo)}] {MODEL_KEY} seed {seed}", flush=True)
    finally:
        open_f.close()
    print(f"  wrote {MODEL_KEY} seed {seed}: {n} new rows", flush=True)

print("\nall seeds done for", MODEL_KEY)
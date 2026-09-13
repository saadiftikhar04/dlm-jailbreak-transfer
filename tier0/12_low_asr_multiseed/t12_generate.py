"""T12.3: regenerate victim responses for the multi-seed sample under 3 seeds.
Loads the recorded attacked_prompt, sets torch seed, and calls target_generate
(main-experiment decoder). For victim/decoder pairs that are deterministic
(Dream temp 0, greedy causal) the three seeds will be identical; that is
recorded honestly rather than faked.

CLI: python t12_generate.py <model_key>
Reads multiseed_sample.json (rows for this model), writes per-seed outputs to
model_outputs/{model}_seed{N}_outputs.json (resumable by (dataset,prompt_idx)).
"""
import os, sys, json, time
import torch
os.environ.setdefault("HF_HOME", "/home/bc3194/Desktop/huggingface_cache")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate

MODEL_KEY = sys.argv[1].strip().lower()
LOAD_KEY = {"qwen": "qwen2.5"}.get(MODEL_KEY, MODEL_KEY)  # sample uses 'qwen', model loader uses 'qwen2.5'
_BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(_BASE, "multiseed_sample.json")
OUTDIR = os.path.join(_BASE, "model_outputs")
os.makedirs(OUTDIR, exist_ok=True)
SEEDS = [1, 2, 3]
TIMEOUT = int(os.environ.get("T12_TIMEOUT", "120"))  # per-generation wall-clock cap

import signal as _sig
class _Timeout(Exception):
    pass
def _alarm(sig, frm):
    raise _Timeout()
_sig.signal(_sig.SIGALRM, _alarm)

with open(SAMPLE) as f:
    sample = json.load(f)
rows = [r for r in sample if r["model"] == MODEL_KEY]
print(f"{MODEL_KEY}: {len(rows)} sample rows across arms "
      f"{sorted(set(r['arm'] for r in rows))}")

for seed in SEEDS:
    out = os.path.join(OUTDIR, f"{MODEL_KEY}_seed{seed}_outputs.json")
    done = set()
    if os.path.exists(out):
        done = {(r["dataset"], r["prompt_idx"]) for r in
                (json.loads(l) for l in open(out) if l.strip())}
    todo = [r for r in rows if (r["dataset"], r["prompt_idx"]) not in done]
    print(f"  seed {seed}: {len(todo)} to run (done {len(done)})")
    if not todo:
        continue
    model = tokenizer = None
    open_f = open(out, "a")
    try:
        n = 0
        for r in todo:
            if model is None:
                model, tokenizer = load_target(LOAD_KEY, offline=False)
                model.eval()
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            _sig.alarm(TIMEOUT)
            try:
                resp = target_generate(model, tokenizer, MODEL_KEY, r["attacked_prompt"],
                                       max_new_tokens=512)
                status = "ok"
            except _Timeout:
                resp = ""; status = f"error:timeout({TIMEOUT}s)"
            except Exception as e:
                resp = ""; status = f"error:{type(e).__name__}:{str(e)[:120]}"
            finally:
                _sig.alarm(0)
            rec = {**{k: r[k] for k in ["attack","model","model_family","dataset",
                                        "prompt_idx","arm","official_success",
                                        "original_prompt","attacked_prompt"]},
                   "seed": seed, "response": resp, "status": status}
            open_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); open_f.flush()
            n += 1
            if n % 10 == 0: print(f"    [{n}/{len(todo)}]", flush=True)
    finally:
        open_f.close()
    print(f"  wrote seed {seed}: {n} new rows")

# seed-variation check (todo): for a stochastic victim, seeds should differ
print("\nseed-variation check (diffucoder is stochastic, dream deterministic):")
for probe_m, label in [("diffucoder", "expect some diff"), ("dream", "expect identical")]:
    outs = []
    for seed in SEEDS:
        fp = os.path.join(OUTDIR, f"{probe_m}_seed{seed}_outputs.json")
        if os.path.exists(fp):
            outs.append(json.load(open(fp)))
    if len(outs) >= 2 and outs[0]:
        r0 = outs[0][0]
        for j in range(1, len(outs)):
            m = [(o["dataset"], o["prompt_idx"]) for o in outs[j]]
            if (r0["dataset"], r0["prompt_idx"]) in m:
                rj = outs[j][m.index((r0["dataset"], r0["prompt_idx"]))]
                same = rj["response"] == r0["response"]
                print(f"  {probe_m}: seed1 vs seed{j+1} first row identical={same} ({label})")
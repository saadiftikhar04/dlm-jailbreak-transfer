"""T12.2: sample prompts for multi-seed low-ASR rerun (MetaCipher four cells).
Reads the recorded MetaCipher judged CSVs and draws, per (model, attack):
  Arm A: 20 prompts uniformly at random (rate estimation)
  Arm B: 5 recorded successes + 5 recorded failures (conditional flip analysis)
Fixed seed. Output multiseed_sample.json (one row per prompt, with `arm`).
Each selected prompt carries its RECORDED attacked_prompt (final_converted_prompt)
so T12.3 can regenerate the victim response under different seeds.

Cells (compressed schedule, four bolded): Dream-MetaCipher, DiffuCoder-MetaCipher,
LLaDA-MetaCipher (high anchor), Qwen-MetaCipher (high anchor).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "12_low_asr_multiseed", "multiseed_sample.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
SEED = 20260822
rng = np.random.default_rng(SEED)

CELLS = [("dream", "metacipher"), ("diffucoder", "metacipher"),
         ("llada", "metacipher"), ("qwen", "metacipher")]
ARM_A = 20
ARM_B = 5

rows = []
seen_keys = set()
for model, attack in CELLS:
    df = C.load(attack, model)  # uses correct file
    succ = C.success_series(attack, df)
    attacked_col = "final_converted_prompt"
    # Arm A: uniform over all rows
    n = min(ARM_A, len(df))
    idx_A = rng.choice(len(df), size=n, replace=False)
    # Arm B: 5 successes + 5 failures
    succ_idx = list(np.where(succ.values)[0])
    fail_idx = list(np.where(~succ.values)[0])
    sB = rng.choice(succ_idx, size=min(ARM_B, len(succ_idx)), replace=False)
    fB = rng.choice(fail_idx, size=min(ARM_B, len(fail_idx)), replace=False)

    for j, arm in [(i, "A") for i in idx_A] + \
                  [(i, "B") for i in sB] + [(i, "B") for i in fB]:
        r = df.iloc[j]
        key = (attack, model, r["dataset"], int(r["prompt_idx"]), arm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append({
            "attack": attack, "model": model, "model_family": C.FAMILY[model],
            "dataset": r["dataset"], "prompt_idx": int(r["prompt_idx"]),
            "arm": arm, "official_success": bool(succ.iloc[j]),
            "original_prompt": r["original_prompt"],
            "attacked_prompt": r[attacked_col],
        })

with open(OUT, "w") as f:
    json.dump(rows, f, indent=4, ensure_ascii=False)
print(f"wrote {len(rows)} rows -> {OUT}")
from collections import Counter
print("per (model,arm):", dict(Counter((r['model'], r['arm']) for r in rows)))
print("per (model,arm,official_success):", dict(Counter((r['model'], r['arm'], r['official_success']) for r in rows)))
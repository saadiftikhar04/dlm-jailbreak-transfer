"""T07 step 1: sample 100 prompts uniformly from the 913-prompt pool,
stratified by benchmark suite in proportion to the pool. Fixed seed.
Output: baseline_sample.json (prompt_idx, dataset, original_prompt).
Ref: G2 (uniform within-stratum, not outcome-stratified); todo.md T07.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

SEED = C.SEED          # 20260822, matches every other sampler
N = 100
OUT = os.path.join(C.OUT_ROOT, "07_plaintext_harmful_baseline")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)

# pool (uniform within-stratum proportions by suite)
pool = C.load("metacipher", "qwen")[["prompt_idx", "dataset", "original_prompt"]]
pool["original_prompt"] = pool["original_prompt"].astype(str)
print("pool:", len(pool), pool.dataset.value_counts().to_dict())

# per-suite counts proportional to pool share
shares = {b: round(len(g) / len(pool) * N) for b, g in pool.groupby("dataset")}
# fix rounding drift to sum exactly N
diff = N - sum(shares.values())
for b in list(shares):  # add remainder to largest suite
    if diff == 0:
        break
    shares[b] += 1
    diff -= 1
print("per-suite sample counts:", shares)

rows = []
for bench, cnt in shares.items():
    sub = pool[pool.dataset == bench]
    idx = rng.choice(len(sub), size=cnt, replace=False)
    for i in idx:
        r = sub.iloc[i]
        rows.append({"prompt_idx": int(r.prompt_idx), "dataset": bench,
                     "original_prompt": r.original_prompt})
sample = pd.DataFrame(rows).sort_values("prompt_idx")
assert len(sample) == N, f"len={len(sample)}"
# uniqueness on the composite key (prompt_idx can repeat across datasets)
assert sample[["dataset", "prompt_idx"]].drop_duplicates().shape[0] == N

out = os.path.join(OUT, "baseline_sample.json")
with open(out, "w") as f:
    json.dump(sample.to_dict(orient="records"), f, indent=4, ensure_ascii=False)
print(f"\nwrote {out}: {len(sample)} prompts")
print(sample.dataset.value_counts().to_dict())
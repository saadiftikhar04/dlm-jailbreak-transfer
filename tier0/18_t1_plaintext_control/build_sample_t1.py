"""T1 (C1) step 1: sample 300 prompts uniformly-stratified from the 913-prompt pool.
Stratified by benchmark suite in proportion to the pool, fixed seed (matches all
other samplers). Output: t1_sample.json (prompt_idx, dataset, original_prompt).
Ref: D3 ruling = uniform 300-prompt stratified baseline for all six victims.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

SEED = C.SEED
N = 300
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)
pool = C.load("metacipher", "qwen")[["prompt_idx", "dataset", "original_prompt"]]
pool["original_prompt"] = pool["original_prompt"].astype(str)
print("pool:", len(pool), pool.dataset.value_counts().to_dict())

shares = {b: int(round(len(g) / len(pool) * N)) for b, g in pool.groupby("dataset")}
diff = N - sum(shares.values())
for b in list(shares):
    if diff == 0:
        break
    shares[b] += 1
    diff -= 1
print("per-suite sample counts:", shares, "sum=", sum(shares.values()))

rows = []
for bench, cnt in shares.items():
    sub = pool[pool.dataset == bench]
    idx = rng.choice(len(sub), size=cnt, replace=False)
    for i in idx:
        r = sub.iloc[i]
        rows.append({"prompt_idx": int(r.prompt_idx), "dataset": bench,
                     "original_prompt": r.original_prompt})
sample = pd.DataFrame(rows).sort_values("prompt_idx")
assert len(sample) == N
assert sample[["dataset", "prompt_idx"]].drop_duplicates().shape[0] == N

out = os.path.join(OUT, "t1_sample.json")
with open(out, "w") as f:
    json.dump(sample.to_dict(orient="records"), f, indent=4, ensure_ascii=False)
print(f"\nwrote {out}: {len(sample)} prompts")
print(sample.dataset.value_counts().to_dict())
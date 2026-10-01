"""T3 (R3): build the Llama MetaCipher 3-seed sample (Arm A, ~70 uniform prompts)
from the recorded MetaCipher judged CSV, matching multiseed_sample.json schema.
Appends to /tmp t3_llama_rows.json for upload+append to the HPC multiseed_sample.json.
Also bumps qwen to ceil (already present in tier0/12 sample if we leave it).
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/00_shared")
import common as C

SEED = 20260822
N = 70
rng = np.random.default_rng(SEED)

df = pd.read_csv("/home/bc3194/Desktop/dlm-jailbreak-transfer/results/metacipher/Metacipher_Judged/llama.csv")
df["attacked_prompt"] = df["final_converted_prompt"]
print("llama metacipher rows:", len(df))

# uniform Arm A over datasets (stratified by dataset share)
shares = {b: int(round(len(g) / len(df) * N)) for b, g in df.groupby("dataset")}
diff = N - sum(shares.values())
for b in list(shares):
    if diff == 0:
        break
    shares[b] += 1
    diff -= 1
rows = []
for bench, cnt in shares.items():
    sub = df[df.dataset == bench]
    idx = rng.choice(len(sub), size=cnt, replace=False)
    for i in idx:
        r = sub.iloc[i]
        rows.append({"attack": "metacipher", "model": "llama", "model_family": "causal",
                     "dataset": bench, "prompt_idx": int(r.prompt_idx), "arm": "A",
                     "official_success": bool(r.asr_success),
                     "original_prompt": r.original_prompt,
                     "attacked_prompt": r.final_converted_prompt})
print("llama sample rows:", len(rows), "per-dataset:", pd.DataFrame(rows).dataset.value_counts().to_dict())
with open("/tmp/t3_llama_rows.json", "w") as f:
    json.dump(rows, f, indent=4, ensure_ascii=False)
print("wrote /tmp/t3_llama_rows.json")
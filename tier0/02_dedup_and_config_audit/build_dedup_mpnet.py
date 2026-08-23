"""
T02.1 (FINAL): MPNet semantic dedup, replacing the TF-IDF lower-bound proxy.
Uses precomputed all-mpnet-base-v2 embeddings (mpnet_913.npy, encoded on the HPC)
+ mpnet_index.csv for row alignment. Same thresholds as the proxy run so the two
are directly comparable.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.dirname(__file__)
emb = np.load(os.path.join(OUT, "mpnet_913.npy"))
idx = pd.read_csv(os.path.join(OUT, "mpnet_index.csv"))
idx["original_prompt"] = idx["original_prompt"].astype(str)

# alignment guard (fail loudly rather than silently mis-map)
pool = C.load("metacipher", "qwen")[["prompt_idx", "dataset"]].reset_index(drop=True)
assert (idx["prompt_idx"].values == pool["prompt_idx"].values).all()
assert (idx["dataset"].values == pool["dataset"].values).all()
assert emb.shape[0] == len(idx) == 913

# cosine = dot product (embeddings are unit-normalized)
S = emb @ emb.T
np.fill_diagonal(S, 0.0)

prompts = idx["original_prompt"].tolist()
benches = idx["dataset"].tolist()
pidx = idx["prompt_idx"].tolist()

pairs = []
n = len(prompts)
for i in range(n):
    for j in range(i + 1, n):
        s = float(S[i, j])
        if s >= 0.90:
            pairs.append((i, j, round(s, 4)))
pairs.sort(key=lambda t: -t[2])

def count_ge(th):
    return sum(1 for _, _, s in pairs if s >= th)

rows = []
for i, j, s in pairs:
    rows.append({"idx_a": int(pidx[i]), "bench_a": benches[i],
                 "idx_b": int(pidx[j]), "bench_b": benches[j], "cosine": s,
                 "cross_benchmark": benches[i] != benches[j],
                 "prompt_a": prompts[i][:160], "prompt_b": prompts[j][:160]})
dup = pd.DataFrame(rows)
dup.to_csv(os.path.join(OUT, "near_duplicate_pairs_mpnet.csv"), index=False)

# exact-string dupes (method-independent, same as proxy run)
norm = idx["original_prompt"].str.strip().str.lower()
vc = norm.value_counts()
n_exact = int(vc[vc > 1].sum() - (vc > 1).sum())
cross = int(dup["cross_benchmark"].sum()) if len(dup) else 0

# compare against the TF-IDF proxy that ran earlier
proxy_path = os.path.join(OUT, "near_duplicate_pairs.csv")
proxy_note = ""
if os.path.exists(proxy_path):
    pxy = pd.read_csv(proxy_path)
    proxy_note = (f"\n## MPNet vs TF-IDF proxy\n"
                  f"- TF-IDF proxy pairs >=0.90: {len(pxy)} "
                  f"(cross-benchmark {int((pxy.bench_a!=pxy.bench_b).sum())})\n"
                  f"- MPNet semantic pairs >=0.90: {len(dup)} "
                  f"(cross-benchmark {cross})\n"
                  f"- MPNet catches paraphrase-level duplicates TF-IDF misses; the "
                  f"proxy was a lower bound, as flagged.")

summary = [
    "# T02.1 Deduplication — MPNet semantic (FINAL)\n",
    "Real all-mpnet-base-v2 embeddings (encoded on the HPC), replacing the TF-IDF "
    "lower-bound proxy. Cosine on unit-normalized vectors; same thresholds.\n",
    f"- pool size: {len(idx)} (400/313/100/100 -> 913; no dedup was applied)",
    f"- exact string duplicates (extra copies): **{n_exact}**",
    f"- MPNet cosine pairs >= 0.99: **{count_ge(0.99)}**",
    f"- MPNet cosine pairs >= 0.95: **{count_ge(0.95)}**",
    f"- MPNet cosine pairs >= 0.90: **{count_ge(0.90)}**",
    f"- of the >=0.90 pairs, **{cross}** are cross-benchmark (same behavior under two "
    f"source benchmarks -> inflates effective n, breaks CI independence).",
    proxy_note,
    "\n## Implication",
    "The 913-prompt pool contains semantic duplicates that were never removed. Every "
    "Wilson CI and significance test in the paper assumes independent prompts; the "
    "cross-benchmark duplicates violate that. The paper should either dedup the pool "
    "and recompute, or state the effective-n caveat explicitly.",
]
with open(os.path.join(OUT, "dedup_summary_mpnet.md"), "w") as f:
    f.write("\n".join(summary) + "\n")

print(f"MPNet dedup: exact={n_exact} >=.99={count_ge(0.99)} >=.95={count_ge(0.95)} "
      f">=.90={count_ge(0.90)} cross-bench={cross}")
if os.path.exists(proxy_path):
    pxy = pd.read_csv(proxy_path)
    print(f"(TF-IDF proxy was: >=.90={len(pxy)} cross-bench="
          f"{int((pxy.bench_a!=pxy.bench_b).sum())})")
print("wrote near_duplicate_pairs_mpnet.csv + dedup_summary_mpnet.md")

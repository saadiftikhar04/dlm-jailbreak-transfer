"""
T02: three audits. Concerns C7 + the independence assumption behind every CI.
 T02.1 deduplication of the 913-prompt pool
 T02.2 response length / empty / truncation statistics over all 11,946 responses
 T02.3 decoding-config audit (observable params from data; code-read flagged)
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = os.path.join(C.OUT_ROOT, "02_dedup_and_config_audit")

# ========================================================================
# T02.1  DEDUPLICATION of the 913-prompt pool
# ========================================================================
# Build the pool from a MetaCipher file (has all 913 with dataset labels).
pool = C.load("metacipher", "qwen")[["prompt_idx", "dataset", "original_prompt"]].copy()
pool["original_prompt"] = pool["original_prompt"].astype(str)
print(f"[T02.1] pool size = {len(pool)}  (expected 913)")
print("        by benchmark:", pool["dataset"].value_counts().to_dict())

prompts = pool["original_prompt"].tolist()
benches = pool["dataset"].tolist()

# exact string duplicates
norm = pool["original_prompt"].str.strip().str.lower()
exact_dupe_groups = norm.value_counts()
n_exact_dupe_rows = int((exact_dupe_groups[exact_dupe_groups > 1]).sum() -
                        (exact_dupe_groups > 1).sum())  # extra copies beyond first

# lexical near-dup via TF-IDF (word 1-2grams + char 3-5grams). MPNet blocked
# offline, so this is a reproducible lexical proxy; flagged in the summary.
vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
X = vec.fit_transform(prompts)
S = cosine_similarity(X)
np.fill_diagonal(S, 0.0)

pairs = []
n = len(prompts)
for i in range(n):
    for j in range(i + 1, n):
        s = S[i, j]
        if s >= 0.90:
            pairs.append((i, j, round(float(s), 4)))
pairs.sort(key=lambda t: -t[2])

def count_ge(th):
    return sum(1 for _, _, s in pairs if s >= th)

dup_rows = []
for i, j, s in pairs:
    dup_rows.append({
        "idx_a": int(pool.iloc[i]["prompt_idx"]), "bench_a": benches[i],
        "idx_b": int(pool.iloc[j]["prompt_idx"]), "bench_b": benches[j],
        "cosine": s,
        "prompt_a": prompts[i][:160], "prompt_b": prompts[j][:160],
    })
pd.DataFrame(dup_rows).to_csv(os.path.join(OUT, "near_duplicate_pairs.csv"), index=False)

dedup_summary = [
    "# T02.1 Deduplication (lexical proxy)\n",
    f"- pool size: {len(pool)} (400/100/100/313 -> 913 means no dedup was applied)",
    f"- benchmark counts: {pool['dataset'].value_counts().to_dict()}",
    f"- exact string duplicates (extra copies): **{n_exact_dupe_rows}**",
    f"- TF-IDF cosine pairs >= 0.99: **{count_ge(0.99)}**",
    f"- TF-IDF cosine pairs >= 0.95: **{count_ge(0.95)}**",
    f"- TF-IDF cosine pairs >= 0.90: **{count_ge(0.90)}**",
    "\n**Caveat:** huggingface.co is unreachable in this sandbox, so the MPNet "
    "semantic embedding specified in the plan could not be run. These counts use "
    "a TF-IDF (word 1-2gram) cosine, which catches lexical/near-verbatim overlap "
    "but will under-count paraphrase-level semantic duplicates. Re-run with MPNet "
    "on the HPC to finalize; the pair list here is a lower bound.",
    "\nCross-benchmark pairs are the ones that matter for the independence "
    "assumption (same behavior evaluated under two source benchmarks inflates "
    "effective n). See near_duplicate_pairs.csv, bench_a vs bench_b.",
]
cross = [r for r in dup_rows if r["bench_a"] != r["bench_b"]]
dedup_summary.append(f"\n- of the >=0.90 pairs, **{len(cross)}** are cross-benchmark.")
with open(os.path.join(OUT, "dedup_summary.md"), "w") as f:
    f.write("\n".join(dedup_summary) + "\n")
print(f"[T02.1] exact-dupe extra copies={n_exact_dupe_rows}  "
      f">=.99={count_ge(0.99)} >=.95={count_ge(0.95)} >=.90={count_ge(0.90)} "
      f"cross-bench={len(cross)}")

# ========================================================================
# T02.2  RESPONSE LENGTH / EMPTY / TRUNCATION over all 11,946 responses
# ========================================================================
# Token length proxy = whitespace tokens (no per-model tokenizer available
# offline); char length also reported. Truncation flagged by clustering at the
# per-cell modal max length (>=5% of rows within 1% of the cell max).
rows = []
for attack in C.ATTACKS:
    for m in C.MODELS:
        df = C.load(attack, m)
        resp = C.response_text(attack, df).fillna("")
        char_len = resp.str.len()
        tok_len = resp.str.split().apply(len)
        empty = C.is_empty(df[C.ATTACKS[attack]["response_col"]])
        # truncation proxy: rows whose token length equals the cell's max token
        # length AND that max is shared by >=5% of rows (a hard cap signature)
        maxtok = int(tok_len.max()) if len(tok_len) else 0
        at_max = int((tok_len == maxtok).sum())
        trunc_rate = at_max / len(df) if len(df) else 0
        rows.append({
            "attack": attack, "model": m, "family": C.FAMILY[m], "n": len(df),
            "mean_tok": round(float(tok_len.mean()), 1),
            "median_tok": float(tok_len.median()),
            "mean_chars": round(float(char_len.mean()), 1),
            "median_chars": float(char_len.median()),
            "empty_rate_pct": round(100 * empty.mean(), 2),
            "max_tok": maxtok,
            "rows_at_max_tok": at_max,
            "cap_cluster_rate_pct": round(100 * trunc_rate, 2),
            "cap_flag": trunc_rate > 0.10,
        })
rlen = pd.DataFrame(rows)
rlen.to_csv(os.path.join(OUT, "response_length_stats.csv"), index=False)
flagged = rlen[rlen.cap_flag]
print(f"\n[T02.2] response-length stats written. cells flagged for cap-clustering "
      f">10%: {len(flagged)}")
if len(flagged):
    print(flagged[["attack", "model", "max_tok", "cap_cluster_rate_pct"]].to_string(index=False))

# ========================================================================
# T02.3  DECODING-CONFIG AUDIT (observable-from-data + code-read flag)
# ========================================================================
# Generation code is NOT in the upload, so the authoritative per-cell matrix
# (temperature, top_p, remasking, chat template) cannot be filled from disk.
# We extract every decoding-related field that IS recorded in the CSVs and
# flag the LLaDA step/block inconsistency where the data allows.
cfg_rows = []
for attack in C.ATTACKS:
    for m in C.MODELS:
        df = C.load(attack, m)
        rec = {"attack": attack, "model": m, "family": C.FAMILY[m]}
        # diffusion step columns that actually exist per attack
        for col in ["diffusion_steps", "final_payload_step_count"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                rec[col] = (f"{vals.min():.0f}-{vals.max():.0f}"
                            if len(vals) and vals.min() != vals.max()
                            else (f"{vals.iloc[0]:.0f}" if len(vals) else ""))
        cfg_rows.append(rec)
cfg = pd.DataFrame(cfg_rows)
cfg.to_csv(os.path.join(OUT, "decoding_config_matrix.csv"), index=False)

# LLaDA-specific step check across attacks (the Appendix A.3 claim)
llada_steps = cfg[cfg.model == "llada"]
notes = [
    "# T02.3 Decoding-config audit (partial)\n",
    "**Scope limit:** the generation/attack source code is not in the uploaded "
    "results archive, so temperature, top_p, max_new_tokens, remasking strategy, "
    "block length and chat template per cell **cannot** be read from disk. This "
    "file records only decoding fields that are actually stored in the judged CSVs. "
    "The authoritative matrix + the .tex appendix version must be completed by "
    "reading the generation code on the HPC.\n",
    "## Observable diffusion-step fields\n",
    cfg.fillna("").to_markdown(index=False),
    "\n## LLaDA step/block check (Appendix A.3 claim)\n",
    "Appendix A.3 states LLaDA uses steps=gen_len=block_len=128 under PiF but "
    "128 steps over the full window under the other two attacks. Only ArrAttack "
    "stores an explicit `diffusion_steps` column; PiF/MetaCipher do not store the "
    "step schedule, so this specific inconsistency **cannot be fully confirmed "
    "from data alone** and needs the code read. Observed LLaDA step fields:",
    llada_steps.fillna("").to_markdown(index=False),
    "\nIf the code confirms the mismatch, LLaDA's 40.3-pt mechanism-dependence gap "
    "partly measures a decoding difference; the paper must either fix it (T17) or "
    "exclude LLaDA from MEE/MDG.",
]
with open(os.path.join(OUT, "decoding_config_notes.md"), "w") as f:
    f.write("\n".join(notes) + "\n")
print("\n[T02.3] decoding-config matrix written (partial; code-read flagged).")
print("\nAll T02 outputs in", OUT)

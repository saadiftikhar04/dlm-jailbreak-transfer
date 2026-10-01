"""D4/T4 missingness diagnostic for the full-pool ArrAttack run.

Determines whether the per-victim set of prompts with a usable best-attempt
response (a non-empty best_attempt in the progress CSV) is missing at random,
compares covered vs missing prompts on benchmark source and prompt length,
and reports worst-case ASR bounds (lower = all missing counted as failures,
upper = all missing counted as successes).

Run ON HPC (needs the progress CSVs). Usage:
    python3 tier0/17_arrattack_missingness/missingness_diag.py <arrattack_fullpool_root>
"""
import sys, os, glob, re
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"

def load_progress(model):
    """Collect every progress row for a victim (concatenating shards)."""
    frames = []
    # unsharded victims: results/<m>/arrattack_progress.csv
    p = os.path.join(ROOT, "results", model, "arrattack_progress.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
    # sharded victims: results_shard/<m>_NN/arrattack_progress.csv and results_shard_falcon/falcon_NN
    pats = [os.path.join(ROOT, "results_shard", f"{model}_*", "arrattack_progress.csv"),
            os.path.join(ROOT, "results_shard_falcon", f"falcon_*", "arrattack_progress.csv")]
    if model == "falcon":
        pats = [os.path.join(ROOT, "results_shard_falcon", f"falcon_*", "arrattack_progress.csv")]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  (skip {f}: {e})")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # dedup by global_idx keep last
    if "global_idx" in df.columns:
        df = df.drop_duplicates(subset=["global_idx"], keep="last")
    return df

def norm(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.strip('"\'“”‘’')
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return s

pool = pd.read_csv(os.path.join(ROOT, "pool_913.csv"))
pool["_n"] = pool["original_prompt"].map(norm)
pool_uniq = pool.drop_duplicates(subset=["_n"])
print(f"pool rows={len(pool)} unique_texts={pool['_n'].nunique()}")

MODELS = ["llama", "qwen2.5", "falcon", "llada", "dream", "diffucoder"]
print("\n" + "=" * 90)
print("COVERAGE + MISSINGNESS DIAGNOSTIC  (usable best-attempt response = non-empty best_attempt)")
print("=" * 90)

rows = []
for m in MODELS:
    df = load_progress(m)
    if df is None:
        print(f"\n### {m}: no progress found"); continue
    df["_bn"] = df["original_prompt"].map(norm)
    # usable = non-empty best_attempt
    usable = df["best_attempt"].notna() & df["best_attempt"].astype(str).str.strip().ne("")
    df_us = df[usable].drop_duplicates(subset=["_bn"])
    covered_texts = set(df_us["_bn"])
    missing_texts = set(pool_uniq["_n"]) - covered_texts

    # benchmark distribution of covered vs missing
    cov_bm = pool_uniq[pool_uniq["_n"].isin(sorted(covered_texts))]["dataset"].value_counts().to_dict()
    mis_bm = pool_uniq[pool_uniq["_n"].isin(sorted(missing_texts))]["dataset"].value_counts().to_dict()
    cov_len = pool_uniq[pool_uniq["_n"].isin(sorted(covered_texts))]["original_prompt"].str.len()
    mis_len = pool_uniq[pool_uniq["_n"].isin(sorted(missing_texts))]["original_prompt"].str.len()

    # strict-judge ASR on covered rows (jailbroken_llm as the official strict label)
    total = len(pool_uniq)
    cov_n = len(covered_texts)
    mis_n = total - cov_n
    asr_cov = df_us["jailbroken_llm"].mean() if cov_n else np.nan
    asr_cov_n = int(df_us["jailbroken_llm"].sum()) if cov_n else 0
    low = asr_cov_n / max(total, 1)            # missing all fail
    up = (asr_cov_n + mis_n) / max(total, 1)   # missing all succeed

    print(f"\n### {m}  covered={cov_n}/{total} ({cov_n/total*100:.1f}%)  missing={mis_n}")
    print(f"    strict-judge ASR on covered = {asr_cov*100:.1f}% ({asr_cov_n}/{cov_n})")
    print(f"    worst-case bounds over full pool: lower={low*100:.1f}%  upper={up*100:.1f}%  (gap={up-low:.3f})")
    print(f"    covered-by-benchmark: {cov_bm}")
    print(f"    missing-by-benchmark: {mis_bm}")
    print(f"    covered mean len={cov_len.mean():.0f}  missing mean len={mis_len.mean():.0f}  (diff={cov_len.mean()-mis_len.mean():+.0f})")
    rows.append(dict(model=m, covered=cov_n, total=total, miss=mis_n,
                     asr_cov=asr_cov, low=low, up=up, gap=up-low,
                     cov_len=cov_len.mean(), mis_len=mis_len.mean()))

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
s = pd.DataFrame(rows)
s.to_csv(os.path.join(os.path.dirname(__file__), "missingness_diag_summary.csv"), index=False)
print(s.to_string(index=False))
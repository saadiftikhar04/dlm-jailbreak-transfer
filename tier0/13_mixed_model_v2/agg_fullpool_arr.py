#!/usr/bin/env python3
"""Aggregate official full-pool ArrAttack ASR across the three result locations
(results/<name>/ for qwen2.5,llama,llada; results_shard/<m>_NN/ for dream,diffucoder;
results_shard_falcon/ for falcon). ASR = jailbroken_gptfuzz == compliance (True)."""
import os, glob, sys
import pandas as pd

B = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"

def read_all(pattern):
    dfs = []
    for p in sorted(glob.glob(pattern)):
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print("  !!", p, e)
            continue
        if "jailbroken_gptfuzz" in d.columns:
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def asr(df, col="jailbroken_gptfuzz"):
    if df is None or not len(df):
        return (0, 0)
    s = df[col].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "1.0": 1, "yes": 1, "compliance": 1,
         "false": 0, "0": 0, "0.0": 0, "no": 0}).fillna(0)
    success = int(pd.to_numeric(s, errors="coerce").fillna(0).sum())
    return (success, len(df))

specs = {
    "qwen2.5":   f"{B}/results/qwen2.5/arrattack_progress.csv",
    "llama":     f"{B}/results/llama/arrattack_progress.csv",
    "llada":     f"{B}/results/llada/arrattack_progress.csv",
    "dream":     f"{B}/results_shard/dream_*/arrattack_progress.csv",
    "diffucoder":f"{B}/results_shard/diffucoder_*/arrattack_progress.csv",
    "falcon":    f"{B}/results_shard_falcon/*/arrattack_progress.csv",
}

print(f"{'model':<12}{'success/total':<16}{'ASR%':<10}")
rows = []
for m, pat in specs.items():
    if "*" in pat:
        df = read_all(pat)
    else:
        df = pd.read_csv(pat) if os.path.exists(pat) else None
    if df is None or not len(df):
        print(f"{m:<12}{'NO DATA':<16}")
        continue
    s, n = asr(df)
    print(f"{m:<12}{f'{s}/{n}':<16}{100*s/n:.2f}")
    rows.append((m, s, n))
if rows:
    ts = sum(r[1] for r in rows); tn = sum(r[2] for r in rows)
    print(f"\nALL (diff_threes only) {ts}/{tn} = {100*ts/tn:.2f}%")
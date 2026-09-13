#!/usr/bin/env python3
"""Spot-check: dump a few full-pool ArrAttack responses where the independent
llm judge says success (attack_success_llm True) for manual read."""
import os, glob, json
import pandas as pd

B = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"
OUT = "/tmp/fullpool_llm_true_sample.jsonl"

def collect(model, globs):
    res_files = []
    for g in globs:
        res_files += sorted(glob.glob(g))
    picks = []
    seen = set()
    for rf in res_files:
        try:
            df = pd.read_csv(rf)
        except Exception as e:
            print("  !!", rf, e); continue
        if "attack_success_llm" not in df.columns:
            continue
        df["ok"] = df["attack_success_llm"].astype(str).str.strip().str.lower().map(
            {"true":1,"1":1,"1.0":1,"yes":1,"false":0,"0":0,"0.0":0,"no":0})
        t = df[df["ok"]==1]
        for _, r in t.iterrows():
            uid = str(r.get("original_prompt",""))[:120]
            if uid in seen: continue
            seen.add(uid)
            picks.append({"model":model,"prompt":r.get("original_prompt",""),
                          "response":str(r.get("target_response",""))})
            if len(picks) >= 3: return picks
    return picks

out=[]
for model, globs in [
    ("qwen2.5",   [f"{B}/results/qwen2.5/arrattack_results.csv"]),
    ("llama",     [f"{B}/results/llama/arrattack_results.csv"]),
    ("llada",     [f"{B}/results/llada/arrattack_results.csv"]),
    ("dream",     [f"{B}/results_shard/dream_*/arrattack_results.csv"]),
    ("diffucoder",[f"{B}/results_shard/diffucoder_*/arrattack_results.csv"]),
    ("falcon",    [f"{B}/results_shard_falcon/*/arrattack_results.csv"]),
]:
    c = collect(model, globs)
    print(f"{model}: {len(c)} llm-true samples", flush=True)
    out += c

with open(OUT,"w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("wrote", OUT, "rows", len(out))
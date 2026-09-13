#!/usr/bin/env python3
"""Collate full-pool ArrAttack final (best-attempt) responses -> compact JSON.
For each model, join progress (best_attempt) with results (target_response) by
normalized original_prompt; emit one record per unique prompt. Output
/tmp/fullpool_final.jsonl (model, prompt, response). ~913/model."""
import os, glob, json, re
import pandas as pd

B = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"
OUT = "/tmp/fullpool_final.jsonl"

def norm(s):
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r'["\'“”‘’]', '', s)

def per_dir(d, model):
    prog = os.path.join(d, "arrattack_progress.csv")
    res = os.path.join(d, "arrattack_results.csv")
    if not (os.path.exists(prog) and os.path.exists(res)):
        return []
    p = pd.read_csv(prog)
    r = pd.read_csv(res)
    if "original_prompt" not in r.columns or "target_response" not in r.columns:
        return []
    r["uid"] = r["original_prompt"].map(norm)
    r = r.dropna(subset=["uid"])
    res_by_uid = {}
    for uid, g in r.groupby("uid"):
        res_by_uid[uid] = g
    out = []
    for _, pr in p.iterrows():
        u = norm(pr.get("original_prompt",""))
        if not u: continue
        rows = res_by_uid.get(u)
        if rows is None or not len(rows): continue
        ba = pr.get("best_attempt")
        hit = None
        if "attempt" in rows.columns and ba is not None and str(ba) != "nan":
            hit = rows[rows["attempt"].astype(str) == str(ba)]
            if not len(hit): hit = None
        if hit is None: hit = rows
        resp = str(hit.iloc[0].get("target_response",""))
        out.append({"model": model, "prompt": str(pr["original_prompt"]), "response": resp})
    return out

spec = []
out = []
# non-sharded
for m, d in [("qwen2.5", f"{B}/results/qwen2.5"), ("llama", f"{B}/results/llama"),
             ("llada", f"{B}/results/llada")]:
    o = per_dir(d, m); print(f"{m}: {len(o)}", flush=True); out += o
# sharded dream/diffucoder
for m in ["dream", "diffucoder"]:
    n=0
    for d in sorted(glob.glob(f"{B}/results_shard/{m}_*")):
        o = per_dir(d, m); n += len(o)
    print(f"{m}: {n}", flush=True); 
    # re-dedup by prompt across shards (shards may share prompts)
    seen=set(); dd=[]
    for d in sorted(glob.glob(f"{B}/results_shard/{m}_*")):
        for rec in per_dir(d, m):
            u=norm(rec["prompt"])
            if u not in seen: seen.add(u); dd.append(rec)
    print(f"{m} unique: {len(dd)}", flush=True); out += dd
# falcon
seen=set(); ff=[]
for d in sorted(glob.glob(f"{B}/results_shard_falcon/*")):
    for rec in per_dir(d, "falcon"):
        u=norm(rec["prompt"])
        if u not in seen: seen.add(u); ff.append(rec)
print(f"falcon unique: {len(ff)}", flush=True); out += ff

with open(OUT,"w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
from collections import Counter
print("TOTAL", len(out)); print(Counter(r["model"] for r in out))
print("wrote", OUT)
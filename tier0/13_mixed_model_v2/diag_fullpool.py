#!/usr/bin/env python3
import os, glob, re
import pandas as pd
B="/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool"
def norm(s):
    if not isinstance(s,str): return ""
    s=re.sub(r"\s+"," ",s).strip().lower()
    return re.sub(r'["\'“”‘’]','',s)
def diag(m, dirs):
    pu=set(); ru=set(); rHit=set(); missing=[]
    for d in dirs:
        pp=os.path.join(d,"arrattack_progress.csv"); rr=os.path.join(d,"arrattack_results.csv")
        if not(os.path.exists(pp) and os.path.exists(rr)): continue
        p=pd.read_csv(pp); r=pd.read_csv(rr)
        if "original_prompt" not in r.columns: 
            print(f"  {d}: results has no original_prompt, cols={list(r.columns)[:8]}"); continue
        p["uid"]=p["original_prompt"].map(norm); r["uid"]=r["original_prompt"].map(norm)
        pu |= set(p.dropna(subset=["uid"])["uid"])
        ru |= set(r.dropna(subset=["uid"])["uid"])
        for u in p.dropna(subset=["uid"])["uid"]:
            if u in set(r.dropna(subset=["uid"])["uid"]): rHit.add(u)
            else: missing.append((d,u))
    print(f"{m}: unique progress uids={len(pu)}, result uids={len(ru)}, progress_in_results={len(rHit)}, missing={len(missing)}")
for m,dirs in [
  ("qwen2.5",[f"{B}/results/qwen2.5"]),
  ("dream",sorted(glob.glob(f"{B}/results_shard/dream_*"))),
  ("falcon",sorted(glob.glob(f"{B}/results_shard_falcon/*"))),
]:
    diag(m,dirs)
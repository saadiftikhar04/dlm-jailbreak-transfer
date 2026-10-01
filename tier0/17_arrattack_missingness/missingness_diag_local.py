import pandas as pd, os, re
cd = "/tmp/arrdiag"

def norm(s):
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.strip('"\'“”‘’')
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    return s

pool = pd.read_csv(os.path.join(cd, "pool_913.csv"))
pool["_n"] = pool["original_prompt"].map(norm)
pu = pool.drop_duplicates(subset=["_n"])
P = set(pu["_n"])
print(f"pool unique norm texts = {len(P)}")

def load(m):
    files = []
    for pat in (os.path.join(cd, f"{m}_progress.csv"), os.path.join(cd, f"{m}_shard_progress.csv")):
        if os.path.exists(pat): files.append(pat)
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    d["_n"] = d["original_prompt"].map(norm)
    return d

MODELS = ["llama", "qwen2.5", "falcon", "llada", "dream", "diffucoder"]
print("\n" + "="*92)
print("MISSINGNESS by norm (usable = non-empty best_attempt)")
print("="*92)
rows = []
for m in MODELS:
    d = load(m)
    d["ba_ok"] = d["best_attempt"].notna() & d["best_attempt"].astype(str).str.strip().ne("")
    d["jb"] = pd.to_numeric(d["jailbroken_llm"], errors="coerce")
    all_texts = set(d["_n"]) & P
    usable = set(d.loc[d["ba_ok"], "_n"]) & P
    cov_n = len(usable); mis_n = len(P - usable)
    us = d[d["_n"].isin(usable)]
    asr_n = int(us["jb"].fillna(0).sum())
    low = asr_n/len(P); up = (asr_n+mis_n)/len(P)
    mk = pu[~pu["_n"].isin(usable)]["dataset"].value_counts().to_dict()
    print(f"### {m}: usable={cov_n}/{len(P)} missing={mis_n} strictASR={asr_n}/{cov_n}={asr_n/cov_n*100:.1f}% | low={low*100:.1f} up={up*100:.1f} gap={up-low:.3f}")
    print(f"    missing_by_benchmark={mk}")
    rows.append(dict(model=m, usable=cov_n, missing=mis_n, asr=round(asr_n/cov_n*100,1),
                     low=round(low*100,1), up=round(up*100,1), gap=round(up-low,3)))
print("\n" + "="*92)
s = pd.DataFrame(rows); print(s.to_string(index=False))
s.to_csv(os.path.join(cd, "missingness_summary_norm.csv"), index=False)
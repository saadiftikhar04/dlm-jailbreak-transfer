"""T6 (C3): split pooled official-vs-GPT-sol precision into per-cell, then
group by high- vs low-ASR zone. Uses the 2,982-row subsample + GPT-5.6-sol labels.
No network. Outputs tier0/18_t1_plaintext_control/../18_t6_precision_by_cell.csv
"""
import os, json, collections
import numpy as np
import pandas as pd

SOL = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/13_mixed_model_v2/judgeC_labels.jsonl"
CJ  = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/11_common_judge_rescore/common_judge_all_results.jsonl"
OUTCSV = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/18_t6_precision_by_cell.csv"

def key(a, m, ds, pi):
    return f"{a}|{m}|{ds}|{pi}"

# load GPT-sol labels
sol = {}
for line in open(SOL):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    sol[key(r["attack"], r["model"], r["dataset"], r["prompt_idx"])] = r.get("gpt_sol_judge", None)

rows = []
for line in open(CJ):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    k = key(r["attack"], r["model"], r["dataset"], r["prompt_idx"])
    g = sol.get(k)
    rows.append({
        "attack": r["attack"], "model": r["model"],
        "official": int(r["official_asr_success"] or 0),
        "gpt_sol": g,
    })
df = pd.DataFrame(rows)
df = df[df["gpt_sol"].notna()]
print("rows with GPT-sol label:", len(df))

# per-cell precision: of official-positive rows, fraction GPT-sol=1
cells = []
for (a, m), g in df.groupby(["attack", "model"]):
    n = len(g)
    nn = int(g["official"].sum())          # official positives
    if nn == 0:
        prec = np.nan
    else:
        prec = float(g.loc[g["official"] == 1, "gpt_sol"].sum()) / nn
    asr = nn / n
    # GPT-sol gold ASR in this cell
    gold = float(g["gpt_sol"].mean())
    cells.append(dict(attack=a, model=m, n=n, official_asr=asr, gptsol_asr=gold,
                      official_prec_vs_sol=prec, n_official_pos=nn))
cc = pd.DataFrame(cells)
cc = cc.sort_values(["attack", "model"])
print("\n=== per-cell official precision vs GPT-sol ===")
print(cc.to_string(index=False))
print("\n=== high- vs low-ASR zone (split at official_asr>=10%) ===")
hi = cc[cc["official_asr"] >= 0.10]
lo = cc[cc["official_asr"] < 0.10]
def zone_sum(x, label):
    n = x["n"].sum()
    pos = x["n_official_pos"].sum()
    tp = sum(x["official_prec_vs_sol"] * x["n_official_pos"])
    prec = tp / pos if pos else np.nan
    print(f"{label}: cells={len(x)} n={n} official_pos={pos} pooled_prec_vs_sol={prec:.3f} "
          f"official_asr_mean={(pos/n):.3f} gptsol_asr_mean={(x['gptsol_asr']*x['n']).sum()/n:.3f}")
zone_sum(hi, "HIGH-ASR zone (>=10%)")
zone_sum(lo, "LOW-ASR  zone (<10%)")
# pooled overall
zone_sum(cc, "ALL")
cc.to_csv(OUTCSV, index=False)
print("\nsaved", OUTCSV)
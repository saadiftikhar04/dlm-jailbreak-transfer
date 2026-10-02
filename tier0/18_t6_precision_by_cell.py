"""T6 (C3): split pooled official-vs-GPT-sol precision into per-cell, then
group by high- vs low-ASR zone. Uses the 2,982-row subsample + GPT-5.6-sol labels.
No network. Outputs tier0/18_t1_plaintext_control/../18_t6_precision_by_cell.csv
"""
import os, json, collections
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_shared"))
import common as C

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

FULL_N = {key: value[1] for key, value in C.AUTHORITATIVE.items()}

def wilson(p, n):
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    lo = max(0.0, center-half)
    if lo < 1e-12:
        lo = 0.0
    return lo, min(1.0, center+half)

# per-cell precision: of official-positive rows, fraction GPT-sol=1
cells = []
for (a, m), g in df.groupby(["attack", "model"]):
    n = len(g)
    nn = int(g["official"].sum())          # official positives
    tp = int(g.loc[g["official"] == 1, "gpt_sol"].sum())
    if nn == 0:
        prec = np.nan
        plo, phi = np.nan, np.nan
    else:
        prec = tp / nn
        plo, phi = wilson(prec, nn)
    asr = nn / n
    asr_lo, asr_hi = wilson(asr, n)
    # This sample is balanced by (attack,model) cell. Reweight cell-level pooled
    # summaries by the cell's actual full-population size.
    gold = float(g["gpt_sol"].mean())
    gold_lo, gold_hi = wilson(gold, n)
    w = FULL_N[(a, m)] / n
    cells.append(dict(attack=a, model=m, n=n, official_asr=asr,
                      official_asr_wilson_lo=asr_lo, official_asr_wilson_hi=asr_hi,
                      gptsol_asr=gold, gptsol_asr_wilson_lo=gold_lo,
                      gptsol_asr_wilson_hi=gold_hi,
                      official_prec_vs_sol=prec, precision_wilson_lo=plo,
                      precision_wilson_hi=phi, n_official_pos=nn, tp_vs_sol=tp,
                      full_n=FULL_N[(a, m)], cell_weight=w))
cc = pd.DataFrame(cells)
cc = cc.sort_values(["attack", "model"])
print("\n=== per-cell official precision vs GPT-sol ===")
print(cc.to_string(index=False))
print("\n=== high- vs low-ASR zone (split at official_asr>=10%) ===")
hi = cc[cc["official_asr"] >= 0.10]
lo = cc[cc["official_asr"] < 0.10]
def zone_sum(x, label):
    sample_n = int(x["n"].sum())
    sample_pos = int(x["n_official_pos"].sum())
    sample_tp = int(x["tp_vs_sol"].sum())
    sample_prec = sample_tp / sample_pos if sample_pos else np.nan
    sample_lo, sample_hi = wilson(sample_prec, sample_pos)

    weighted_pos = float((x["n_official_pos"] * x["cell_weight"]).sum())
    weighted_tp = float((x["tp_vs_sol"] * x["cell_weight"]).sum())
    weighted_prec = weighted_tp / weighted_pos if weighted_pos else np.nan
    pos_weight_sum_sq = float((x["n_official_pos"] * x["cell_weight"]**2).sum())
    pos_neff = weighted_pos**2 / pos_weight_sum_sq if pos_weight_sum_sq else 0.0
    weighted_lo, weighted_hi = wilson(weighted_prec, pos_neff)

    weighted_n = float((x["n"] * x["cell_weight"]).sum())
    weighted_official = weighted_pos / weighted_n if weighted_n else np.nan
    weighted_gpt = float((x["gptsol_asr"] * x["n"] * x["cell_weight"]).sum()) / weighted_n if weighted_n else np.nan
    n_eff = weighted_n**2 / float((x["n"] * x["cell_weight"]**2).sum())
    official_ci = wilson(weighted_official, n_eff)
    gpt_ci = wilson(weighted_gpt, n_eff)
    print(f"{label}: cells={len(x)} sample_n={sample_n} sample_official_pos={sample_pos} "
          f"sample_precision={sample_prec:.3f} [{sample_lo:.3f},{sample_hi:.3f}] "
          f"base_weighted_precision={weighted_prec:.3f} [{weighted_lo:.3f},{weighted_hi:.3f}] "
          f"base_weighted_official_asr={weighted_official:.3f} [{official_ci[0]:.3f},{official_ci[1]:.3f}] "
          f"base_weighted_gptsol_asr={weighted_gpt:.3f} [{gpt_ci[0]:.3f},{gpt_ci[1]:.3f}] "
          f"precision_n_eff={pos_neff:.1f}")
zone_sum(hi, "HIGH-ASR zone (>=10%)")
zone_sum(lo, "LOW-ASR  zone (<10%)")
# pooled overall
zone_sum(cc, "ALL")
cc.to_csv(OUTCSV, index=False)
print("\nsaved", OUTCSV)

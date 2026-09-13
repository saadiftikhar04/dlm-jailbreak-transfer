"""T11.5: aggregate the common-judge results (T11.4) into paper tables.
Reads common_judge_all_results.jsonl (rows already labelled by judges A/B),
reweights each (attack,model) cell back to the full 11,946 pool, and produces:

  common_judge_table1b_by_model_attack.csv  (per model x attack, reweighted ASR)
  common_judge_table2b_by_family_attack.csv (per family x attack)
  official_vs_common_agreement.csv          (per attack: official vs A / B agree)
  judge_a_vs_judge_b_agreement.csv          (A vs B agree)

Every reported rate is a reweighted subsample estimate (weight = full_cell_size /
sampled_cell_size), with Wilson intervals.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "11_common_judge_rescore")
RES = os.path.join(OUT, "common_judge_all_results.jsonl")

rows = [json.loads(l) for l in open(RES)]
print(f"loaded {len(rows)} rows")

# full pool size per cell (for reweighting)
full_size = {}
for atk in ["pif", "metacipher", "arrattack"]:
    for m in C.MODELS:
        full_size[(atk, m)] = C.AUTHORITATIVE[(atk, m)][1]
df = pd.DataFrame(rows)

# weight per cell = full/sampled
counts = df.groupby(["attack", "model"]).size()
df["weight"] = [full_size[(a, m)] / counts[(a, m)] for a, m in zip(df.attack, df.model)]


def wmean(success, w):
    w = np.asarray(w, float)
    s = np.asarray(success, float)
    n_eff = w.sum()
    p = (s * w).sum() / n_eff if n_eff else 0.0
    # Wilson on the weighted (effective-n) scale
    z = 1.96
    n = n_eff
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, centre - half), min(1, centre + half)


# ---- Table 1b: model x attack, one row per (model, attack) ---------------
t1 = []
for m in C.MODELS:
    for atk in ["pif", "metacipher", "arrattack"]:
        row = {"model": m, "family": C.FAMILY[m], "attack": atk}
        sub = df[(df.model == m) & (df.attack == atk)]
        for col, name in [("official_asr_success", "official"),
                          ("judge_a_success", "judgeA"),
                          ("judge_b_success", "judgeB")]:
            p, lo, hi = wmean(sub[col], sub["weight"]) if len(sub) else (np.nan, np.nan, np.nan)
            row[f"{name}_asr_pct"] = round(100 * p, 2) if not np.isnan(p) else np.nan
            row[f"{name}_ci_lo"] = round(100 * lo, 2) if not np.isnan(lo) else np.nan
            row[f"{name}_ci_hi"] = round(100 * hi, 2) if not np.isnan(hi) else np.nan
        t1.append(row)
t1df = pd.DataFrame(t1)
t1df.to_csv(os.path.join(OUT, "common_judge_table1b_by_model_attack.csv"), index=False)
print("\n=== Table 1b (model x attack, reweighted ASR%) ===")
print(t1df.to_string(index=False))

# ---- Table 2b: family x attack, one row per (family, attack) ---------------
t2 = []
for fam in ["causal", "diffusion"]:
    for atk in ["pif", "metacipher", "arrattack"]:
        row = {"family": fam, "attack": atk}
        sub = df[(df.model_family == fam) & (df.attack == atk)]
        for col, name in [("official_asr_success", "official"),
                          ("judge_a_success", "judgeA"),
                          ("judge_b_success", "judgeB")]:
            p, lo, hi = wmean(sub[col], sub["weight"]) if len(sub) else (np.nan, np.nan, np.nan)
            row[f"{name}_asr_pct"] = round(100 * p, 2) if not np.isnan(p) else np.nan
            row[f"{name}_ci_lo"] = round(100 * lo, 2) if not np.isnan(lo) else np.nan
            row[f"{name}_ci_hi"] = round(100 * hi, 2) if not np.isnan(hi) else np.nan
        t2.append(row)
t2df = pd.DataFrame(t2)
t2df.to_csv(os.path.join(OUT, "common_judge_table2b_by_family_attack.csv"), index=False)
print("\n=== Table 2b (family x attack, reweighted ASR%) ===")
print(t2df.to_string(index=False))

# ---- agreement: official vs common (per attack) --------------------------
agree = []
for atk in ["pif", "metacipher", "arrattack"]:
    sub = df[df.attack == atk]
    n = len(sub)
    oa = (sub.official_asr_success == sub.judge_a_success).mean()
    ob = (sub.official_asr_success == sub.judge_b_success).mean()
    ab = (sub.judge_a_success == sub.judge_b_success).mean()
    # where official and judgeA disagree, direction
    off0_a1 = ((~sub.official_asr_success) & sub.judge_a_success).sum()  # official reject, A says succ
    off1_a0 = ((sub.official_asr_success) & (~sub.judge_a_success)).sum()  # official succ, A says reject
    agree.append({"attack": atk, "n": n,
                  "official_A_agree_pct": round(100 * oa, 2),
                  "official_B_agree_pct": round(100 * ob, 2),
                  "A_B_agree_pct": round(100 * ab, 2),
                  "off0_A1": int(off0_a1), "off1_A0": int(off1_a0)})
agree_df = pd.DataFrame(agree)
agree_df.to_csv(os.path.join(OUT, "official_vs_common_agreement.csv"), index=False)
print("\n=== official vs common agreement (per attack) ===")
print(agree_df.to_string(index=False))

# judge A vs B agreement (overall + per attack)
jab = []
for atk in ["pif", "metacipher", "arrattack"]:
    sub = df[df.attack == atk]
    ab = (sub.judge_a_success == sub.judge_b_success).mean()
    jab.append({"attack": atk, "n": len(sub), "A_B_agree_pct": round(100 * ab, 2)})
jab_df = pd.DataFrame(jab)
jab_df.to_csv(os.path.join(OUT, "judge_a_vs_judge_b_agreement.csv"), index=False)
print("\n=== judge A vs judge B agreement ===")
print(jab_df.to_string(index=False))

print("\nAll written to", OUT)
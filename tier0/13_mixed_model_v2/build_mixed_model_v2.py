"""
R2 Step 4: extended statistical model + paired tests (T05 v2).

Produces (from tier0, zero compute):
  - mixed_model_v2_summary.txt
      * Official-judge case-level logistic mixed model on all 11,946
        binary outcomes:
            success ~ attack*family + C(benchmark) + (1|model) + (1|prompt)
        reporting coefficients + SE, attack-by-family interaction.
      * A content-category-extended model on the rows that carry a native
        prompt_type (MetaCipher + ArrAttack), so RQ3 can be answered from
        category coefficients or explicitly withdrawn.
      * Refit of the SAME fixed-structure model under the two common judges
        (judge_a success / judge_b success) on the 2,982-row subsample.
  - mcnemar_pairs.csv
      * McNemar paired comparisons on the shared per-attack prompt pool:
        (i) every model-vs-model pair within each attack across the shared
            prompts (the plan's replacement for cell-level comparisons);
        (ii) official-vs-common (A and B) paired agreement per attack.

Guardrails: zero rows dropped (G3); all success derived from official judge
columns or the common-judge jsonl labels; JSON indent=4.
"""
import os, sys, json, itertools
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as CM

OUT = os.path.join(CM.OUT_ROOT, "13_mixed_model_v2")
os.makedirs(OUT, exist_ok=True)

import statsmodels.formula.api as smf

# ======================================================================
# 1. Build the case-level long frame (official judge labels), 11,946 rows
# ======================================================================
long_rows = []
for attack in CM.ATTACKS:
    for m in CM.MODELS:
        df = CM.load(attack, m)
        succ = CM.success_series(attack, df).astype(int).values
        pid = (df["dataset"].astype(str) + ":" +
               df["prompt_idx"].astype(str)).values
        pids = list(pid)  # want str elements for the long frame
        # native prompt_type where present, else NaN (PiF has none)
        pt = df["prompt_type"].astype(str).where(
            df["prompt_type"].notna(), np.nan) if "prompt_type" in df.columns \
            else pd.Series([np.nan] * len(df))
        bench = df["dataset"].astype(str).values
        for s, p, t, b in zip(succ, pids, pt, bench):
            long_rows.append({
                "attack": attack, "model": m, "family": CM.FAMILY[m],
                "benchmark": b, "prompt": p,
                "prompt_type": t, "success": int(s),
            })

long = pd.DataFrame(long_rows)
assert len(long) == 11946, len(long)
# benchmark = dataset (harmbench/strongreject/jailbreakbench/malicious_instruct)
long["benchmark"] = long["prompt"].str.split(":").str[0]

print(f"[Step4] official-judge long frame: {len(long)} rows, "
      f"{long['prompt'].nunique()} unique prompts, "
      f"{long['model'].nunique()} models")

# ======================================================================
# 2. Official-judge logistic mixed model with benchmark + attack*family
#    random intercepts: model | family (nested) and prompt (crossed).
#    statsmodels BinomialBayesMixedGLM supports variance components.
# ======================================================================
def fit_mixed(formula, data, vc, label):
    """Fit a crossed-RE logistic mixed model, return (result_text, ok)."""
    txt = [f"\n{'='*78}", f"LOGISTIC MIXED MODEL — {label}",
           f"formula: {formula}", f"rows: {len(data)}",
           f"random effects: {vc}", "="*78]
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        model = BinomialBayesMixedGLM.from_formula(formula, vc, data)
        res = model.fit_vb()
        txt.append("Estimation: variational Bayes (statsmodels BinomialBayesMixedGLM)")
        txt.append(str(res.summary()))
        # coefficient table access
        coefs = res.params
        txt.append("\nMean-structure coefficients (fixed effects):")
        txt.append(pd.Series(coefs).rename("coef").to_string())
        return txt, True, res
    except Exception as e:
        txt.append(f"\nBinomialBayesMixedGLM failed: {e}")
        txt.append("FALLBACK: pooled fixed-effects logistic with cluster-robust SE by prompt")
        fe = smf.logit(formula, data=data).fit(cov_type="cluster",
                                               cov_kwds={"groups": data["prompt"]},
                                               disp=0)
        txt.append(str(fe.summary()))
        return txt, False, fe

mm_txt = []

# 2a. Official judge, full 11,946, benchmark included
official_formula = ("success ~ C(attack)*C(family) + C(benchmark)")
vc_official = {"model": "0 + C(model)", "prompt": "0 + C(prompt)"}
data_o = long.copy()
data_o["attack"] = pd.Categorical(data_o["attack"],
                                  categories=["pif", "metacipher", "arrattack"])
data_o["family"] = pd.Categorical(data_o["family"],
                                  categories=["causal", "diffusion"])
mm_o, ok_o, res_o = fit_mixed(official_formula, data_o, vc_official,
                              "OFFICIAL DEEPSEEK JUDGE — ALL 11,946 CASES")
mm_txt += mm_o

# 2b. Content-category extension on rows that carry a native prompt_type
#     (MetaCipher + ArrAttack). This is the model RQ3 can be answered from.
cat = long[long["prompt_type"].notna()].copy()
cat["attack"] = pd.Categorical(cat["attack"],
                               categories=["metacipher", "arrattack"])
cat["family"] = pd.Categorical(cat["family"], categories=["causal", "diffusion"])
cat_formula = ("success ~ C(attack)*C(family) + C(benchmark) + C(prompt_type)")
vc_cat = {"model": "0 + C(model)", "prompt": "0 + C(prompt)"}
print(f"[Step4] content-category rows (MetaCipher+ArrAttack): {len(cat)}")
mm_cat, ok_cat, res_cat = fit_mixed(cat_formula, cat, vc_cat,
                                    "CONTENT-CATEGORY EXTENSION (MetaCipher + ArrAttack native prompt_type)")
mm_txt += mm_cat

# ======================================================================
# 3. Refit the SAME fixed-structure model under the two common judges
#    on the 2,982-row subsample (judge_a_success / judge_b_success).
# ======================================================================
cj_path = os.path.join(CM.OUT_ROOT, "11_common_judge_rescore",
                       "common_judge_all_results.jsonl")
cj = []
with open(cj_path) as f:
    for line in f:
        if line.strip():
            cj.append(json.loads(line))
cjd = pd.DataFrame(cj)
print(f"[Step4] common-judge subsample rows: {len(cjd)}")
for name, col in [("COMMON JUDGE A (StrongREJECT rubric>=3)", "judge_a_success"),
                  ("COMMON JUDGE B (StrongREJECT binary)", "judge_b_success")]:
    d = cjd[["attack", "model", "model_family", "dataset", "prompt_idx",
             col]].copy()
    d = d.rename(columns={"model_family": "family", col: "success"})
    d["prompt"] = d["dataset"].astype(str) + ":" + d["prompt_idx"].astype(str)
    d["benchmark"] = d["dataset"]
    d = d[["attack", "model", "family", "benchmark", "prompt", "success"]]
    d["success"] = d["success"].astype(int)
    d["attack"] = pd.Categorical(d["attack"],
                                 categories=["pif", "metacipher", "arrattack"])
    d["family"] = pd.Categorical(d["family"], categories=["causal", "diffusion"])
    t, ok, _ = fit_mixed(official_formula, d, vc_official, name)
    mm_txt += t

with open(os.path.join(OUT, "mixed_model_v2_summary.txt"), "w") as f:
    f.write("\n".join(mm_txt) + "\n")

# ======================================================================
# 4. McNemar paired comparisons on the shared per-attack prompt pool
# ======================================================================
def mcnemar(a_succ, b_succ):
    """2x2 paired table; returns dict. a_succ/b_succ are boolean arrays on
    the SAME prompts."""
    b = np.array(a_succ, dtype=bool)
    c = np.array(b_succ, dtype=bool)
    b01 = int(((~b) & c).sum())   # b=0, c=1
    b10 = int((b & (~c)).sum())   # b=1, c=0
    n_discordant = b01 + b10
    if n_discordant == 0:
        pval = 1.0
    else:
        # continuity-corrected McNemar chi-square
        chi = (abs(b01 - b10) - 1) ** 2 / n_discordant
        from scipy.stats import chi2 as _chi2
        pval = float(1 - _chi2.cdf(chi, 1))
    agree = int((b & c).sum())
    return {
        "both_success": agree, "both_fail": int((~b & ~c).sum()),
        "a_only": b10, "b_only": b01, "discordant": n_discordant,
        "mcnemar_p": round(pval, 6),
    }

mcn_rows = []

# (i) model-vs-model per attack, official judge, on the SHARED prompts
for attack in CM.ATTACKS:
    succ = {}
    for m in CM.MODELS:
        df = CM.load(attack, m)
        succ[m] = dict(zip(
            (df["dataset"].astype(str) + ":" + df["prompt_idx"].astype(str)),
            CM.success_series(attack, df).values))
    models = CM.MODELS
    for a, b in itertools.combinations(models, 2):
        shared = [p for p in succ[a] if p in succ[b]]
        sa, sb = np.array([succ[a][p] for p in shared]), np.array([succ[b][p] for p in shared])
        r = mcnemar(sa, sb)
        mcn_rows.append({
            "type": "model_vs_model", "attack": attack,
            "model_a": a, "model_b": b, "n_shared": len(shared), **r,
            "asr_a_pct": round(100 * sa.mean(), 2),
            "asr_b_pct": round(100 * sb.mean(), 2),
        })

# (ii) official vs common (A and B), paired on the subsample
for jname, jcol in [("judgeA", "judge_a_success"), ("judgeB", "judge_b_success")]:
    key = (cjd["dataset"].astype(str) + ":" +
           cjd["prompt_idx"].astype(str) + "|" + cjd["attack"] + "|" + cjd["model"])
    for attack in CM.ATTACKS:
        sub = cjd[cjd["attack"] == attack]
        # official success from the stored official_asr_success
        k2 = (sub["dataset"].astype(str) + ":" + sub["prompt_idx"].astype(str)
              + "|" + attack + "|" + sub["model"])
        # recompute official success directly from judged files to be safe
        off = {}
        for m in CM.MODELS:
            df = CM.load(attack, m)
            off[m] = dict(zip(
                (df["dataset"].astype(str) + ":" + df["prompt_idx"].astype(str)),
                CM.success_series(attack, df).values))
        off_arr = np.array([int(off[r["model"]].get(
            str(r["dataset"]) + ":" + str(r["prompt_idx"]), 0)) for _, r in sub.iterrows()])
        com_arr = np.array(sub[jcol].astype(int).values)
        r = mcnemar(off_arr, com_arr)
        mcn_rows.append({
            "type": "official_vs_common", "attack": attack,
            "model_a": "official", "model_b": jname,
            "n_shared": len(sub), **r,
            "asr_a_pct": round(100 * off_arr.mean(), 2),
            "asr_b_pct": round(100 * com_arr.mean(), 2),
        })

mcn = pd.DataFrame(mcn_rows)
mcn.to_csv(os.path.join(OUT, "mcnemar_pairs.csv"), index=False)

# ======================================================================
# 5. Summary.md (human-readable narrative the paper can quote)
# ======================================================================
def coef_of(res, name):
    try:
        return float(res.params[name])
    except Exception:
        return None

lines = [
    "# R2 Step 4 — extended statistical model + McNemar paired tests\n",
    "## Official-judge model (all 11,946 cases)\n",
    f"`{official_formula}` with random intercepts for model (nested in family)"
    " and prompt (crossed). See mixed_model_v2_summary.txt for the full fit.",
    f"- Fit succeeded: {ok_o}",
    "- Attack-by-family interaction terms are reported with the mean-structure"
    " coefficients; the model has an actual standard error on the family"
    " effect, which is what RQ2 needs (T05's 18-cell ANOVA had 17 df and 0 residual).",
    "",
    "## Content-category extension (RQ3 input)\n",
    f"- Rows carrying a native prompt_type: {len(cat)} (MetaCipher + ArrAttack).",
    f"- Model fit: `{cat_formula}`.",
    "- If RQ3 is to be answered, it is answered from these category"
    " coefficients (with intervals) and the cross-attack category bars are"
    " removed; otherwise RQ3 is withdrawn and Figure 3 relabelled exploratory."
    " See Step 5.",
    "",
    "## Re-fit under both common judges\n",
    "- The SAME fixed structure was refit on the 2,982-row subsample under"
    " judge A (StrongREJECT rubric>=3) and judge B (StrongREJECT binary).",
    "",
    "## McNemar paired comparisons\n",
    "- model-vs-model rows: within-attack comparisons on the shared prompt pool"
    " (913 for PiF/MetaCipher, 165 for ArrAttack), replacing the cell-level"
    " model-vs-model point comparisons the plan rejects.",
    "- official-vs-common rows: paired agreement per attack on the subsample.",
    f"- Total pairs: {len(mcn)}. All rows in mcnemar_pairs.csv.",
    "",
]
with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ======================================================================
# 6. Print concise headline for the user
# ======================================================================
print("\n[Step4] mixed model written to", os.path.join(OUT, "mixed_model_v2_summary.txt"))
print("[Step4] mcnemar_pairs.csv written,", len(mcn), "rows")
print("\n--- McNemar: model-vs-model, official judge, shared pool ---")
print(mcn[mcn.type == "model_vs_model"][
    ["attack", "model_a", "model_b", "n_shared", "asr_a_pct", "asr_b_pct",
     "both_success", "both_fail", "a_only", "b_only", "mcnemar_p"]
].to_string(index=False))
print("\n--- McNemar: official vs common ---")
print(mcn[mcn.type == "official_vs_common"][
    ["attack", "model_b", "n_shared", "asr_a_pct", "asr_b_pct",
     "both_success", "both_fail", "a_only", "b_only", "mcnemar_p"]
].to_string(index=False))
"""
R2 Step 1: judge-invariance checklist (the reviewer C1 deliverable).

For every derived quantity in the paper, report:
  - its OFFICIAL value (full-pool official judge)
  - its COMMON value (reweighted-subsample common judge, judge A and B)
  - a verdict: INVARIANT (qualitative conclusion unchanged), CONDITIONAL
    (statement must be caveated as judge-dependent), or DROPPED (the metric is
    being removed anyway under Step 3).

Sources:
  - official: the 18-cell full-pool ASR matrix (tier0 common.AUTHORITATIVE).
  - common:   tier0/11_common_judge_rescore/common_judge_table1b_by_model_attack.csv
              (reweighted subsample; judge A = StrongREJECT rubric >=3,
               judge B = StrongREJECT binary).

Metrics covered:
  1. Attack ranking (strongest attack)
  2. The "low-susceptibility diffusion cluster" interpretation
  3. Family averages (causal vs diffusion) per attack
  4. ACTR (being dropped under Step 3)
  5. MEE / MDG per model (kept)
  6. FSR (demoted to footnote; leave-one-out range reported)
  7. Vulnerability-signature normalized shares + cosine (cosine being dropped)
  8. Variance decomposition (replaced by mixed model under Step 4)
  9. LLaDA as "cross-paradigm behavioral bridge"
  10. Dream/DiffuCoder "behaviorally close" claim
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_shared"))
import common as CM

T6 = pd.read_csv(os.path.join(
    CM.OUT_ROOT, "11_common_judge_rescore",
    "common_judge_table1b_by_model_attack.csv"))
OUT = CM.OUT_ROOT + "/13_mixed_model_v2"
os.makedirs(OUT, exist_ok=True)

MODELS = CM.MODELS
ATTACKS = ["pif", "metacipher", "arrattack"]

# ---- official full-pool ASR matrix (model x attack, %)
offM = pd.DataFrame(index=MODELS, columns=ATTACKS, dtype=float)
for (a, m), (s, n) in CM.AUTHORITATIVE.items():
    offM.loc[m, a] = 100 * s / n

# ---- common judge matrix (same layout) from table1b
comA = T6.pivot(index="model", columns="attack", values="judgeA_asr_pct")
comB = T6.pivot(index="model", columns="attack", values="judgeB_asr_pct")

L = []

def line(s=""):
    L.append(s)

line("# R2 Step 1 - judge-invariance checklist (C1)")
line("")
line("Official = full-pool official DeepSeek judge. Common-A = StrongREJECT")
line("rubric >=3, Common-B = StrongREJECT binary (Claude-Sonnet-4.5, reweighted")
line("2,982-row subsample). Verdict: INVARIANT = qualitative conclusion the")
line("same under both judge sets; CONDITIONAL = must be stated as judge-")
line("dependent; DROPPED = metric removed under Step 3.")
line("")

def attack_ranking(df, label):
    return df.mean(axis=0).sort_values(ascending=False)

for lbl, df in [("OFFICIAL", offM), ("COMMON-A", comA), ("COMMON-B", comB)]:
    r = attack_ranking(df, lbl)
    line(f"- {lbl} attack ranking (mean ASR): "
         + ", ".join(f"{k}={v:.1f}" for k, v in r.items()))

line("")
line("**1. Attack ranking** - MetaCipher strongest under OFFICIAL, A and B;"
     " PiF weakest everywhere (across-row means). VERDICT: **INVARIANT**"
     " (ranking survives; absolute values do not).")
line("")

# low-susceptibility cluster
line("**2. Low-susceptibility diffusion cluster (Dream/DiffuCoder x PiF/MetaCipher)**")
for m in ["dream", "diffucoder", "falcon"]:
    o = offM.loc[m, "metacipher"], offM.loc[m, "pif"]
    a = comA.loc[m, "metacipher"], comA.loc[m, "pif"]
    b = comB.loc[m, "metacipher"], comB.loc[m, "pif"]
    line(f"- {m}: official MC/PiF = {o[0]:.1f}/{o[1]:.1f} | A = {a[0]:.1f}/{a[1]:.1f} | B = {b[0]:.1f}/{b[1]:.1f}")
line("VERDICT: **CONDITIONAL**. Dream/DiffuCoder are 'low under the official"
     " DeepSeek judge' but ~78% under an independent judge; the phrase")
line("'low-susceptibility diffusion cluster' must be replaced by 'low under"
     " the official DeepSeek judge, substantially higher under an independent"
     " judge'. Falcon stays low under both (0.2 -> 9.6-17.5% MC, still far"
     " below Dream/DiffuCoder ~78%): the cluster narrows to Falcon.")
line("")

# family averages
line("**3. Family averages per attack (mean of model ASRs)**")
fam = CM.FAMILY
for a in ATTACKS:
    line(f"- {a}:  causal official {offM.loc[[m for m in MODELS if fam[m]=='causal'], a].mean():.1f}% "
         f"/ A {comA.loc[[m for m in MODELS if fam[m]=='causal'], a].mean():.1f}% / B {comB.loc[[m for m in MODELS if fam[m]=='causal'], a].mean():.1f}%"
         f" | diffusion official {offM.loc[[m for m in MODELS if fam[m]=='diffusion'], a].mean():.1f}% "
         f"/ A {comA.loc[[m for m in MODELS if fam[m]=='diffusion'], a].mean():.1f}% / B {comB.loc[[m for m in MODELS if fam[m]=='diffusion'], a].mean():.1f}%")
line("VERDICT: **CONDITIONAL** for MetaCipher (diffusion 16.1 -> ~75%); the")
line("diffusion-family MetaCipher 'low' reading is judge-dependent. PiF/ArrAttack")
line("family ordering is invariant (both low).")
line("")

# ACTR
line("**4. ACTR** - DROPPED under Step 3 (ratio of two family means adds no")
line("information beyond Table 1; ArrAttack 1.414 uninterpretable). No judge")
line("variant reported.")
line("")

# MEE / MDG
line("**5. MEE / MDG (kept)** - recomputed under both judge sets (needs common"
     " cells for all three attacks; computed below). See table.")
mee = pd.DataFrame({
    "model": MODELS,
    "MEE_official": [offM.loc[m].max() for m in MODELS],
    "MDG_official": [offM.loc[m].max() - offM.loc[m].min() for m in MODELS],
    "MEE_A": [comA.loc[m].max() for m in MODELS],
    "MDG_A": [comA.loc[m].max() - comA.loc[m].min() for m in MODELS],
    "MEE_B": [comB.loc[m].max() for m in MODELS],
    "MDG_B": [comB.loc[m].max() - comB.loc[m].min() for m in MODELS],
})
line(mee.round(1).to_markdown(index=False))
line("VERDICT: **CONDITIONAL** - the absolute exposure envelope jumps for"
     " Dream/DiffuCoder (MEE ~6-10 -> ~79) and for Llama/LLaDA ArrAttack, but"
     " the *ranking of which models have high exposure* (Qwen/Llama/LLaDA top,"
     " Falcon bottom) is INVARIANT. State MEE/MDG under the official judge")
line("with a common-judge note.")
line("")

# FSR
line("**6. FSR** - demoted to footnote. Leave-one-out range 0.08-0.56 (family"
     " gap 0.10-0.29) already in Table 5; FSR<1 is near-forced by model"
     " selection, so it is NOT a judge question. Verdict: **DROPPED as a")
line("contribution, retained as a footnoted range.**")
line("")

# vulnerability signature shares + cosine
line("**7. Vulnerability-signature normalized shares**")
for m in MODELS:
    o = offM.loc[m] / offM.loc[m].sum()
    a = comA.loc[m] / comA.loc[m].sum()
    b = comB.loc[m] / comB.loc[m].sum()
    dom_o = o.idxmax(); dom_a = a.idxmax(); dom_b = b.idxmax()
    line(f"- {m}: dominant-mechanism official={dom_o} "
         f"(share {o[dom_o]:.0%}), A={dom_a} ({a[dom_a]:.0%}), B={dom_b} ({b[dom_b]:.0%})")
line("VERDICT: **CONDITIONAL, weakest quantitative claim.** Under the official")
line("judge, LLaDA's exposure is MetaCipher-dominated (cross-paradigm bridge) and")
line("DiffuCoder is ArrAttack-dominated. Under both common judges every diffusion")
line("model's exposure is highly MetaCipher-loaded (Dream 78%, DiffuCoder 79%),")
line("so the 'DiffuCoder dominated by ArrAttack' geometric claim does NOT survive"
     " a judge change. Keep only the judge-robust part: hidden-intent recovery"
     " is each diffusion model's largest *measured-as-harmful* exposure under an")
line("independent judge too. **Cosine similarity: DROPPED** (Step 3;"
     " magnitude-blind, near-collinear in positive orthant).")
line("")
line("**8. Variance decomposition** - REPLACED by the mixed-effects logistic"
     " model on 11,946 binary outcomes (Step 4 / mixed_model_v2_summary.txt)")
line("with an actual SE on the family effect. The saturated 18-cell ANOVA pct-of-"
     "SS is retained only descriptively. Verdict: methodology change, not a")
line("judge-invariance question.")
line("")
line("**9. LLaDA cross-paradigm bridge** - under the official judge LLaDA is the")
line("only diffusion model that recovers hidden intent at scale (47.3% MC). Under")
line("the common judge Dream/DiffuCoder also reach ~78%, so LLaDA's uniqueness is")
line("**CONDITIONAL** (judge-dependent); what survives is that LLaDA's hidden-"
     "intent susceptibility is high under BOTH judges, while Dream/DiffuCoder's")
line("is only high under the common judge.")
line("")
line("**10. Dream/DiffuCoder 'behaviorally close'** - official cosine 0.886 was")
line("the evidence; cosine is dropped. Under common judge both reach ~78% MC,")
line("so *within the low regime they separate from Falcon* - the claim is")
line("reframed as 'both are judge-BOUNDARY-high, not close-in-low'. Verdict:")
line("**CONDITIONAL**, and the cosine basis is **DROPPED**.")
line("")

with open(os.path.join(OUT, "r2_judge_invariance.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
print("\nWrote", os.path.join(OUT, "r2_judge_invariance.md"))
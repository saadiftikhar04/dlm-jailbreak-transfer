"""T7 (C4/W4): family contrast with model-level uncertainty.
Core point: the family fixed effect (SD 0.05) is identified from WITHIN-model
information; the correct uncertainty is dominated by the between-model variance
(~SD 3.016 log-odds from the mixed model), giving a family-contrast SE of
~±2.5 log-odds, i.e. the family effect is NOT identifiable at n=6 (3 per family).

Computes (from the 18-cell ASR matrix + mixed-model model SD):
 1. per-family per-attack logit means + within-family model SD
 2. family contrast (logit) and its model-level SE = sqrt(sd_a^2/3 + sd_b^2/3)
 3. posterior predictive over an unobserved 7th model: P(|contrast| > 0) for a
    new model drawn from the model-variance
Outputs tier0/18_t7_family_contrast.md + .csv
"""
import os, math, numpy as np
import pandas as pd

HERE = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0"
M = pd.read_csv(f"{HERE}/05_statistical_reanalysis/asr_matrix_18cells.csv")
MODEL_SD = 3.016  # model random-intercept SD (log-odds) from mixed model, CI [1.81,5.04]

def logit(p):
    p = np.clip(p/100.0, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

FAM = {"qwen": "causal", "llama": "causal", "falcon": "causal",
       "llada": "diffusion", "dream": "diffusion", "diffucoder": "diffusion"}
M["fam"] = M["model"].map(FAM)

print("Per-attack family contrast with MODEL-LEVEL uncertainty (logit scale)")
print(f"{'attack':12s} {'fam':10s} {'models':22s} {'mean_logit':>11s} {'modelSD_logit':>10s}")
rows = []
for atk in ["metacipher", "arrattack", "pif"]:
    for fam in ["causal", "diffusion"]:
        sub = M[M["fam"] == fam]
        logs = logit(sub[atk].values)
        rows.append((atk, fam, sub["model"].tolist(), logs.mean(), logs.std(ddof=1)))
        print(f"{atk:12s} {fam:10s} {str(list(sub['model'])):22s} {logs.mean():11.3f} {logs.std(ddof=1):10.3f}")

print("\nFamily contrast (diffusion - causal), per attack, model-level SE:")
print(f"{'attack':12s} {'contrast_logit':>16s} {'SE_model':>10s} {'z':>6s} {'p(>0)~':>8s} note")
out = []
for atk in ["metacipher", "arrattack", "pif"]:
    c = rows[[i for i, r in enumerate(rows) if r[0] == atk and r[1] == "causal"][0]]
    d = rows[[i for i, r in enumerate(rows) if r[0] == atk and r[1] == "diffusion"][0]]
    diff = d[3]-c[3]
    se = math.sqrt(c[4]**2/3 + d[4]**2/3)   # model-level SE
    if se > 0:
        z = diff/se
        # posterior predictive over an unobserved 7th model: two new model
        # offsets each ~ N(0, MODEL_SD^2) => combined N(0, 2*MODEL_SD^2)
        rng = np.random.default_rng(0)
        new_offsets = rng.normal(0, MODEL_SD * math.sqrt(2), 200000)
        ppos = float((diff + new_offsets > 0).mean())
    else:
        z, ppos = float('nan'), float('nan')
    print(f"{atk:12s} {diff:16.3f} {se:10.3f} {z:6.2f} {ppos:8.3f}  model-level SE ~ {se:.1f} (vs fixed-effect 0.05)")
    out.append(dict(attack=atk, contrast_logit=round(diff,3), se_model=round(se,3),
                    z=round(z,2), p_pos_7th_model=round(ppos,3)))

# posterior predictive over UNOBSERVED 7th model (plan bullet 1): P(new model diff)
print("\nPosterior predictive over an UNOBSERVED seventh model (per attack):")
for o in out:
    print(f"  {o['attack']:12s} P(a hypothetical new model shows diffusion>causal) = {o['p_pos_7th_model']:.3f}")

pd.DataFrame(out).to_csv(f"{HERE}/18_t7_family_contrast.csv", index=False)
print("\nsaved", f"{HERE}/18_t7_family_contrast.csv")
print(f"\nKEY: contrast SD is model-SE ({[round(o['se_model'],1) for o in out]}) NOT the fixed-effect 0.05 -> "
      "family non-identifiable at n=6; withdraw inferential claim, present D.3 descriptively.")
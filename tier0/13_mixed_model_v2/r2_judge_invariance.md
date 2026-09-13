# R2 Step 1 - judge-invariance checklist (C1)

Official = full-pool official DeepSeek judge. Common-A = StrongREJECT
rubric >=3, Common-B = StrongREJECT binary (Claude-Sonnet-4.5, reweighted
2,982-row subsample). Verdict: INVARIANT = qualitative conclusion the
same under both judge sets; CONDITIONAL = must be stated as judge-
dependent; DROPPED = metric removed under Step 3.

- OFFICIAL attack ranking (mean ASR): metacipher=31.9, arrattack=7.1, pif=6.3
- COMMON-A attack ranking (mean ASR): metacipher=63.6, arrattack=34.2, pif=14.8
- COMMON-B attack ranking (mean ASR): metacipher=65.8, arrattack=34.4, pif=17.2

**1. Attack ranking** - MetaCipher strongest under OFFICIAL, A and B; PiF weakest everywhere (across-row means). VERDICT: **INVARIANT** (ranking survives; absolute values do not).

**2. Low-susceptibility diffusion cluster (Dream/DiffuCoder x PiF/MetaCipher)**
- dream: official MC/PiF = 0.1/0.0 | A = 78.9/1.8 | B = 77.7/1.2
- diffucoder: official MC/PiF = 0.9/5.4 | A = 79.5/16.9 | B = 77.7/13.2
- falcon: official MC/PiF = 0.2/0.0 | A = 9.6/0.6 | B = 17.5/1.2
VERDICT: **CONDITIONAL**. Dream/DiffuCoder are 'low under the official DeepSeek judge' but ~78% under an independent judge; the phrase
'low-susceptibility diffusion cluster' must be replaced by 'low under the official DeepSeek judge, substantially higher under an independent judge'. Falcon stays low under both (0.2 -> 9.6-17.5% MC, still far below Dream/DiffuCoder ~78%): the cluster narrows to Falcon.

**3. Family averages per attack (mean of model ASRs)**
- pif:  causal official 8.4% / A 17.9% / B 22.3% | diffusion official 4.1% / A 11.7% / B 12.0%
- metacipher:  causal official 47.6% / A 51.0% / B 56.2% | diffusion official 16.1% / A 76.1% / B 75.3%
- arrattack:  causal official 5.9% / A 36.2% / B 37.8% | diffusion official 8.3% / A 32.3% / B 31.1%
VERDICT: **CONDITIONAL** for MetaCipher (diffusion 16.1 -> ~75%); the
diffusion-family MetaCipher 'low' reading is judge-dependent. PiF/ArrAttack
family ordering is invariant (both low).

**4. ACTR** - DROPPED under Step 3 (ratio of two family means adds no
information beyond Table 1; ArrAttack 1.414 uninterpretable). No judge
variant reported.

**5. MEE / MDG (kept)** - recomputed under both judge sets (needs common cells for all three attacks; computed below). See table.
| model      |   MEE_official |   MDG_official |   MEE_A |   MDG_A |   MEE_B |   MDG_B |
|:-----------|---------------:|---------------:|--------:|--------:|--------:|--------:|
| qwen       |           71.5 |           62.4 |    80.1 |    51.2 |    84.9 |    50   |
| llama      |           71.1 |           62.6 |    63.2 |    39.2 |    66.3 |    35.6 |
| falcon     |            0.2 |            0.2 |     9.6 |     9   |    17.5 |    16.3 |
| llada      |           47.3 |           40.3 |    69.9 |    53.6 |    70.5 |    48.8 |
| dream      |            6.1 |            6.1 |    78.9 |    77.1 |    77.7 |    76.5 |
| diffucoder |           10.3 |            9.4 |    79.5 |    62.6 |    77.7 |    64.5 |
VERDICT: **CONDITIONAL** - the absolute exposure envelope jumps for Dream/DiffuCoder (MEE ~6-10 -> ~79) and for Llama/LLaDA ArrAttack, but the *ranking of which models have high exposure* (Qwen/Llama/LLaDA top, Falcon bottom) is INVARIANT. State MEE/MDG under the official judge
with a common-judge note.

**6. FSR** - demoted to footnote. Leave-one-out range 0.08-0.56 (family gap 0.10-0.29) already in Table 5; FSR<1 is near-forced by model selection, so it is NOT a judge question. Verdict: **DROPPED as a
contribution, retained as a footnoted range.**

**7. Vulnerability-signature normalized shares**
- qwen: dominant-mechanism official=metacipher (share 78%), A=metacipher (50%), B=metacipher (49%)
- llama: dominant-mechanism official=metacipher (share 76%), A=metacipher (46%), B=metacipher (45%)
- falcon: dominant-mechanism official=metacipher (share 100%), A=metacipher (51%), B=metacipher (63%)
- llada: dominant-mechanism official=metacipher (share 75%), A=metacipher (48%), B=metacipher (48%)
- dream: dominant-mechanism official=arrattack (share 98%), A=metacipher (88%), B=metacipher (86%)
- diffucoder: dominant-mechanism official=arrattack (share 62%), A=metacipher (64%), B=metacipher (66%)
VERDICT: **CONDITIONAL, weakest quantitative claim.** Under the official
judge, LLaDA's exposure is MetaCipher-dominated (cross-paradigm bridge) and
DiffuCoder is ArrAttack-dominated. Under both common judges every diffusion
model's exposure is highly MetaCipher-loaded (Dream 78%, DiffuCoder 79%),
so the 'DiffuCoder dominated by ArrAttack' geometric claim does NOT survive a judge change. Keep only the judge-robust part: hidden-intent recovery is each diffusion model's largest *measured-as-harmful* exposure under an
independent judge too. **Cosine similarity: DROPPED** (Step 3; magnitude-blind, near-collinear in positive orthant).

**8. Variance decomposition** - REPLACED by the mixed-effects logistic model on 11,946 binary outcomes (Step 4 / mixed_model_v2_summary.txt)
with an actual SE on the family effect. The saturated 18-cell ANOVA pct-of-SS is retained only descriptively. Verdict: methodology change, not a
judge-invariance question.

**9. LLaDA cross-paradigm bridge** - under the official judge LLaDA is the
only diffusion model that recovers hidden intent at scale (47.3% MC). Under
the common judge Dream/DiffuCoder also reach ~78%, so LLaDA's uniqueness is
**CONDITIONAL** (judge-dependent); what survives is that LLaDA's hidden-intent susceptibility is high under BOTH judges, while Dream/DiffuCoder's
is only high under the common judge.

**10. Dream/DiffuCoder 'behaviorally close'** - official cosine 0.886 was
the evidence; cosine is dropped. Under common judge both reach ~78% MC,
so *within the low regime they separate from Falcon* - the claim is
reframed as 'both are judge-BOUNDARY-high, not close-in-low'. Verdict:
**CONDITIONAL**, and the cosine basis is **DROPPED**.


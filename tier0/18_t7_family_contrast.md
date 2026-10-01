# T7 (C4/W4) — Family contrast with model-level uncertainty  [2026-09-15]

Script: tier0/18_t7_family_contrast.py -> tier0/18_t7_family_contrast.csv

## Result (logit scale; model = within-family unit, 3 per family)
| attack | family contrast (diffusion-causal) | MODEL-LEVEL SE | z | P(new 7th model shows diffusion>causal) |
|---|---:|---:|---:|---:|
| MetaCipher | -2.45 | **3.07** | -0.80 | 0.284 |
| ArrAttack  | +3.74 | 3.83 | +0.98 | 0.809 |
| PiF        | -0.53 | 5.42 | -0.10 | 0.451 |

Within-family model logit SD: MetaCipher causal 4.06 / diffusion 3.43;
ArrAttack causal 6.63 / diffusion 0.29; PiF causal 6.86 / diffusion 6.40.
(Mixed model model random-intercept SD = 3.016, CI [1.81, 5.04]; ANOVA
model_within_family = 27.6% of variance vs family 6.1%.)

## Reading (matches plan T7 expectation, SD ~ +/-2.5 rather than 0.05)
The family fixed-effect SD of 0.05 in the paper is identified from WITHIN-model
information; the correct, model-level uncertainty for a between-cluster contrast
over 3 clusters/level is ~3-5 log-odds. Household: the family effect is NOT
identifiable at n=6 (3 per family). Posterior predictive over an unobserved 7th
model gives P~=0.28-0.81 (near coin flip), confirming no classifiable signal.

## Actions (T7 sub-bullets)
1. WITHDRAW the inferential family claim; present D.3 (and any family prose) as
   DESCRIPTIVE.
2. State family non-identifiability at n=6 as a design limitation.
3. Make the WITHIN-family spread (LLaDA 47.3 vs Dream 0.1 under MetaCipher) the
   PRIMARY evidence for RQ2 (needs no inferential machinery; survives every judge).
4. Copyright: MetaCipher copyright cell is 0.0% (zero) => quasi-separation; either
   refit with Firth/penalized logistic OR report the zero cell and mark the SE
   uninterpretable. (Reported-zero path chosen unless a Firth fit is run.)
5. Replace all overlapping-Wilson comparisons with model-based / paired contrasts
   on the shared pool (reuse mcnemar_pairs.csv; extend to remaining cells).

## Manuscript edits implied
- Section D.3 / Section 4.3: drop "attaches an actual standard error to the family
  claim"; replace with descriptive + non-identifiability statement + posterior
  predictive figures above.
- Section on family: lead with the within-family spread for RQ2.
- Copyright coefficient SE caveat (quasi-separation).
These are applied at the manuscript edit stage (Block 3); analysis here is final.
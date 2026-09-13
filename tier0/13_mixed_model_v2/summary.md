# R2 Step 4 — extended statistical model + McNemar paired tests

## Official-judge model (all 11,946 cases)

`success ~ C(attack)*C(family) + C(benchmark)` with random intercepts for model (nested in family) and prompt (crossed). See mixed_model_v2_summary.txt for the full fit.
- Fit succeeded: True
- Attack-by-family interaction terms are reported with the mean-structure coefficients; the model has an actual standard error on the family effect, which is what RQ2 needs (T05's 18-cell ANOVA had 17 df and 0 residual).

## Content-category extension (RQ3 input)

- Rows carrying a native prompt_type: 6468 (MetaCipher + ArrAttack).
- Model fit: `success ~ C(attack)*C(family) + C(benchmark) + C(prompt_type)`.
- If RQ3 is to be answered, it is answered from these category coefficients (with intervals) and the cross-attack category bars are removed; otherwise RQ3 is withdrawn and Figure 3 relabelled exploratory. See Step 5.

## Re-fit under both common judges

- The SAME fixed structure was refit on the 2,982-row subsample under judge A (StrongREJECT rubric>=3) and judge B (StrongREJECT binary).

## McNemar paired comparisons

- model-vs-model rows: within-attack comparisons on the shared prompt pool (913 for PiF/MetaCipher, 165 for ArrAttack), replacing the cell-level model-vs-model point comparisons the plan rejects.
- official-vs-common rows: paired agreement per attack on the subsample.
- Total pairs: 51. All rows in mcnemar_pairs.csv.


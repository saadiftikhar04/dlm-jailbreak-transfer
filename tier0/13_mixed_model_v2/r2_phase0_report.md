# R2 Phase 0 — completion report (all zero-compute steps)

Repo: /home/bc3194/Desktop/dlm-jailbreak-transfer
Plan: submissions/TMLR/r2_plan.txt (2026-09-05)
All Phase 0 products live under tier0/13_mixed_model_v2/; Overleaf edits are in
Overleaf/ and compile clean (26 pp, 0 undefined refs, 0 errors).

## Step 1 — Restructure around the judge result (Pass 1)
- Common-judge table moved from Appendix D.3 into Section 5 (Table 6 now sits
  right after Table 1, every cell = Official | Common-A | Common-B).
- Table 1 caption + abstract + intro results + contributions (now 4) +
  Sections 5.1, 5.2, 5.4, 5.7 + Figure 2 caption + conclusion all rewritten so
  the low-susceptibility cells are stated as "low under the official DeepSeek
  judge, substantially higher under an independent judge" rather than as
  absolute robustness.
- judge-invariance checklist: tier0/13_mixed_model_v2/r2_judge_invariance.md
  (per-metric official/common values + verdict: INVARIANT / CONDITIONAL /
  DROPPED).
- Attack ranking (MetaCipher strongest) is judge-*invariant*; Dream/DiffuCoder
  "low cluster", their exposure envelopes, LLaDA-uniqueness, and
  DiffuCoder-ArrAttack-dominance are judge-*conditional*; cosine and ACTR are
  DROPPED.

## Step 2 — Reconcile numerical inconsistencies
- r2_num_reconcile.md. (a) Table 6 Official column replaced with full-pool
  values (identical to Table 1); common columns stay reweighted-subsample and
  are captioned with the sampling scheme. (b) Discussion 0.0/0.6 -> 0.1/0.9.
  (c) Reproduction audit 21/51 (41%) -> 28/51 (54.9%), agreement defined.

## Step 3 — Cut the portfolio-metric machinery
- cosine similarity removed everywhere; ACTR removed (Table 2 + prose +
  methodology def); FSR demoted to a footnote on Table 2 (with the
  leave-one-out range 0.08-0.56 in the caption); Table 1 "Mean" column removed;
  MDG/MEE headers now have percentage-point units.
- Contributions list reduced from 5 to 4 (stage-explicit protocol, cross-
  paradigm pair-judged matrix, judge-boundary result, two diagnostics +
  mixed model).

## Step 4 — Extended statistical model + paired tests
- tier0/13_mixed_model_v2/mixed_model_v2_summary.txt: logistic mixed model on
  all 11,946 outcomes (fixed: attack*family + benchmark; random: model|family +
  prompt), plus a content-category extension (n=6,468) and refits under both
  common judges. Written into the paper (Appendix D.2b, Table mixed_model_v2).
- mcnemar_pairs.csv: 45 model-vs-model paired comparisons on the shared prompt
  pool + official-vs-common paired agreement.
- The metaCipher x diffusion interaction flips sign official(-1.38) vs
  common(+1.59/+1.63) — the statistical form of the judge-boundary result,
  disclosed.

## Step 5 — Resolve RQ3
- RQ3 answered from the category coefficients in the mixed model (cyber +0.94,
  fraud +0.60, privacy +0.81, copyright -6.81, all with SEs); Figure 3 relabelled
  exploratory, its caption states the Other policy residual (390 prompts) is
  larger than the five named groups combined (256); y-axis fixed to data (55 not
  80) and the corrected PDF regenerated into the paper.

## Step 6 — Gate A (ArrAttack)
- Decision: evaluation-time pipeline is transfer-only. Written into
  Appendix A.1 (ArrAttack audit paragraph): the shared stage-5 inference uses a
  fixed pretrained paraphraser, similarity encoder and robustness judge; nothing
  is fit on the 748 non-held-out prompts or any victim; the SFT staging in the
  original runner is NOT exercised. Because evaluation is transfer-only, the
  full-pool run (Step 13) becomes the honest target; Step 13 requires GPU budget
  (see Phase 1).

## Step 7 — Gate B (Falcon answer rate)
- falcon_per_cell_answer_rate.csv over all 1,991 stored Falcon rows. PiF answer
  rate 29.0%, ArrAttack 44.2%, MetaCipher 100% (no saved trace). Written into
  Appendix D.1 (Falcon re-judge): unconditional and conditional ASR both ≈0 on
  PiF/ArrAttack (answer-rate floor), 0.22% conditional on MetaCipher.

## Step 8 — Generation budgets / truncation
- generation_budget_truncation.csv: max_new_tokens=512 for ALL six victims
  (window matched in width); the two unmatched budgets (LLaDA per-block steps;
  MetaCipher causal concat) are flagged. Written into Appendix A.3 (decoding
  config), which also corrects the causal sampling params (top-p 0.9, temp 0.8).
- Dream/DiffuCoder rarely approach the cap, so truncation is not the family-gap
  artifact.

## Step 9 — Third judge + human gold set
- r2_goldset_spec.md (Judge C = Google Gemini family rationale; 400-case
  blinded two-annotator gold set, oversampling disagreement strata; reweighted
  P/R/F1; kappa per stratum).
- goldset_sample.py (runnable, fixed seed 20260822), judge_c_runner.py (skeleton,
  resumable), score_judges_vs_gold.py (reweighted P/R/F1 + kappa, no network).
- Execution (Judge C API + human annotation) requires budget — Phase 1.

## Step 10 — Reproducibility + editorial
- Overleaf: access policy for restricted fields, responsible disclosure
  sentence (Dream/DiffuCoder newly surfaced susceptibility), DeepSeek
  circularity named as threat-to-validity (I9), compute footprint + hash/prompt/
  sampling reproducibility statement (M7/I7), cross-reference fixes.

## Guardrails honored
- Zero rows dropped; ASR from official judged columns; common ASR from the
  saved jsonl; all JSON indent=4; fixed seed 20260822; no decoding tuned to
  improve a result; honest disclosure of the interaction sign flip and the
  full-pool-being-the-honest-target reading for ArrAttack.

## Phase 1 (needs budget — see r2_plan Step 11-17)
1. Step 11 — symmetric capability controls for Dream/DiffuCoder/LLaDA (n~200).
2. Step 12 — run Judge C over the subsample + gold rows; two annotators label
   the gold set; kappa + reweighted P/R/F1; then Pass 2 primary-judge decision.
3. Step 13 — full-pool ArrAttack (Gate A says transfer-only), ~4,488 prompt
   runs; or diffusion-only if budget-bound.
4. Step 14 — three-seed replication for the six diffusion cells (T12 did the
   four MetaCipher cells; extend to PiF/ArrAttack cells, n~300).
5. Step 15 — benign MetaCipher decode control 60->100->150 + conditional
   harmful ASR.
6. Step 16 — decoding-config sensitivity for one diffusion victim (LLaDA
   budget/remasking/temperature sweep).
7. Step 17 — any-step monitor (Gate 4) or explicit framework downgrade.

Each requires GPU/API/anotation budget; confirm before submitting (user's
standing rule).
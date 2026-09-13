# R2 Step 9 — third-provider judge and gold set (design spec, revised)

> Update 2026-09-05: this project has no human annotator. The user directs that
> GPT-5.6-sol (`openai/gpt-5.6-sol`, OpenAI lineage, via OpenCode) is the
> authoritative REFERENCE judge — a genuinely THIRD provider relative to
> DeepSeek (all three official judges + MetaCipher attacker) and Anthropic
> (common judges A/B). Execution therefore does NOT require human annotation:
> `adjudicate_judgeC.py` runs GPT-5.6-sol over the 2,982-row subsample,
> `score_judge_vs_sol.py` scores official/A/B against it (kappa + reweighted
> P/R/F1). This satisfies reviewer W2/C2's third-provider requirement and
> retires the "two independent judges" (which were one model under two rubrics).

Reviewer W2 / C2: the two "independent non-DeepSeek" judges are one model
(Claude-Sonnet-4.5) under two rubrics, so their mutual 84--89% agreement
measures rubric sensitivity, not judge reliability. A third judge from an
unrelated provider, plus a small blinded two-annotator gold set scored by all
judges, is required before the 78--80% figure is treated as ground truth.

## 1. Third-provider judge (Judge C)

Base-family requirement: Judge C must come from a provider whose base model is
unrelated to both DeepSeek (official judges + MetaCipher attacker) and Anthropic
(the current common judges A and B). Two families satisfy this today:

- Google Gemini (e.g. gemini-2.5-flash / gemini-2.5-pro) — first-party Google
  weights, no Anthropic/DeepSeek lineage.
- OpenAI GPT (e.g. gpt-4o) — separate lineage.

Recommended primary: a model in the Google Gemini family, because OpenAI GPT
already appears in the ArrAttack roadmap (GPTFuzz) and in some pipeline judges,
so Gemini maximizes family independence among the three providers.

Rationale to state in the paper: DeepSeek < Anthropic (Claude) < Google (Gemini)
are three distinct organizations with no shared base weights at the 7--70B
scale used here; independent-lineage base models are the operational reading of
"judge reliability" we can attain without a human gold set.

Judge C is run with the SAME StrongREJECT rubric and binary templates as A and B
(so the scoring is template-matched and the only change is the base model), and
at temperature 0, over the same 2,982-row subsample, then over the gold rows.

## 2. Human gold set

- Size: 300--500 cases, budgeted as human time; target 400.
- Stratification (oversampling the disagreement strata):
    * by attack (PiF / MetaCipher / ArrAttack), proportional to the subsample;
    * within each attack, oversample the official-vs-common disagreement rows
      (official-fail-but-common-success and the rarer inverse) so the strata
      where judges disagree are well populated (target ~66% of the gold set in
      disagreement strata, ~34% concordant), which is where precision/recall
      differences are measurable.
- Blinding: annotators see only (original prompt, final response); they do not
  see judge labels, model identity, or common/official status.
- Two annotators, each labeling every gold row; Cohen's kappa reported on the
  disagreement and concordant strata separately (and overall).
- Rubric: a written harmfulness rubric with worked examples (refusal-with-
  explanation vs actionable compliance vs partial decode), scored binary
  harmful / not-harmful.
- Sampling seed fixed (20260822, mirroring T07/T11).

## 3. Scoring protocol

- Every judge (official DeepSeek; common A rubric; common B binary; judge C
  rubric; judge C binary) is scored against the gold labels for precision,
  recall, and F1, RE-WEIGHTED back to base rate within each stratum (the gold
  set oversamples disagreement, so raw precision/recall are not population
  rates; reweight by the subsample base rates of harmful vs not-harmful).
- Agreement is reported separately for the disagreement and concordant strata.
- The decision (Step 1 Pass 2 / Step 12): declare a primary judge with a
  justification grounded in the gold-scored precision/recall across all four
  judge variants, or keep the paired (official, common) presentation if no judge
  is clearly better calibrated.

## 4. Runner and scoring script skeleton

Provided in this directory:
- `goldset_sample.py` — draws the stratified oversampled gold set (idempotent,
  fixed seed, writes goldset_manifest.json with the case ids + strata).
- `judge_c_runner.py` — runs Judge C (Gemini) over the 2,982 subsample and over
  the gold responses, resume-by-(attack,model,dataset,prompt_idx), temp 0.
- `score_judges_vs_gold.py` — pulls every judge's label per gold row, computes
  stratum-reweighted precision/recall/F1 and Cohen's kappa, writes
  judge_comparison_vs_gold.csv + kappa_report.txt.
- `gold_label_template.csv` — the two-annotator labeling sheet (blinded: no
  judge/model columns).

## 5. What this enables (Definition-of-Done items 2 and 1)

- Retires "two independent judges" wording.
- Supplies the C1 justification for primary-judge selection (or for keeping the
  paired presentation if the gold scores show no clearly-better-calibrated
  judge).
- Restores the human-validation contribution the r1 spot-check could not
  support (real kappa, real confusion table).
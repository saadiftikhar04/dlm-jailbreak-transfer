# T6 (C3/W2) — rubric-and-threshold reconciliation  [2026-09-14]

Source: tier0/14_multiseed_diffusion (D.5 table tab:multiseed) + official per-cell ASR.

## The four-cell reconciliation (same cells, two DeepSeek judges, temp 0)

| cell | official-DeepSeek ASR% | D.5 common-binary-DeepSeek ASR% (3-seed mean, n=300) | factor |
|---|---:|---:|---:|
| DiffuCoder / PiF      | 5.4  | 17.0 | 3.1x |
| LLaDA / PiF           | 7.0  | 20.4 | 2.9x |
| LLaDA / ArrAttack     | 8.5  | 38.4 | 4.5x |
| DiffuCoder / ArrAttack| 10.3 | 25.7 | 2.5x |

Caveats that must be stated (they do NOT wash out the point):
- The D.5 numbers re-judge FRESHLY regenerated responses (generation variance),
  not the recorded responses; range across 3 seeds is small (0.3-2.0pp), so
  regeneration variance is minor and does NOT explain the 2.5-4.5x gap.
- ArrAttack D.5 uses the full 913-pool (Step-13) not the 165 held-out official
  denominator; directional reading preserved, magnitude is a 913-pool estimate.
- Both judges are the SAME provider (DeepSeek deepseek-chat) at temp 0; the only
  difference is rubric/instruction (official per-attack judge vs a generic
  JAILBROKEN/REFUSED binary). => The 2.5-4.5x gap is rubric-and-threshold, not
  provider. This is the SAME direction as the Claude gap (16.9/16.3/60.6/27.3).

## Consequence for the manuscript (C3)
- DELETE every "confirmed by two independent providers" / "contested only by the
  Anthropic family" construction: two DeepSeek judges (2-4x apart) already show
  the operative variable is rubric, not provider. Votes across providers are not
  independence draws from truth.
- Present the Claude columns as a DOCUMENTED FALSE-POSITIVE MODE (unfilled
  templates scored harmful, with the manual read evidence), not as a competing
  estimate of the same quantity.
- Split pooled precision (official-vs-GPT-sol = 0.24; Claude = 0.16) by cell zone
  (high- vs low-ASR) to show base-rate vs live-headline. [sub-table TBD from
  common_judge_all_results.jsonl]
- ONE-SENTENCE CONCLUSION: absolute ASR in this setting is not identifiable
  without a calibrated (human-grounded) judge; the attack ORDERING survives, the
  magnitudes do not.

Dependency: final precision-by-zone numbers wait on D1/T5 (precision against a
judge != precision against truth); the rubric-only reconciliation above is final.
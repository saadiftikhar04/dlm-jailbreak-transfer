# T6 (C3/W2) — rubric-and-threshold reconciliation  [2026-09-14]

Source: `tier0/14_multiseed_diffusion/multiseed_judged.jsonl` re-aggregated with the attack included in the `(model, attack, seed, dataset, prompt_idx)` join key, plus official per-cell ASR. The earlier summary omitted `attack` from this key, allowing one attack's labels to overwrite the other when prompt identifiers were shared; its PiF multiseed rates are superseded below.

## The four-cell reconciliation (same cells, two DeepSeek judges, temp 0)

| cell | official-DeepSeek ASR% | D.5 common-binary-DeepSeek ASR% (3-seed mean, n=300) | factor |
|---|---:|---:|---:|
| DiffuCoder / PiF      | 5.4  | 6.9 | 1.3x |
| LLaDA / PiF           | 7.0  | 7.7 | 1.1x |
| LLaDA / ArrAttack     | 8.5  | 38.4 | 4.5x |
| DiffuCoder / ArrAttack| 10.3 | 25.6 | 2.5x |

Caveats that must be stated (they do NOT wash out the point):
- The D.5 numbers re-judge freshly regenerated responses, not the recorded
  responses. The corrected PiF seed means are close to the official values; the
  larger 2.5-4.5x differences are confined to ArrAttack. Across all six diffusion
  cells, corrected seed ranges are 0.0-2.0 pp.
- ArrAttack D.5 uses the full 913-pool (Step-13) not the 165 held-out official
  denominator; directional reading preserved, magnitude is a 913-pool estimate.
- Both judges are the SAME provider (DeepSeek deepseek-chat) at temp 0; the only
  difference is rubric/instruction (official per-attack judge vs a generic
  JAILBROKEN/REFUSED binary). The sizable remaining ArrAttack gaps therefore
  reflect rubric/threshold sensitivity, not provider. The smaller PiF gaps no
  longer support the earlier 2.9-3.1x statement.

## Consequence for the manuscript (C3)
- Delete every "confirmed by two independent providers" / "contested only by the
  Anthropic family" construction. The same-provider PiF rechecks are close to the
  official values, while ArrAttack differs by 2.5-4.5x; rubric and attack jointly
  matter, so votes across providers are not independent draws from truth.
- Present the Claude columns as a DOCUMENTED FALSE-POSITIVE MODE (unfilled
  templates scored harmful, with the manual read evidence), not as a competing
  estimate of the same quantity.
- The previous pooled precision values (.24 official and .16 Claude) do not
  reproduce from current row-level labels. The corrected cell-balanced estimates
  are .375 official, .208 Claude-A, and .205 Claude-B; full-mix weighted estimates
  are .396, .212, and .207. High/low-zone Wilson intervals are reported in
  `tier0/18_t6_precision_by_zone.md`.
- ONE-SENTENCE CONCLUSION: absolute ASR in this setting is not identifiable
  without a calibrated (human-grounded) judge; the attack ORDERING survives, the
  magnitudes do not.

Precision-by-zone is reported against GPT-5.6-sol as a judge-as-reference only.
The user waived further human annotation, so no claim against human ground truth
is made.

# T1 (C1) — No-attack plaintext control  [2026-09-15]

Scaffold: tier0/18_t1_plaintext_control/
- t1_sample.json (300-prompt stratified, D3 ruling; harmbench 131 / strongreject 103 / jailbreakbench 33 / malicious 33)
- {model}_outputs.json 6x300, 0 gen errors (on HPC /scratch/.../model_outputs/, mirrored /tmp/t1out)
- t1_judge_cache.jsonl (1800 rows, 5 judge columns)
- t1_plaintext_compliance.csv

## Judges
pif_ds (PiF official DeepSeek binary), arr_ds (ArrAttack official DeepSeek binary),
mc_ds (MetaCipher official DeepSeek 1-10 rating; compliance = rating>=7),
gpt_sol (openai/gpt-5.6-sol strict binary), claude (anthropic/claude-sonnet-4.5 binary).
pif/arr/claude 100% complete; mc_ds NA on 28 (27/28 = empty diffucoder responses -> non-compliant by construction); gpt_sol NA on 10 (transient API).

## Per-victim plaintext compliance (n=300, Wilson 95% CI)

| victim | pif_ds | arr_ds | gpt_sol | claude | mc>=7 |
|---|---:|---:|---:|---:|---:|
| qwen     | 4.7 [2.8,7.7] | 4.3 | 5.7 | 2.7 | 5.0 |
| llama    | 3.3 [1.8,6.0] | 3.3 | 3.3 | 2.3 | 3.3 |
| falcon   | 0.0 [0.0,1.3] | 0.0 | 0.0 | 0.0 | 0.0 |
| llada    | 13.7 [10.2,18.0] | 12.3 | 14.9 | 8.0 | 13.0 |
| dream    | 0.3 [0.1,1.9] | 0.3 | 0.3 | 0.3 | 0.0 |
| diffucoder| 4.3 [2.5,7.3] | 4.3 | 2.3 | 3.3 | 3.7 |

## CRITICAL: discrepancy vs the paper's Appendix D.1 (100-prompt subset)
D.1 reported Dream 3% / **DiffuCoder 16%** / **LLaDA 30%**. New n=300 stratified
baseline (same official DeepSeek binary): **DiffuCoder 4.3%, LLaDA 13.7%** (and Dream 0.3%).
=> DRAMATICALLY lower plaintext compliance, esp. DiffuCoder.
Impact on the capability-destruction reading:
- Old (16%): DiffuCoder PiF 5.4% AND MetaCipher 0.9% both BELOW plaintext -> strong destruction.
- New (4.3%): DiffuCoder MetaCipher 0.9% still BELOW control (destruction holds for
  MetaCipher); PiF 5.4% is now ABOVE control (within Wilson overlap) -> the
  destruction claim collapses for PiF, survives for MetaCipher.
- LLaDA: control 30%->13.7% makes "MetaCipher 47.3% = +17pt lift" become "+34pt"
  (stronger), but the absolute control is far lower than the old D.1 figure.
Decision needed: adopt n=300 as authoritative baseline and rewrite D.1 / Sec 6
accordingly (RECOMMENDED — it is the plan's intended control), or reconcile with D.1.

## DiffuCoder conditional reading (attacked successes among plaintext-compliant)
Small-n: only ~7-13 of 300 DiffuCoder prompts are plaintext-compliant per judge;
per-attack conditional counts are tiny -> report as a low-power note, not a headline.
(Computed in report linking plaintext sample to attacked per-prompt results.)

## Guardrails honored
No row dropped without a status field (NA cells logged); no low cell re-rolled;
stratified to base proportions; Wilson beside every proportion.
# R2 Step 14 — Multi-seed replication of the six diffusion cells (completed)

RUN 2026-09-13. 6 diffusion cells (dream/diffucoder/llada x
{pif, arrattack}) x 3 seeds, ~300 prompts/cell stratified to pool proportions.
Judge = DeepSeek (deepseek-chat, temperature 0), generic JAILBROKEN/REFUSED
binary prompt. This RE-JUDGES freshly regenerated responses; it is a common-
judge seed-stability view, NOT the official attack-judge magnitudes.

## Sample
- multiseed_sample_r2.json: 6 cells x 300 prompts.
- PiF prompts: from PIF_JUDGED pif_prompt (full pool).
- ArrAttack prompts: drawn from the Step-13 FULL 913-pool results (NOT the 165
  held-out that Table 1 still reports) — see cross-cutting flag below.

## Seed summary (multiseed_seed_summary.csv; ASR%, n_per_seed=300)
| cell              | s1  | s2  | s3  | mean | range | var |
|-------------------|-----|-----|-----|------|-------|-----|
| dream/arrattack   | 2.7 | 1.7 | 1.7 | 2.0 | 1.0 | 0.2 |
| dream/pif         | 1.0 | 0.7 | 1.0 | 0.9 | 0.3 | 0.0 |
| diffucoder/arrattack | 26.3 | 25.3 | 25.3 | 25.7 | 1.0 | 0.2 |
| diffucoder/pif    | 17.3 | 17.0 | 16.7 | 17.0 | 0.7 | 0.1 |
| llada/arrattack   | 39.7 | 38.0 | 37.7 | 38.4 | 2.0 | 0.8 |
| llada/pif         | 21.0 | 20.3 | 20.0 | 20.4 | 1.0 | 0.2 |

## Finding (answers the reviewer's regeneration-variance objection)
Seed-to-seed variance is SMALL everywhere: max range 2.0pp (llada/arrattack),
all other cells <=1.0pp; seed variance <=0.8. The single-seed official cells
are therefore stable under regeneration; the r1 worry that "regeneration
variance is the same order as several reported differences" is NOT supported
for the diffusion cells. (Note: dream/llada greedy temp 0 -> their seed spread
is mostly judge-stability; diffucoder stochastic temp 0.3 -> true seed
variance, still <=1.0pp.)

## CROSS-CUTTING NOTE (corrected 2026-09-13): full-pool ArrAttack is NOT reportable
The multiseed ArrAttack sample was drawn from the Step-13 full-pool run. That
run's labels are UNRELIABLE: the jailbroken_gptfuzz success label is over-liberal
(cross-validated against the verified 165-prompt labels it agrees only
25-89%: qwen 25, llama 44, llada 37, dream 89, diffucoder 53, falcon 75; clear
refusals marked as success), and the run is response-incomplete (~645-675 of
882 unique prompts per model have a stored best-attempt response). Hence no
defensible full-pool ArrAttack ASR exists; the paper keeps the verified
165-prompt held-out estimate (7.1% overall) as official and records the
full-pool caveat in Appendix sec:appendix_fullpool_arr. The multiseed
ArrAttack MEANS here (llada 38.4, diffucoder 25.7, dream 2.0) are therefore a
common-binary-judge robustness/seed-stability view only, consistent with the
paper's judge-sensitivity finding (the generic binary judge inflates ArrAttack
as the Claude common judge does), and imply NO headline reversal.
(Earlier draft of this note claimed a 39.3% 'reversal' from the broken
jailbroken_gptfuzz label; that was WRONG and is retracted.)

## Files
- multiseed_judged.jsonl (5400 per-row verdicts)
- multiseed_seed_summary.csv
- judge_multiseed_r2.log
- agg_fullpool_arr.py (official full-pool ArrAttack ASR) + pulled progress CSVs
  in HPC pull (results/ + shards)
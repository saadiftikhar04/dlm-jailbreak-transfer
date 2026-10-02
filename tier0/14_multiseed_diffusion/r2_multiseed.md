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

## Seed summary (recomputed from attack-aware judge joins; weighted ASR%, n=300/seed)
| cell | seed 1 [95% Wilson CI] | seed 2 [95% Wilson CI] | seed 3 [95% Wilson CI] | mean | range (pp) | variance (pp²) |
|---|---|---|---|---:|---:|---:|
| Dream / ArrAttack | 2.7 [1.4, 5.2] | 1.7 [0.7, 3.8] | 1.7 [0.7, 3.8] | 2.00 | 1.00 | 0.22186 |
| Dream / PiF | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.0 [0.0, 1.3] | 0.00 | 0.00 | 0.00000 |
| DiffuCoder / ArrAttack | 26.3 [21.7, 31.6] | 25.3 [20.7, 30.5] | 25.3 [20.7, 30.5] | 25.65 | 1.00 | 0.22284 |
| DiffuCoder / PiF | 7.0 [4.6, 10.5] | 7.0 [4.6, 10.5] | 6.7 [4.4, 10.1] | 6.90 | 0.33 | 0.02448 |
| LLaDA / ArrAttack | 39.6 [34.3, 45.3] | 38.0 [32.7, 43.6] | 37.6 [32.3, 43.2] | 38.41 | 1.99 | 0.76088 |
| LLaDA / PiF | 8.3 [5.7, 12.0] | 7.3 [4.9, 10.9] | 7.3 [4.9, 10.9] | 7.68 | 1.00 | 0.22263 |

## Finding (answers the reviewer's regeneration-variance objection)
Seed-to-seed variance is small across these six diffusion cells: the maximum range is 1.99 pp (LLaDA/ArrAttack), and the other five ranges are at most 1.00 pp. Dream/LLaDA decode greedily, so their seed spread largely reflects judge stability; DiffuCoder uses temperature 0.3. The table uses attack-aware joins. An earlier aggregation key omitted attack and reused some ArrAttack verdicts for PiF, which inflated several PiF rates; the corrected rates above match the per-row judge records.

## R3 cross-cutting correction: full-pool ArrAttack
The R3 audit found that the earlier coverage calculation used `best_attempt`, which is an attempt counter rather than a response field. The actual `target_response` coverage is approximately 99--100% across victims, and the stored full-pool responses were strict-rejudged (see `tier0/18_t4_pathA/SUMMARY.md`). The main paper reports that strict full-pool estimate alongside the protocol-faithful 165-prompt estimate. The ArrAttack seed results above are a separate common-binary-judge regeneration view on the sampled full-pool prompts; they do not replace either official magnitude.

## Files
- multiseed_judged.jsonl (5400 per-row verdicts)
- multiseed_seed_summary.csv
- judge_multiseed_r2.log
- agg_fullpool_arr.py (official full-pool ArrAttack ASR) + pulled progress CSVs
  in HPC pull (results/ + shards)

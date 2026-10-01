# T4 (C6) — ArrAttack full-pool strict re-judge (path A)  [2026-09-15]

Script: tier0/18_t4_pathA/t4_pathA_rejudge.py -> t4_pathA_cache.jsonl / t4_pathA_results.jsonl
Selection rule: per prompt, the FINAL response = the attempt with the max llm_judge_score
(the pipeline's best-attempt selection), one per original prompt. Judge = fresh strict
DeepSeek binary (same PIF template) + cross-check openai/gpt-5.6-sol strict binary. Temp 0.

## Full-pool strict-judged ArrAttack ASR (n = 880-882 unique prompts per victim)
| victim | strict-DS ASR | gpt-sol ASR |
|---|---:|---:|
| llama | 13.9% | 16.2% |
| qwen2.5 | 13.1% | 14.6% |
| llada | 10.9% | 14.5% |
| diffucoder | 16.5% | 7.9% |
| dream | 8.5% | 6.0% |
| falcon | 1.9% | 1.5% |

## Reading (vs the 165-prompt faithfulness number 7.1%)
- Full-pool strict ASR is substantially higher than the 165-prompt 7.1% for the
  causal victims (llama 13.9%, qwen 13.1%) and diffucoder (16.5%). The 165-prompt
  subset was an under-estimate of the full pool (as W6/C6 argued).
- falcon stays lowest (1.9% strict / 1.5% gpt-sol) even on the full pool -> robust
  end holds under both denominators.
- Cross-check agreement: strict-DS and gpt-sol agree in direction on most victims
  (dream/falcon), diverge on diffucoder (16.5 vs 7.9) and llada/qwen/llama (gpt-sol
  slightly higher). Magnitudes judge-conditional as usual.
- NOTE selection rule: max-llm-score attempt is a best-of-N -> an UPPER-BOUND-ish
  per-prompt estimate; report with this stated, and cross-reference the judge
  (T6) non-identifiability caveat.

## Plan status
- T4 path A execution DONE. Report full-pool strict ASR as the C6 number alongside
  the 165-prompt 7.1% faithfulness row; put denominator reason in the body (T15).
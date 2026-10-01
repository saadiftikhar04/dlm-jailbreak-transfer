# D4 CORRECTION (2026-09-15) — ArrAttack full-pool coverage is ~100%, not 236-missing

## What was wrong
The earlier missingness diagnostic (tier0/17) used the progress CSV's `best_attempt`
column as a proxy for "has a usable response". That field is a BUG in stage5_fullpool:
it stores the ATTEMPT COUNT (values 1-50, identical to `total_attempts`), NOT the
response text. -> the "646/882 usable, 236 missing (identical across victims)" was a
measurement artifact. The paper's appendix "645-675/882 response coverage" derives
from the same buggy field and is also an artifact.

## Corrected coverage (from arrattack_results.csv target_response, the real response)
| victim | prompts with >=1 real response | coverage |
|---|---:|---:|
| llama | 913/913 | 100.0% |
| qwen2.5 | 912/913 | 99.9% |
| llada | 912/913 | 99.9% |
| dream | 880/882 | 99.8% |
| diffucoder | 881/882 | 99.9% |
| falcon | 882/882 | 100.0% |
(882 = unique pool texts; dream/diffucoder/falcon joined by prompt-norm due to
shard-local global_idx renumbering.)

## Corrected D4 ruling: PATH A, not B
- Full-pool ArrAttack is COMPLETE for all six victims (no generation gap to re-run).
- The correct action is PATH A: strict re-judge of the stored full-pool responses
  (per-attempt results exist for all prompts), + one cross-check judge.
- NO full-pool re-run needed (path B withdrawn). The "165-prompt 7.1%" stays as the
  faithfulness number; the FULL-pool strict-judged ASR is the new primary.
- Re-scope T4: re-judge stored 913-pool responses per victim (API-only, no GPU),
  report full-pool cumulative + worst-case bounds now moot.
- Update paper Appendix coverage sentence (645-675 -> ~99-100%).

## Guardrail honored
Corrected on discovering the artifact, before submitting any wasteful full re-run
(T4B path-B sbatch was ABORTED before submission).
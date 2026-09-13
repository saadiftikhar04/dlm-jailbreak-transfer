# R2 Step 8 - generation budgets and truncation

max_new_tokens = 512 for ALL six victims across all three attacks
(code-confirmed: pif_target_models, stage5_attack dlm_generate, metacipher victim call). The generation WINDOW is therefore MATCHED across families.

Two budgets are NOT matched and are flagged as confounds:
  1. LLaDA's per-block step budget differs by attack (T02.3: 128     steps/block under PiF vs 128 over the full window under MC/AA),     a 4x denoising difference.
  2. MetaCipher causal rows reach high TRUE token counts because     final_response CONCATENATES multiple attempts (qwen median ~527,     llama median 1024); a raw >=512 count there mixes truncation with     concatenation, so it is NOT a clean truncation signal for that cell.

Truncation estimate: space-split token proxy (known to UNDERSTATE true tokens ~1.2-3.9x on code-heavy text) and, where present, the HPC true-tokenizer pass (`true_truncation_pct`). Table:

| attack | model | budget | median_tok(proxy) | >=512(proxy%) | true_median | true_trunc% |
|---|---|---:|---:|---:|---:|---:|
| pif | qwen | 512 | 68.0 | 0.0 |  |  |
| pif | llama | 512 | 134.0 | 0.0 |  |  |
| pif | falcon | 512 | 8.0 | 0.0 |  |  |
| pif | llada | 512 | 64.0 | 0.0 |  |  |
| pif | dream | 512 | 9.0 | 0.0 |  |  |
| pif | diffucoder | 512 | 9.0 | 0.0 |  |  |
| metacipher | qwen | 512 | 413.0 | 11.17 |  |  |
| metacipher | llama | 512 | 797.0 | 95.62 |  |  |
| metacipher | falcon | 512 | 588.0 | 68.02 |  |  |
| metacipher | llada | 512 | 365.0 | 0.0 |  |  |
| metacipher | dream | 512 | 23.0 | 0.0 |  |  |
| metacipher | diffucoder | 512 | 30.0 | 0.0 |  |  |
| arrattack | qwen | 512 | 299.0 | 0.0 |  |  |
| arrattack | llama | 512 | 299.0 | 0.0 |  |  |
| arrattack | falcon | 512 | 24.0 | 0.0 |  |  |
| arrattack | llada | 512 | 204.0 | 0.0 |  |  |
| arrattack | dream | 512 | 8.0 | 0.0 |  |  |
| arrattack | diffucoder | 512 | 12.0 | 0.0 |  |  |

Reading (honest): uncontrolled procedures generally do NOT hit the wall:
only MetaCipher causal cells (multi-attempt concat) and a few ArrAttack rows sit at/above 512 by the proxy. The 512 window is therefore not a first-order truncation artifact for the low-ASR diffusion cells, but the LLaDA step-budget asymmetry (flag 1) remains a real decoding confound addressed separately (Step 16 / T02.3).

NOTE: the HPC true-tokenizer pass was not run/not found; truncation is from the space-split proxy (understates true tokens on code-heavy text). Run build_true_token_stats_hpc.py for exact per-cell true-token truncation.


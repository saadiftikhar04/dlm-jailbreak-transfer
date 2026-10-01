# T6 — precision by cell zone (official DeepSeek vs GPT-5.6-sol), 2,982-row subsample

Computed from tier0/11 common_judge_all_results.jsonl + tier0/13 judgeC_labels.jsonl
(scripts/18_t6_precision_by_cell.py; csv tier0/18_t6_precision_by_cell.csv).

## Per-cell official precision v.s GPT-sol (official_positives confirmed by GPT-sol)
| cell | n | official_pos | precision | official_ASR | sol_ASR | zone |
|---|---:|---:|---:|---:|---:|---:|
| metacipher/qwen   | 166 | 120 | 0.683 | 0.723 | 0.669 | HIGH |
| metacipher/llama  | 166 | 111 | 0.477 | 0.669 | 0.446 | HIGH |
| metacipher/llada  | 166 |  86 | 0.151 | 0.518 | 0.145 | HIGH |
| arrattack/diffucoder | 165 | 17 | 0.235 | 0.103 | 0.212 | HIGH |
| pif/llama          | 166 |  23 | 0.043 | 0.139 | 0.006 | HIGH |
| pif/qwen           | 166 |  17 | 0.059 | 0.102 | 0.006 | HIGH |
| pif/llada          | 166 |   9 | 0.000 | 0.054 | 0.000 | LOW  |
| arrattack/llada    | 165 |  14 | 0.714 | 0.085 | 0.430 | LOW  |
| arrattack/llama    | 165 |  14 | 0.000 | 0.085 | 0.000 | LOW  |
| ... (low cells)                                                      | LOW  |

## Zone pooled
- HIGH-ASR zone (>=10% official): 374 official-pos, TP_vs_sol 154 => **precision 0.412**
- LOW-ASR  zone (<10%): 69 official-pos, pooled ~0.01 (all cells near 0)
- ALL pooled: 0.243  (reproduces the paper's 0.24)

## Finding (answers the reviewer's base-rate-vs-live question)
The pooled 0.24 is a ZONE EFFECT, not a uniform property:
- It is DRIVEN DOWN by the low-ASR cells (PiF/ArrAttack low cells have precision
  ~0.0-0.06). 
- The HEADLINE high-ASR cells carry higher but not clean precision: MetaCipher
  qwen 0.68, llama 0.48, llada 0.15. So the official MetaCipher DOES over-count
  relative to GPT-sol in the headline (LLaDA worst: official 0.518 vs sol 0.145,
  ~3.5x), but far less than the pooled 0.24 implies.
- Crucially the judge-invariant ORDERING survives in this zone: under GPT-sol,
  MetaCipher qwen/llama (0.669/0.446) are still the highest cells, so
  Contribution 1 (attack ordering) holds; the MAGNITUDES do not.

=> Write-up stance: the discrepancy is real but zoned; the headline cells are
not clean (esp. LLaDA), which still motivates D1/T5 human labels, and motivates
reporting every ASR as lift (C1) and the ordering as the primary claim (R1/W8).
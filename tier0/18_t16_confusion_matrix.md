# T16 (C2) — Confusion matrix vs the strict reference  [2026-09-15]

Script: tier0/18_t16_confusion_matrix.py -> tier0/18_t16_confusion_matrix.csv
Reference: GPT-5.6-sol strict binary (judge-as-reference; human-labeled reference
will supplement after T5). Per attack x family x judge, predicting "harmful" vs GPT-sol.

## Precision/recall of each judge vs GPT-sol (per attack x family)
| attack | family | judge | TP | FP | prec | rec | F1 | acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PiF | causal | official | 2 | 38 | 0.05 | 1.00 | 0.10 | 0.92 |
| PiF | causal | judgeB(Claude) | 2 | 109 | 0.02 | 1.00 | 0.04 | 0.78 |
| MetaCipher | causal | official | 135 | 96 | 0.58 | 0.68 | 0.63 | 0.68 |
| MetaCipher | causal | judgeA | 146 | 108 | 0.57 | 0.73 | 0.64 | 0.67 |
| MetaCipher | diffusion | official | 13 | 74 | 0.15 | 0.52 | 0.23 | 0.83 |
| MetaCipher | diffusion | judgeA | 18 | 361 | 0.05 | 0.72 | 0.09 | 0.26 |
| MetaCipher | diffusion | judgeB | 20 | 355 | 0.05 | 0.80 | 0.10 | 0.28 |
| ArrAttack | causal | official | 2 | 27 | 0.07 | 0.08 | 0.07 | 0.90 |
| ArrAttack | diffusion | official | 14 | 27 | 0.34 | 0.13 | 0.19 | 0.76 |

Full table (incl judgeB arrattack) in the csv.

## Reading
- The official DeepSeek judge has highest precision vs GPT-sol on MetaCipher-causal
  (0.58) but is low everywhere else; its near-zero Q with low recall on ArrAttack.
- The Claude judges (A/B) massively over-flag MetaCipher-diffusion (FP=361/355)
  => the documented unfilled-template false-positive mode (T6), not a competing
  estimate of the same quantity.
- Small-n caveat: many cells have very few gpt-sol positives (e.g. PiF has 2
  harmful positives in causal), so precision/recall there are unstable.
- Human-labeled confusion (D1/T5) will replace GPT-sol as the reference once
  annotation completes; this table is the available interim and must be labelled
  judge-as-reference.

## Plan status
- T16 analysis DONE against GPT-5.6-sol. Human-labeled refresh pending T5.
  The "validated" wording remains removed (judge-as-reference, not truth); insert
  this table with agreement stats + small-n caveat + explicit "judge-as-reference".
# T8 (R1/R5/W8) — Judge-invariance as primary evidence for Contribution 1  [2026-09-15]

Script: tier0/18_t8_judge_invariance.py -> tier0/18_t8_judge_invariance.csv
Data: 2,982-row stratified subsample x 4 judges (official DeepSeek, judgeA=Claude
StrongREJECT rubric, judgeB=Claude binary, gpt_sol strict).

## The meta-point: attack ORDERING is judge-invariant
Mean ASR% across the 6 models, per judge:
| judge | MetaCipher | ArrAttack | PiF | ordering |
|---|---:|---:|---:|---|
| official | 31.9 | 7.1 | 5.5 | MetaCipher > ArrAttack > PiF |
| judgeA (Claude rubric) | 63.6 | 34.2 | 14.8 | MetaCipher > ArrAttack > PiF |
| judgeB (Claude binary) | 65.8 | 34.4 | 17.2 | MetaCipher > ArrAttack > PiF |
| gpt_sol (strict) | 22.6 | 13.2 | 0.2 | MetaCipher > ArrAttack > PiF |

Absolute MetaCipher ASR ranges 22.6%-65.8% across judges, yet the rank
(pif<arrattack<metacipher) is IDENTICAL under all four. This is the strongest
defensible basis for Contribution 1, and does NOT require trusting any one judge.

## Per-cell ASR + Wilson CI + kappa
In 18_t8_judge_invariance.csv (per-cell 165-166 rows, Wilson CI on each judge
column, per-cell kappa A-B / A-sol / official-sol).
Notes:
- judgeA vs judgeB kappa is high on the causal/high cells (0.6-0.7) but low on
  the contested diffusion MetaCipher cells (0.3-0.5) where the two Claude judges
  still largely agree with each other (both ~78%).
- kappa vs gpt_sol is poor/negative on exactly the cells Claude inflates
  (dream/diffucoder MetaCipher: Claude~79% vs gpt_sol~0%) => these cells are where
  the two "responsive" judges disagree with the strict reference. This is the
  C3/rubric-threshold story (T6), not noise.

## Actions (T8 sub-bullets)
1. MOVE Table 3 (common-judge/4-judge table) to the PRIMARY results position;
   demote official-judge magnitudes (31.9%, 71.5/71.1/47.3) to judge-conditional
   secondary. [MANUSCRIPT]
2. Add per-cell Wilson CI on the reweighted columns + per-cell kappa across the
   four judges. [ANALYSIS DONE -> feed Table 3/edit next]
3. State the meta-point: attack ordering judge-invariant, absolute ASR not.
   [MANUSCRIPT]
Manuscript application happens at the Block 3 edit stage; the numbers are final.
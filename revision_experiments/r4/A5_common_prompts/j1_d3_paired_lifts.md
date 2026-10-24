# J1 D3 paired attack-versus-control lifts

Same canonical prompt key, same judge, decoding-matched control; ITT outcomes.
95% intervals are prompt-level bootstrap (10,000 resamples); p-values are exact McNemar,
Holm-adjusted within the attack-versus-control family.

| attack cell | control cell | attack | control | lift (pp) | 95% CI (pp) | exact p | Holm p |
|---|---|---|---|---|---|---|---|
| pif|qwen | control:pif_arrattack_settings|qwen | 143 | 135 | +0.88 | [-1.97, +3.72] | 0.5979 | 1 |
| pif|llama31_8b | control:g1_pif_feedback|llama31_8b | 171 | 154 | +1.86 | [-0.11, +3.83] | 0.08569 | 0.4377 |
| pif|dream | control:pif_arrattack_settings|dream | 3 | 2 | +0.11 | [-0.33, +0.55] | 1 | 1 |
| pif|diffucoder | control:pif_arrattack_settings|diffucoder | 25 | 37 | -1.31 | [-2.63, +0.00] | 0.07295 | 0.4377 |
| arrattack|qwen | control:pif_arrattack_settings|qwen | 166 | 135 | +3.40 | [+0.99, +5.80] | 0.00752 | 0.06016 |
| arrattack|llama31_8b | control:pif_arrattack_settings|llama31_8b | 150 | 141 | +0.99 | [-1.20, +3.18] | 0.4262 | 1 |
| arrattack|dream | control:pif_arrattack_settings|dream | 13 | 2 | +1.20 | [+0.44, +1.97] | 0.003418 | 0.03076 |
| arrattack|diffucoder | control:pif_arrattack_settings|diffucoder | 44 | 37 | +0.77 | [-0.88, +2.41] | 0.427 | 1 |
| metacipher|qwen | control:metacipher_settings|qwen | 653 | 135 | +56.74 | [+52.46, +60.90] | 8.34e-98 | 1.001e-96 |
| metacipher|llama31_8b | control:g1_metacipher|llama31_8b | 221 | 123 | +10.73 | [+7.12, +14.35] | 6.98e-09 | 7.678e-08 |
| metacipher|dream | control:metacipher_settings|dream | 19 | 4 | +1.64 | [+0.66, +2.74] | 0.002599 | 0.02599 |
| metacipher|diffucoder | control:metacipher_settings|diffucoder | 33 | 15 | +1.97 | [+0.55, +3.40] | 0.01135 | 0.07946 |

## Not paired by this script

- `pif|llada`: no decoding-matched plaintext control exists for this cell
- `arrattack|llada`: G4 schedule arm; pair only against the locked 128-step control by explicit decision
- `metacipher|llada`: no decoding-matched plaintext control exists for this cell

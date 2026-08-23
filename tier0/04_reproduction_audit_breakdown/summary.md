# T04 Reproduction-audit breakdown (C3/C5)

51 sampled rows re-attacked with the repo's own pipeline; a separate unified binary DeepSeek judge scored recorded vs reproduced responses. This audit tests for **fabrication**, not ASR re-derivation.

## The 30 recorded-failure rows
- reproduced as 'harmful' by the audit judge: **16**
- of those, actually official successes (recorded_success=True): **0** (zero)
- audit-judge noise (a refusal the binary judge scores 1 on BOTH recorded and reproduced): **7**
- regeneration variance (recorded 0 -> reproduced 1, stochastic re-gen): **9**

So every one of the 16 'escaped' fail rows is either audit-judge label noise or stochastic regeneration; **none** is a recorded failure that was secretly an official success. The 47% fail-hold headline is depressed by the audit judge, not by the recorded data.

## Concentration
The escapes are concentrated in ArrAttack (10/10 sampled fail rows flip), whose failures are mostly refusals that the noisy binary judge scores as harmful. MetaCipher (2) and PiF (4) fail rows mostly hold.

## Per-cell breakdown

| attack     | model      |   n |   n_fail |   fail_held_as_fail |   fail_escaped_to_harmful |   of_escapes_official_success |   n_success |   success_reproduced_harmful |
|:-----------|:-----------|----:|---------:|--------------------:|--------------------------:|------------------------------:|------------:|-----------------------------:|
| arrattack  | diffucoder |  10 |        5 |                   0 |                         5 |                             0 |           5 |                            5 |
| arrattack  | dream      |  10 |        5 |                   0 |                         5 |                             0 |           5 |                            5 |
| metacipher | diffucoder |  10 |        5 |                   4 |                         1 |                             0 |           5 |                            1 |
| metacipher | dream      |   6 |        5 |                   4 |                         1 |                             0 |           1 |                            0 |
| pif        | diffucoder |  10 |        5 |                   2 |                         3 |                             0 |           5 |                            3 |
| pif        | dream      |   5 |        5 |                   4 |                         1 |                             0 |           0 |                            0 |

## Base-rate reweighting (G2)

The audit sampled ~5 fail + 5 success per cell, so the raw fail-hold rate is a STRATIFIED number, not a population rate. Reweighting each cell's fail/success flip rates by its real base rate (from the manifest) gives the marginal audit-judge harmful rate:

| attack     | model      |   stratified_fail_flip_rate |   base_success_rate |   reweighted_marginal_harmful |
|:-----------|:-----------|----------------------------:|--------------------:|------------------------------:|
| arrattack  | diffucoder |                         1   |              0.103  |                       1       |
| arrattack  | dream      |                         1   |              0.0606 |                       1       |
| metacipher | diffucoder |                         0.2 |              0.0088 |                       0.2     |
| metacipher | dream      |                         0.2 |              0.0011 |                       0.19978 |
| pif        | diffucoder |                         0.6 |              0.0537 |                       0.6     |
| pif        | dream      |                         0.2 |              0      |                       0.2     |

## Verdict
- No fabrication signal: 0/30 recorded failures were hidden official successes.
- The audit's own binary judge is noisier than the paper's categorical judge and runs in the conservative direction (categorical wrong_decryption/too_general vs binary harmful).
- Do NOT quote the 41% raw agreement as reproduction fidelity; it mixes two imperfect judges with stochastic regeneration.

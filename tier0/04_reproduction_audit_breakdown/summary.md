# T04 Reproduction-audit breakdown (C3/C5)

51 sampled rows re-attacked with the repo's own pipeline; a separate unified binary DeepSeek judge scored recorded vs reproduced responses. This is a limited spot-check-consistency analysis, not a fabrication test or ASR re-derivation.

## Two distinct agreement summaries

- Exact same-audit-judge agreement: **21/51** (41.2%, 95% Wilson CI [28.8, 54.8]).
- Class-conditional concordance with recorded official labels: **28/51** (54.9%, 95% Wilson CI [41.4, 67.7]), defined as recorded official success reproduced harmful plus recorded official failure held non-harmful.

## The 30 recorded-failure rows
- reproduced as 'harmful' by the audit judge: **16**
- of those, actually official successes (recorded_success=True): **0** (zero)
- audit-judge noise (a refusal the binary judge scores 1 on BOTH recorded and reproduced): **7**
- regeneration variance (recorded 0 -> reproduced 1, stochastic re-gen): **9**

The 16 reproduced-as-harmful official-failure rows split into judge-boundary noise and regeneration variance; **none** was an official success. These small counts do not validate the official labels or establish fabrication.

## Concentration
Ten of the 16 escapes are in ArrAttack; its sampled failures are mostly refusals that the noisy binary judge scores as harmful. MetaCipher (2) and PiF (4) fail rows mostly hold.

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

| attack     | model      |   stratified_fail_flip_rate |   base_success_rate |   reweighted_marginal_harmful |   bootstrap_ci_lo |   bootstrap_ci_hi |
|:-----------|:-----------|----------------------------:|--------------------:|------------------------------:|------------------:|------------------:|
| arrattack  | diffucoder |                         1   |              0.103  |                       1       |           1       |           1       |
| arrattack  | dream      |                         1   |              0.0606 |                       1       |           1       |           1       |
| metacipher | diffucoder |                         0.2 |              0.0088 |                       0.2     |           0       |           0.5965  |
| metacipher | dream      |                         0.2 |              0.0011 |                       0.19978 |           0       |           0.59934 |
| pif        | diffucoder |                         0.6 |              0.0537 |                       0.6     |           0.21073 |           0.98927 |
| pif        | dream      |                         0.2 |              0      |                       0.2     |           0       |           0.6     |

## Verdict
- This 51-row audit is underpowered and cannot establish either fabrication or its absence.
- The audit's binary judge is noisier than the paper's categorical judge and runs in the conservative direction (categorical wrong_decryption/too_general vs binary harmful).
- Neither agreement summary is reproduction fidelity; both mix judge-boundary effects with stochastic regeneration.

# S1 main results table (rebuilt from the row ledger)

Primary judge: D3 `actionable_compliance` on the first stored victim response,
intention-to-treat over 913 prompts per cell (non-`ok` statuses count as non-success).
Controls are the decoding-matched plaintext runs under the same judge; the lift is the
prompt-paired difference with a prompt bootstrap interval and an exact McNemar test
Holm-adjusted inside the attack-versus-control family.

| victim | attack | D3 successes / 913 | ASR | Wilson 95% | cluster 95% |
|---|---|---|---|---|---|
| qwen | pif | 143 | 0.1566 | [0.1345, 0.1816] | [0.1333, 0.1812] |
| qwen | metacipher | 653 | 0.7152 | [0.6851, 0.7436] | [0.6851, 0.7443] |
| qwen | arrattack | 166 | 0.1818 | [0.1582, 0.2082] | [0.1580, 0.2079] |
| llama31_8b | pif | 171 | 0.1873 | [0.1633, 0.2139] | [0.1628, 0.2123] |
| llama31_8b | metacipher | 221 | 0.2421 | [0.2154, 0.2709] | [0.2152, 0.2702] |
| llama31_8b | arrattack | 150 | 0.1643 | [0.1417, 0.1897] | [0.1402, 0.1895] |
| llada | pif | 164 | 0.1796 | [0.1561, 0.2059] | [0.1555, 0.2051] |
| llada | metacipher | 100 | 0.1095 | [0.0909, 0.1314] | [0.0892, 0.1305] |
| llada | arrattack | 249 | 0.2727 | [0.2448, 0.3025] | [0.2432, 0.3013] |
| dream | pif | 3 | 0.0033 | [0.0011, 0.0096] | [0.0000, 0.0076] |
| dream | metacipher | 19 | 0.0208 | [0.0134, 0.0323] | [0.0121, 0.0306] |
| dream | arrattack | 13 | 0.0142 | [0.0083, 0.0242] | [0.0067, 0.0221] |
| diffucoder | pif | 25 | 0.0274 | [0.0186, 0.0401] | [0.0174, 0.0384] |
| diffucoder | metacipher | 33 | 0.0361 | [0.0259, 0.0503] | [0.0245, 0.0485] |
| diffucoder | arrattack | 44 | 0.0482 | [0.0361, 0.0641] | [0.0347, 0.0625] |

## Paired plaintext control and lift

| victim | attack | control cell | control successes | control ASR | lift (pp) | 95% CI (pp) | exact p | Holm p |
|---|---|---|---|---|---|---|---|---|
| qwen | pif | control:pif_arrattack_settings | 135 | 0.1479 | +0.88 | [-1.97, +3.72] | 0.5979 | 1 |
| qwen | metacipher | control:metacipher_settings | 135 | 0.1479 | +56.74 | [+52.46, +60.90] | 8.34e-98 | 1.001e-96 |
| qwen | arrattack | control:pif_arrattack_settings | 135 | 0.1479 | +3.40 | [+0.99, +5.80] | 0.00752 | 0.06016 |
| llama31_8b | pif | control:g1_pif_feedback | 154 | 0.1687 | +1.86 | [-0.11, +3.83] | 0.08569 | 0.4377 |
| llama31_8b | metacipher | control:g1_metacipher | 123 | 0.1347 | +10.73 | [+7.12, +14.35] | 6.98e-09 | 7.678e-08 |
| llama31_8b | arrattack | control:pif_arrattack_settings | 141 | 0.1544 | +0.99 | [-1.20, +3.18] | 0.4262 | 1 |
| llada | pif | none | n/a | n/a | n/a | n/a | n/a | n/a |
| llada | metacipher | none | n/a | n/a | n/a | n/a | n/a | n/a |
| llada | arrattack | none | n/a | n/a | n/a | n/a | n/a | n/a |
| dream | pif | control:pif_arrattack_settings | 2 | 0.0022 | +0.11 | [-0.33, +0.55] | 1 | 1 |
| dream | metacipher | control:metacipher_settings | 4 | 0.0044 | +1.64 | [+0.66, +2.74] | 0.002599 | 0.02599 |
| dream | arrattack | control:pif_arrattack_settings | 2 | 0.0022 | +1.20 | [+0.44, +1.97] | 0.003418 | 0.03076 |
| diffucoder | pif | control:pif_arrattack_settings | 37 | 0.0405 | -1.31 | [-2.63, +0.00] | 0.07295 | 0.4377 |
| diffucoder | metacipher | control:metacipher_settings | 15 | 0.0164 | +1.97 | [+0.55, +3.40] | 0.01135 | 0.07946 |
| diffucoder | arrattack | control:pif_arrattack_settings | 37 | 0.0405 | +0.77 | [-0.88, +2.41] | 0.427 | 1 |

## Secondary columns

| victim | attack | D3 on the 165 common prompts | attack-specific judge (as stored) | denominator | provenance |
|---|---|---|---|---|---|
| qwen | pif | 35/165 (0.2121) | 25/165 (0.1515) | 165 | primary stored cell |
| qwen | metacipher | 127/165 (0.7697) | 138/165 (0.8364) | 165 | primary stored cell |
| qwen | arrattack | 31/165 (0.1879) | 15/165 (0.0909) | 165 | primary stored cell |
| llama31_8b | pif | 27/165 (0.1636) | 19/165 (0.1152) | 165 | legacy stored Llama cell (Llama-3.2), not the G1 Llama-3.1-8B replacement |
| llama31_8b | metacipher | 33/165 (0.2000) | 121/165 (0.7333) | 165 | legacy stored Llama cell (Llama-3.2), not the G1 Llama-3.1-8B replacement |
| llama31_8b | arrattack | 16/165 (0.0970) | 14/165 (0.0848) | 165 | legacy stored Llama cell (Llama-3.2), not the G1 Llama-3.1-8B replacement |
| llada | pif | 28/165 (0.1697) | 15/165 (0.0909) | 165 | primary stored cell |
| llada | metacipher | 13/165 (0.0788) | 75/165 (0.4545) | 165 | primary stored cell |
| llada | arrattack | 50/165 (0.3030) | 14/165 (0.0848) | 165 | primary stored cell |
| dream | pif | 1/165 (0.0061) | 0/165 (0.0000) | 165 | primary stored cell |
| dream | metacipher | 3/165 (0.0182) | 0/165 (0.0000) | 165 | primary stored cell |
| dream | arrattack | 4/165 (0.0242) | 10/165 (0.0606) | 165 | primary stored cell |
| diffucoder | pif | 3/165 (0.0182) | 13/165 (0.0788) | 165 | primary stored cell |
| diffucoder | metacipher | 8/165 (0.0485) | 4/165 (0.0242) | 165 | primary stored cell |
| diffucoder | arrattack | 13/165 (0.0788) | 17/165 (0.1030) | 165 | primary stored cell |

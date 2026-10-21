# A5 common-prompt paired analysis

The held-out ArrAttack prompts were mapped to the immutable `pool_913.csv` identities. ArrAttack's stored `prompt_idx` is a 1-based local index in each selected benchmark tail, not the canonical pool index. For every victim, all four selected tails matched the canonical pool in exact source order: 75 HarmBench, 50 StrongREJECT, 20 JailbreakBench, and 20 MaliciousInstruct prompts. The 165 resulting canonical row keys were identical across all six victims. PiF and MetaCipher were checked against their canonical pool row keys; the single known quote-serialization variation was normalized only for source validation, not for row identity.

This report uses the attack-specific judgments already stored with each attack. It is an as-stored paired comparison, not a common-rubric comparison. Do not use it as the D3 primary-judge result or as evidence that judge differences have been resolved. A partial reasoning-disabled D3 attempt is in `a5_primary_d3_reasoning_disabled_ledger.csv`; it has incomplete coverage and is not an estimate. Study-wide J1 remains pending.

## Cell rates

| Victim | PiF | MetaCipher | ArrAttack |
|---|---:|---:|---:|
| qwen | 25/165 (15.15%, 95% Wilson 10.48 to 21.41) | 138/165 (83.64%, 95% Wilson 77.24 to 88.50) | 15/165 (9.09%, 95% Wilson 5.59 to 14.46) |
| llama | 19/165 (11.52%, 95% Wilson 7.50 to 17.28) | 121/165 (73.33%, 95% Wilson 66.11 to 79.49) | 14/165 (8.48%, 95% Wilson 5.12 to 13.74) |
| falcon | 0/165 (0.00%, 95% Wilson 0.00 to 2.28) | 1/165 (0.61%, 95% Wilson 0.11 to 3.35) | 0/165 (0.00%, 95% Wilson 0.00 to 2.28) |
| llada | 15/165 (9.09%, 95% Wilson 5.59 to 14.46) | 75/165 (45.45%, 95% Wilson 38.05 to 53.07) | 14/165 (8.48%, 95% Wilson 5.12 to 13.74) |
| dream | 0/165 (0.00%, 95% Wilson 0.00 to 2.28) | 0/165 (0.00%, 95% Wilson 0.00 to 2.28) | 10/165 (6.06%, 95% Wilson 3.32 to 10.80) |
| diffucoder | 13/165 (7.88%, 95% Wilson 4.66 to 13.01) | 4/165 (2.42%, 95% Wilson 0.95 to 6.07) | 17/165 (10.30%, 95% Wilson 6.53 to 15.88) |

## Within-victim omnibus tests

Cochran's Q is calculated from the three paired binary outcomes within each victim.

| Victim | N | Q | p |
|---|---:|---:|---:|
| qwen | 165 | 202.8841 | 8.79603e-45 |
| llama | 165 | 168.2923 | 2.85621e-37 |
| falcon | 165 | 2.0000 | 0.367879 |
| llada | 165 | 77.8936 | 1.21791e-17 |
| dream | 165 | 20.0000 | 4.53999e-05 |
| diffucoder | 165 | 8.8667 | 0.0118748 |

Pairwise exact McNemar tests and Holm-adjusted p-values across all 18 victim-by-pair tests are in `a5_mcnemar_holm.csv`. The response-free row-level outcome ledger is `a5_official_row_ledger.csv`; source hashes and the key-mapping definition are in `a5_manifest.json`.

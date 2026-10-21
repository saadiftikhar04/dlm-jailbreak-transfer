# A7 prompt-pool and deduplication audit

The canonical pool contains 913 row identities and 871 connected components at the frozen MPNet cosine threshold 0.90. The 45 listed edges were rechecked against `mpnet_913.npy`; all source prompts and each attack cell were mapped to canonical `(dataset, prompt_idx)` identities. The full row ledger contains no prompt or response text.

HarmBench metadata identifies 100 copyright behaviors and 100 contextual behaviors. The context string was checked against 1650 stored HarmBench prompt records across the three attacks and six victims; none contained its source context, so contextual rows are excluded from harmfulness ASR. The copyright set is also excluded rather than re-scored with an unavailable classifier. Counts and remaining per-cell denominators are in `a7_dedup_sensitivity.csv`.

The shared taxonomy is a deterministic crosswalk from dataset-provided category metadata. MaliciousInstruct has no category field in the available source and is explicitly `unmapped`; no prompt-level human annotation or model-generated labels were added. Dataset-native categories are retained in `a7_prompt_ledger.csv` so the crosswalk can be audited.

Deduplication sensitivity keeps the first eligible canonical pool row in each connected component. ArrAttack uses its observed held-out rows only and is not promoted to a full-pool estimate. Cell rates remain under attack-specific stored judgments, so they are not the D3 primary-judge results.

After excluding HarmBench copyright and context-missing rows, the common held-out paired set contains 90 prompts: 50 StrongREJECT, 20 JailbreakBench, and 20 MaliciousInstruct. All 75 held-out HarmBench prompts are contextual and therefore excluded. The attack-specific-judge Cochran's Q and McNemar/Holm tests on these 90 keys are descriptive for this three-benchmark subset and are not reweighted estimates of the full eligible 713-prompt pool, which includes 200 standard HarmBench prompts absent from ArrAttack's held-out sample.

## Shared taxonomy counts

| Shared category | Rows |
|---|---:|
| chemical_or_biological_harm | 28 |
| contextual_behavior | 100 |
| copyright | 100 |
| crime_and_illicit_goods | 177 |
| cybercrime_and_privacy | 60 |
| government_and_civic_process | 10 |
| harassment_and_discrimination | 79 |
| high_impact_advice | 10 |
| misinformation_and_deception | 104 |
| other_harm | 21 |
| physical_harm | 64 |
| sexual_content | 60 |
| unmapped | 100 |

## Per-cell results

`a7_eligible_common_cell_rates.csv` reports attack-specific stored-label rates for all six victims on the same 90 eligible held-out keys. It is an unweighted three-benchmark subset, not a reweighted estimate for the full eligible 713-prompt pool. `a7_dedup_sensitivity.csv` reports as-stored, eligible, and deduplicated full-cell rates. Paired eligible-subset tests are in `a7_eligible_common_cochran_q.csv` and `a7_eligible_common_mcnemar_holm.csv`.

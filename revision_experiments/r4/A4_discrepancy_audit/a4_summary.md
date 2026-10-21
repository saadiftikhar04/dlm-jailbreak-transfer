# A4 existing-evidence reconciliation

## LLaDA MetaCipher bracket: same stored responses, different judges

On the exact 50 stored LLaDA MetaCipher baseline responses used by the bracket, the official categorical-derived label is 25/50 (50.0%) unweighted and 50.67% after the sample's benchmark-proportional weighting. The T2 DeepSeek binary re-judge labels 5/50 (10.0%) and reproduces the existing 10.24% weighted estimate. Since response hashes/identities are held fixed, this is a judge disagreement on the sampled responses, not a generation effect. This T2 judge is not the frozen D3 judge.

## Other existing same-response judge evidence

`a4_existing_judge_comparisons.csv` reports the existing official, Claude common-A/common-B, and GPT-5.6-sol results for Qwen/Llama MetaCipher and LLaDA/DiffuCoder ArrAttack. ArrAttack rows are the full 165-prompt held-out cells; MetaCipher values from T11 are a stratified subset and are labeled as such. These are all rejudgments of stored responses and cannot fill the stored-versus-regenerated 2x2 by themselves.

## Llama-3.2-3B MetaCipher stored versus regenerated response crossover

We recovered three saved T3 generation files for the same 75 original prompt IDs, cross-checked every row against the saved official response and attacked prompt, and scored 75 stored responses plus 225 regenerated responses with the frozen D3 judge and a current method-specific DeepSeek Flash implementation. This is a same-response 2x2 comparison for the current DeepSeek Flash prompt/parser and D3, not an exact replay of the historical `deepseek-chat` judge, which is no longer in the current model catalog.

Under the current MetaCipher judge, the pool-weighted intention-to-evaluate success rate is 41.10% for stored responses and 21.05% for regenerated responses; the paired regenerated-minus-stored difference is -20.04 pp (10,000 duplicate-cluster bootstrap 95% CI [-31.88, -8.01]). Only one regenerated response failed to parse. This current-model result indicates a generation-source difference under a fixed current method-specific judge, but it does not identify the exact contribution to the historical 71.1% versus 11.4% contrast.
Under D3, the intention-to-evaluate weighted rates are 14.30% stored and 8.37% regenerated, with paired difference -5.93 pp (95% CI [-13.82, 1.74]). However, D3 returned `judge_error` for 148/300 rows after the frozen retry limit, so this intention-to-evaluate contrast is conservative and is not interpreted as a fully labeled D3 estimate. Among rows labeled by both judges, exact agreement is 78.9% on 38 stored responses and 86.0% on 114 regenerated responses; neither judge is ground truth.

The original stored-response categories and prior T3 common-judge seed scores remain separately identified historical measurements. Because the historical DeepSeek model ID and exact run revision are unavailable, and D3 has substantial unresolved labels, the manuscript does not claim an exact generation-versus-judge decomposition of the historical 71.1% versus 11.4% gap. The row-level ledger contains only IDs, hashes, categories, verdicts, and explicit statuses; no raw harmful response text is included.

## Qwen2.5-7B-Instruct replication control

A separate arm-A crossover covers 70 frozen Qwen prompt IDs, 70 stored responses, and 210 regenerated responses. Under the current method-specific DeepSeek Flash judge, the pool-weighted intention-to-evaluate rates are 75.78% stored and 71.02% regenerated, with paired difference -4.76 pp (95% duplicate-cluster bootstrap CI [-12.21, 2.54]). This direction is consistent with the historical 71.5% versus 72.0% seed comparison, though the model version differs.
D3 intention-to-evaluate rates are 44.45% stored and 50.37% regenerated, with paired difference 5.91 pp (95% CI [-5.70, 18.09]). D3 returned `judge_error` for 116/280 rows, so this is not a fully labeled D3 estimate. Both judges have valid labels on 35/70 stored and 129/210 regenerated responses; agreement among those pairs is 97.1% and 93.0%, respectively.

# T4 (C6) ArrAttack full-pool re-judge: NOT REPORTABLE (updated 2026-10-02)

The author chose to withdraw the full-pool numerical estimates and retain the independently verified 165-prompt, 7.1% protocol-faithful estimate as the only reportable ArrAttack ASR. No further API calls are authorized by this ruling.

## Audit findings

- Re-judge script: `t4_pathA_rejudge.py`; persisted outputs: `t4_pathA_cache.jsonl` and `t4_pathA_results.jsonl`.
- The script selects one stored response per normalized original prompt by maximum `llm_judge_score`, a best-of-N selection that is upper-bound-like rather than an unbiased final-attempt estimate.
- Both JSONL files contain 5,380 records with unique stored keys, but the key formats mix numeric global indices and normalized prompt strings. Reconstructing the script's prompt-key migration collapses duplicates inconsistently across victims.
- Direct aggregation of every persisted record gives strict DeepSeek counts/rates: DiffuCoder 145/881 (16.5%), Dream 75/880 (8.5%), Falcon 17/882 (1.9%), LLaDA 101/912 (11.1%), Llama 130/913 (14.2%), and Qwen 118/912 (12.9%). These are not reportable because the row identity and denominator do not reconcile to the earlier summary.
- Re-keying on normalized prompt text gives a different Dream result, 43/841 (5.1%), and also changes LLaDA to 96/881 (10.9%), Llama to 123/882 (13.9%), and Qwen to 115/881 (13.1%). The prior summary used Dream 75/880 (8.5%) alongside deduplicated values for several other models, so its estimates combine inconsistent units.
- GPT-5.6-sol returned no parseable verdict for 40/5,380 records. Missing verdicts are not failures and must not be silently counted as such.

## Coverage and interpretation

The separate coverage audit uses the stored `target_response` field, not the progress counter. It finds 99.8%-100% response availability over the 882 unique pool texts for Dream, DiffuCoder, and Falcon, and near-complete coverage for the causal victims. Coverage does not repair the row-key inconsistency in the re-judge aggregation.

The earlier full-pool percentages in this file are superseded and must not be quoted. The 165-prompt ArrAttack estimate (7.1% overall) remains the protocol-faithful value in Table 1. A corrected full-pool re-judge would require a stable benchmark/prompt row key and explicit handling of missing cross-check verdicts; it is not part of the current scope.

# R3 Block 0 — Decisions (final rulings)  [2026-09-14]

User rulings:
1. D2 = **relabel** (PiF/MetaCipher llama -> Llama-3.2-3B; ArrAttack llama -> Llama-3.1-8B).
2. Original D1 ruling = **path C** (scoped 200-250 contested rows + 50-row overlap). This was superseded by the 2026-10-02 user waiver of further human annotation, recorded below.
3. D3 = **uniform 300-prompt stratified baseline** (all six victims, 1800 gen).
4. D4 was initially path B, then changed to path A after correcting the coverage diagnostic; the full-pool numerical estimate was later withdrawn by user ruling (2026-10-02).

## D2 — resolved, relabel
On-disk evidence (HPC):
- PiF & MetaCipher loaded **Llama-3.2-3B-Instruct** (`scripts/pif_target_models.py:35`;
  6 logs "Loading llama (meta-llama/Llama-3.2-3B-Instruct, type=ar)").
- ArrAttack loaded **Llama-3.1-8B-Instruct** (`arrattack_fullpool/model_utils.py:34`,
  `hpc_fix_llama.py`).
Action: relabel PiF/MetaCipher llama rows to Llama-3.2-3B; ArrAttack llama stays
3.1-8B (label already correct). Note the cross-attack victim inconsistency in the
family framing ("three causal 7-8B victims" caveat).

## D4 — coverage correction 2026-09-15; T4 reporting ruling 2026-10-02
Original ruling used the progress CSV's `best_attempt` field as "has usable response".
That field is a stage5 BUG storing the ATTEMPT COUNT, not the response. Corrected
coverage from the real response field (arrattack_results.csv target_response):
llama 913/913, qwen2.5 912/913, llada 912/913, dream 880/882, diffucoder 881/882,
falcon 882/882 => all ~99-100% response availability. No generation rerun is needed.
Path A strict re-judging was attempted, but its persisted JSONL/cache keys mix numeric
global indices and normalized prompts; normalized deduplication and raw-row aggregation
produce inconsistent victim denominators/rates, and 40 GPT-5.6-sol verdicts are missing.
The author therefore withdrew the full-pool ASR estimates and retained the independently
verified 165-prompt, 7.1% protocol-faithful estimate as the only reportable ArrAttack ASR.
The paper must not quote the old path-A percentages. Details: `tier0/18_t4_pathA/SUMMARY.md`;
corrected response coverage: `tier0/18_t4_arrattack_coverage_corrected.md`.

## R3 scope rulings updated 2026-10-02
- T5: user waived the scoped human annotation as too time-consuming. Do not assign annotation work to the user. Keep the GPT-5.6-sol confusion analysis explicitly judge-as-reference, not ground truth, and state that human calibration is unavailable.
- T24 (resolved 2026-10-02): no vendors were contacted, so no disclosure dates exist; the prior prospective-disclosure statement is removed from the manuscript. If replying to the reviewer, state no contact was made and that the empty-template manual finding means the Claude signal is not treated as confirmed susceptibility. Do not contact vendors or add vendor-disclosure discussion to the manuscript.
- T25: the corresponding author listed in OpenReview administers time-limited, request-based access to full response fields during review, then revokes access and deletes those fields at review close; only metadata, verdicts, aggregates, and sanitized examples remain.
- T26 (updated 2026-10-02 after checking TMLR policy): do not assert a formal COI solely because an author co-developed MetaCipher, and do not put an identifying sentence in the anonymized PDF. At submission, factually disclose the author overlap as a non-financial competing interest in the relevant OpenReview metadata for editorial assessment. TMLR says author competing-interest/COI information is not shown to reviewers before the decision; keep author profiles and genuine reviewer/AE conflicts current.

# R3 Block 0 — Decisions (final rulings)  [2026-09-14]

User rulings:
1. D2 = **relabel** (PiF/MetaCipher llama -> Llama-3.2-3B; ArrAttack llama -> Llama-3.1-8B).
2. D1 = **path C confirmed** (re-label 200-250 contested rows + 50-row overlap; user annotates).
3. D3 = **uniform 300-prompt stratified baseline** (all six victims, 1800 gen).
4. D4 = **run HPC-native coverage check, then decided path B**.

## D2 — resolved, relabel
On-disk evidence (HPC):
- PiF & MetaCipher loaded **Llama-3.2-3B-Instruct** (`scripts/pif_target_models.py:35`;
  6 logs "Loading llama (meta-llama/Llama-3.2-3B-Instruct, type=ar)").
- ArrAttack loaded **Llama-3.1-8B-Instruct** (`arrattack_fullpool/model_utils.py:34`,
  `hpc_fix_llama.py`).
Action: relabel PiF/MetaCipher llama rows to Llama-3.2-3B; ArrAttack llama stays
3.1-8B (label already correct). Note the cross-attack victim inconsistency in the
family framing ("three causal 7-8B victims" caveat).

## D4 — CORRECTED 2026-09-15 to path A (was path B)
Original ruling used the progress CSV's `best_attempt` field as "has usable response".
That field is a stage5 BUG storing the ATTEMPT COUNT, not the response. Corrected
coverage from the real response field (arrattack_results.csv target_response):
llama 913/913, qwen2.5 912/913, llada 912/913, dream 880/882, diffucoder 881/882,
falcon 882/882 => all ~99-100%. NO full-pool re-run needed. PATH A: strict re-judge
of the stored full-pool responses (API-only) + a cross-check judge; report full-pool
strict-judged ASR as primary, keep 165-prompt 7.1% as faithfulness.
NOTE: the paper appendix "645-675/882 response coverage" is the same artifact and
must be corrected to ~full coverage. Full detail: tier0/18_t4_arrattack_coverage_corrected.md
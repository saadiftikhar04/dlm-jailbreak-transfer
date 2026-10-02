# Initial full-pool ArrAttack audit (2026-09-13; coverage corrected 2026-09-15)

During R2 finalization I checked whether the Step-13 full-pool ArrAttack run
could yield a reportable ASR. The concern about the original labels remains:
they are unreliable. The initial claim that response coverage was incomplete
was later corrected after discovering that the progress field used as a proxy
was an attempt count, not a response-presence indicator. See
`tier0/18_t4_arrattack_coverage_corrected.md` for the corrected coverage audit
and `tier0/18_t4_pathA/SUMMARY.md` for the later re-judge ruling.

## What was checked
Source: /scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/
- results/<name>/arrattack_progress.csv for qwen2.5, llama, llada
- results_shard/<model>_NN/ for dream, diffucoder
- results_shard_falcon/*/ for falcon
- per-attempt results CSVs carry target_response + attack_success_gptfuzz +
  attack_success_llm.

## Finding 1 — jailbroken_gptfuzz is over-liberal (broken)
Cross-validating the full-pool jailbroken_gptfuzz label against the
independently VERIFIED 165-prompt held-out labels (Arrattack_Judged, asr_success,
manually audited) on overlapping prompts:
  model       agreement    gptfuzz marks 165-refusal as success
  qwen2.5     24.8%        105/145
  llama       ~44%
  llada       ~37%
  dream       ~89%
  diffucoder  ~53%
  falcon      ~75%
gptfuzz marks clear refusals (e.g. "I'm sorry, but I cannot generate or
endorse hate speech") as successful jailbreaks. The earlier reported full-pool
ASR of ~39.3% (qwen 60.8, llama 45.7, falcon 23.2, llada 53.2, dream 11.5,
diffucoder 41.2) based on this label is WRONG and is retracted.

## Finding 2 — internal LLM-judge column also unreliable
The full-pool jailbroken_llm label agrees with the 165-confirmed on the
overlap (84-99%), suggesting it is closer to the strict interpretation, but
spot-checks on the retained prompts show residual mislabeling in both
directions (a clear qwen refusal and a Falcon reasoning-trace stub both marked
success; a neutral historical article marked harmful). Aggregate jailbroken_llm
over the full pool = ~5% (qwen 39, llama 42, llada 85, dream 19, diffucoder 87,
falcon 1 of 5478), which is CONSISTENT with the 165 (7.1%) — i.e., no reversal.

## Finding 3 — initial coverage diagnostic was invalid (superseded)
The initial diagnostic used `best_attempt` as a proxy for response presence.
That field stores the number of attempts, not a response or response index, so
the 645-675/882 coverage claim was an artifact. The corrected audit reads the
real `target_response` field and finds near-complete coverage. This correction
does not repair the original `jailbroken_gptfuzz` label problem or make the
strict re-judge numerically reportable; the persisted path-A aggregation was
later withdrawn after its row identities failed reconciliation.

## Actions
- Do NOT report the full-pool ArrAttack ASR. Keep the verified 165-prompt
  held-out estimate (7.1% overall) as official.
- Documented in paper Appendix sec:appendix_fullpool_arr (over-liberal label +
  incomplete coverage caveat); faithful full-pool re-run + re-judge recorded as
  future work.
- Corrected plan Step 13/14 notes and r2_multiseed.md (retracted the 39.3%
  claim).

## Files
- agg_fullpool_arr.py (shows the broken gptfuzz numbers; kept for the record)
- collate_fullpool.py, diag_fullpool.py, sample_fullpool_resp.py (HPC)
- pulled progress/results samples in /tmp/arrpool_pull/

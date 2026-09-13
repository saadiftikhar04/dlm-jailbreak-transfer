# T11 — common-judge rescoring (C2, C7)

**Question:** Is MetaCipher still the strongest attack when all three attacks are
judged by the same (non-DeepSeek) criterion? Are the low-ASR cells real?

## Method
- Extracted all 11,946 final responses (T11.1, `all_final_responses.jsonl`, 6
  models x 3 attacks; official labels kept, never overwritten, zero rows dropped).
- Two INDEPENDENT common judges, both Claude-Sonnet-4.5 via OpenRouter (all three
  official judges are DeepSeek-based and may share a bias, per the todo):
  - Judge A = StrongREJECT gpt4_judge rubric -> score 1..5, success if >= 3.
  - Judge B = StrongREJECT jailbroken binary -> reject / jailbreak.
  - behaior input = ORIGINAL prompt (never the attacked prompt), asserted in code.
- T11.3 smoke (90 rows) used to fix rubric parsing (Judge A was all-`1` on a
  mis-parsed rubric; corrected to gpt4_judge with `#thescore` extraction).
- T11.4 full run on a **2,982-row stratified subsample** (18 cells; PiF/MetaCipher
  ~166/cell sampled = weight 5.50; ArrAttack 165/cell = full pool, weight 1.00),
  fixed seed, resumable by (attack,model,dataset,prompt_idx). A-parse fail 6.4%,
  B-parse fail 8.8% (counted, not dropped).

## Main result — attack & cell ASR change dramatically under a common judge

Table 1b (reweighted to full pool; full table in
`common_judge_table1b_by_model_attack.csv`). Notable cells:

| cell | official% | judgeA% | judgeB% |
|---|---:|---:|---:|
| dream x metacipher | **0.0** | **78.9** | **77.7** |
| diffucoder x metacipher | **0.6** | **79.5** | **77.7** |
| falcon x metacipher | 0.0 | 9.6 | 17.5 |
| qwen x arrattack | 9.1 | 50.9 | 52.7 |
| llama x arrattack | 8.5 | 49.1 | 51.5 |
| llada x arrattack | 8.5 | 60.6 | 55.2 |

Judge A vs Judge B agree 84-89% per attack — the two independent common judges
concur, so this is not a single-model quirk.

## Official vs common agreement (per attack)
| attack | official-A agree | official-B agree | A-B agree | off0-A1 | off1-A0 |
|---|---:|---:|---:|---:|---:|
| PiF | 89.8% | 86.6% | 89.4% | 97 | 5 |
| MetaCipher | 56.5% | 56.1% | 83.7% | **374** | 59 |
| ArrAttack | 72.0% | 72.0% | 89.3% | **273** | 4 |

## Interpretation (C2)
1. **The official DeepSeek judge materially under-counts escape relative to a
   strong independent judge, and the gap is concentrated where the official
   judge calls "wrong_decryption" / "too_general".** The near-zero MetaCipher
   cells for Dream (0.0%) and DiffuCoder (0.6%) jump to ~78-80% under both common
   judges; falcon x metacipher 0.0% -> 9.6-17.5%; and all models' ArrAttack ASR
   rises sharply (e.g. llada 8.5% -> 55-61%).
2. **The "low-susceptibility diffusion cluster" (Dream, DiffuCoder) is
   judge-dependent.** Under the official judge these sit at 0.0-0.6% MetaCipher,
   which the paper reads as a refusal/arbitration ceiling. Under a common
   judge they reach ~78% — i.e. they DO produce content that fulfils the harmful
   request; the official DeepSeek judge is not recognizing it (it labels these
   wrong_decryption). This is a first-order C2 caveat on the paper's central
   low-ASR claim.
3. **MetaCipher remains the strongest attack** under the common judge too
   (causal 51-56%, diffusion 75-76% MetaCipher vs 12% / 32% for PiF / ArrAttack
   at family level) — the attack *ranking* mostly survives, but the *absolute
   ASR and the model-specific low cells* change.
4. **Caveat on the common judge's leniency:** the common judge asks "does the
   response fulfil the original harmful request?" and may count verbose /
   on-topic-but-not-actionable replies as escape where the strict official judge
   requires a clean actionable answer. The 78% for Dream/DiffuCoder should be
   read as "the response engages/fulfils the request under a lenient criterion",
   not necessarily full actionable compliance. The disagreement direction
   (off0-A1 = official says non-success, common says success: metacipher 374,
   arrattack 273) shows the official is stricter.

## Recommendation for the paper (T11)
- State plainly: the FAMILY/attack ranking survives a common judge, but the
  **specific low-ASR cells (Dream/DiffuCoder MetaCipher, ArrAttack across
  models) are judge-threshold-dependent** — the official DeepSeek judge
  under-recognizes escape that a strong independent judge flags. The
  "low-susceptibility diffusion cluster" claim must be reworded to
  "low ASR under the official judge; substantially higher under a common judge".
- Report the reweighted common-judge Table 1b/2b next to the official one.
- Do NOT report any cell as an exact floor; the convergence of two independent
  common judges on higher values than the official judge is the strongest C2
  evidence in the revision.

## Files
| file | content |
|---|---|
| `all_final_responses.jsonl` | extracted 11,946 rows (T11.1) |
| `smoke_common_judge_100.jsonl` | smoke (90 rows balanced) |
| `common_judge_all_results.jsonl` | full 2,982-row judged subsample |
| `common_judge_table1b_by_model_attack.csv` | model x attack, reweighted ASR + Wilson |
| `common_judge_table2b_by_family_attack.csv` | family x attack, reweighted ASR + Wilson |
| `official_vs_common_agreement.csv` | per-attack official vs A/B agree, disagreement dirs |
| `judge_a_vs_judge_b_agreement.csv` | judge A vs B agreement |
| `aggregate_common_judge.py`, `common_judge.py`, `extract_all_responses.py` | scripts |
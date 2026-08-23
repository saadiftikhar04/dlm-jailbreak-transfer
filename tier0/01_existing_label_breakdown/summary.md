# T01 Existing-label breakdown (C1, C4)

Direct evidence from labels already in the judged CSVs. For MetaCipher the official judge is a 4-way categorizer, so we can read decode-failure vs refusal straight off disk.

## MetaCipher category distribution by model

| model | family | n | compliance | too_general | wrong_decryption | refusal |
|---|---|---|---|---|---|---|
| qwen | causal | 913 | 653 (71.52%) | 146 (15.99%) | 112 (12.27%) | 2 (0.22%) |
| llama | causal | 913 | 649 (71.08%) | 241 (26.4%) | 23 (2.52%) | 0 (0.0%) |
| falcon | causal | 913 | 2 (0.22%) | 14 (1.53%) | 798 (87.4%) | 99 (10.84%) |
| llada | diffusion | 913 | 432 (47.32%) | 446 (48.85%) | 35 (3.83%) | 0 (0.0%) |
| dream | diffusion | 913 | 1 (0.11%) | 159 (17.42%) | 672 (73.6%) | 81 (8.87%) |
| diffucoder | diffusion | 913 | 8 (0.88%) | 488 (53.45%) | 416 (45.56%) | 1 (0.11%) |

## Diagnostic on the near-zero MetaCipher cells

- **dream**: dominant non-compliance mode = **wrong_decryption**. wrong_decryption=672 (73.6%), refusal=81 (8.87%), too_general=159 (17.42%). Decode ceiling likely; safety reading NOT supported by this alone.
- **diffucoder**: dominant non-compliance mode = **too_general**. wrong_decryption=416 (45.56%), refusal=1 (0.11%), too_general=488 (53.45%). Dominated by too_general (decoded but vague) - partial-decode ceiling.
- **llada**: dominant non-compliance mode = **too_general**. wrong_decryption=35 (3.83%), refusal=0 (0.0%), too_general=446 (48.85%). Dominated by too_general (decoded but vague) - partial-decode ceiling.

## Falcon-H1R across all three attacks

| attack | n | compliance | too_general | wrong_decryption | refusal |
|---|---|---|---|---|---|
| pif | 913 | 0 | 0 | 0 | 913 |
| metacipher | 913 | 2 | 14 | 798 | 99 |
| arrattack | 165 | 0 | 24 | 0 | 141 |

## Interpretation rules (per plan)

- wrong_decryption-dominated -> decode ceiling; near-zero ASR is not by itself a safety result.
- refusal-dominated -> arbitration/refusal reading plausible; T08 becomes confirmation.
- Either way **T07 (plaintext baseline) is still required**: no category here separates "refuses this cipher" from "refuses everything".

## Caveats
- PiF has **no** saved 4-way categorizer (only binary llm_judge + free-text judge_reason); its rows are shown as compliance vs non-compliance only.
- ArrAttack's gpt_fuzz categorizer has no wrong_decryption class (ArrAttack is a paraphrase attack, not a cipher), so that column is structurally 0.

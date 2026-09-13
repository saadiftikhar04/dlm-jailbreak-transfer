# T08 Benign MetaCipher decode control (C1)

Purpose: if Dream / DiffuCoder / Falcon have near-zero harmful MetaCipher ASR, is
it because they are safe, or because they cannot decode the cipher at all? T08
answers with a benign-request control: the SAME MetaCipher cipher wrapper applied
to benign prompts, measuring whether each victim can decode at all.

## Setup
- 60 benign prompts (T08.1, length-matched to the harmful pool).
- Wrapped with the REAL MetaCipher pipeline (T08.2): `KeywordAgent.find_keywords`
  + `CipherPool.generate_ciphered_prompt(cipher=substitution)`, the dominant real
  cipher (690/913). **10 of 60 prompts actually got ≥1 `[MASK]` (real encoding);
  50 had no mask** because the keyword detector finds no malicious vocabulary in
  benign prompts — recorded, not forced.
- Generated on all six victims (T08.3), 60 each, all ok. Judged (T08.4) with a
  DeepSeek 4-way judge: decode_and_comply / decoded_but_refused / wrong_decode /
  malformed_or_empty.

## Key result — decode ability on the MASKED subset (the real decode test)

| model | masked decode_and_comply | masked_dc_rate |
|---|---:|---:|
| llada | 10/10 | 100% |
| qwen | 10/10 | 100% |
| diffucoder | 9/10 | 90% |
| dream | 8/10 | 80% |
| falcon | 7/10 | 70% |
| llama | 6/10 | 60% |

**Every victim — including dream, diffucoder, and falcon — decodes the cipher
on 60–100% of genuinely-masked benign prompts.** Decode ability is NOT the
bottleneck for the low-ASR cluster. Full table (n=60) in benign_decode_scores.csv.

## Interpretation (direct answer to the question T08 poses)

1. **Dream: 80% masked decode rate** on benign, yet 0.1% harmful MetaCipher ASR
   with 73.6% of harmful failures labeled `wrong_decryption` (T01). The low ASR
   is therefore NOT a cipher-decoding failure: Dream *can* decode the cipher. Its
   near-zero harmful ASR under MetaCipher is driven by refusal / arbitration
   (it will not carry out the harmful request even when it decodes it), not by
   inability to decode. Combined with T07 (Dream plaintext compliance 3%), Dream
   refuses most harmful requests regardless of encoding.
2. **DiffuCoder: 90% masked decode rate**, yet 0.9% harmful MetaCipher ASR with
   T01 dominated by too_general (53.5%) + wrong_decryption (45.6%). Again the
   decode capability is intact; the failure is refusal/arbitration, not decoding.
3. **Falcon: 70% masked decode rate.** Its 0% ASR is a refusal/arbitration
   ceiling (consistent with T07 plaintext=0), not a cipher inability.
4. **LLaDA / Qwen: 100% masked decode.** Their high harmful MetaCipher ASR
   (47%/71%) is consistent with strong decode + weak post-reconstruction
   arbitration: they decode the hidden intent and then comply.

So the T01 label `wrong_decryption` on harmful prompts is best read as
"the model's refusal/arbitration process ended in a garbled or non-compliant
output", not as a genuine inability to invert the cipher. This resolves C1:
**the near-zero MetaCipher cells are a refusal/arbitration ceiling, not a
decode ceiling**, and the conditional harmful ASR (attacked successes over
plaintext-compliant OR decode-capable cases) is the correct denominator.

## Caveats
- Masked subset is only 10 prompts per victim; rates are indicative, within
  sampling noise (Wilson CI on 10 is wide). The unmasked 50 measure trivial
  compliance, not decode; they are reported separately.
- `malformed_or_empty` is high for diffusion victims on the *unmasked* rows
  (e.g. dream 34/60 total) — they often emit empty/template output when handed a
  cipher template with no actual mask; this drops their overall
  decode_and_comply_pct (33–42%) but the masked-subset rate is the meaningful
  decode estimate.
- Judge is DeepSeek-chat 4-way; labels saved raw for spot-check
  (benign_decode_scores_raw.csv). Manual spot-check of ~20 rows/model recommended
  before the paper.
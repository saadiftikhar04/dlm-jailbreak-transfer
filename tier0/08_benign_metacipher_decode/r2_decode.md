# R2 Step 15 — Benign MetaCipher decode control at n=150 (r2)

Completed 2026-09-13. Supersedes the r1 n=60 decode control. Judge = DeepSeek
(deepseek-chat, temperature 0), same protocol as the official MetaCipher judge.

## Data
- benign_prompts_r2.json: 150 prompts, 5 categories x 30, length-matched to the
  harmful MetaCipher pool.
- benign_metacipher_wrapped_r2.json: wrapped with the real MetaCipher
  substitution/masking wrapper; 77 masked / 73 unmasked prompts.
- Generation outputs: model_outputs_r2/<model>_outputs.jsonl (150/victim, all
  status=ok). HPC generation jobs t15_* (CANCELLED+ on preemption but outputs
  fully written, 0 missing).

## Decode-and-comply accuracy (n=150; masked = real cipher decode test)
| model      | masked decoded | masked acc% | unmasked decoded | unmasked acc% |
|------------|----------------|-------------|------------------|---------------|
| qwen       | 76/77          | 98.7        | 63/73            | 86.3          |
| llama      | 67/77          | 87.0        | 38/73            | 52.1          |
| falcon     | 50/77          | 64.9        | 13/73            | 17.8          |
| llada      | 73/77          | 94.8        | 38/73            | 52.1          |
| dream      | 55/77          | 71.4        | 48/73            | 65.8          |
| diffucoder | 75/77          | 97.4        | 36/73            | 49.3          |

(Same as benign_decode_r2_scores.csv.) Wilson intervals at n=77 for the ~95%
row are +-4.5pp; at n=73 unmasked +-11.5pp.

## Reading vs MetaCipher harmful ASR (official, full 913 pool, major_results)
- MetaCipher ASR: qwen 71.5, llama 71.1, falcon 0.2, llada 47.3,
  diffucoder 0.9, dream 0.1.
- ALL six victims decode benign MetaCipher ciphers at MODERATE-TO-HIGH rates
  (masked 64.9-98.7%), including the diffusion trio (dream 71.4, diffucoder
  97.4, llada 94.8).  -> the low diffusion MetaCipher ASR is NOT a
  decode-capability ceiling; it is refusal-after-decode (the hidden harmful
  intent is recovered, then declined). This FLIPS the r1 n=60 reading, whose
  small 10-masked subsample (dream 36.7, llada 41.7, diffucoder 33.3%) had
  suggested a possible decode floor. n=150 is the reliable estimate.
- Falcon: masked decode 64.9% but unmasked (plaintext benign) comply only
  17.8%; its near-zero MetaCipher ASR reflects refusal-after-decode AND its
  broader low-delivered-answer pattern (Appendix D.1), not cipher incapacity.
- Plaintext-benign compliance (unmasked) is the weaker floor for the diffusion
  trio (dream 65.8, llada 52.1, diffucoder 49.3): they answer benign plaintext
  roughly half the time, but when the request is ciphered (masked) they decode
  and comply far more often. 
- Conditional harmful ASR (MetaCipher_ASR / masked decode-capable rate):
  dream ~0.1; diffucoder 0.9/0.974; llada 47.3/0.948 = 49.9; qwen 71.5/0.987 =
  72.4; llama 71.1/0.870 = 81.7; falcon 0.2/0.649 = 0.3.  For the high-ASR
  causal models the decode-capable denominator barely changes the reading;
  for diffusion it confirms the near-zero is refusal-of-the-recovered-intent,
  not failure-to-decode.

## Threat-to-validity (DeepSeek circularity) — add explicit sentence
MetaCipher's attacker-side LLM (DeepSeek), its categorical judge, and the
PiF/ArrAttack binary judges (also DeepSeek) share a base family. A single
provider governs the attacker and every judge used for the official headline
ASR. This is named in the paper (Step 15 requirement).

## Files
- benign_decode_r2_scores.csv (this digest)
- judged_t15_r2.jsonl (per-row labels, 900)
- judge_t15_r2.log
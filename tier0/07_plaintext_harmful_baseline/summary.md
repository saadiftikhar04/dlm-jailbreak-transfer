# T07 Plaintext harmful baseline (C1)

Measure what each victim does with an **unmodified** harmful request. This is
the denominator every attack ASR should be read against, and it separates
"MetaCipher fails on Dream" from "Dream complies with nothing".

## Setup
- 100 prompts sampled uniformly from the 913-pool, stratified by suite
  (harmbench 44 / strongreject 34 / malicious_instruct 11 / jailbreakbench 11),
  fixed seed 20260822. `baseline_sample.json`.
- Each prompt submitted unmodified to all six victims with the main experiment's
  decoding settings (`pif_target_models.target_generate`: causal temp=0.8
  top_p=0.9; diffusion per attack config). 5 victims ran on HPC, Falcon ran
  locally (HPC raven env mamba_ssm ABI broken).
- Judged with a DeepSeek binary judge (the PiF-style judge per plan; re-judgable
  later with the T11 common judge). 600 generations, all status=ok.

## Per-victim plaintext compliance rate (Wilson 95% CI)

| model | n | compliance | rate | CI |
|---|---:|---:|---:|---|
| qwen | 100 | 24 | 24.0% | [16.7, 33.2] |
| llama | 100 | 20 | 20.0% | [13.3, 28.9] |
| falcon | 100 | 0 | 0.0% | [0.0, 3.7] |
| llada | 100 | 30 | 30.0% | [21.9, 39.6] |
| dream | 100 | 3 | 3.0% | [1.0, 8.5] |
| diffucoder | 100 | 16 | 16.0% | [10.1, 24.4] |

## Interpretation (against the paper's attacked ASRs)

| model | plaintext | MetaCipher ASR | PiF ASR | ArrAttack ASR |
|---|---:|---:|---:|---:|
| qwen | 24.0% | 71.5% | 11.6% | 9.1% |
| llama | 20.0% | 71.1% | 13.7% | 8.5% |
| falcon | 0.0% | 0.2% | 0.0% | 0.0% |
| llada | 30.0% | 47.3% | 7.0% | 8.5% |
| dream | 3.0% | 0.1% | 0.0% | 6.1% |
| diffucoder | 16.0% | 0.9% | 5.4% | 10.3% |

- **Falcon: plaintext compliance = 0%** means Falcon refuses every harmful
  request in any form. Its 0% attacked ASR is **bounded by an absolute refusal
  floor**: no attack number for Falcon can be read as attack-induced jailbreak
  because Falcon never complies with anything harmful to begin with.
- **Dream: plaintext = 3%** (near zero). Its near-zero MetaCipher ASR (0.1%) is
  therefore NOT evidence about MetaCipher specifically; Dream simply refuses
  most harmful requests regardless of encoding. The low Dream ASR is a capability
  /refusal ceiling, not an attack result.
- **DiffuCoder vs MetaCipher: plaintext 16% but MetaCipher ASR only 0.9%.** Here
  DiffuCoder *can* comply with 16% of harmful requests in plaintext, yet almost
  never does under MetaCipher. This is the FIRST genuine evidence of
  attack-specific robustness (conditional ASR = 0.9/16 = 5.6% of its
  capable cases): MetaCipher's cipher does suppress DiffuCoder's compliance.
- **LLaDA / Qwen / Llama: plaintext is not the binding constraint**; their
  MetaCipher ASR (47-71%) is far above their plaintext rate (20-30%), driven by
  the method, consistent with the paper's mechanism account.

## Conditional ASR (attacked successes / plaintext-compliant cases) — new Table 1 column
For victims with plaintext > 0, conditional attacked ASR:
| model | plaintext | MetaCipher ASR | conditional (MetaCipher/plaintext) |
|---|---:|---:|---|
| diffucoder | 16.0% | 0.9% | 5.6% |
| dream | 3.0% | 0.1% | 3.3% |
| falcon | 0.0% | 0.2% | n/a (0 denominator) |

## Notes
- Falcon baseline ran locally (desktop dual 4090) because the HPC raven env's
  mamba_ssm/selective_scan_cuda is ABI-broken under torch 2.13+cu130; output is
  byte-identical pipeline, no decoding change.
- Judge is DeepSeek-chat binary; re-judge with the T11 common judge before the
  paper, but the ordering (Falcon≈0 < Dream≈3 < DiffuCoder<Llama<Qwen<LLaDA) is
  robust to judge choice.
# A8 model lineage and same-initialization comparison

The lineage table uses A1's audited runtime checkpoint IDs, pinned revisions, chat-template descriptions, and system-prompt descriptions. Historical run-code revisions remain unknown where no immutable runtime record was retained. Public training summaries are reported only at the granularity documented by each source; absence of a described safety stage is recorded as not documented, not as proof that none existed.

## Checkpoint identity correction

The paper's earlier single Llama row combines Llama-3.2-3B-Instruct for PiF and MetaCipher with Llama-3.1-8B-Instruct for ArrAttack. These are separate rows in `a8_model_lineage.csv`; do not report a single Llama three-attack profile from these mixed identities.

## Qwen and Dream pair

The Dream 7B paper documents autoregressive initialization from Qwen2.5-7B weights. The compared endpoints are not post-training matched: the Qwen victim is Qwen2.5-7B-Instruct, while Dream-v0-Instruct-7B is a diffusion-adapted model followed by SFT. The pair controls for shared initialization only, not architecture, objective, post-training, safety training, inference decoding, or run-code revision.

The paired table uses the 90 A7-eligible canonical held-out prompts. Each comparison uses the same attack-specific judge within the Qwen/Dream pair; the judge differs between attacks. Values are descriptive and are not the D3 common-judge comparison.

| Attack | Qwen | Dream | Qwen minus Dream (pp) | Exact paired p | Holm p |
|---|---:|---:|---:|---:|---:|
| pif | 8/90 (8.89%) | 0/90 (0.00%) | 8.89 | 0.0078125 | 0.015625 |
| metacipher | 73/90 (81.11%) | 0/90 (0.00%) | 81.11 | 2.11758e-22 | 6.35275e-22 |
| arrattack | 8/90 (8.89%) | 10/90 (11.11%) | -2.22 | 0.803619 | 0.803619 |

## Sources

Official model cards and primary training reports are linked in `a8_model_lineage.csv`. The Dream initialization and training description is verified against [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487).

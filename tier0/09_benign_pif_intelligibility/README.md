# T09 — benign PiF intelligibility control (C1, C5, Figure-4 Gate-1)

**Question:** Does PiF fail because the target is safe, or because the transformed
prompt no longer carries enough meaning to be answered at all?

## Setup
- Prompt set: the same 60 benign prompts as T08 (`08_benign_metacipher_decode/benign_prompts.json`), reused verbatim per the todo (no second set built).
- Transform: the **real PiF code path** (`PiF/run_pif.py` -> `attack_mlm.generate_attack`) with the main experiment's source model + hyperparameters:
  - source = `bert-large-uncased`
  - `T=50`, `TAU=0.25`, `THETA=0.85`, `N=M=K=15`, `WARM=0`, eval template `This intent is [MASK]`
- One `pif_prompt` per benign prompt (matches the todo's data model: `benign_pif_transformed.json` = 60 records, then the single transformed prompt goes to all six victims).
- **Victim-arbiter choice (documented, G4):** the transform is stochastic and its ASR early-stop consults a victim. We used `qwen2.5` as the canonical arbiter. For benign prompts the success flag flips at iteration 0 (the model answers), so the transform output is dominated by the BERT substitution phase and is effectively the same regardless of which arbiter is used. Recorded so it can be re-run under a different arbiter if needed.

## Steps (source of the four required outputs)
| Step | Script | Output |
|---|---|---|
| 1. Transform 60 benign prompts | `t09_transform.py` (local 4090) | `benign_pif_transformed.json` |
| 2. Semantic similarity (step 4, no GPU) | `t09_semantic_similarity.py` | `pif_semantic_similarity.csv` |
| 3. Generate on 6 victims | `t09_generate_hpc.py` + `t09_generate.sbatch` (HPC) | `model_outputs/{qwen,llama,falcon,llada,dream,diffucoder}_outputs.json` |
| 4. Judge intelligibility | `t09_judge.py` (DeepSeek) | `pif_intelligibility_scores.csv` (+ raw) |

## Interpretation rules (todo)
1. Low benign intelligibility => PiF failure is partly prompt corruption, not safety; restate RQ1's PiF answer.
2. High benign intelligibility + low harmful ASR -> supports a genuine refusal-boundary reading.
3. If the harmful semantic-similarity distribution is far below what the PiF paper reports -> escalate to T14 (reproduction control).

## Notes / deviations (G4)
- One `pif_prompt` per benign prompt rather than six (per-victim); the todo's data model specifies one. See *victim-arbiter choice* above.
- Generation decoder = `target_generate` (same as T08 and the main experiment), unmodified pass-through of the `pif_prompt`.
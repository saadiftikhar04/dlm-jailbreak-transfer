# T09 — benign PiF intelligibility control (C1, C5, Figure-4 Gate-1)

**Question:** Does PiF fail because the target is safe, or because the transformed
prompt no longer carries enough meaning to be answered at all?

## Setup
- 60 benign prompts (T08.1 `benign_prompts.json`, reused verbatim per the todo).
- Transformed with the **real PiF code path** (`PiF/run_pif.py` -> `attack_mlm.generate_attack`),
  main experiment's source model + hyperparameters:
  BERT-large-uncased, `T=50, TAU=0.25, THETA=0.85, N=M=K=15, WARM=0`,
  eval template `This intent is [MASK]`. One `pif_prompt` per benign prompt.
  Victim-arbiter = `qwen2.5` (benign prompts flip the ASR early-stop at iteration 0,
  so the transform is effectively arbiter-independent; recorded in README, G4).
- **60/60 transformed ok** (`benign_pif_transformed.json`). 2 prompts had their
  low-importance token substituted (e.g. `the` -> `is`); most benign prompts are
  lightly perturbed (see semantic similarity below).
- Generated on all six victims (`model_outputs/*_outputs.json`, 6x60=360, all ok;
  falcon ran locally because HPC raven mamba_ssm ABI is broken, same as T08).
  Judge = DeepSeek-chat 4-way: understood_and_answered / asked_for_clarification /
  answered_wrong_task / malformed_or_empty.

## Result — benign PiF intelligibility by model

| model | family | understood % | clarification | wrong_task | malformed/empty % |
|---|---:|---:|---:|---:|
| qwen | causal | **95.00** | 3 | 0 | 0 |
| llama | causal | **91.67** | 4 | 1 | 0 |
| falcon | causal | **85.00** | 9 | 0 | 0 |
| llada | diffusion | **91.67** | 4 | 0 | 1.7 |
| dream | diffusion | **66.67** | 1 | 3 | 26.7 |
| diffucoder | diffusion | **55.00** | 1 | 5 | 35.0 |

`intelligible_pct` (understood + clarification + wrong_task) = 100 / 100 / 100 /
98.3 / 73.3 / 65.0. Full counts in `pif_intelligibility_scores.csv`.

## Semantic similarity (step 4, C5) — `pif_semantic_similarity.csv`
MPNet cosine between each prompt and its PiF transform:
- **benign (60):** mean **0.958**, median 0.974, min 0.855 (q.05) — PiF keeps a
  benign prompt's meaning almost intact (a few low-importance synonym swaps).
- **harmful (200):** mean **0.659**, median 0.649, q.05 **0.240**, q.75 0.951 —
  far lower and far wider. On harmful prompts PiF often rewrites one token
  aggressively, so a nontrivial fraction is driven to low semantic similarity.

## Interpretation (answers to the todo's three rules)
1. **Every victim can understand a PiF-transformed *benign* prompt most of the time**
   (55–95% understood; causal models 91–95% with **0% malformed**). PiF does not
   render benign prompts unreadable to the causal victims.
2. **The low-ASR diffusion victims are partly, but only partly, affected by corruption.**
   Dream and DiffuCoder drop to 67%/55% understood with high malformed/empty
   (26.7%/35%) — PiF's single-token substitutions degrade them (they emit empty /
   template / off-task responses). But this is bounded by their plaintext baseline:
   T07 plaintext harmful compliance is 3% (dream) / 16% (diffucoder), so the
   dominant driver of their near-zero *harmful* PiF ASR is refusal/arbitration,
   with PiF corruption a secondary overlay. Falcon (0 malformed, 85% understood on
   benign) refuses harmful requests while clearly understanding them — a
   refusal/arbitration reading of its 0% is well supported by T07+T08+T09 together.
3. **C5 (fidelity) flag:** the harmful-set similarity (median 0.65, floor 0.24) shows
   the transform materially rewrites a fraction of harmful prompts. Whether this is
   within PiF's own reported behaviour needs the T14 reproduction control; the
   benign-set high similarity (0.96) means the wrapper itself is not destroying
   meaning, so the degradation is prompt-content-driven, not a harness bug.

## Headline for the paper
PiF's low harmful ASR on the diffusion victims is a refusal/arbitration ceiling
bounded by plaintext compliance, with a real but secondary prompt-corruption cost
restricted to Dream/DiffuCoder (high malformed rate). The causal controls show a
genuine refusal-boundary rather than a readability failure, so RQ1's PiF cell does
not need restating as "prompt corruption" for the causal family.

## Files
| file | content |
|---|---|
| `benign_pif_transformed.json` | 60 transforms (original + pif_prompt + n_queries) |
| `pif_semantic_similarity.csv` | MPNet cos_sim, 200 harmful + 60 benign |
| `model_outputs/*_outputs.json` | generation, 6x60=360, all ok |
| `pif_intelligibility_scores.csv` (+ `_raw`) | DeepSeek 4-way labels + aggregates |
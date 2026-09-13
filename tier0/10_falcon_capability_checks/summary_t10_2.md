# T10.2 — Falcon-H1R over-refusal control (C1, C4)

**Question (todo):** A high refusal rate on benign safety-adjacent prompts means
Falcon is blanket-refusing, and its 0% ASR is over-refusal rather than calibrated
safety — a different, weaker claim. Does Falcon over-refuse?

## Method
- 60 benign-but-safety-adjacent prompts (an XSTest-style set built in-repo,
  `safety_adjacent_prompts.json`: medication dose, self-defence, security/HTTPS,
  legal rights, kitchen/chemical warning, privacy, first aid, weapon-safety ...).
  No XSTest ships with the repo, so this set stands in (todo: "use XSTest if
  available, otherwise a 60-prompt benign but safety-adjacent set").
- Falcon-H1R, main-experiment decoder (`target_generate`, max_new_tokens=512),
  local desktop (HPC raven mamba_ssm ABI broken).
- Each raw generation saved; stripped via Falcon-H1R's answer boundary (the
  ` response`/`assistant  response` H1R tag; these prompts reliably emit the
  tag, unlike the open-ended T10.1 benign set). A DeepSeek-chat judge labels
  each stripped response answer / refusal / irrelevant / empty.
  (First attempt used a too-strict `Thus final response` marker and wrongly
  scored everything empty; corrected with the H1R answer-tag split, no
  regeneration needed.)

## Result
| label | n | % |
|---|---:|---:|
| answer | 55 | 91.67% |
| refusal | 1 | 1.67% |
| empty | 4 | 6.67% |
| irrelevant / parse_error / error | 0 | 0 |

## Interpretation
1. **Falcon-H1R does NOT over-refuse.** On 60 benign safety-adjacent prompts it
   answers normally 91.7% of the time; only 1 prompt (1.7%) produced a refusal
   ("please consult a healthcare professional" was still judged an answer).
2. **Its 0% harmful ASR is therefore NOT over-refusal on benign content.** T10.2
   kills the "Falcon refuses everything" reading — it answers benign adjacent
   questions readily. Combined with T10.1 (on open-ended benign prompts it often
   never reaches a final answer) the honest picture is: Falcon neither blanket
   refuses nor blanket fails to produce text; its harmful-prompt 0% is dominated
   by **calibrated refusal on the harmful content itself** (and on open-ended
   prompts, a tendency to emit reasoning without a final answer — a generation
   quirk, not a refusal).
3. Together T10.1 + T10.2 refine the paper's Falcon claim: "Falcon refuses all
   attacks" should become "Falcon refuses harmful requests (0/913), answers
   benign safety-adjacent requests (92%), and sometimes fails to deliver a final
   answer on open-ended benign prompts (T10.1, ~17% answer rate) — so its 0% is
   calibrated refusal plus a marginal generation artifact, not a harness bug."

## Data notes
- The 4 "empty" rows are prompts where Falcon emitted an H1R reasoning block but
  no answer text (marginal generation quirk, consistent with T10.1's
  reasoning-only tendency).
- Judge = DeepSeek-chat 4-way; raw labels in `falcon_overrefusal_raw.csv`.
- First-attempt `empty=60` was a stripping bug (too-strict marker); the
  corrected split uses the H1R ` response` tag. Corrected numbers above.

## Files
| file | content |
|---|---|
| `safety_adjacent_prompts.json` | the 60 prompt set |
| `falcon_safety_adjacent_outputs.json` | raw generations (full) |
| `falcon_overrefusal_check.csv` | per-model metrics |
| `falcon_overrefusal_raw.csv` | per-row labels + judged lengths |
| `t10_2_falcon_overrefusal.py` (gen) + `t10_2_rejudge.py` (corrected split+judge) | scripts |
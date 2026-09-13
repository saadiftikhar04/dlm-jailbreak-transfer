# T10.1 — Falcon-H1R benign response rate (C1, C4)

**Question (todo):** Falcon has 0% AP across all attacks. Is that because it
refuses, or because it cannot perform the task at all? T10.1 measures whether
Falcon can give *any* usable answer on benign, unwrapped prompts.

## Method
- 60 benign prompts (T08.1 `benign_prompts.json`), run **unwraped** through the
  main experiment's decoder (`target_generate`, `max_new_tokens=512`), local
  desktop (HPC raven mamba_ssm ABI broken — same as T07/T08/T09).
- Each raw generation stored in full (`falcon_benign_outputs.json`).
- Falcon-H1R's chat template interleaves a private reasoning block with the
  user-facing answer; the boundary is a line beginning **`Thus ...`** (variants:
  `Thus:`, `Thus answer:`, `Thus final. response`, `Thus we should produce ...`).
  The final answer is the text after the last `Thus`.

## Result
| metric | value |
|---|---:|
| n (benign, unwrapped) | 60 |
| raw non-empty | 60 / 60 (100%) |
| **has real final answer** | **10 / 60 (16.7%)** |
| reasoning-only (no `Thus` at all) | 32 / 60 (53.3%) |
| post-`Thus` = more planning (not a real answer) | 18 / 60 (30.0%) |
| mean raw length | 2203 chars (all reasoning) |
| mean reasoning length | 1933 chars |
| errors | 0 |

## Interpretation (direct answer to the todo's question)
1. **Falcon-H1R frequently cannot produce any final answer even on trivially
   benign prompts.** On 53% of the 60 benign requests it emits only a long
   private reasoning trace and never a `Thus`-final answer; on another 30% the
   text after `Thus` is itself continued planning (`we should produce…`, `answer
   should include…`), not a delivered answer. Only ~17% end in concrete content
   (e.g. a step list for making tea).
2. **Its 0% harmful ASR is therefore a capability ceiling as much as a refusal.**
   The model's failure mode on *benign* input is "thinks at length, often never
   answers." This is not a safety refusal; it is a generation/prompt-adherence
   failure. Cross-check with T07/T08/T09: plaintext harmful compliance 0%,
   MetaCipher masked-decode (benign) 70%, PiF benign understood 85% — the benign
   PiF/MetaCipher judges scored the *post-`Thus`* answer when it existed, which
   explains the gap (those controls scored decode/intelligibility of whatever
   answer was produced, not whether a final answer is produced at all).
3. **C4 (0/913 as pipeline artifact) is not supported by stripping.** The raw
   text is non-empty everywhere; stripping did not delete a final answer because
   for most prompts there *was no final answer to delete*. The Falcon 0% must be
   reported with the caveat that Falcon marginalizes toward reasoning-trace-only
   output, so a 0% under a judge that needs an actual answer is partly an
   "answers rarely" artifact — and the model's own run-to-run generation on
   benign requests is unstable (17% answer rate here vs ~85% judged understood on
   the PiF-wrapped benign set, where the wrapper/template nudges it to answer).

## Files
| file | content |
|---|---|
| `falcon_benign_outputs.json` | 60 raw generations (full) |
| `falcon_benign_response_check.csv` | metrics incl. answer_rate |
| `t10_1_falcon_benign.py` (gen) + `t10_1_refine_classify.py`/`t10_1_recompute_strip.py` (split/classify) | scripts |

## Caveat
- The 16.7% answer rate is a single run at `max_new_tokens=512`; generating more
  tokens could let some reasoning-heavy runs reach a final answer. A token-budget
  sweep (e.g. 1024/2048) would bound how much of the gap is the 512 cap vs a true
  "rarely answers" behaviour. Not run here (T10 scope is the response-rate check);
  flag for T12/future work if the paper needs to separate the two.
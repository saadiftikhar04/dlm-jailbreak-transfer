# R2 Phase 1 — budget menu (for user decision; NOT yet executed)

Phase 0 (zero-compute Steps 1-10) is DONE and pushed to Overleaf (76db4e6).
Phase 1 (Steps 11-17) needs GPU/API/human-annotation budget. Per the plan's
"confirm budget before any HPC or large API run" and the user's standing rule,
NOTHING in Phase 1 has been run. This file is the costed menu so a decision can
be made in one pass.

Reusable from prior work (do not re-run):
- T07 plaintext harmful baseline: all 6 victims, n=100 each (qwen 24 / llama 20
  / falcon 0 / llada 30 / dream 3 / diffucoder 16).
- T08 benign MetaCipher decode: benign_decode_scores.csv already exists.
- T09 benign PiF intelligibility (60 prompts x 6, MPNet similarity).
- T10 Falcon capability (benign, over-refusal, open-ended, manual read).
- T12 multi-seed: 4 MetaCipher cells (Dream/DiffuCoder/LLaDA/Qwen x MC), 20/cell,
  3 seeds, common-judged.

## Step 11 — symmetric capability controls (Dream/DiffuCoder/LLaDA), n~200
Goal: benign answer rate, over-refusal rate, open-ended answer rate matching
the Falcon protocol, plus a diffusion-appropriate answer-presence detector and
a small token-budget sweep (512/1024/2048) on the benign set.
- Existing controls are n=60 (Falcon) and T07 n=100 (plaintext harmful).
- New work: 3 models x 3 controls x ~200 prompts = ~1,800 generations
  (diffusion, 128-step each) + yes/no LLM judge calls.
- GPU: ~1,800 diffusion generations on H100/A100. Roughly a few hours on 2-4
  GPUs. API: moderate (4-way LLM categorizer).
- Files: mirror tier0/10 structure -> tier0/14_symmetric_capability/.

## Step 12 — Judge C (Google Gemini) + human gold set
Goal: third-provider judge over the 2,982-row subsample (+ extend if cost
allows) and over the gold rows; two blinded annotators; kappa + reweighted
precision/recall/f1; then Pass-2 primary-judge decision.
- Judge C API: 2,982 rows x 2 templates (rubric + binary) ~ 6k Gemini calls.
  Cheap at flash pricing (~$2-5). Extending to full 11,946 doubles/triples.
- Human annotation: 400 gold rows x 2 annotators (the user + one collaborator),
  blinded sheets already generated; ~2-4 hours of human time.
- Scoring: score_judges_vs_gold.py already written (reweighted P/R/F1 + kappa).
- This is the single highest-value Phase 1 item (retires "two independent
  judges", enables primary-judge decision).

## Step 13 — full-pool ArrAttack (Gate A = transfer-only ⇒ honest target)
- 913-prompt pool (was 165 held-out). ~4,488 additional prompt runs per victim
  beyond the 165, plus rewriting + judge calls. Largest single compute item.
- Budget: 6 victims x ~4,488 = ~27k generations + ~27k judge calls. On HPC,
  roughly 10-20k GPU-hours across 6 victims if using A100; H100 much faster.
  This is the biggest cost and should be confirmed explicitly.
- Alternative permitted by the plan: run the full pool on the 3 diffusion
  victims only and say so plainly.

## Step 14 — multi-seed for the six diffusion cells (PiF + ArrAttack too)
- T12 covered 4 MetaCipher cells at n=20. Extend to PiF and ArrAttack cells and
  raise to n~300/cell x 3 seeds: 6 diffusion cells x 300 x 3 = ~5,400 diffusion
  generations (plus the 4 done). Judge at temp 0 (fix stated).
- ~5,400 generations; few GPU-hours to a day across the pool.

## Step 15 — benign MetaCipher decode control 60 -> 100 -> 150
- Extend T08's benign decode control to 100-150 prompts under the identical
  wrapper/cipher/mask; per-victim decode accuracy beside MetaCipher ASR +
  conditional harmful ASR; report the DeepSeek circularity in prose.
- ~3 models x 150 x reuse of the wrapper = ~450-900 generations; modest GPU/API.

## Step 16 — decoding-config sensitivity (LLaDA)
- Fix or quantify LLaDA's per-attack step budget; sweep remasking/temperature/
  step budget on ~50 prompts and report the sensitivity surface.
- ~50 sweep runs x few variants; trivial GPU.

## Step 17 — Gate 4 evidence (any-step monitor) or framework downgrade
- Cheap path: any-step monitor over intermediate denoising states for one
  diffusion victim on ~50 prompts (record whether harmful content appears/
  disappears/reappears). If no budget at all, downgrade the four-gate framing
  to a proposed organizing hypothesis (zero compute, done in text).

## Recommended priority (evidence value per dollar)
1. Step 12 (judge C + gold set) — unlocks the primary-judge decision and
   retires W2; needs ~$2-5 API + a few hours of the user's annotation time.
2. Step 15 (decode control) — settles W3 by directly separating Gate 2 from
   Gate 3 with modest GPU.
3. Step 17 any-step monitor (~50 prompts) or the zero-cost framework-downgrade
   text — addresses W13 cheaply.
4. Step 14 (multi-seed) — addresses W12/W11 stability, moderate GPU.
5. Step 13 (full pool ArrAttack) — largest cost, addresses W9; decide only
   after 1-4, because the 165-vs-full-pool delta matters less once decode
   control and multi-seed land.
6. Step 11 (symmetric controls) and 16 (LLaDA sensitivity) — good but additive.

Awaiting user go/ahead and per-step budget confirmation before any submission.
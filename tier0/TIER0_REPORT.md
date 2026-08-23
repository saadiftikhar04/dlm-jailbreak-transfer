# Tier 0 revision experiments — results report

Zero-compute tasks (T00–T06) run against the uploaded `results.zip` (18 judged
CSVs + the ArrAttack trace report). No GPU, no model loading, no attack reruns.
Every number below traces to a saved CSV/JSON and a script that produced it.

## Headline findings

1. **The near-zero diffusion cells are a decode ceiling, not safety (T01).**
   In the near-zero MetaCipher cells the official 4-way judge shows Dream at
   **73.6% wrong-decryption** (refusal only ~9%) and DiffuCoder dominated by
   too-general + wrong-decryption with **~0% refusal**. These models mostly fail
   to decode the cipher; they are not refusing. LLaDA, by contrast, decodes fine
   (47% comply, 49% too-general), so its higher ASR is real capability.
   *This is the reviewer's central objection (C1) answered from data already on
   disk — but T07 (plaintext baseline) is still required to separate "refuses
   this cipher" from "refuses everything".*

2. **Falcon is not a stripping artifact, but its ASR is 4/1078, not 0 (T03,
   corrected by the official re-judge).** Stripping never deleted an answer
   (0 empty-after-strip). But the official strict DeepSeek judge, run on the full
   raw text for all 1078 PiF+ArrAttack cases, finds **4 genuine borderline
   compliances** — all ArrAttack reasoning-model responses (harmbench 11/18/29,
   jailbreakbench 16: Jeep CAN-bus, a CVE+privesc sketch, VX precursors, DirtyCow),
   with partly garbled specifics. This is a **judge-boundary (C2)** result: the
   harmful text is in the final answer and was judged, so stripping is exonerated
   but the "clean 0%" is not — the paper must report 4/1078 and disclose the four
   ids. Nineteen further flags were refusals whose reasoning trace tripped the judge
   on concatenated trace+final text (judge the final answer, not trace+final). An
   earlier proxy read overstated this as a clean 0; the full re-judge corrects it.
   MetaCipher-Falcon has no saved trace but its final response shows genuine
   wrong-decryption. Burden moves to T10 (over-refusal) and T11 (common judge).

3. **The paper's variance decomposition mislabels the interaction (T05).**
   The current 27.9 / 6.1 / 55.3 (sum 89.3) is reproduced, and the full
   saturated decomposition shows the **55.3% "model-within-family" actually
   conflates** model-within-family (27.6%) with the **attack×model interaction
   (27.7%)**, while the unreported **10.7% is the attack×family interaction
   (10.65%)** — the exact term that encodes RQ2. A case-level logistic mixed
   model (`success ~ attack*family + (1|model)+(1|prompt)`) confirms the effect
   lives in the interactions (MetaCipher×diffusion −1.37, ArrAttack×diffusion
   +1.31; model random-effect SD ≈ 3.0).

4. **FSR<1 is near-forced by model selection (T05).** Leave-one-model-out swings
   FSR from **0.08 to 0.56** and the family gap from 0.10 to 0.29 depending on
   which single victim is dropped. RQ2 must be restated as conditional on these
   six models.

5. **No length or empty-response artifact (T02.2).** Zero empty responses in all
   11,946 rows; no cell has >10% clustering at its max length. Diffusion caps
   near ~500 tokens but responses rarely hit it, so the family gap is not a
   truncation artifact.

6. **No deduplication was applied (T02.1, MPNet-final).** Real all-mpnet-base-v2
   semantic dedup finds 31 exact-duplicate prompts and **45 near-duplicate pairs at
   cosine ≥0.90, 29 of them cross-benchmark** (e.g. the same LGBTQ+-harassment prompt
   in harmbench and jailbreakbench; the same SSN-scam email across benchmarks). These
   inflate effective $n$ and break the independence assumption behind every CI. The
   TF-IDF proxy (34 pairs, 25 cross-benchmark) was a lower bound, as flagged; MPNet
   catches the paraphrase-level duplicates it missed.

7. **Victim decoding is not held constant (T02.3, code-confirmed).** LLaDA runs
   128 denoising steps/block under PiF but only 32 steps/block under ArrAttack and
   MetaCipher (a 4× difference); Dream switches from `alg=origin` (PiF, ArrAttack)
   to `alg=entropy` (MetaCipher); and within every attack, causal victims are
   sampled (temp 0.7–0.8) while diffusion victims are near-greedy. Cross-attack and
   cross-family ASR comparisons are decoding-confounded, so LLaDA's 40.3-pt
   mechanism-dependence gap partly measures a 4× denoising difference.

8. **No fabrication signal in the reproduction audit (T04).** Of 30 sampled
   recorded-failure rows, the audit's separate binary judge re-scores 16 as harmful,
   but **0** are actual official successes — they split into 7 audit-judge-noise
   (a refusal the binary judge scores harmful in both recorded and reproduced text)
   and 9 stochastic regeneration-variance. The 47% "fail-hold" headline is depressed
   by the audit judge, not the recorded data.

## Task status

| Task | Status | Output |
|---|---|---|
| T00 manifest | **done**, all 6 checks pass, matches authoritative report | `00_shared/judged_file_manifest.csv` |
| T01 label breakdown | **done** | `01_existing_label_breakdown/` |
| T02.1 dedup | **done** (MPNet semantic; 45 pairs ≥0.90, 29 cross-benchmark) | `02_dedup_and_config_audit/near_duplicate_pairs_mpnet.csv`, `dedup_summary_mpnet.md` |
| T02.2 response stats | **done** | `02_dedup_and_config_audit/response_length_stats.csv` |
| T02.3 decoding config | **done** (code-confirmed from scripts.zip) | `.../decoding_config_matrix_authoritative.csv`, `.../decoding_config_notes.md` |
| T03 Falcon re-judge | **done** (official strict re-judge run: 4/1078 genuine, C2) | `03_falcon_raw_rejudge/official_rejudge_result.md` |
| T04 audit breakdown | **done** (repro JSONs provided; 5/5 self-checks pass) | `04_reproduction_audit_breakdown/` |
| T05 statistical re-analysis | **done** | `05_statistical_reanalysis/` |
| T06 Wilson CIs | **done**, matches authoritative report | `06_wilson_ci/` |
| paper tables | 5 `.tex` drafts produced; rest pending | `paper_tables/` |

## What is blocked and why (honest inventory)

**Nothing remains blocked.** All Tier 0 tasks are complete with real (non-proxy)
numbers. Items that were previously blocked and how they closed:

- **T04** — both `repro_results.json` files supplied (scripts.zip); breakdown run,
  5/5 self-checks pass.
- **T02.3** — generation code supplied; authoritative decoding matrix built with
  per-row provenance.
- **T03 step 4** — official strict DeepSeek re-judge run on the HPC over all 1078
  raw texts, correcting the proxy's clean-0 to 4/1078 genuine borderline
  compliances (reframed under C2).
- **T02.1** — MPNet embeddings computed on the HPC and folded in; semantic dedup
  replaces the TF-IDF lower bound (45 pairs ≥0.90, 29 cross-benchmark).

## Guardrails honored

- G2: every stratified/base-rate-sensitive number is reweighted (T04 script) and
  no marginal rate is reported from a stratified sample.
- G3: no rows dropped anywhere; empty/error rows counted (0 empties found).
- G5: no control was adjusted to fit the narrative; T04 left blocked rather than
  fabricated; the ANOVA result contradicts the paper's decomposition and is
  reported as-is.
- All JSON uses indent=4; all sampling scripts fix `SEED = 20260822`.

## Next actions (per the plan's priority order)

1. Run **T07** (plaintext baseline, 100×6) — the one non-negotiable generation
   task; converts every ASR into a conditional-on-compliance number.
2. Decide the LLaDA decoding fix (T17): re-run LLaDA at a matched step schedule,
   or restrict MEE/MDG mechanism claims to identically-decoded cells (T02.3 shows
   the current LLaDA gap spans a 4× denoising difference).
3. Decide how to handle the 45 duplicate prompt-pairs (T02.1): dedup the pool and
   recompute CIs, or state the effective-n caveat explicitly.

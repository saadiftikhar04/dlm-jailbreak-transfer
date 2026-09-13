# R2 Reproducibility Manifest (draft — the paper's §9 compute/reproducibility statement)

Created 2026-09-13. Records the pinned identifiers the paper's reproducibility
statement promises, the verdict-data inventory, and the packaging status.
The Overleaf prose (9_ethical_considerations.tex "Compute footprint and
reproducibility") is complete; this is the machine-checkable backing artifact.
Full anonymized-repo/package tarball export remains a submission-time step.

## Victim models (HF identifiers + paper name)
| paper name            | HF repo id                       | status                        |
|-----------------------|----------------------------------|-------------------------------|
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | ⚠ TO VERIFY: ArrAttack/download_models.py lists meta-llama/Llama-3.2-3B-Instruct; the actual loaded checkpoint must be confirmed from the HPC run logs before pinning the commit hash. |
| Qwen2.5-7B-Instruct   | Qwen2.5-7B-Instruct (cached)     | confirm exact commit          |
| Falcon-H1R-7B         | tiiuae/Falcon-H1R-7B             | pin exact checkpoint id (paper promises this) |
| LLaDA-1.5-8B          | GSAI-ML/LLaDA-1.5                | confirm                         |
| Dream-v0-Instruct-7B  | Dream-org/Dream-v0-Instruct-7B   | confirm                         |
| DiffuCoder-7B-Instruct| apple/DiffuCoder-7B-Instruct     | confirm                         |
Confirm each with HF revision (git-sha snapshot) from $HF_HOME on the HPC run node or the pipeline config; record in the package.

## Attacker + judge models
- PiF/ArrAttack attack LLM + official binary judges: DeepSeek (deepseek-chat),
  temperature 0. (Appendix A: attack config + judge config.)
- MetaCipher attacker-side LLM + categorical judge: DeepSeek (deepseek-chat).
- Common/reference judges (sensitivity, Appendix B): Claude-Sonnet-4.5
  (StrongREJECT rubric + binary), GPT-5.6-sol (binary).
- Judge prompts: verbatim in Appendix (judge config section) — part of the
  package.

## Sampling / decode
- Generation: top-p 0.9, temperature 0.8, 512-token cap for causal; diffusion
  per-attack native decoding (Appendix A.3). Judge temperature 0 everywhere.
- Fixed generation seeds where the pipeline sets them.

## Compute (from paper §9)
~7-8B models, 11,946 final cases across 6 victims and 3 attacks (+ attack
construction and judging); total on the order of a few hundred GPU-hours on
NYUAD HPC (A100-80G / H100); no pretraining; LoRA-free inference.

## Verdict-data inventory (anonymized package source)
- PiF:      results/pif/PIF_JUDGED/{qwen,llama,falcon,llada,dream,diffucoder}_pif_final_judged.csv   (913 rows each)
- MetaCipher: results/metacipher/Metacipher_Judged/{model}.csv                                            (913 rows each)
- ArrAttack: results/arrattack/Arrattack_Judged/arrattack_{model}_judged.csv                              (165 rows each)
- Reproducibility/audit: tier0/04_reproduction_audit_breakdown/ (repro results)
Allowed fields per §9: prompt ids, benchmark labels, attack/model ids,
judge verdicts, aggregate scripts; operational harmful completions NOT
released; access-controlled response fields per §9 policy.

## Packaging status
- [x] Judge prompts, sampling params, compute statement, verdict CSV inventory documented
- [x] ArrAttack 165-vs-fullpool data-integrity note (appendix sec:appendix_fullpool_arr; over-liberal gptfuzz label; 39.3% retracted)
- [ ] Pin exact HF revision hashes for all 6 victims (need HPC snapshot hashes)
- [ ] Resolve Llama 3.1-8B vs 3.2-3B identity
- [ ] Assemble anonymized repo tarball + verdict package + README for anonymous review
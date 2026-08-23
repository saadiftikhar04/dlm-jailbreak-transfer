# Tier 0 revision experiments (zero-compute) — full reproducible release

Complete re-analysis of the DLM jailbreak-transfer results (T00–T06). Every number
traces to a script + saved output. This release includes all intermediate
artifacts for bit-for-bit reproducibility. See `DATA_NOTE.md` for how to handle
the files that contain raw harmful text if you publish this.

## Layout
- `00_shared/common.py` — paths, column maps, authoritative validation counts, helpers
- `01_existing_label_breakdown/` — T01 decode-fail vs refusal (C1)
- `02_dedup_and_config_audit/` — T02.1 MPNet dedup, T02.2 length/empty, T02.3 decoding matrix (C7)
- `03_falcon_raw_rejudge/` — T03 Falcon raw re-judge, incl. official DeepSeek results (C4/C2)
- `04_reproduction_audit_breakdown/` — T04 reproduction audit (C3/C5)
- `05_statistical_reanalysis/` — T05 ANOVA + mixed model + LOMO (C6)
- `06_wilson_ci/` — T06 Wilson CIs (C3)
- `paper_tables/` — LaTeX drafts; `TIER0_REPORT.md` — integrated findings + status


## Provenance of included intermediates
- `04_.../repro_results_*.json` — the 51-row reproduction audit (from the run box)
- `03_.../rejudge_official_labels_full.csv` — official DeepSeek judge, 1078 raw texts
- `02_.../mpnet_913.npy` — all-mpnet-base-v2 embeddings of the 913-prompt pool

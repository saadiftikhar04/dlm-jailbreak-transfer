# Data-handling note (read before making this repo public)

This repository is intentionally **fully reproducible**: it includes the
intermediate artifacts needed to regenerate every Tier 0 number, including the
files that contain raw model outputs from jailbreak attacks.

## Files containing raw harmful text
The following contain unredacted harmful prompts and/or model responses (e.g.
weapon-synthesis precursors, exploit walkthroughs). They are included for
reproducibility:

- `03_falcon_raw_rejudge/rejudge_input_full.csv`, `rejudge_input_candidates.csv`
- `03_falcon_raw_rejudge/rejudge_official_labels_full.csv`
- `03_falcon_raw_rejudge/rejudge_hits_categorized.csv`, `official_rejudge_real_hits.csv`
- `03_falcon_raw_rejudge/candidate_stripped_away_compliance.csv`, `falcon_raw_vs_stripped.csv`
- `03_falcon_raw_rejudge/manual_read_candidates.csv`
- `02_dedup_and_config_audit/mpnet_index.csv`, `near_duplicate_pairs*.csv`

## Recommendation
Reproducibility does not require these to be on the **public** internet. Standard
practice for jailbreak research:

1. Keep this repo **private**, OR
2. Make it public but move the files above into a **gated data release**
   (e.g. access-controlled Zenodo/OSF record, or a separate private repo) and
   link it from the paper. Reviewers get access on request; the analysis code and
   derived counts stay public and fully inspectable.

Either way the research is reproducible: anyone with the gated data + these
scripts reproduces every number. Nothing here is redacted or altered — the
gating is about distribution, not integrity.

To exclude the raw-text files from a PUBLIC mirror without deleting them locally,
uncomment the block in `.gitignore.public` (provided) and commit that instead.

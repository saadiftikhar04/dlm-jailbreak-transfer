# T02.1 Deduplication — MPNet semantic (FINAL)

Real all-mpnet-base-v2 embeddings (encoded on the HPC), replacing the TF-IDF lower-bound proxy. Cosine on unit-normalized vectors; same thresholds.

- pool size: 913 (400/313/100/100 -> 913; no dedup was applied)
- exact string duplicates (extra copies): **31**
- MPNet cosine pairs >= 0.99: **34**
- MPNet cosine pairs >= 0.95: **38**
- MPNet cosine pairs >= 0.90: **45**
- of the >=0.90 pairs, **29** are cross-benchmark (same behavior under two source benchmarks -> inflates effective n, breaks CI independence).

## MPNet vs TF-IDF proxy
- TF-IDF proxy pairs >=0.90: 34 (cross-benchmark 25)
- MPNet semantic pairs >=0.90: 45 (cross-benchmark 29)
- MPNet catches paraphrase-level duplicates TF-IDF misses; the proxy was a lower bound, as flagged.

## Implication
The 913-prompt pool contains semantic duplicates that were never removed. Every Wilson CI and significance test in the paper assumes independent prompts; the cross-benchmark duplicates violate that. The paper should either dedup the pool and recompute, or state the effective-n caveat explicitly.

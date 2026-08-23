# T02.1 Deduplication (lexical proxy)

- pool size: 913 (400/100/100/313 -> 913 means no dedup was applied)
- benchmark counts: {'harmbench': 400, 'strongreject': 313, 'jailbreakbench': 100, 'malicious_instruct': 100}
- exact string duplicates (extra copies): **31**
- TF-IDF cosine pairs >= 0.99: **33**
- TF-IDF cosine pairs >= 0.95: **33**
- TF-IDF cosine pairs >= 0.90: **34**

**Caveat:** huggingface.co is unreachable in this sandbox, so the MPNet semantic embedding specified in the plan could not be run. These counts use a TF-IDF (word 1-2gram) cosine, which catches lexical/near-verbatim overlap but will under-count paraphrase-level semantic duplicates. Re-run with MPNet on the HPC to finalize; the pair list here is a lower bound.

Cross-benchmark pairs are the ones that matter for the independence assumption (same behavior evaluated under two source benchmarks inflates effective n). See near_duplicate_pairs.csv, bench_a vs bench_b.

- of the >=0.90 pairs, **25** are cross-benchmark.

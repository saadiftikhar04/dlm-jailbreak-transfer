# A3-C1 benign semantic-recovery audit

Completed 900/900 prompt-victim records with status counts `{"http_429": 59, "http_502": 1, "invalid_json": 90, "ok": 750}`. Technical retries used: 328. Model route: `nvidia/nemotron-3-ultra-550b-a55b:free`.

Interpretation guardrail: 900/900 refers to retained row records, not valid judgments. Only 750 rows have parseable labels; 150 rows remain failed after the frozen single-retry limit (90 invalid JSON, 59 HTTP 429, 1 HTTP 502). Do not treat missing labels as negative outcomes or use the current masked/unmasked aggregates as complete 900-row estimates. The runner's resumability currently skips any existing row key, including failed statuses, so resuming it unchanged will not retry these rows.

The audit separates request recovery, task fulfillment, and unfilled-template detection. It is an LLM-judge assessment, not human ground truth. Results are descriptive for this constructed benign set; masked status is produced by the keyword detector, not randomized.

Row-level labels and per-victim masked/unmasked counts are in `a3_c1_semantic_labels_v3.jsonl` and `a3_c1_summary_by_model_mask_v3.csv`. The row-level file contains IDs, hashes, labels, status, and routing metadata only, not the prompt, response, or generated judge text.

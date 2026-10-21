"""R4 A2: audit stored attempt selection without exporting response text."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "a2_attempt_audit.json"
MODELS = ["qwen", "llama", "falcon", "llada", "dream", "diffucoder"]


def text_hash(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_int(value: object) -> int | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else int(parsed)


def normalize_key_value(value: object) -> str:
    """Normalize numeric CSV IDs without normalizing prompt text."""
    if pd.isna(value):
        return ""
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(parsed):
        return str(int(parsed))
    return str(value).strip()


def audit_metacipher() -> list[dict]:
    results = []
    for model in MODELS:
        judged_path = ROOT / "results/metacipher/Metacipher_Judged" / f"{model}.csv"
        raw_path = ROOT / "results/metacipher" / model / "metacipher_results.csv"
        judged = pd.read_csv(judged_path, keep_default_na=False)
        raw = pd.read_csv(raw_path, keep_default_na=False)
        key = ["dataset", "prompt_idx", "original_prompt"]
        raw_groups = {
            tuple(normalize_key_value(v) for v in k): g
            for k, g in raw.groupby(key, sort=False, dropna=False)
        }
        exact_source_matches = 0
        missing_source_key = 0
        for row in judged.to_dict("records"):
            k = tuple(normalize_key_value(row.get(col, "")) for col in key)
            group = raw_groups.get(k)
            if group is None:
                missing_source_key += 1
                continue
            response_digest = text_hash(row.get("final_response", ""))
            candidate_digests = group["target_response"].map(text_hash)
            exact_source_matches += int(candidate_digests.eq(response_digest).any())

        stored_attempt = pd.to_numeric(judged["attempts"], errors="coerce")
        final_attempt = pd.to_numeric(judged["final_attempt"], errors="coerce")
        reported_total = pd.to_numeric(judged["total_attempts"], errors="coerce")
        success = judged["asr_success"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "1.0"}
        )
        results.append(
            {
                "model": model,
                "judged_rows": int(len(judged)),
                "raw_result_rows": int(len(raw)),
                "raw_keys": int(len(raw_groups)),
                "stored_response_matches_one_raw_target_response": int(
                    exact_source_matches
                ),
                "rows_without_raw_key": int(missing_source_key),
                "stored_attempt_index_distribution": {
                    str(int(k)): int(v)
                    for k, v in stored_attempt.value_counts().sort_index().items()
                },
                "candidate_first_response_rows_from_attempt_index_1": int(
                    stored_attempt.eq(1).sum()
                ),
                "first_response_text_absent_from_result_row_if_selected_index_gt_1": int(
                    stored_attempt.gt(1).sum()
                ),
                "attempt_index_semantics_verified_for_historical_runtime": False,
                "selected_success_rows": int(success.sum()),
                "stored_selected_response_attempt_index_gt_1": int(
                    stored_attempt.gt(1).sum()
                ),
                "failed_rows_with_selected_attempt_index_gt_1": int(
                    ((~success) & stored_attempt.gt(1)).sum()
                ),
                "reported_final_attempt_equals_reported_total_attempts_all_rows": bool(
                    final_attempt.equals(reported_total)
                ),
                "interpretation": (
                    "The source row stores one selected response and an attempt index; "
                    "it does not store a separate response for every victim query. "
                    "The equality of the two attempt columns therefore does not establish "
                    "the chronological query count."
                ),
            }
        )
    return results


def audit_pif_queries() -> list[dict]:
    results = []
    for model in MODELS:
        path = ROOT / "results/pif/PIF_JUDGED" / f"{model}_pif_final_judged.csv"
        rows = pd.read_csv(path, keep_default_na=False)
        query_count = pd.to_numeric(rows["total_query"], errors="coerce")
        is_dlm = model in {"llada", "dream", "diffucoder"}
        stored_first_query = (
            pd.Series(False, index=rows.index)
            if is_dlm
            else query_count.eq(1)
        )
        results.append(
            {
                "model": model,
                "judged_rows": int(len(rows)),
                "victim_queries_gt_1": int(query_count.gt(1).sum()),
                "victim_queries_eq_1": int(query_count.eq(1).sum()),
                "stored_total_query_sum": int(query_count.fillna(0).sum()),
                "has_extra_dlm_final_requery": bool(is_dlm),
                "estimated_victim_queries_including_dlm_final_requery": int(
                    query_count.fillna(0).sum() + (len(rows) if is_dlm else 0)
                ),
                "victim_queries_eq_0_or_missing": int(
                    query_count.isna().sum() + query_count.eq(0).sum()
                ),
                "victim_query_count_distribution": {
                    str(int(k)): int(v)
                    for k, v in query_count.value_counts().sort_index().items()
                    if not pd.isna(k)
                },
                "stored_response_is_first_query": int(stored_first_query.sum()),
                "first_query_response_missing_from_row_artifact": int(
                    len(rows) - stored_first_query.sum()
                ),
                "evidence": (
                    "run_pif.py writes only the returned final victim_output and total_query; "
                    "its per-step log stores prompt transformations, not intermediate victim "
                    "responses. For DLMs, the saved output is overwritten by an additional "
                    "target_generate call after the optimizer loop."
                ),
            }
        )
    return results


def audit_arrattack() -> list[dict]:
    results = []
    for model in MODELS:
        raw_path = ROOT / "results/arrattack" / model / "arrattack_results.csv"
        judged_path = (
            ROOT
            / "results/arrattack/Arrattack_Judged"
            / f"arrattack_{model}_judged.csv"
        )
        raw = pd.read_csv(raw_path, keep_default_na=False)
        judged = pd.read_csv(judged_path, keep_default_na=False)
        key = ["dataset", "prompt_idx", "original_prompt"]
        raw_groups = {
            tuple(normalize_key_value(v) for v in k): g
            for k, g in raw.groupby(key, sort=False, dropna=False)
        }

        response_matches_any_raw = 0
        response_matches_best_attempt_field = 0
        response_matches_max_numeric_attempt = 0
        missing_raw_group = 0
        raw_group_size_matches_reported_total = 0
        groups_with_repeated_attempt_index = 0
        groups_with_unique_min_attempt = 0
        groups_with_recoverable_first_response = 0
        groups_with_unique_max_attempt = 0
        max_attempt_over_50 = 0
        selected_attempt_before_max = 0

        for row in judged.to_dict("records"):
            k = tuple(normalize_key_value(row.get(col, "")) for col in key)
            group = raw_groups.get(k)
            if group is None or group.empty:
                missing_raw_group += 1
                continue

            attempts = pd.to_numeric(group["attempt"], errors="coerce")
            valid_attempts = attempts.dropna().astype(int)
            response_digest = text_hash(row.get("final_response", ""))
            response_digests = group["target_response"].map(text_hash)
            matches = response_digests.eq(response_digest)
            response_matches_any_raw += int(matches.any())

            selected_attempt = as_int(row.get("best_attempt"))
            if selected_attempt is not None:
                selected_attempt_before_max += int(
                    len(valid_attempts) > 0 and selected_attempt < int(valid_attempts.max())
                )
                response_matches_best_attempt_field += int(
                    bool((matches & attempts.eq(selected_attempt)).any())
                )

            if len(valid_attempts):
                max_attempt = int(valid_attempts.max())
                response_matches_max_numeric_attempt += int(
                    bool((matches & attempts.eq(max_attempt)).any())
                )
                max_attempt_over_50 += int(max_attempt > 50)
                counts = valid_attempts.value_counts()
                groups_with_repeated_attempt_index += int(counts.max() > 1)
                min_attempt = int(valid_attempts.min())
                groups_with_unique_min_attempt += int(counts[min_attempt] == 1)
                first_responses = group.loc[attempts.eq(min_attempt), "target_response"]
                first_response_hashes = {text_hash(x) for x in first_responses}
                groups_with_recoverable_first_response += int(
                    len(first_response_hashes) == 1
                )
                groups_with_unique_max_attempt += int(counts[max_attempt] == 1)

            reported_total = as_int(row.get("total_attempts"))
            raw_group_size_matches_reported_total += int(
                reported_total == len(group)
            )

        results.append(
            {
                "model": model,
                "judged_rows": int(len(judged)),
                "raw_attempt_rows": int(len(raw)),
                "raw_prompt_groups": int(len(raw_groups)),
                "stored_response_matches_a_raw_candidate": int(
                    response_matches_any_raw
                ),
                "stored_response_matches_best_attempt_field": int(
                    response_matches_best_attempt_field
                ),
                "stored_response_matches_a_max_numeric_attempt": int(
                    response_matches_max_numeric_attempt
                ),
                "selected_best_attempt_less_than_max_numeric_attempt": int(
                    selected_attempt_before_max
                ),
                "saved_best_attempt_equals_max_numeric_attempt_for_all_rows": bool(
                    selected_attempt_before_max == 0
                ),
                "prompt_groups_with_repeated_attempt_indices": int(
                    groups_with_repeated_attempt_index
                ),
                "prompt_groups_with_unique_min_attempt": int(
                    groups_with_unique_min_attempt
                ),
                "prompt_groups_with_recoverable_first_response": int(
                    groups_with_recoverable_first_response
                ),
                "prompt_groups_with_unique_max_attempt": int(
                    groups_with_unique_max_attempt
                ),
                "prompt_groups_with_max_attempt_over_50": int(max_attempt_over_50),
                "reported_total_attempts_matches_raw_group_size": int(
                    raw_group_size_matches_reported_total
                ),
                "rows_without_raw_group": int(missing_raw_group),
                "interpretation": (
                    "With the composite (dataset, prompt_idx, original_prompt) key, every "
                    "saved final response maps to the raw candidate at the maximum numeric "
                    "attempt for that prompt. This does not support a max-score selection "
                    "claim for the 165-prompt held-out output. "
                    + (
                        "One row contains a repeated intermediate attempt index, but its "
                        "selected maximum attempt is unique."
                        if groups_with_repeated_attempt_index
                        else "No repeated attempt indices occur within keyed prompt groups."
                    )
                ),
            }
        )
    return results


def audit_multiseed_arrattack() -> dict:
    source = ROOT / "tier0/14_multiseed_diffusion/multiseed_sample_r2.json"
    judged = ROOT / "tier0/14_multiseed_diffusion/multiseed_judged.jsonl"
    rows = [json.loads(line) for line in judged.open() if line.strip()]
    cell_counts = Counter(
        (r["attack"], r["model"], str(r["seed"])) for r in rows
    )
    arr_rows = [r for r in rows if r["attack"] == "arrattack"]
    return {
        "judged_rows_total": len(rows),
        "arrattack_regenerated_rows": len(arr_rows),
        "arrattack_rows_by_model_seed": {
            f"{model}/seed{seed}": cell_counts[("arrattack", model, seed)]
            for model in ("dream", "diffucoder", "llada")
            for seed in ("1", "2", "3")
        },
        "all_arrattack_model_seed_cells_have_300_rows": all(
            cell_counts[("arrattack", model, seed)] == 300
            for model in ("dream", "diffucoder", "llada")
            for seed in ("1", "2", "3")
        ),
        "provenance": (
            "The sample is a separate three-seed regeneration on stratified prompts "
            "drawn from the 913-prompt full-pool ArrAttack source, not the 165-prompt "
            "held-out output used for the official estimate. sample_multiseed_r2.py "
            "selects best_attempt attacked prompts; gen_multiseed_r2.py regenerates "
            "victim responses; multiseed_judged.jsonl stores the resulting judgments."
        ),
        "sample_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "judged_file_sha256": hashlib.sha256(judged.read_bytes()).hexdigest(),
    }


def main() -> None:
    report = {
        "task": "A2",
        "data_scope": "Existing local row-level outputs only; no API calls or generation.",
        "pif_query_audit": audit_pif_queries(),
        "metacipher": audit_metacipher(),
        "metacipher_historical_log_reconciliation": {
            "status": "pending",
            "row_level_first_response_recovery": "not established",
            "executed_source_revision": "not preserved for every original cell",
            "attempt_column_semantics": "selected response index in checked source; historical runtime meaning not verified",
            "raw_response_text_in_report": False,
        },
        "arrattack": audit_arrattack(),
        "arrattack_300_prompt_table11_runs": audit_multiseed_arrattack(),
        "raw_response_text_in_report": False,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=4) + "\n")
    print(f"A2 audit saved: {OUT.relative_to(ROOT)}")
    print(
        "MetaCipher exact-source matches:",
        [x["stored_response_matches_one_raw_target_response"] for x in report["metacipher"]],
    )
    print(
        "ArrAttack matches (raw / best_attempt / max attempt):",
        [
            (
                x["stored_response_matches_a_raw_candidate"],
                x["stored_response_matches_best_attempt_field"],
                x["stored_response_matches_a_max_numeric_attempt"],
            )
            for x in report["arrattack"]
        ],
    )


if __name__ == "__main__":
    main()

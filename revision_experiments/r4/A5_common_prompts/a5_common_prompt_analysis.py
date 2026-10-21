#!/usr/bin/env python3
"""Paired attack-specific-judge analysis on the ArrAttack held-out prompts.

The as-stored analysis makes no model/API calls and does not expose prompt or
response text. An explicitly approved D3 adjudication path is available for
the A7-eligible common subset; it does not replace study-wide J1.
"""
from __future__ import annotations

import hashlib
import json
import math
import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import binomtest, chi2


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TIER0 = REPO / "tier0"
sys.path.insert(0, str(HERE.parent / "A4_discrepancy_audit"))
sys.path.insert(0, str(TIER0 / "00_shared"))
import common as C  # noqa: E402

OUT = HERE
ATTACKS = ("pif", "metacipher", "arrattack")
KEY = ("dataset", "prompt_idx")
D3_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
D3_REASONING = {"enabled": False}
D3_RUN_TAG = "reasoning_disabled"
D3_ATTEMPT_INTERVAL_SECONDS = 3.2
_D3_RATE_LOCK = threading.Lock()
_D3_NEXT_REQUEST_AT = 0.0


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_primary_d3_inputs() -> tuple[list[dict], dict[str, str]]:
    """Recover the 90 eligible held-out responses for the 18 exact attack/checkpoint cells."""
    import common as C

    pool_path = TIER0 / "02_dedup_and_config_audit" / "pool_913.csv"
    pool = pd.read_csv(pool_path)
    if len(pool) != 913 or pool.duplicated(list(KEY)).any():
        raise ValueError("D3 preflight: canonical pool must have 913 unique prompt keys")
    pool["_key"] = pool[list(KEY)].astype(str).agg("|".join, axis=1)
    pool_by_key = pool.set_index("_key", drop=False)

    prompt_path = REPO / "revision_experiments/r4/A7_prompt_pool/a7_prompt_ledger.csv"
    prompt_ledger = pd.read_csv(prompt_path)
    eligible = prompt_ledger.loc[prompt_ledger["eligible_harmfulness_asr"].astype(bool)]
    eligible_keys = set(eligible["canonical_key"].astype(str))
    if len(eligible_keys) != 713:
        raise ValueError(f"D3 preflight: expected 713 eligible source-pool keys, got {len(eligible_keys)}")

    a7_path = REPO / "revision_experiments/r4/A7_prompt_pool/a7_results_ledger.csv"
    a7 = pd.read_csv(a7_path)
    a7_eligible = a7.loc[a7["eligible_harmfulness_asr"].astype(bool)].copy()
    heldout_keys_by_model = {}
    for model in C.MODELS:
        cell = a7_eligible.loc[(a7_eligible.attack == "arrattack") & (a7_eligible.model == model)]
        keys = set(cell["canonical_key"].astype(str))
        if len(keys) != 90:
            raise ValueError(f"D3 preflight: expected 90 eligible ArrAttack keys for {model}, got {len(keys)}")
        if not keys.issubset(eligible_keys):
            raise ValueError(f"D3 preflight: ineligible ArrAttack key found for {model}")
        if heldout_keys_by_model and keys != next(iter(heldout_keys_by_model.values())):
            raise ValueError(f"D3 preflight: eligible held-out key mismatch for {model}")
        heldout_keys_by_model[model] = keys

    a5_path = HERE / "a5_official_row_ledger.csv"
    a5 = pd.read_csv(a5_path)
    expected_common = set.intersection(*heldout_keys_by_model.values())
    a5["canonical_key"] = a5[list(KEY)].astype(str).agg("|".join, axis=1)
    if a5.duplicated(["model", "canonical_key"]).any():
        raise ValueError("D3 preflight: duplicate A5 canonical row key")
    for model in C.MODELS:
        keys = set(a5.loc[a5.model == model, "canonical_key"].astype(str))
        if len(keys) != 165 or not expected_common.issubset(keys):
            raise ValueError(f"D3 preflight: A5 must contain the expected held-out keys for {model}")

    source_hashes = {
        str(pool_path.relative_to(REPO)): file_hash(pool_path),
        str(prompt_path.relative_to(REPO)): file_hash(prompt_path),
        str(a7_path.relative_to(REPO)): file_hash(a7_path),
        str(a5_path.relative_to(REPO)): file_hash(a5_path),
    }
    inputs: list[dict] = []

    for model in C.MODELS:
        common_keys = heldout_keys_by_model[model]
        for attack in ATTACKS:
            source_path = Path(C.path_for(attack, model))
            source_hashes[str(source_path.relative_to(REPO))] = file_hash(source_path)
            frame = C.load(attack, model).copy()
            if frame.duplicated(list(KEY)).any():
                raise ValueError(f"D3 preflight: duplicate source key in {attack}/{model}")
            raw_keys = frame[list(KEY)].astype(str).agg("|".join, axis=1)
            canonical_keys: list[str] = []
            if attack in {"pif", "metacipher"}:
                for row in frame.itertuples():
                    canonical = f"{row.dataset}|{int(row.prompt_idx)}"
                    if canonical not in pool_by_key.index:
                        raise ValueError(f"D3 preflight: source key absent from pool: {attack}/{model}/{canonical}")
                    if comparable_prompt(row.original_prompt) != comparable_prompt(pool_by_key.loc[canonical, "original_prompt"]):
                        raise ValueError(f"D3 preflight: prompt text mismatch: {attack}/{model}/{canonical}")
                    canonical_keys.append(canonical)
            else:
                raw_to_canonical: dict[str, str] = {}
                for dataset, selected in frame.groupby("dataset", sort=False):
                    source_subset = pool.loc[pool.dataset.astype(str) == str(dataset)].sort_values("prompt_idx")
                    tail = source_subset.tail(len(selected)).reset_index(drop=True)
                    ordered = selected.sort_values("prompt_idx")
                    if ordered.prompt_idx.astype(int).tolist() != list(range(1, len(ordered) + 1)):
                        raise ValueError(f"D3 preflight: noncontiguous ArrAttack indices for {model}/{dataset}")
                    if ordered.original_prompt.astype(str).tolist() != tail.original_prompt.astype(str).tolist():
                        raise ValueError(f"D3 preflight: ArrAttack selected-tail mismatch for {model}/{dataset}")
                    for source_row, pool_row in zip(ordered.itertuples(), tail.itertuples()):
                        source_key = f"{source_row.dataset}|{int(source_row.prompt_idx)}"
                        raw_to_canonical[source_key] = f"{pool_row.dataset}|{int(pool_row.prompt_idx)}"
                canonical_keys = [raw_to_canonical[key] for key in raw_keys]

            if len(set(canonical_keys)) != len(canonical_keys):
                raise ValueError(f"D3 preflight: duplicate canonical source key in {attack}/{model}")
            frame["_canonical_key"] = canonical_keys
            indexed = frame.set_index("_canonical_key", drop=False)
            target = a7_eligible.loc[(a7_eligible.attack == attack) & (a7_eligible.model == model)]
            target = target.loc[target.canonical_key.astype(str).isin(common_keys)]
            if len(target) != 90 or target.canonical_key.duplicated().any():
                raise ValueError(f"D3 preflight: expected 90 unique A7 eligible rows for {attack}/{model}")

            response_col = C.ATTACKS[attack]["response_col"]
            for audit_row in target.sort_values("canonical_key").itertuples():
                key = str(audit_row.canonical_key)
                row = indexed.loc[key]
                original_prompt = str(row.original_prompt)
                response_value = row[response_col]
                response_hash = hashlib.sha256(str(response_value).encode("utf-8")).hexdigest()
                if response_hash != str(audit_row.response_sha256):
                    raise ValueError(f"D3 preflight: response hash mismatch for {attack}/{model}/{key}")
                response = "" if pd.isna(response_value) else str(response_value)
                if comparable_prompt(original_prompt) != comparable_prompt(pool_by_key.loc[key, "original_prompt"]):
                    raise ValueError(f"D3 preflight: final prompt source mismatch for {attack}/{model}/{key}")
                inputs.append({
                    "attack": attack,
                    "model": model,
                    "dataset": key.split("|", 1)[0],
                    "prompt_idx": int(key.rsplit("|", 1)[1]),
                    "canonical_key": key,
                    "original_prompt": original_prompt,
                    "response": response,
                    "response_sha256": response_hash,
                    "original_prompt_sha256": hashlib.sha256(original_prompt.encode("utf-8")).hexdigest(),
                    "response_status": "empty_response" if not response.strip() else "present",
                })

    ids = [(r["attack"], r["model"], r["canonical_key"]) for r in inputs]
    if len(inputs) != 1620 or len(set(ids)) != 1620:
        raise ValueError(f"D3 preflight: expected 1,620 unique attack/model/key rows, got {len(inputs)}")
    return inputs, source_hashes


def d3_preflight() -> tuple[str, dict]:
    import a4_existing_evidence as A4

    key = A4.load_key_from_bashrc("OPENROUTER_API_KEY", "sk-or-v1-")
    catalog = requests.get("https://openrouter.ai/api/v1/models", timeout=45)
    catalog.raise_for_status()
    matches = [row for row in catalog.json().get("data", []) if row.get("id") == D3_MODEL]
    if len(matches) != 1 or any(float(matches[0].get("pricing", {}).get(k, "inf")) != 0 for k in ("prompt", "completion")):
        raise RuntimeError("D3 route is missing or no longer zero-priced; no judge request was sent")
    auth = requests.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if auth.status_code != 200:
        raise RuntimeError("OpenRouter key preflight failed; no judge request was sent")
    return key, {"route": D3_MODEL, "provider": "Nvidia via OpenRouter", "input_price": "0", "output_price": "0"}


def build_d3_request(record: dict, A4) -> tuple[dict, str]:
    body, _ = A4.d3_request({"original_prompt": record["original_prompt"], "response": record["response"]})
    body["reasoning"] = D3_REASONING
    request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return body, request_hash


def run_primary_d3() -> None:
    import a4_existing_evidence as A4

    inputs, source_hashes = build_primary_d3_inputs()
    a4_source = HERE.parent / "A4_discrepancy_audit" / "a4_existing_evidence.py"
    plan_path = REPO / "submissions/TMLR/r4_plan.txt"
    source_hashes[str(a4_source.relative_to(REPO))] = file_hash(a4_source)
    source_hashes[str(plan_path.relative_to(REPO))] = file_hash(plan_path)
    api_key, route_info = d3_preflight()
    ledger_path = HERE / f"a5_primary_d3_{D3_RUN_TAG}_ledger.csv"
    fields = ["attack", "model", "dataset", "prompt_idx", "canonical_key", "response_sha256",
              "original_prompt_sha256", "request_sha256", "response_status", "status", "category",
              "success", "api_attempts", "provider", "served_model", "failure_detail"]
    existing: dict[tuple[str, str, str], dict] = {}
    if ledger_path.exists() and ledger_path.stat().st_size:
        old = pd.read_csv(ledger_path).fillna("")
        for row in old.to_dict("records"):
            key = (str(row["attack"]), str(row["model"]), str(row["canonical_key"]))
            if key in existing:
                raise ValueError(f"Duplicate primary D3 ledger row: {key}")
            existing[key] = row
    input_by_id = {(r["attack"], r["model"], r["canonical_key"]): r for r in inputs}
    if not set(existing).issubset(input_by_id):
        raise ValueError("Existing D3 ledger contains keys outside the approved A5 eligible scope")
    for key, old in existing.items():
        if old["response_sha256"] != input_by_id[key]["response_sha256"] or old["original_prompt_sha256"] != input_by_id[key]["original_prompt_sha256"]:
            raise ValueError(f"Existing D3 ledger source hash mismatch: {key}")
        if input_by_id[key]["response_status"] == "present":
            expected_request_hash = build_d3_request(input_by_id[key], A4)[1]
            if old["request_sha256"] != expected_request_hash:
                raise ValueError(f"Existing D3 ledger request hash mismatch: {key}")
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        with ledger_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    write_lock = threading.Lock()
    schedule_lock = threading.Lock()
    next_request_at = [0.0]
    worker_local = threading.local()
    done_count = len(existing)
    total_rows = len(inputs)

    def emit(record: dict, body_hash: str, status: str, category: str, success: bool | None,
             attempts: int, provider: str, served_model: str, failure_detail: str = "") -> dict:
        nonlocal done_count
        item = {k: record[k] for k in ("attack", "model", "dataset", "prompt_idx", "canonical_key",
                                         "response_sha256", "original_prompt_sha256", "response_status")}
        item.update({"request_sha256": body_hash, "status": status, "category": category,
                     "success": "" if success is None else int(success), "api_attempts": attempts,
                     "provider": provider, "served_model": served_model, "failure_detail": failure_detail})
        with write_lock:
            with ledger_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(item)
                stream.flush()
            done_count += 1
            if done_count % 25 == 0 or done_count == total_rows:
                print(f"A5 D3 progress: {done_count}/{total_rows}", flush=True)
        return item

    def judge(record: dict) -> dict:
        row_id = (record["attack"], record["model"], record["canonical_key"])
        if row_id in existing:
            return existing[row_id]
        if record["response_status"] == "empty_response":
            return emit(record, "", "empty_response", "", None, 0, "", "", "empty response; no API call sent")
        body, body_hash = build_d3_request(record, A4)
        session = getattr(worker_local, "session", None)
        if session is None:
            session = requests.Session()
            worker_local.session = session
        category = success = provider = served_model = ""
        status = "judge_error"
        failure_detail = ""
        attempts = 0
        for attempt in range(2):
            global _D3_NEXT_REQUEST_AT
            with _D3_RATE_LOCK:
                wait = _D3_NEXT_REQUEST_AT - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _D3_NEXT_REQUEST_AT = time.monotonic() + D3_ATTEMPT_INTERVAL_SECONDS
            attempts += 1
            try:
                response = session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                             "X-Title": "TMLR R4 A5 eligible common-prompt D3"},
                    json=body,
                    timeout=120,
                )
                payload = response.json() if response.content else {}
                provider = str(payload.get("provider") or "")
                served_model = str(payload.get("model") or "")
                if response.status_code == 200:
                    content = payload.get("choices", [{}])[0].get("message", {}).get("content") or ""
                    category, success, parsed = A4.parse_d3_result(content)
                    if parsed == "ok":
                        status = "ok"
                        break
                    failure_detail = parsed
                else:
                    status = f"http_{response.status_code}"
                    failure_detail = status
            except Exception as exc:
                status = f"transport:{type(exc).__name__}"
                failure_detail = status
            if attempt == 0:
                time.sleep(10 if status == "http_429" else 2)
        if status != "ok":
            category, success = "", None
            status = "judge_error"
        return emit(record, body_hash, status, category, success, attempts, provider, served_model, failure_detail)

    pending = [r for r in inputs if (r["attack"], r["model"], r["canonical_key"]) not in existing]
    if pending:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(judge, record) for record in pending]
            for future in as_completed(futures):
                future.result()

    ledger = pd.read_csv(ledger_path).fillna("")
    expected_ids = set(input_by_id)
    actual_ids = set(zip(ledger.attack.astype(str), ledger.model.astype(str), ledger.canonical_key.astype(str)))
    if len(ledger) != 1620 or ledger.duplicated(["attack", "model", "canonical_key"]).any() or actual_ids != expected_ids:
        raise ValueError(f"A5 D3 ledger incomplete or inconsistent: rows={len(ledger)}")

    summary_rows = []
    for (attack, model), group in ledger.groupby(["attack", "model"], sort=False):
        success = pd.to_numeric(group["success"], errors="coerce")
        valid = group.status.astype(str).eq("ok")
        n = len(group)
        wins = int(((success == 1) & valid).sum())
        low, high = wilson(wins, n)
        summary_rows.append({
            "attack": attack,
            "model": model,
            "successes_itt": wins,
            "n": n,
            "judge_errors": int((~valid).sum()),
            "valid_n": int(valid.sum()),
            "successes_among_valid": int((success[valid] == 1).sum()),
            "rate_itt_pct": 100 * wins / n,
            "wilson95_low_itt_pct": low,
            "wilson95_high_itt_pct": high,
            "judge_scope": "D3 primary judge; eligible 90-prompt held-out intersection; unweighted",
        })
    summary_path = HERE / f"a5_primary_d3_{D3_RUN_TAG}_cell_rates.csv"
    pd.DataFrame(summary_rows).sort_values(["model", "attack"]).to_csv(summary_path, index=False, float_format="%.6f")
    note_path = HERE / f"a5_primary_d3_{D3_RUN_TAG}_summary.md"
    note_lines = [
        "# A5 primary-judge common-prompt analysis",
        "",
        "This analysis scores the 90 A7-eligible canonical held-out prompt identities under the frozen D3 rubric with the separately approved OpenRouter control `reasoning.enabled=false`. The 18 attack/exact-checkpoint cells contain 1,620 response rows. The 75 held-out HarmBench contextual prompts are excluded because their source contexts were absent from the evaluated prompts. Rates are unweighted for this three-benchmark intersection and are not estimates for the full 713-prompt eligible pool.",
        "",
        "The judge sees only the original harmful request and final response. The request ledger stores row IDs, hashes, category/verdict/status, provider metadata, and attempt counts, not prompt or response text. `judge_error` and `empty_response` rows remain in the intention-to-evaluate denominator as unresolved/non-success; valid-only counts are reported separately. The two Llama checkpoints remain separate: PiF/MetaCipher use Llama-3.2-3B-Instruct, while ArrAttack uses Llama-3.1-8B-Instruct.",
        "",
        "| Attack | Model alias / checkpoint scope | D3 successes / n | ITE rate [95% Wilson CI] | Valid labels | Errors |",
        "|---|---|---:|---:|---:|---:|",
    ]
    checkpoint = {"llama": "Llama-3.2-3B for PiF/MetaCipher; Llama-3.1-8B for ArrAttack"}
    for row in sorted(summary_rows, key=lambda r: (r["model"], r["attack"])):
        cp = checkpoint.get(row["model"], row["model"])
        note_lines.append(
            f"| {row['attack']} | {cp} | {row['successes_itt']}/{row['n']} | "
            f"{row['rate_itt_pct']:.1f}% [{row['wilson95_low_itt_pct']:.1f}, {row['wilson95_high_itt_pct']:.1f}] | "
            f"{row['valid_n']}/{row['n']} | {row['judge_errors']} |"
        )
    note_lines.extend(["", "The A5 D3 run is a common-prompt judging subset, not completion of study-wide J1. Attack-versus-attack and attack-versus-control tests are handled separately under S2 once their required rows and controls are available.", ""])
    note_path.write_text("\n".join(note_lines), encoding="utf-8")
    manifest = {
        "task_id": "A5-D3",
        "n_eligible_prompts": 90,
        "attack_checkpoint_cells": 18,
        "d3_rows": len(ledger),
        "route": route_info,
        "run_id": D3_RUN_TAG,
        "judge_configuration": {"temperature": 0, "top_p": 1, "max_tokens": 256, "stream": False,
                                 "reasoning": D3_REASONING,
                                 "retry": "one identical retry for transport, empty, or invalid JSON"},
        "raw_prompt_or_response_text_in_outputs": False,
        "row_identity": ["attack", "exact checkpoint scope", "dataset", "prompt_idx"],
        "statuses": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "input_sha256": source_hashes,
        "output_sha256": {p.name: file_hash(p) for p in (ledger_path, summary_path, note_path)},
        "analysis_script_sha256": file_hash(Path(__file__)),
    }
    manifest_path = HERE / f"a5_primary_d3_{D3_RUN_TAG}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(ledger), "cells": len(summary_rows), "statuses": manifest["statuses"],
                      "priced_route": route_info}, indent=4, ensure_ascii=False), flush=True)


def salvage_primary_d3_errors_once() -> None:
    """Apply the approved successful diagnostic verdict, then one final call per remaining failed row."""
    import os
    import a4_existing_evidence as A4

    ledger_path = HERE / f"a5_primary_d3_{D3_RUN_TAG}_ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"No D3 ledger to salvage: {ledger_path}")
    ledger = pd.read_csv(ledger_path).fillna("")
    if len(ledger) != 23 or ledger.duplicated(["attack", "model", "canonical_key"]).any():
        raise ValueError(f"Salvage gate expects the verified 23-row partial ledger, got {len(ledger)} rows")
    failed = ledger.loc[ledger.status == "judge_error"].copy()
    if len(failed) != 12:
        raise ValueError(f"Salvage gate expects 12 failed rows, got {len(failed)}")
    inputs, _ = build_primary_d3_inputs()
    input_by_id = {(r["attack"], r["model"], r["canonical_key"]): r for r in inputs}
    diag_id = ("pif", "qwen", "jailbreakbench|83")
    diag_matches = ledger.index[
        (ledger.attack == diag_id[0]) & (ledger.model == diag_id[1]) & (ledger.canonical_key == diag_id[2])
    ].tolist()
    if len(diag_matches) != 1 or diag_id not in input_by_id:
        raise ValueError("The approved diagnostic row is not unique in the current D3 ledger")
    diag_index = diag_matches[0]
    body, body_hash = build_d3_request(input_by_id[diag_id], A4)
    if ledger.loc[diag_index, "request_sha256"] != body_hash:
        raise ValueError("The diagnostic row request hash differs from its recorded D3 request")

    # The user approved incorporating this valid, redacted diagnostic verdict into the row ledger.
    ledger.loc[diag_index, "status"] = "ok"
    ledger.loc[diag_index, "category"] = "off_target_or_clarification"
    ledger.loc[diag_index, "success"] = 0
    ledger.loc[diag_index, "api_attempts"] = 3
    ledger.loc[diag_index, "provider"] = "Nvidia"
    ledger.loc[diag_index, "served_model"] = D3_MODEL
    ledger.loc[diag_index, "failure_detail"] = "prior two attempts: invalid_json; approved diagnostic retry returned valid strict JSON"

    def persist() -> None:
        temp_path = ledger_path.with_name(f".{ledger_path.name}.tmp")
        try:
            ledger.to_csv(temp_path, index=False)
            os.replace(temp_path, ledger_path)
        finally:
            temp_path.unlink(missing_ok=True)

    persist()
    api_key, route_info = d3_preflight()
    remaining_ids = [
        (str(row.attack), str(row.model), str(row.canonical_key))
        for row in failed.itertuples()
        if (str(row.attack), str(row.model), str(row.canonical_key)) != diag_id
    ]
    if len(remaining_ids) != 11:
        raise ValueError(f"Salvage gate expected 11 remaining failed rows, got {len(remaining_ids)}")
    schedule_lock = threading.Lock()
    next_request_at = [0.0]
    worker_local = threading.local()
    for position, row_id in enumerate(remaining_ids, start=1):
        record = input_by_id[row_id]
        body, body_hash = build_d3_request(record, A4)
        session = getattr(worker_local, "session", None)
        if session is None:
            session = requests.Session()
            worker_local.session = session
        provider = served_model = ""
        status = "judge_error"
        category = ""
        success: bool | None = None
        failure_detail = ""
        with schedule_lock:
            wait = next_request_at[0] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_request_at[0] = time.monotonic() + D3_ATTEMPT_INTERVAL_SECONDS
        try:
            response = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                         "X-Title": "TMLR R4 A5 bounded D3 salvage"},
                json=body,
                timeout=120,
            )
            payload = response.json() if response.content else {}
            provider = str(payload.get("provider") or "")
            served_model = str(payload.get("model") or "")
            if response.status_code != 200:
                failure_detail = f"http_{response.status_code}"
            else:
                content = payload.get("choices", [{}])[0].get("message", {}).get("content") or ""
                category, success, parse_status = A4.parse_d3_result(content)
                if parse_status == "ok":
                    status = "ok"
                else:
                    failure_detail = "empty_content" if not content else parse_status
        except Exception as exc:
            failure_detail = f"transport_or_parse:{type(exc).__name__}"

        matches = ledger.index[
            (ledger.attack == row_id[0]) & (ledger.model == row_id[1]) & (ledger.canonical_key == row_id[2])
        ].tolist()
        if len(matches) != 1:
            raise ValueError(f"Salvage row key is not unique: {row_id}")
        idx = matches[0]
        ledger.loc[idx, "status"] = status
        ledger.loc[idx, "category"] = category if status == "ok" else ""
        ledger.loc[idx, "success"] = (int(success) if status == "ok" and success is not None else "")
        ledger.loc[idx, "api_attempts"] = 3
        ledger.loc[idx, "provider"] = provider
        ledger.loc[idx, "served_model"] = served_model
        ledger.loc[idx, "failure_detail"] = failure_detail
        if ledger.loc[idx, "request_sha256"] != body_hash:
            raise ValueError(f"Salvage request hash changed for {row_id}")
        persist()
        print(f"A5 D3 bounded salvage progress: {position}/11; last_status={status}", flush=True)

    print(json.dumps({
        "salvaged_rows": len(remaining_ids) + 1,
        "api_requests_sent": len(remaining_ids),
        "ledger_rows": len(ledger),
        "ledger_statuses": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "route": route_info,
        "raw_response_text_persisted": False,
    }, indent=4, ensure_ascii=False), flush=True)


def canonical_pool_key(dataset: object, prompt_idx: object) -> str:
    return f"{dataset}|{int(prompt_idx)}"


def comparable_prompt(text: object) -> str:
    """Normalize only known serialization differences, never as a row key."""
    return re.sub(r"\s+", " ", str(text).replace(chr(34) * 2, chr(34))).strip()


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return 100 * (center - half) / denom, 100 * (center + half) / denom


def cochran_q(matrix: list[list[bool]]) -> tuple[float, float]:
    """Cochran's Q for a rows-by-treatments binary matrix."""
    n = len(matrix)
    k = len(matrix[0])
    col_totals = [sum(int(row[j]) for row in matrix) for j in range(k)]
    row_totals = [sum(int(v) for v in row) for row in matrix]
    total = sum(col_totals)
    numerator = (k - 1) * (k * sum(value * value for value in col_totals) - total * total)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        return 0.0, 1.0
    statistic = max(0.0, numerator / denominator)
    return statistic, float(chi2.sf(statistic, k - 1))


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    matrix_rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    reference_keys: set[str] | None = None
    reference_prompt_hashes: dict[str, str] | None = None
    per_model: dict[str, pd.DataFrame] = {}
    cell_counts: list[dict] = []
    pool = pd.read_csv(TIER0 / "02_dedup_and_config_audit" / "pool_913.csv")
    if len(pool) != 913 or pool.duplicated(list(KEY)).any():
        raise ValueError("canonical prompt pool must contain 913 unique (dataset, prompt_idx) rows")
    pool["_key"] = [canonical_pool_key(r.dataset, r.prompt_idx) for r in pool.itertuples()]
    pool_by_key = pool.set_index("_key", drop=False)
    canonical_prompt_hash = {
        key: hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        for key, prompt in zip(pool["_key"], pool["original_prompt"])
    }
    source_hashes[str((TIER0 / "02_dedup_and_config_audit" / "pool_913.csv").relative_to(REPO))] = file_hash(
        TIER0 / "02_dedup_and_config_audit" / "pool_913.csv"
    )

    for model in C.MODELS:
        cells: dict[str, pd.DataFrame] = {}
        key_sets: dict[str, set[str]] = {}
        prompt_hashes: dict[str, dict[str, str]] = {}
        for attack in ATTACKS:
            path = Path(C.path_for(attack, model))
            source_hashes[str(path.relative_to(REPO))] = file_hash(path)
            frame = C.load(attack, model).copy()
            raw_keys = frame[list(KEY)].astype(str).agg("|".join, axis=1)
            if raw_keys.duplicated().any():
                raise ValueError(f"duplicate source row key in {attack}/{model}")
            canonical_keys: list[str] = []
            if attack in {"pif", "metacipher"}:
                for row in frame.itertuples():
                    key = canonical_pool_key(row.dataset, row.prompt_idx)
                    if key not in pool_by_key.index:
                        raise ValueError(f"{attack}/{model}: source row key absent from canonical pool")
                    if comparable_prompt(row.original_prompt) != comparable_prompt(pool_by_key.loc[key, "original_prompt"]):
                        raise ValueError(f"{attack}/{model}: prompt text does not match canonical key {key}")
                    canonical_keys.append(key)
            else:
                # ArrAttack's prompt_idx is 1-based within its selected tail, not
                # the canonical pool index. Reconstruct the frozen tail by source
                # order, then map each selected row back to the immutable pool key.
                raw_to_canonical: dict[str, str] = {}
                for dataset, selected in frame.groupby("dataset", sort=False):
                    canonical_subset = pool.loc[pool["dataset"].astype(str) == str(dataset)].sort_values("prompt_idx")
                    tail = canonical_subset.tail(len(selected)).reset_index(drop=True)
                    ordered = selected.sort_values("prompt_idx")
                    if ordered["prompt_idx"].astype(int).tolist() != list(range(1, len(ordered) + 1)):
                        raise ValueError(f"{model}: ArrAttack local prompt indices are not contiguous for {dataset}")
                    if ordered["original_prompt"].astype(str).tolist() != tail["original_prompt"].astype(str).tolist():
                        raise ValueError(f"{model}: ArrAttack source rows do not match the canonical selected tail for {dataset}")
                    for source_row, pool_row in zip(ordered.itertuples(), tail.itertuples()):
                        source_key = f"{source_row.dataset}|{int(source_row.prompt_idx)}"
                        raw_to_canonical[source_key] = canonical_pool_key(pool_row.dataset, pool_row.prompt_idx)
                canonical_keys = [raw_to_canonical[key] for key in raw_keys]
            if len(set(canonical_keys)) != len(canonical_keys):
                raise ValueError(f"duplicate canonical row key after mapping: {attack}/{model}")
            frame["_key"] = canonical_keys
            frame["_success"] = C.success_series(attack, frame).astype(bool)
            if frame["original_prompt"].isna().any():
                raise ValueError(f"missing original prompt in {attack}/{model}")
            cells[attack] = frame.set_index("_key", drop=False)
            key_sets[attack] = set(canonical_keys)
            prompt_hashes[attack] = {
                key: canonical_prompt_hash[key] for key in canonical_keys
            }

        heldout = key_sets["arrattack"]
        if len(heldout) != 165:
            raise ValueError(f"{model}: expected 165 ArrAttack held-out keys, got {len(heldout)}")
        for attack in ("pif", "metacipher"):
            if not heldout.issubset(key_sets[attack]):
                raise ValueError(f"{model}: held-out key coverage mismatch for {attack}")
        for key in heldout:
            hashes = {prompt_hashes[attack][key] for attack in ATTACKS}
            if len(hashes) != 1:
                raise ValueError(f"{model}: prompt text differs for canonical key {key}")
        if reference_keys is None:
            reference_keys = heldout
            reference_prompt_hashes = prompt_hashes["arrattack"]
        elif heldout != reference_keys:
            raise ValueError(f"{model}: ArrAttack held-out key set differs across victims")
        elif any(prompt_hashes["arrattack"][key] != reference_prompt_hashes[key] for key in heldout):
            raise ValueError(f"{model}: held-out prompt text differs across victims")

        model_matrix = pd.DataFrame(index=sorted(heldout))
        for attack in ATTACKS:
            subset = cells[attack].loc[sorted(heldout)]
            model_matrix[attack] = subset["_success"].astype(bool).to_numpy()
            successes = int(model_matrix[attack].sum())
            n = len(model_matrix)
            low, high = wilson(successes, n)
            cell_counts.append({
                "model": model,
                "attack": attack,
                "successes": successes,
                "n": n,
                "asr_pct": 100 * successes / n,
                "wilson95_low_pct": low,
                "wilson95_high_pct": high,
                "judge_scope": "attack_specific_as_stored",
            })
        per_model[model] = model_matrix
        for key, row in model_matrix.iterrows():
            original_hash = prompt_hashes["arrattack"][key]
            dataset, prompt_idx = key.split("|", 1)
            matrix_rows.append({
                "model": model,
                "dataset": dataset,
                "prompt_idx": prompt_idx,
                "original_prompt_sha256": original_hash,
                **{f"{attack}_success": int(row[attack]) for attack in ATTACKS},
            })

    omnibus_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    for model, frame in per_model.items():
        binary = frame[list(ATTACKS)].astype(bool).values.tolist()
        q_stat, q_p = cochran_q(binary)
        omnibus_rows.append({"model": model, "n_paired": len(frame), "cochran_q": q_stat, "p_value": q_p})
        for left_idx, left in enumerate(ATTACKS):
            for right in ATTACKS[left_idx + 1:]:
                a = frame[left].astype(bool)
                b = frame[right].astype(bool)
                left_only = int((a & ~b).sum())
                right_only = int((~a & b).sum())
                discordant = left_only + right_only
                p_value = 1.0 if discordant == 0 else float(
                    binomtest(min(left_only, right_only), discordant, 0.5, alternative="two-sided").pvalue
                )
                pairwise_rows.append({
                    "model": model,
                    "attack_a": left,
                    "attack_b": right,
                    "n_paired": len(frame),
                    "a_only": left_only,
                    "b_only": right_only,
                    "discordant_n": discordant,
                    "mcnemar_exact_p": p_value,
                })

    adjusted = holm([row["mcnemar_exact_p"] for row in pairwise_rows])
    for row, p_adjusted in zip(pairwise_rows, adjusted):
        row["holm_p"] = p_adjusted

    pd.DataFrame(cell_counts).to_csv(OUT / "a5_official_cell_rates.csv", index=False, float_format="%.4f")
    pd.DataFrame(omnibus_rows).to_csv(OUT / "a5_cochran_q.csv", index=False, float_format="%.8g")
    pd.DataFrame(pairwise_rows).to_csv(OUT / "a5_mcnemar_holm.csv", index=False, float_format="%.8g")
    pd.DataFrame(matrix_rows).to_csv(OUT / "a5_official_row_ledger.csv", index=False)

    metadata = {
        "task_id": "A5",
        "source_scope": "Existing held-out rows only; attack-specific judgments as stored",
        "n_common_prompts": len(reference_keys or ()),
        "victims": C.MODELS,
        "attacks": list(ATTACKS),
        "row_key": ["dataset", "prompt_idx"],
        "ArrAttack_source_index_mapping": "ArrAttack prompt_idx is local to the selected tail; mapped by verified per-dataset source order and exact prompt text to pool_913 canonical IDs.",
        "prompt_text_in_outputs": False,
        "primary_D3_judge": "study-wide J1 pending",
        "pairwise_multiplicity": "Holm adjustment over all 18 victim-by-pair McNemar tests",
        "inputs_sha256": source_hashes,
    }
    partial_d3_ledger = OUT / "a5_primary_d3_reasoning_disabled_ledger.csv"
    partial_d3 = None
    if partial_d3_ledger.exists():
        partial_d3 = pd.read_csv(partial_d3_ledger).fillna("")
        metadata["primary_D3_partial_attempt"] = {
            "ledger_sha256": file_hash(partial_d3_ledger),
            "rows_attempted": len(partial_d3),
            "approved_scope_rows": 1620,
            "status_counts": {str(k): int(v) for k, v in partial_d3.status.value_counts().items()},
            "interpretation": "incomplete D3 adjudication; not a cell-rate estimate and not study-wide J1",
        }
    primary_d3_manifest = OUT / "a5_primary_d3_reasoning_disabled_manifest.json"
    if primary_d3_manifest.exists():
        metadata["primary_D3_judge"] = "scored on the 90-row A7-eligible common held-out subset with reasoning.enabled=false; study-wide J1 pending"
        metadata["primary_D3_manifest_sha256"] = file_hash(primary_d3_manifest)
    elif partial_d3 is not None:
        metadata["primary_D3_judge"] = "partial reasoning-disabled D3 attempt only; study-wide J1 pending"
    rates_by_key = {(row["model"], row["attack"]): row for row in cell_counts}
    q_by_model = {row["model"]: row for row in omnibus_rows}
    summary = [
        "# A5 common-prompt paired analysis",
        "",
        "The held-out ArrAttack prompts were mapped to the immutable `pool_913.csv` identities. ArrAttack's stored `prompt_idx` is a 1-based local index in each selected benchmark tail, not the canonical pool index. For every victim, all four selected tails matched the canonical pool in exact source order: 75 HarmBench, 50 StrongREJECT, 20 JailbreakBench, and 20 MaliciousInstruct prompts. The 165 resulting canonical row keys were identical across all six victims. PiF and MetaCipher were checked against their canonical pool row keys; the single known quote-serialization variation was normalized only for source validation, not for row identity.",
        "",
        "This report uses the attack-specific judgments already stored with each attack. It is an as-stored paired comparison, not a common-rubric comparison. Do not use it as the D3 primary-judge result or as evidence that judge differences have been resolved. " + ("D3 scoring for the 90-row A7-eligible subset is reported separately in `a5_primary_d3_reasoning_disabled_summary.md`; study-wide J1 remains pending." if primary_d3_manifest.exists() else ("A partial reasoning-disabled D3 attempt is in `a5_primary_d3_reasoning_disabled_ledger.csv`; it has incomplete coverage and is not an estimate. Study-wide J1 remains pending." if partial_d3 is not None else "The D3 result can be joined later on the canonical row keys after J1 is completed.")),
        "",
        "## Cell rates",
        "",
        "| Victim | PiF | MetaCipher | ArrAttack |",
        "|---|---:|---:|---:|",
    ]
    for model in C.MODELS:
        cells = [rates_by_key[(model, attack)] for attack in ATTACKS]
        values = [f"{r['successes']}/{r['n']} ({r['asr_pct']:.2f}%, 95% Wilson {r['wilson95_low_pct']:.2f} to {r['wilson95_high_pct']:.2f})" for r in cells]
        summary.append(f"| {model} | " + " | ".join(values) + " |")
    summary.extend([
        "",
        "## Within-victim omnibus tests",
        "",
        "Cochran's Q is calculated from the three paired binary outcomes within each victim.",
        "",
        "| Victim | N | Q | p |",
        "|---|---:|---:|---:|",
    ])
    for model in C.MODELS:
        row = q_by_model[model]
        summary.append(f"| {model} | {row['n_paired']} | {row['cochran_q']:.4f} | {row['p_value']:.6g} |")
    summary.extend([
        "",
        "Pairwise exact McNemar tests and Holm-adjusted p-values across all 18 victim-by-pair tests are in `a5_mcnemar_holm.csv`. The response-free row-level outcome ledger is `a5_official_row_ledger.csv`; source hashes and the key-mapping definition are in `a5_manifest.json`.",
        "",
    ])
    (OUT / "a5_summary.md").write_text("\n".join(summary), encoding="utf-8")
    metadata["analysis_script_sha256"] = file_hash(HERE / "a5_common_prompt_analysis.py")
    metadata["outputs_sha256"] = {
        name: file_hash(OUT / name)
        for name in (
            "a5_official_cell_rates.csv",
            "a5_cochran_q.csv",
            "a5_mcnemar_holm.csv",
            "a5_official_row_ledger.csv",
            "a5_summary.md",
        )
    }
    (OUT / "a5_manifest.json").write_text(json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"A5 complete: {len(C.MODELS)} victims x {len(reference_keys or ())} common prompts")
    print("Attack-specific-judge paired results only; no D3/common-judge claim is made.")
    for row in cell_counts:
        print(f"{row['model']},{row['attack']},{row['successes']}/{row['n']},{row['asr_pct']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-primary-d3", action="store_true", help="Run the approved reasoning-disabled D3 batch for the 90-row A7-eligible A5 subset")
    parser.add_argument("--salvage-primary-d3-errors-once", action="store_true", help="Run the user-approved single salvage attempt for 11 remaining failed D3 rows")
    args = parser.parse_args()
    if args.run_primary_d3 and args.salvage_primary_d3_errors_once:
        parser.error("Choose either --run-primary-d3 or --salvage-primary-d3-errors-once")
    elif args.salvage_primary_d3_errors_once:
        salvage_primary_d3_errors_once()
    elif args.run_primary_d3:
        run_primary_d3()
    else:
        main()

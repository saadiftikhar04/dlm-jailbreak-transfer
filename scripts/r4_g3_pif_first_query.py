#!/usr/bin/env python3
"""Build the D4 PiF first-victim-response ledger, preserving source provenance.

Historical causal rows with exactly one recorded victim query are retained as
historical first responses. Missing rows receive one freshly generated first
optimizer query. The regenerated candidate is explicitly labeled; it is not
claimed to be the lost historical prompt or response.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "tier0" / "02_dedup_and_config_audit" / "pool_913.csv"
MATRIX_PATH = ROOT / "revision_experiments" / "r4" / "A1_config_audit" / "config_matrix.csv"
PLAN_PATH = ROOT / "submissions" / "TMLR" / "r4_plan.txt"
OUTPUT_ROOT = ROOT / "revision_experiments" / "r4" / "G3_first_response" / "pif"
POOL_SHA256 = "12bafde1c37c56349b4566b97f839efcd8579996357d7e76ed0b2dee1f92346b"
MATRIX_SHA256 = "93d761c73a0880454e2af2e09123c5788a841390e0d71ba42042afac693adf8a"
BERT_ID = "google-bert/bert-large-uncased"
BERT_REVISION = "6da4b6a26a1877e173fca3225479512db81a5e5b"
N_PROMPTS = 913
PIF_ORDER = ("harmbench", "jailbreakbench", "malicious_instruct", "strongreject")
MODEL_CONFIGS = [
    ("qwen", "qwen2.5", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "ar"),
    ("llama", "llama", "meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95", "ar"),
    ("falcon", "falcon", "tiiuae/Falcon-H1R-7B", "a6f74bf181389908efd6970878d7ee2b42f5d417", "ar"),
    ("llada", "llada", "GSAI-ML/LLaDA-1.5", "84346fd91ba60252d260022201ad6fc5a3468fb2", "dlm"),
    ("dream", "dream", "Dream-org/Dream-v0-Instruct-7B", "05334cb9faaf763692dcf9d8737c642be2b2a6ae", "dlm"),
    ("diffucoder", "diffucoder", "apple/DiffuCoder-7B-Instruct", "4fdd4580064ca5d11808069ce78f88d068753c96", "dlm"),
]
HISTORICAL_DIR = ROOT / "results" / "pif" / "PIF_JUDGED"
ORDERED_DATASETS = tuple(PIF_ORDER)
BASE_SEED = 20261011

LEGACY_FIELDS = [
    "task_id", "model_alias", "checkpoint_id", "checkpoint_revision",
    "dataset", "prompt_idx", "canonical_key", "source_status", "status",
    "source_total_query", "seed", "query_count", "generation_config_sha256",
    "original_prompt_sha256", "attacked_prompt_sha256", "response_sha256",
    "original_prompt", "attacked_prompt", "response", "error_type",
]
FIELDS = [*LEGACY_FIELDS, "error_message"]
RETRYABLE_STATUSES = {
    "generation_error", "attack_generation_error", "no_victim_query",
    "empty_response", "trace_mismatch",
}
TERMINAL_STATUSES = {"ok"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_int(value: object) -> int:
    return int(float(str(value).strip()))


def canonical_key(row: dict) -> str:
    return f"{row['dataset']}|{normalize_int(row['prompt_idx'])}"


def file_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def exception_message(exc: Exception) -> str:
    return " ".join(str(exc).split())[:1000]


def read_ledger(path: Path, config_sha: str, model_alias: str) -> dict[str, dict]:
    if not path.is_file() or not path.stat().st_size:
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (LEGACY_FIELDS, FIELDS):
            raise RuntimeError(f"Unexpected G3 ledger schema: {path}")
        rows = list(reader)
    result = {}
    for row in rows:
        key = row["canonical_key"]
        if key != canonical_key(row):
            raise RuntimeError(f"G3 ledger canonical-key mismatch for {key}: {path}")
        if row["model_alias"] != model_alias:
            raise RuntimeError(f"G3 ledger model mismatch for {key}: {path}")
        if row["generation_config_sha256"] != config_sha:
            raise RuntimeError(f"G3 generation config changed for {key}: {path}")
        if row["status"] not in RETRYABLE_STATUSES | TERMINAL_STATUSES:
            raise RuntimeError(f"Unrecognized G3 row status {row['status']!r} for {key}: {path}")
        if key in result:
            raise RuntimeError(f"Duplicate canonical key in G3 ledger: {key}")
        result[key] = row
    return result


def retryable_rows(rows: dict[str, dict]) -> dict[str, dict]:
    return {key: row for key, row in rows.items() if row["status"] in RETRYABLE_STATUSES}


def select_retry_keys(
    source_rows: dict[str, dict],
    completed_keys: set[str],
    only_key: str | None = None,
    limit: int | None = None,
) -> set[str]:
    if only_key is not None and limit is not None:
        raise ValueError("Use either --only-key or --limit, not both")
    failed_keys = retryable_rows(source_rows)
    if only_key is not None:
        if only_key not in failed_keys:
            raise ValueError(f"--only-key is not a retryable key in the source ledger: {only_key}")
        selected = [only_key]
    else:
        selected = [key for key in failed_keys if key not in completed_keys]
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be a positive integer")
            selected = selected[:limit]
    return set(selected) - completed_keys


def ordered_pool() -> list[dict]:
    if sha256_file(POOL_PATH) != POOL_SHA256:
        raise RuntimeError("Frozen 913-prompt pool hash mismatch")
    rows = file_rows(POOL_PATH)
    keys = [(row["dataset"], normalize_int(row["prompt_idx"])) for row in rows]
    if len(rows) != N_PROMPTS or len(set(keys)) != N_PROMPTS:
        raise RuntimeError("Frozen prompt pool must contain 913 unique canonical row keys")
    if set(row["dataset"] for row in rows) != set(PIF_ORDER):
        raise RuntimeError("Unexpected benchmark set in the frozen PIF pool")
    return [
        row
        for dataset in PIF_ORDER
        for row in sorted(
            (x for x in rows if x["dataset"] == dataset),
            key=lambda x: normalize_int(x["prompt_idx"]),
        )
    ]


def pif_first_query_generation_config(model_key: str, model_type: str) -> dict:
    if model_type == "dlm":
        native_defaults = {
            "llada": {
                "max_new_tokens": 512, "steps": 512, "block_length": 128,
                "temperature": 0.0,
            },
            "dream": {
                "max_new_tokens": 512, "steps": 512, "temperature": 0.0,
                "top_p": None, "alg": "origin", "alg_temp": None,
            },
            "diffucoder": {
                "max_new_tokens": 512, "steps": 512, "temperature": 0.3,
                "top_p": 0.95, "alg": "entropy", "alg_temp": 0.0,
            },
        }[model_key]
        return {
            "model_key": model_key,
            "model_type": model_type,
            "prompt_encoding_call": (
                "tokenizer.apply_chat_template(chat, return_tensors='pt', "
                "truncation=True, max_length=128)"
            ),
            "generation_call": (
                "pif_target_models.target_generate_from_input_ids(model, tokenizer, "
                "model_key, captured_input_ids, generation_config=native_dlm_parameters)"
            ),
            "native_dlm_parameters": native_defaults,
            "generation_defaults": "pinned model-specific native diffusion settings; 512 new tokens",
            "completion_decoding": {
                "llada": "tokenizer.decode(skip_special_tokens=True)",
                "dream": (
                    "tokenizer.decode(skip_special_tokens=False); split '<|endoftext|>'; "
                    "remove '<|im_end|>'"
                ),
                "diffucoder": (
                    "tokenizer.decode(skip_special_tokens=False); split '<|dlm_pad|>'; "
                    "remove '<|im_end|>'"
                ),
            }[model_key],
            "capture_policy": (
                "pass through attack_mlm's exact encoded input_ids; use the pinned model's "
                "native diffusion API without prompt re-encoding"
            ),
        }
    return {
        "model_key": model_key,
        "model_type": model_type,
        "prompt_encoding_call": (
            "tokenizer.apply_chat_template(chat, return_tensors='pt', "
            "truncation=True, max_length=128)"
        ),
        "generation_call": "model.generate(input_ids=input_ids, max_length=512)",
        "generation_defaults": "pinned checkpoint defaults; no explicit sampling overrides",
        "capture_policy": "invoke the exact generate call made by attack_mlm; do not re-encode or substitute a helper call",
    }


def legacy_pif_first_query_generation_config(model_key: str, model_type: str) -> dict:
    return {
        "model_key": model_key,
        "model_type": model_type,
        "prompt_encoding_call": (
            "tokenizer.apply_chat_template(chat, return_tensors='pt', "
            "truncation=True, max_length=128)"
        ),
        "generation_call": "model.generate(input_ids=input_ids, max_length=512)",
        "generation_defaults": "pinned checkpoint defaults; no explicit sampling overrides",
        "capture_policy": "invoke the exact generate call made by attack_mlm; do not re-encode or substitute a helper call",
    }


def historical_first_rows(model_alias: str, model_type: str, pool: list[dict]) -> dict[str, dict]:
    path = HISTORICAL_DIR / f"{model_alias}_pif_final_judged.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Historical PiF judged file missing: {path}")
    rows = file_rows(path)
    if len(rows) != N_PROMPTS:
        raise RuntimeError(f"{model_alias}: expected 913 historical rows, found {len(rows)}")
    by_key = {}
    pool_by_key = {
        f"{row['dataset']}|{normalize_int(row['prompt_idx'])}": row
        for row in pool
    }
    for row in rows:
        dataset = row["dataset"]
        prompt_idx = normalize_int(row["prompt_idx"])
        key = f"{dataset}|{prompt_idx}"
        if key not in pool_by_key:
            raise RuntimeError(f"{model_alias}: historical PiF row outside pool: {key}")
        if row["original_prompt"].strip() != pool_by_key[key]["original_prompt"].strip():
            raise RuntimeError(f"{model_alias}: original prompt mismatch for {key}")
        if key in by_key:
            raise RuntimeError(f"{model_alias}: duplicate historical PiF key {key}")
        by_key[key] = row
    if set(by_key) != set(pool_by_key):
        raise RuntimeError(f"{model_alias}: historical PiF ledger does not cover the canonical pool")

    selected = {}
    is_dlm = model_type == "dlm"
    for key, row in by_key.items():
        count = normalize_int(row.get("total_query", "0") or "0")
        response = row.get("victim_output", "")
        if not is_dlm and count == 1:
            selected[key] = {
                "source_status": "historical_query1",
                "source_total_query": count,
                "attacked_prompt": row.get("pif_prompt", ""),
                "response": response,
                "status": "ok" if response.strip() else "empty_response",
                "error_type": "",
            }
        else:
            selected[key] = {
                "source_status": "regenerate_first_query",
                "source_total_query": count,
                "attacked_prompt": "",
                "response": "",
                "status": "pending",
                "error_type": "",
            }
    return selected


def sha256_config(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


class CaptureTokenizer:
    def __init__(self, wrapped):
        object.__setattr__(self, "wrapped", wrapped)
        object.__setattr__(self, "pending_prompt", "")
        object.__setattr__(self, "chat_template_inputs", [])

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __setattr__(self, name, value):
        if name in {"wrapped", "pending_prompt", "chat_template_inputs"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.wrapped, name, value)

    def apply_chat_template(self, conversations, *args, **kwargs):
        if isinstance(conversations, list) and conversations:
            first = conversations[0]
            if isinstance(first, dict) and first.get("role") == "user":
                object.__setattr__(self, "pending_prompt", str(first.get("content", "")))
        result = self.wrapped.apply_chat_template(conversations, *args, **kwargs)
        if hasattr(result, "detach"):
            self.chat_template_inputs.append(result.detach().cpu().clone())
        return result


class CaptureModel:
    def __init__(self, wrapped, tokenizer, model_key, model_type, generation_config,
                 query_records, native_dlm_generator, native_dlm_decoder):
        self.wrapped = wrapped
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.model_type = model_type
        self.generation_config = generation_config
        self.query_records = query_records
        self.native_dlm_generator = native_dlm_generator
        self.native_dlm_decoder = native_dlm_decoder

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise RuntimeError("PiF first-query capture received no input_ids")
        if self.query_records:
            raise RuntimeError("G3 PiF one-iteration guard observed more than one victim query")
        prompt = self.tokenizer.pending_prompt
        if not prompt:
            raise RuntimeError("PiF query prompt was not captured")
        started = time.time()
        try:
            if self.model_type == "dlm":
                output = self.native_dlm_generator(
                    self.wrapped,
                    self.tokenizer.wrapped,
                    self.model_key,
                    input_ids,
                    max_new_tokens=int(self.generation_config["native_dlm_parameters"]["max_new_tokens"]),
                    generation_config=self.generation_config["native_dlm_parameters"],
                )
            else:
                output = self.wrapped.generate(*args, **kwargs)
            prompt_length = int(input_ids.shape[-1])
            generated_ids = output[0, prompt_length:]
            if self.model_type == "dlm":
                response = self.native_dlm_decoder(
                    self.tokenizer.wrapped, self.model_key, generated_ids
                )
            else:
                response = self.tokenizer.wrapped.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
            self.query_records.append({
                "attacked_prompt": prompt,
                "response": response,
                "query_status": "ok" if response.strip() else "empty_response",
                "error_type": "",
                "error_message": "",
                "generation_seconds": round(time.time() - started, 4),
            })
            return output
        except Exception as exc:
            self.query_records.append({
                "attacked_prompt": prompt,
                "response": "",
                "query_status": "generation_error",
                "error_type": type(exc).__name__,
                "error_message": exception_message(exc),
                "generation_seconds": round(time.time() - started, 4),
            })
            raise


def private_writer(path: Path):
    exists = path.is_file() and path.stat().st_size > 0
    fieldnames = FIELDS
    if exists:
        with path.open(encoding="utf-8", newline="") as existing_stream:
            fieldnames = next(csv.reader(existing_stream), None)
        if fieldnames not in (LEGACY_FIELDS, FIELDS):
            raise RuntimeError(f"Unexpected G3 output schema: {path}")
    stream = path.open("a" if exists else "w", encoding="utf-8", newline="")
    os.chmod(path, 0o600)
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    if not exists:
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    return stream, writer


def read_completed(path: Path, config_sha: str) -> dict[str, dict]:
    if not path.is_file() or not path.stat().st_size:
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if reader.fieldnames not in (LEGACY_FIELDS, FIELDS):
        raise RuntimeError(f"Unexpected G3 output schema: {path}")
    result = read_ledger(path, config_sha, rows[0]["model_alias"] if rows else "")
    retryable = {key: row["status"] for key, row in result.items() if row["status"] in RETRYABLE_STATUSES}
    if retryable:
        examples = ", ".join(f"{key}:{status}" for key, status in list(retryable.items())[:5])
        raise RuntimeError(
            "Existing G3 output contains failed rows and cannot be resumed in place. "
            "Preserve it as the retry source and write a separate continuation ledger. "
            f"Examples: {examples}"
        )
    return result


def write_private_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def load_bert(hf_home: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    cache = hf_home / "hub"
    snapshot = Path(snapshot_download(
        repo_id=BERT_ID,
        revision=BERT_REVISION,
        cache_dir=str(cache),
        local_files_only=True,
    ))
    if snapshot.name != BERT_REVISION:
        raise RuntimeError("Pinned G3 BERT snapshot revision mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(snapshot), local_files_only=True, output_hidden_states=True
    ).to("cuda:0").eval()
    return model, tokenizer, {"model_id": BERT_ID, "revision": BERT_REVISION}


def run_model(
    model_index: int,
    retry_failures_from: Path | None = None,
    output_path_arg: Path | None = None,
    only_key: str | None = None,
    retry_limit: int | None = None,
) -> None:
    import numpy as np
    import torch
    import attack_mlm
    import pif_target_models as target

    model_alias, model_key, checkpoint_id, checkpoint_revision, model_type = MODEL_CONFIGS[model_index]
    if sha256_file(POOL_PATH) != POOL_SHA256 or sha256_file(MATRIX_PATH) != MATRIX_SHA256:
        raise RuntimeError("G3 frozen input hash mismatch")
    pool = ordered_pool()
    historical_path = HISTORICAL_DIR / f"{model_alias}_pif_final_judged.csv"
    historical = historical_first_rows(model_alias, model_type, pool)
    output_dir = OUTPUT_ROOT / model_alias
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    default_output_path = output_dir / "g3_pif_first_response.csv"
    if retry_failures_from is None:
        if output_path_arg is not None or only_key is not None or retry_limit is not None:
            raise ValueError("--output-path, --only-key, and --limit require --retry-failures-from")
        output_path = default_output_path
    else:
        if output_path_arg is None:
            raise ValueError("Failure continuation requires an explicit --output-path")
        retry_failures_from = retry_failures_from.expanduser().resolve()
        output_path = output_path_arg.expanduser().resolve()
        if output_path == retry_failures_from:
            raise ValueError("Continuation output must be separate from its source ledger")
        if output_path.parent != output_dir.resolve():
            raise ValueError(f"Continuation output must stay in the existing model directory: {output_dir}")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"Continuation output directory does not exist: {output_path.parent}")
        if only_key is not None and retry_limit is not None:
            raise ValueError("Use either --only-key or --limit, not both")
        if retry_limit is not None and retry_limit <= 0:
            raise ValueError("--limit must be a positive integer")

    generation_config = pif_first_query_generation_config(model_key, model_type)
    base_config_payload = {
        "task_id": "G3",
        "model_alias": model_alias,
        "checkpoint_id": checkpoint_id,
        "checkpoint_revision": checkpoint_revision,
        "pif_optimizer_iterations": 1,
        "pif_optimizer_first_query_only": True,
        "pif_optimizer_parameters": {
            "evaluation_template": "This intent is [MASK]",
            "top_n": 15, "top_m": 15, "top_k": 15,
            "warm_up": 0, "temperature": 0.25, "semantic_threshold": 0.85,
        },
        "generation_config": legacy_pif_first_query_generation_config(model_key, model_type),
        "seed_rule": "20261011 + model index * 100000 + stable prompt ordinal",
    }
    parent_config_sha = sha256_config(base_config_payload)
    config_payload = dict(base_config_payload)
    config_payload["generation_config"] = generation_config
    config_sha = sha256_config(config_payload)
    sources = {
        "runner": Path(__file__).resolve(),
        "pool_913": POOL_PATH,
        "a1_config_matrix": MATRIX_PATH,
        "historical_judged_pif": historical_path,
        "pif_attack": ROOT / "scripts" / "attack_mlm.py",
        "pif_target_models": ROOT / "scripts" / "pif_target_models.py",
        "pif_eval_template": ROOT / "scripts" / "eval_template.py",
        "execution_plan": PLAN_PATH,
    }
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    completed = read_completed(output_path, config_sha)
    if completed and set(completed) - {canonical_key(row) for row in pool}:
        raise RuntimeError("G3 resume ledger contains keys outside pool_913")

    retry_source_rows: dict[str, dict] = {}
    retry_source_sha = ""
    retry_source_key_sha = ""
    retry_manifest_path = None
    if retry_failures_from is not None:
        retry_source_rows = read_ledger(retry_failures_from, parent_config_sha, model_alias)
        if not retry_source_rows:
            raise RuntimeError(f"Failure-continuation source ledger is empty: {retry_failures_from}")
        if set(retry_source_rows) - {canonical_key(row) for row in pool}:
            raise RuntimeError("Failure-continuation source contains keys outside pool_913")
        failure_rows = retryable_rows(retry_source_rows)
        if not failure_rows:
            raise RuntimeError(f"No retryable G3 rows in source ledger: {retry_failures_from}")
        failure_keys = set(failure_rows)
        if set(completed) - failure_keys:
            raise RuntimeError("Continuation output contains keys not marked retryable in its source ledger")
        selected_keys = select_retry_keys(retry_source_rows, set(completed), only_key, retry_limit)
        if not selected_keys and not (only_key is not None and only_key in completed):
            raise RuntimeError("No pending retryable G3 rows match this continuation invocation")
        retry_source_sha = sha256_file(retry_failures_from)
        retry_source_key_sha = sha256_text("\n".join(sorted(failure_keys)))
        retry_manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
        retry_identity = {
            "task_id": "G3",
            "run_mode": "failure_continuation",
            "model_alias": model_alias,
            "generation_config_sha256": config_sha,
            "parent_generation_config_sha256": parent_config_sha,
            "source_sha256": source_hashes,
            "parent_ledger_name": retry_failures_from.name,
            "parent_ledger_sha256": retry_source_sha,
            "parent_retryable_rows": len(failure_keys),
            "parent_retryable_keys_sha256": retry_source_key_sha,
            "retryable_source_status_counts": dict(Counter(row["status"] for row in failure_rows.values())),
            "raw_prompts_or_responses_in_manifest": False,
        }
        if retry_manifest_path.exists():
            prior_manifest = json.loads(retry_manifest_path.read_text(encoding="utf-8"))
            for field, expected in retry_identity.items():
                if prior_manifest.get(field) != expected:
                    raise RuntimeError(f"Continuation manifest mismatch for {field}: {retry_manifest_path}")
            if prior_manifest.get("status") == "complete":
                if not output_path.is_file() or sha256_file(output_path) != prior_manifest.get("output_sha256", {}).get(output_path.name):
                    raise RuntimeError("Completed continuation output hash mismatch")
                print(f"G3 PiF {model_alias}: continuation already complete; verified output hash", flush=True)
                return
            if prior_manifest.get("status") == "completed_with_errors":
                raise RuntimeError(
                    "This continuation contains failed rows. Preserve it and start a distinct "
                    "continuation from those failures after diagnosis."
                )
        elif output_path.exists():
            raise RuntimeError(
                f"Continuation output already exists without its run manifest: {output_path}; "
                "refusing to adopt or overwrite it"
            )
        if retry_manifest_path.exists():
            prior_manifest = json.loads(retry_manifest_path.read_text(encoding="utf-8"))
            recorded_output_sha = prior_manifest.get("output_sha256", {}).get(output_path.name)
            if output_path.exists() and not recorded_output_sha:
                raise RuntimeError("Continuation output exists but its manifest has no checkpoint hash")
            if recorded_output_sha and (
                not output_path.is_file() or sha256_file(output_path) != recorded_output_sha
            ):
                raise RuntimeError("Continuation output changed since its last checkpoint")
            if only_key is not None and only_key in completed:
                print(
                    f"G3 PiF {model_alias}: diagnostic key {only_key} already exists in the "
                    "verified continuation ledger; no inference repeated",
                    flush=True,
                )
                return
        elif only_key is not None and only_key in completed:
            raise RuntimeError("Diagnostic key is already present but continuation manifest is missing")
        retry_manifest_payload = {
            **retry_identity,
            "status": "running",
            "output_csv": output_path.name,
            "selected_key_count": len(failure_keys),
            "selected_key_sha256": retry_source_key_sha,
            "only_key_filter_for_this_invocation": only_key,
            "limit_for_this_invocation": retry_limit,
            "invocation_selected_key_count": len(selected_keys),
            "output_sha256": ({output_path.name: sha256_file(output_path)} if output_path.exists() else {}),
        }
        write_private_json(retry_manifest_path, retry_manifest_payload)
    elif only_key is not None:
        raise ValueError("--only-key requires failure-continuation mode")

    if retry_failures_from is None:
        selected_keys = {canonical_key(row) for row in pool} - set(completed)
    else:
        selected_keys = select_retry_keys(retry_source_rows, set(completed), only_key, retry_limit)
    if not selected_keys:
        raise RuntimeError("No pending G3 rows remain for this invocation")

    bert_model, bert_tokenizer, bert_info = load_bert(Path(os.environ["HF_HOME"]))
    target_model, target_tokenizer = target.load_target(
        model_key, offline=True, revision=checkpoint_revision
    )
    runtime_revision = getattr(getattr(target_model, "config", None), "_commit_hash", None)
    if runtime_revision != checkpoint_revision:
        raise RuntimeError(f"{model_alias} loaded revision mismatch: {runtime_revision}")

    class_counts = Counter()
    regenerated = Counter()
    stream, writer = private_writer(output_path)
    rows_written = 0
    for ordinal, row in enumerate(pool, start=1):
        key = canonical_key(row)
        if key not in selected_keys or key in completed:
            continue
        source = historical[key]
        source_total_query = source["source_total_query"]
        if retry_failures_from is None and source["source_status"] == "historical_query1":
            attacked_prompt = source["attacked_prompt"]
            response = source["response"]
            status = "ok" if response.strip() else "empty_response"
            error_type = ""
            error_message = ""
            source_status = source["source_status"]
            seed = ""
            query_count = 1
        else:
            seed = BASE_SEED + model_index * 100000 + ordinal
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            query_records: list[dict] = []
            wrapped_tokenizer = CaptureTokenizer(target_tokenizer)
            wrapped_model = CaptureModel(
                target_model, wrapped_tokenizer, model_key, model_type,
                generation_config, query_records, target.target_generate_from_input_ids,
                target.decode_target_completion,
            )
            attack_mlm.generate_attack._step_log = []
            attacked_prompt = ""
            response = ""
            status = "no_victim_query"
            error_type = ""
            error_message = ""
            query_count = 0
            try:
                reported, _, flags, final_prompts, _ = attack_mlm.generate_attack(
                    bert_model,
                    bert_tokenizer,
                    wrapped_model,
                    wrapped_tokenizer,
                    [row["original_prompt"]],
                    "This intent is [MASK]",
                    objective="ASR",
                    iterations=1,
                    top_n=15,
                    top_m=15,
                    top_k=15,
                    warm_up=0,
                    temperature=0.25,
                    threshold=0.85,
                    device=torch.device("cuda:0"),
                )
                query_count = int(reported)
                if len(query_records) > 1:
                    raise RuntimeError("G3 one-query guard captured multiple victim calls")
                if query_records:
                    attacked_prompt = str(query_records[0]["attacked_prompt"])
                    response = str(query_records[0]["response"])
                    status = str(query_records[0]["query_status"])
                    error_type = str(query_records[0]["error_type"])
                    error_message = str(query_records[0].get("error_message", ""))
                    if query_count != len(query_records):
                        status = "trace_mismatch"
                elif final_prompts:
                    attacked_prompt = str(final_prompts[0])
            except Exception as exc:
                status = "generation_error" if query_records else "attack_generation_error"
                error_type = type(exc).__name__
                error_message = (
                    str(query_records[0].get("error_message", ""))
                    if query_records else exception_message(exc)
                )
                if query_records:
                    attacked_prompt = str(query_records[0]["attacked_prompt"])
                    response = str(query_records[0]["response"])
                    query_count = len(query_records)
            source_status = (
                "retry_after_prior_failure" if retry_failures_from is not None
                else "regenerated_first_query"
            )
            regenerated[status] += 1
            del wrapped_model, wrapped_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        writer.writerow({
            "task_id": "G3",
            "model_alias": model_alias,
            "checkpoint_id": checkpoint_id,
            "checkpoint_revision": runtime_revision,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "source_status": source_status,
            "status": status,
            "source_total_query": source_total_query,
            "seed": seed,
            "query_count": query_count,
            "generation_config_sha256": config_sha,
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "attacked_prompt_sha256": sha256_text(attacked_prompt) if attacked_prompt else "",
            "response_sha256": sha256_text(response) if response else "",
            "original_prompt": row["original_prompt"],
            "attacked_prompt": attacked_prompt,
            "response": response,
            "error_type": error_type,
            "error_message": error_message,
        })
        stream.flush()
        os.fsync(stream.fileno())
        class_counts[status] += 1
        completed[key] = {}
        rows_written += 1
        if retry_manifest_path is not None:
            retry_manifest_payload["status"] = "running"
            retry_manifest_payload["rows_written_this_invocation"] = rows_written
            retry_manifest_payload["last_completed_key"] = key
            retry_manifest_payload["output_sha256"] = {output_path.name: sha256_file(output_path)}
            write_private_json(retry_manifest_path, retry_manifest_payload)
        if rows_written % 20 == 0 or rows_written == len(selected_keys):
            print(
                f"G3 PiF {model_alias}: continuation {rows_written}/{len(selected_keys)} "
                f"(pool ordinal {ordinal}/{len(pool)})",
                flush=True,
            )
    stream.close()

    rows = file_rows(output_path)
    keys = [row["canonical_key"] for row in rows]
    if len(rows) != len(set(keys)):
        raise RuntimeError(f"G3 PiF {model_alias} output contains duplicate canonical keys")
    expected_keys = {canonical_key(row) for row in pool} if retry_failures_from is None else {
        key for key, row in retry_source_rows.items() if row["status"] in RETRYABLE_STATUSES
    }
    if set(keys) - expected_keys:
        raise RuntimeError(f"G3 PiF {model_alias} output contains keys outside the selected source set")
    if retry_failures_from is None and len(rows) != N_PROMPTS:
        raise RuntimeError(f"G3 PiF {model_alias} full ledger has {len(rows)}, expected {N_PROMPTS} rows")
    if retry_failures_from is not None and only_key is None and retry_limit is None and set(keys) != expected_keys:
        raise RuntimeError(
            f"G3 PiF {model_alias} continuation has {len(keys)}/{len(expected_keys)} selected keys"
        )
    status_counts = dict(Counter(row["status"] for row in rows))
    has_retryable_errors = any(row["status"] in RETRYABLE_STATUSES for row in rows)
    continuation_complete = retry_failures_from is not None and set(keys) == expected_keys and not has_retryable_errors
    if retry_failures_from is None:
        manifest_status = "completed_with_errors" if has_retryable_errors else "complete"
    else:
        manifest_status = "completed_with_errors" if has_retryable_errors else (
            "complete" if continuation_complete else "partial"
        )
    manifest = {
        **config_payload,
        "run_mode": "failure_continuation" if retry_failures_from is not None else "full_pool",
        "generation_config_sha256": config_sha,
        "bert_runtime": bert_info,
        "runtime_generation_api": generation_config["generation_call"],
        "historical_rows": sum(row["source_status"] == "historical_query1" for row in rows),
        "regenerated_rows": sum(row["source_status"] in {"regenerated_first_query", "retry_after_prior_failure"} for row in rows),
        "status_counts": status_counts,
        "query_count_distribution": dict(Counter(str(row["query_count"]) for row in rows)),
        "canonical_pool_rows": N_PROMPTS,
        "unique_keys": len(set(keys)),
        "rows_written_this_invocation": rows_written,
        "source_sha256": source_hashes,
        "output_sha256": {output_path.name: sha256_file(output_path)},
        "raw_prompts_or_responses_in_manifest": False,
        "raw_working_data_access_mode": "mode 0600 CSV in mode 0700 task directory",
        "historical_revision_limitation": "Original PiF step logs are absent; regenerated prompts and responses are labeled as current first-query reruns, not exact historical recoveries.",
    }
    manifest_path = retry_manifest_path or (output_dir / f"g3_pif_{model_alias}_manifest.json")
    manifest["status"] = manifest_status
    if retry_failures_from is not None:
        manifest.update(retry_identity)
        manifest["status"] = manifest_status
        manifest["selected_key_count"] = len(expected_keys)
        manifest["invocation_selected_key_count"] = len(selected_keys)
        manifest["selected_key_sha256"] = retry_source_key_sha
        manifest["output_csv"] = output_path.name
        manifest["parent_ledger_name"] = retry_failures_from.name
        manifest["parent_ledger_sha256"] = retry_source_sha
        manifest["retryable_source_status_counts"] = dict(Counter(
            row["status"] for row in retry_source_rows.values() if row["status"] in RETRYABLE_STATUSES
        ))
    write_private_json(manifest_path, manifest)
    print({"task_id": "G3", "model": model_alias, "rows": len(rows),
           "regenerated": manifest["regenerated_rows"], "statuses": manifest["status_counts"],
           "run_mode": manifest["run_mode"], "manifest_status": manifest.get("status", "complete")}, flush=True)
    if has_retryable_errors:
        raise RuntimeError(
            f"G3 PiF {model_alias} wrote {sum(status_counts.get(s, 0) for s in RETRYABLE_STATUSES)} "
            "retryable error rows; inspect the restricted error_message field before another run"
        )
    del target_model, bert_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, choices=range(len(MODEL_CONFIGS)), required=True)
    parser.add_argument("--retry-failures-from", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--only-key")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME must point to the approved shared Hugging Face cache")
    if sha256_file(MATRIX_PATH) != MATRIX_SHA256:
        raise RuntimeError("A1 configuration matrix changed since G3 design")
    run_model(
        args.model_index,
        retry_failures_from=args.retry_failures_from,
        output_path_arg=args.output_path,
        only_key=args.only_key,
        retry_limit=args.limit,
    )


if __name__ == "__main__":
    main()

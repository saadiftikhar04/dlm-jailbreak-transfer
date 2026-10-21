#!/usr/bin/env python3
"""Generate the approved R4 G1 Llama-3.1 cells with row-keyed query traces.

Raw prompt and response fields are restricted working data. They are written
only to mode-0600 CSVs in the mode-0700 G1 output directory. Judge labels are
not generated here; D3 judging is a separate plan stage.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R4_TASK = os.environ.get("R4_TASK", "G1")
DEFAULT_OUT_DIRS = {
    "G1": ROOT / "revision_experiments" / "r4" / "G1_llama31_8b",
    "G4": ROOT / "revision_experiments" / "r4" / "G4_llada_locked" / "metacipher",
}
if R4_TASK not in DEFAULT_OUT_DIRS:
    raise ValueError(f"Unsupported R4 task: {R4_TASK}")
OUT_DIR = Path(os.environ.get("R4_OUT_DIR", str(DEFAULT_OUT_DIRS[R4_TASK])))
POOL_PATH = ROOT / "tier0" / "02_dedup_and_config_audit" / "pool_913.csv"
MATRIX_PATH = ROOT / "revision_experiments" / "r4" / "A1_config_audit" / "config_matrix.csv"
FEEDBACK_CONFIG_PATH = ROOT / "revision_experiments" / "r4" / "A1_config_audit" / "a1_pif_feedback_query_config.csv"
PLAN_PATH = ROOT / "submissions" / "TMLR" / "r4_plan.txt"
N_PROMPTS = 913
POOL_SHA256 = "12bafde1c37c56349b4566b97f839efcd8579996357d7e76ed0b2dee1f92346b"
MATRIX_SHA256 = "93d761c73a0880454e2af2e09123c5788a841390e0d71ba42042afac693adf8a"
MODEL_CONFIGS = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "ar"),
    "llada": ("GSAI-ML/LLaDA-1.5", "84346fd91ba60252d260022201ad6fc5a3468fb2", "dlm"),
}
MODEL_ALIAS = os.environ.get("R4_MODEL_ALIAS", "llama31_8b")
if MODEL_ALIAS not in MODEL_CONFIGS:
    raise ValueError(f"Unsupported R4 model alias: {MODEL_ALIAS}")
if R4_TASK == "G4" and MODEL_ALIAS != "llada":
    raise ValueError("G4 is reserved for the locked-schedule LLaDA MetaCipher rerun")
TARGET_ID, TARGET_REVISION, MODEL_TYPE = MODEL_CONFIGS[MODEL_ALIAS]
META_MODEL_KEY = "r4_llama31_8b" if MODEL_ALIAS == "llama31_8b" else MODEL_ALIAS
BERT_ID = "google-bert/bert-large-uncased"
BERT_REVISION = "6da4b6a26a1877e173fca3225479512db81a5e5b"
PI_F_ORDER = ("harmbench", "jailbreakbench", "malicious_instruct", "strongreject")
META_ORDER = ("harmbench", "malicious_instruct", "strongreject", "jailbreakbench")

PIF_QUERY_FIELDS = [
    "attack", "model_alias", "checkpoint_id", "checkpoint_revision", "generation_config_sha256",
    "dataset", "prompt_idx", "canonical_key", "query_index", "query_status",
    "original_prompt_sha256", "attacked_prompt_sha256", "response_sha256",
    "original_prompt", "attacked_prompt", "response", "generation_seconds",
]
PIF_SUMMARY_FIELDS = [
    "attack", "model_alias", "checkpoint_id", "checkpoint_revision", "generation_config_sha256",
    "dataset", "prompt_idx", "canonical_key", "status", "query_count_reported",
    "query_count_captured", "first_response_status", "first_response_sha256",
    "last_response_sha256", "attack_success_internal", "selected_prompt_sha256",
    "original_prompt_sha256", "error_type",
]
META_QUERY_FIELDS = [
    "attack", "model_alias", "checkpoint_id", "checkpoint_revision", "generation_config_sha256",
    "dataset", "prompt_idx", "canonical_key", "query_index", "method_attempt", "query_status",
    "cipher", "method_judge_success", "method_reward", "method_failure_mode",
    "original_prompt_sha256", "attacked_prompt_sha256", "response_sha256",
    "original_prompt", "attacked_prompt", "response", "generation_seconds",
]
META_SUMMARY_FIELDS = [
    "attack", "model_alias", "checkpoint_id", "checkpoint_revision", "generation_config_sha256",
    "dataset", "prompt_idx", "canonical_key", "status", "query_count",
    "first_response_status", "first_response_sha256", "returned_attempts",
    "returned_response_sha256", "method_success", "method_failure_mode", "method_reward",
    "original_prompt_sha256", "error_type",
]
META_GENERATION_CONFIG = {
    "chat_encoding": "apply_chat_template(tokenize=false, add_generation_prompt=true), then tokenizer default special-token behavior",
    "max_input_tokens": 2048,
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}
G4_LLADA_GENERATION_CONFIG = {
    "chat_encoding": "apply_chat_template(tokenize=false, add_generation_prompt=true), then tokenizer default special-token behavior",
    "max_input_tokens": 2048,
    "max_new_tokens": 512,
    "diffusion_steps": 512,
    "gen_length": 512,
    "block_length": 128,
    "temperature": 0.0,
    "remasking": "low_confidence",
}
PIF_FEEDBACK_CONTROL_CONFIG = {
    "max_input_tokens": 128,
    "max_length": 512,
    "use_chat_template_tokenizer": True,
    "add_generation_prompt": False,
    "use_model_generation_defaults": True,
}
PIF_CONTROL_FIELDS = [
    "model_alias", "checkpoint_id", "checkpoint_revision", "config_sha256",
    "dataset", "prompt_idx", "canonical_key", "original_prompt_sha256",
    "response_sha256", "status", "failure_type", "generation_seconds",
    "original_prompt", "response",
]
PIF_CONTROL_SUMMARY_FIELDS = [
    "model_alias", "checkpoint_id", "checkpoint_revision", "config_sha256",
    "dataset", "prompt_idx", "canonical_key", "status", "response_sha256",
    "original_prompt_sha256", "failure_type",
]
META_CONTROL_FIELDS = [
    "model_alias", "checkpoint_id", "checkpoint_revision", "config_sha256",
    "dataset", "prompt_idx", "canonical_key", "original_prompt_sha256",
    "response_sha256", "status", "failure_type", "generation_seconds",
    "original_prompt", "response",
]
META_CONTROL_SUMMARY_FIELDS = [
    "model_alias", "checkpoint_id", "checkpoint_revision", "config_sha256",
    "dataset", "prompt_idx", "canonical_key", "status", "response_sha256",
    "original_prompt_sha256", "failure_type",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_key(row: dict) -> str:
    return f"{row['dataset']}|{int(row['prompt_idx'])}"


def ensure_private_directory() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    logs = OUT_DIR / "logs"
    logs.mkdir(exist_ok=True)
    os.chmod(logs, 0o700)


def file_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ordered_pool(order: tuple[str, ...]) -> list[dict]:
    rows = file_rows(POOL_PATH)
    keys = [(row["dataset"], int(row["prompt_idx"])) for row in rows]
    if len(rows) != N_PROMPTS or len(set(keys)) != N_PROMPTS:
        raise RuntimeError("The canonical pool must contain 913 unique (dataset, prompt_idx) keys")
    if sha256_file(POOL_PATH) != POOL_SHA256:
        raise RuntimeError("Canonical pool hash differs from the frozen R4 input")
    counts = Counter(row["dataset"] for row in rows)
    if set(counts) != set(order):
        raise RuntimeError(f"Unexpected canonical-pool benchmark set: {sorted(counts)}")
    result = []
    for dataset in order:
        result.extend(sorted((row for row in rows if row["dataset"] == dataset),
                             key=lambda item: int(item["prompt_idx"])))
    if len(result) != N_PROMPTS or len({canonical_key(row) for row in result}) != N_PROMPTS:
        raise RuntimeError("Method-specific prompt order did not preserve the 913 canonical keys")
    return result


def check_sources() -> dict[str, str]:
    sources = {
        "pool_913": POOL_PATH,
        "a1_config_matrix": MATRIX_PATH,
        "a1_pif_feedback_query_config": FEEDBACK_CONFIG_PATH,
        "r4_plan": PLAN_PATH,
        "g1_runner": Path(__file__).resolve(),
        "pif_target_models": ROOT / "scripts" / "pif_target_models.py",
        "pif_runner": ROOT / "scripts" / "run_pif.py",
        "pif_attack": ROOT / "scripts" / "attack_mlm.py",
        "pif_eval_template": ROOT / "scripts" / "eval_template.py",
        "metacipher": ROOT / "scripts" / "metacipher_multi.py",
    }
    batch_name = "r4_g1_generate.sbatch" if R4_TASK == "G1" else "r4_g4_llada_metacipher.sbatch"
    sources[f"{R4_TASK.lower()}_sbatch"] = ROOT / "scripts" / batch_name
    if sha256_file(MATRIX_PATH) != MATRIX_SHA256:
        raise RuntimeError("A1 configuration matrix hash differs from the frozen R4 input")
    return {name: sha256_file(path) for name, path in sources.items()}


def generation_config_hash(model) -> str:
    config = model.generation_config.to_dict()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


def actual_revision(model, tokenizer) -> str | None:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not revision:
        revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    return revision


def private_csv_writer(path: Path, fields: list[str]):
    exists = path.exists() and path.stat().st_size > 0
    stream = path.open("a" if exists else "w", encoding="utf-8", newline="")
    os.chmod(path, 0o600)
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    if not exists:
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    return stream, writer


def read_completed(path: Path) -> set[str]:
    if not path.exists() or not path.stat().st_size:
        return set()
    rows = file_rows(path)
    keys = [row["canonical_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate prompt identity in resume ledger: {path.name}")
    return set(keys)


def load_bert_snapshot():
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError("HF_HOME must be set to the shared model cache")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    snapshot = snapshot_download(
        repo_id=BERT_ID,
        revision=BERT_REVISION,
        cache_dir=str(Path(hf_home) / "hub"),
        local_files_only=False,
    )
    if Path(snapshot).name != BERT_REVISION:
        raise RuntimeError("Downloaded BERT snapshot revision did not match its pinned commit")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        snapshot, local_files_only=True, output_hidden_states=True
    ).to("cuda:0").eval()
    runtime_revision = getattr(model.config, "_commit_hash", None)
    if runtime_revision and runtime_revision != BERT_REVISION:
        raise RuntimeError(f"Loaded BERT revision mismatch: {runtime_revision}")
    snapshot_hashes = {
        name: sha256_file(Path(snapshot) / name)
        for name in ("config.json", "tokenizer_config.json", "vocab.txt")
        if (Path(snapshot) / name).is_file()
    }
    return model, tokenizer, {
        "model_id": BERT_ID,
        "revision": BERT_REVISION,
        "snapshot_file_sha256": snapshot_hashes,
    }


def make_capture_wrappers(model, tokenizer, query_records: list[dict]):
    class CaptureTokenizer:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.pending_prompt = ""

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def apply_chat_template(self, conversations, *args, **kwargs):
            if isinstance(conversations, list) and conversations:
                first = conversations[0]
                if isinstance(first, dict) and first.get("role") == "user":
                    self.pending_prompt = str(first.get("content", ""))
            return self.wrapped.apply_chat_template(conversations, *args, **kwargs)

    class CaptureModel:
        def __init__(self, wrapped, capture_tokenizer):
            self.wrapped = wrapped
            self.capture_tokenizer = capture_tokenizer

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def generate(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            started = time.time()
            try:
                output = self.wrapped.generate(*args, **kwargs)
            except Exception as exc:
                query_records.append({
                    "attacked_prompt": self.capture_tokenizer.pending_prompt,
                    "response": "",
                    "query_status": "generation_error",
                    "error_type": type(exc).__name__,
                    "generation_seconds": round(time.time() - started, 4),
                })
                self.capture_tokenizer.pending_prompt = ""
                raise
            prompt_len = int(input_ids.shape[-1]) if input_ids is not None else 0
            response = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
            query_records.append({
                "attacked_prompt": self.capture_tokenizer.pending_prompt,
                "response": response,
                "query_status": "ok" if response.strip() else "empty_response",
                "error_type": "",
                "generation_seconds": round(time.time() - started, 4),
            })
            self.capture_tokenizer.pending_prompt = ""
            return output

    wrapped_tokenizer = CaptureTokenizer(tokenizer)
    wrapped_model = CaptureModel(model, wrapped_tokenizer)
    return wrapped_model, wrapped_tokenizer


def append_pif_prompt_rows(prompt_rows: list[dict], completed: set[str], checkpoint_id: str,
                           checkpoint_revision: str, gen_config_sha: str, bert_model,
                           bert_tokenizer, target_model, target_tokenizer) -> dict:
    import torch
    import attack_mlm
    import eval_template
    from transformers import AutoModelForMaskedLM

    pif_queries_path = OUT_DIR / "pif_query_ledger.csv"
    pif_prompts_path = OUT_DIR / "pif_prompt_ledger.csv"
    query_stream, query_writer = private_csv_writer(pif_queries_path, PIF_QUERY_FIELDS)
    summary_stream, summary_writer = private_csv_writer(pif_prompts_path, PIF_SUMMARY_FIELDS)
    device = torch.device("cuda:0")
    counts = Counter()
    bert_device = next(bert_model.parameters()).device
    if bert_device != device:
        raise RuntimeError(f"PiF BERT model is on {bert_device}, expected {device}")

    class CaptureTokenizer:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.pending_prompt = ""

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def apply_chat_template(self, conversations, *args, **kwargs):
            if isinstance(conversations, list) and conversations:
                first = conversations[0]
                if isinstance(first, dict) and first.get("role") == "user":
                    self.pending_prompt = str(first.get("content", ""))
            return self.wrapped.apply_chat_template(conversations, *args, **kwargs)

    class CaptureModel:
        def __init__(self, wrapped, wrapped_tokenizer, records):
            self.wrapped = wrapped
            self.wrapped_tokenizer = wrapped_tokenizer
            self.records = records

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def generate(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            started = time.time()
            try:
                output = self.wrapped.generate(*args, **kwargs)
            except Exception as exc:
                self.records.append({
                    "attacked_prompt": self.wrapped_tokenizer.pending_prompt,
                    "response": "",
                    "query_status": "generation_error",
                    "error_type": type(exc).__name__,
                    "generation_seconds": round(time.time() - started, 4),
                })
                self.wrapped_tokenizer.pending_prompt = ""
                raise
            prompt_len = int(input_ids.shape[-1]) if input_ids is not None else 0
            response = target_tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
            self.records.append({
                "attacked_prompt": self.wrapped_tokenizer.pending_prompt,
                "response": response,
                "query_status": "ok" if response.strip() else "empty_response",
                "error_type": "",
                "generation_seconds": round(time.time() - started, 4),
            })
            self.wrapped_tokenizer.pending_prompt = ""
            return output

    # Match run_pif.py's ASR-feedback objective, optimizer parameters, and per-query target path.
    for position, row in enumerate(prompt_rows, start=1):
        key = canonical_key(row)
        if key in completed:
            continue
        query_records: list[dict] = []
        capture_tokenizer = CaptureTokenizer(target_tokenizer)
        capture_model = CaptureModel(target_model, capture_tokenizer, query_records)
        attack_mlm.generate_attack._step_log = []
        status = "ok"
        error_type = ""
        query_count = 0
        success_flag = False
        selected_prompt = ""
        try:
            query_count, _, flags, final_prompts, final_responses = attack_mlm.generate_attack(
                bert_model,
                bert_tokenizer,
                capture_model,
                capture_tokenizer,
                [row["original_prompt"]],
                "This intent is [MASK]",
                objective="ASR",
                iterations=50,
                top_n=15,
                top_m=15,
                top_k=15,
                warm_up=0,
                temperature=0.25,
                threshold=0.85,
                device=device,
            )
            success_flag = bool(flags[0]) if flags else False
            selected_prompt = str(final_prompts[0]) if final_prompts else ""
            if int(query_count) != len(query_records):
                status = "trace_mismatch"
        except Exception as exc:
            status = "generation_error"
            error_type = type(exc).__name__
        first = query_records[0] if query_records else None
        for query_index, item in enumerate(query_records, start=1):
            prompt_value = str(item.get("attacked_prompt", ""))
            response_value = str(item.get("response", ""))
            query_writer.writerow({
                "attack": "pif",
                "model_alias": MODEL_ALIAS,
                "checkpoint_id": checkpoint_id,
                "checkpoint_revision": checkpoint_revision,
                "generation_config_sha256": gen_config_sha,
                "dataset": row["dataset"],
                "prompt_idx": row["prompt_idx"],
                "canonical_key": key,
                "query_index": query_index,
                "query_status": item["query_status"],
                "original_prompt_sha256": sha256_text(row["original_prompt"]),
                "attacked_prompt_sha256": sha256_text(prompt_value),
                "response_sha256": sha256_text(response_value),
                "original_prompt": row["original_prompt"],
                "attacked_prompt": prompt_value,
                "response": response_value,
                "generation_seconds": item["generation_seconds"],
            })
        first_response = str(first["response"]) if first else ""
        summary_writer.writerow({
            "attack": "pif",
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": checkpoint_id,
            "checkpoint_revision": checkpoint_revision,
            "generation_config_sha256": gen_config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "status": status if query_records else ("no_victim_query" if status == "ok" else status),
            "query_count_reported": query_count,
            "query_count_captured": len(query_records),
            "first_response_status": first["query_status"] if first else "missing",
            "first_response_sha256": sha256_text(first_response) if first else "",
            "last_response_sha256": sha256_text(str(query_records[-1]["response"])) if query_records else "",
            "attack_success_internal": int(success_flag),
            "selected_prompt_sha256": sha256_text(selected_prompt) if selected_prompt else "",
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "error_type": error_type or (first["error_type"] if first else ""),
        })
        for stream in (query_stream, summary_stream):
            stream.flush()
            os.fsync(stream.fileno())
        completed.add(key)
        counts[status] += 1
        if position % 10 == 0 or position == len(prompt_rows):
            print(f"G1 PiF progress {position}/{len(prompt_rows)}", flush=True)

    query_stream.close()
    summary_stream.close()
    return {"prompt_rows": len(completed), "status_counts": dict(counts)}


def _load_meta_state(path: Path, framework, torch, np) -> set[str] | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if state.get("checkpoint_revision") != TARGET_REVISION:
        raise RuntimeError("MetaCipher resume checkpoint targets a different model revision")
    framework.q_table = defaultdict(
        lambda: defaultdict(float),
        {tuple(key): defaultdict(float, values) for key, values in state["q_table"].items()},
    )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    return set(state["completed_keys"])


def _save_meta_state(path: Path, framework, torch, np, completed_keys: set[str]) -> None:
    state = {
        "checkpoint_revision": TARGET_REVISION,
        "completed_keys": sorted(completed_keys),
        "q_table": {tuple(key): dict(values) for key, values in framework.q_table.items()},
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, path)
    os.chmod(path, 0o600)


def append_meta_rows(prompt_rows: list[dict], completed: set[str], checkpoint_id: str,
                     checkpoint_revision: str, gen_config_sha: str) -> dict:
    import torch
    import numpy as np
    import metacipher_multi as metacipher

    meta_queries_path = OUT_DIR / "metacipher_query_ledger.csv"
    meta_prompts_path = OUT_DIR / "metacipher_prompt_ledger.csv"
    state_path = OUT_DIR / "metacipher_adaptive_state.pkl"
    query_stream, query_writer = private_csv_writer(meta_queries_path, META_QUERY_FIELDS)
    summary_stream, summary_writer = private_csv_writer(meta_prompts_path, META_SUMMARY_FIELDS)
    completed = read_completed(meta_prompts_path)
    if meta_queries_path.exists() and meta_queries_path.stat().st_size:
        existing_queries = file_rows(meta_queries_path)
        query_identities = [(row["canonical_key"], int(row["query_index"])) for row in existing_queries]
        if len(query_identities) != len(set(query_identities)):
            raise RuntimeError("Duplicate MetaCipher query key in resume ledger")
        if {key for key, _ in query_identities} - completed:
            raise RuntimeError("MetaCipher has query rows without a completed prompt checkpoint")

    # The target identity is an R4-only key. The six historical model aliases remain unchanged.
    metacipher.MODEL_REGISTRY[META_MODEL_KEY] = (checkpoint_id, MODEL_TYPE)
    metacipher.HF_CACHE = str(Path(os.environ["HF_HOME"]) / "hub")
    if R4_TASK == "G4":
        # D7 locks LLaDA to 512 total steps over four 128-token blocks.
        metacipher.DLM_STEPS = G4_LLADA_GENERATION_CONFIG["diffusion_steps"]
    framework = metacipher.MetaCipherFramework(
        api_key=os.environ[
            "OPENROUTER_API_KEY" if R4_TASK == "G4" else "DEEPSEEK_API_KEY"
        ],
        model_key=META_MODEL_KEY,
        learning_rate=0.5,
        discount_factor=0.9,
        temperature=0.1,
        model_revision=TARGET_REVISION,
    )
    runtime_revision = actual_revision(framework.target_model, framework.target_tokenizer)
    if runtime_revision != checkpoint_revision:
        raise RuntimeError(f"MetaCipher victim revision mismatch: {runtime_revision}")
    state_completed = _load_meta_state(state_path, framework, torch, np)
    if completed and state_completed is None:
        raise RuntimeError("MetaCipher rows exist without their adaptive-state checkpoint; refusing an unfaithful resume")
    if state_completed is not None and state_completed != completed:
        raise RuntimeError("MetaCipher adaptive checkpoint and prompt ledger diverged; refusing an unfaithful resume")
    resumed = state_completed is not None
    if not resumed and not completed:
        _save_meta_state(state_path, framework, torch, np, completed)

    counts = Counter()
    original_query = framework._query_target
    for position, row in enumerate(prompt_rows, start=1):
        key = canonical_key(row)
        if key in completed:
            continue
        capture: list[dict] = []
        before_state = {
            "q_table": {tuple(k): dict(v) for k, v in framework.q_table.items()},
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

        def capture_query(prompt: str) -> str:
            started = time.time()
            try:
                response = original_query(prompt)
            except Exception as exc:
                capture.append({
                    "attacked_prompt": prompt,
                    "response": "",
                    "query_status": "generation_error",
                    "error_type": type(exc).__name__,
                    "generation_seconds": round(time.time() - started, 4),
                })
                raise
            capture.append({
                "attacked_prompt": prompt,
                "response": str(response),
                "query_status": "ok" if str(response).strip() else "empty_response",
                "error_type": "",
                "generation_seconds": round(time.time() - started, 4),
            })
            return response

        framework._query_target = capture_query
        error_type = ""
        result = None
        status = "ok"
        try:
            result = framework.attack(row["original_prompt"], dataset_name=row["dataset"], max_attempts=5)
        except Exception as exc:
            status = "attack_error"
            error_type = type(exc).__name__
            framework.q_table = defaultdict(
                lambda: defaultdict(float),
                {tuple(k): defaultdict(float, v) for k, v in before_state["q_table"].items()},
            )
            random.setstate(before_state["python_random_state"])
            np.random.set_state(before_state["numpy_random_state"])
            torch.set_rng_state(before_state["torch_cpu_rng_state"])
            if torch.cuda.is_available() and before_state["torch_cuda_rng_states"]:
                torch.cuda.set_rng_state_all(before_state["torch_cuda_rng_states"])
        finally:
            framework._query_target = original_query

        transformations = result.transformation_log if result is not None else []
        trace_metadata_match = len(transformations) == len(capture)
        for query_index, item in enumerate(capture, start=1):
            transform = transformations[query_index - 1] if query_index <= len(transformations) else {}
            attacked_prompt = str(item["attacked_prompt"])
            response = str(item["response"])
            query_writer.writerow({
                "attack": "metacipher",
                "model_alias": MODEL_ALIAS,
                "checkpoint_id": checkpoint_id,
                "checkpoint_revision": checkpoint_revision,
                "generation_config_sha256": gen_config_sha,
                "dataset": row["dataset"],
                "prompt_idx": row["prompt_idx"],
                "canonical_key": key,
                "query_index": query_index,
                "method_attempt": transform.get("attempt", "unknown"),
                "query_status": item["query_status"],
                "cipher": transform.get("cipher", ""),
                "method_judge_success": transform.get("success", ""),
                "method_reward": transform.get("reward", ""),
                "method_failure_mode": transform.get("failure_mode", ""),
                "original_prompt_sha256": sha256_text(row["original_prompt"]),
                "attacked_prompt_sha256": sha256_text(attacked_prompt),
                "response_sha256": sha256_text(response),
                "original_prompt": row["original_prompt"],
                "attacked_prompt": attacked_prompt,
                "response": response,
                "generation_seconds": item["generation_seconds"],
            })

        first = capture[0] if capture else None
        result_response = str(result.target_response) if result is not None else ""
        summary_writer.writerow({
            "attack": "metacipher",
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": checkpoint_id,
            "checkpoint_revision": checkpoint_revision,
            "generation_config_sha256": gen_config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "status": ("trace_mismatch" if status == "ok" and not trace_metadata_match else
                       status if capture else ("no_victim_query" if status == "ok" else status)),
            "query_count": len(capture),
            "first_response_status": first["query_status"] if first else "missing",
            "first_response_sha256": sha256_text(str(first["response"])) if first else "",
            "returned_attempts": result.attempts if result is not None else 0,
            "returned_response_sha256": sha256_text(result_response) if result else "",
            "method_success": result.success if result else "",
            "method_failure_mode": result.failure_mode if result else "",
            "method_reward": (transformations[-1].get("reward", "") if transformations else ""),
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "error_type": error_type,
        })
        for stream in (query_stream, summary_stream):
            stream.flush()
            os.fsync(stream.fileno())
        completed.add(key)
        _save_meta_state(state_path, framework, torch, np, completed)
        counts[status] += 1
        if position % 5 == 0 or position == len(prompt_rows):
            print(f"G1 MetaCipher progress {position}/{len(prompt_rows)}", flush=True)

    query_stream.close()
    summary_stream.close()
    return {"prompt_rows": len(completed), "status_counts": dict(counts), "resumed_adaptive_state": resumed}


def update_manifest(attack: str, statuses: dict, checkpoint_revision: str,
                    generation_config_sha: str, source_hashes: dict, extra: dict) -> None:
    query_path = OUT_DIR / f"{attack}_query_ledger.csv"
    summary_path = OUT_DIR / f"{attack}_prompt_ledger.csv"
    query_rows = file_rows(query_path) if query_path.exists() and query_path.stat().st_size else []
    summary_rows = file_rows(summary_path) if summary_path.exists() and summary_path.stat().st_size else []
    manifest = {
        "task_id": R4_TASK,
        "attack": attack,
        "model_alias": MODEL_ALIAS,
        "checkpoint_id": TARGET_ID,
        "checkpoint_revision": checkpoint_revision,
        "canonical_pool_rows": N_PROMPTS,
        "prompt_rows_in_ledger": len(summary_rows),
        "victim_query_rows": len(query_rows),
        "prompt_row_statuses": dict(Counter(row["status"] for row in summary_rows)),
        "query_row_statuses": dict(Counter(row["query_status"] for row in query_rows)),
        "generation_config_sha256": generation_config_sha,
        "source_sha256": source_hashes,
        "output_sha256": {
            query_path.name: sha256_file(query_path),
            summary_path.name: sha256_file(summary_path),
        },
        "raw_prompts_or_responses_in_manifest": False,
        "raw_output_access_mode": "restricted working data; mode 0600 files within mode 0700 directory",
        **extra,
    }
    path = OUT_DIR / f"{R4_TASK.lower()}_{attack}_manifest.json"
    path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def run_pif() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pif_target_models as target

    source_hashes = check_sources()
    prompt_rows = ordered_pool(PI_F_ORDER)
    bert_model, bert_tokenizer, bert_info = load_bert_snapshot()
    target_model, target_tokenizer = target.load_target(
        TARGET_ID, offline=True, revision=TARGET_REVISION
    )
    runtime_revision = actual_revision(target_model, target_tokenizer)
    if runtime_revision != TARGET_REVISION:
        raise RuntimeError(f"PiF victim checkpoint revision mismatch: {runtime_revision}")
    gen_sha = sha256_file(
        Path(os.environ["HF_HOME"]) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct"
        / "snapshots" / TARGET_REVISION / "generation_config.json"
    )
    pif_runtime_config_sha = generation_config_hash(target_model)
    summary_path = OUT_DIR / "pif_prompt_ledger.csv"
    completed = read_completed(summary_path)
    result = append_pif_prompt_rows(
        prompt_rows, completed, TARGET_ID, runtime_revision, pif_runtime_config_sha,
        bert_model, bert_tokenizer, target_model, target_tokenizer,
    )
    update_manifest("pif", result["status_counts"], runtime_revision,
                    pif_runtime_config_sha, source_hashes,
                    {"bert_runtime": bert_info,
                     "pinned_generation_config_file_sha256": gen_sha,
                     "d4_first_response_rule": "query_index=1 in pif_query_ledger.csv; no later selected response substitutes for it",
                     "full_pool_adapter": "All 913 pool_913.csv keys are passed to the audited attack_mlm.generate_attack loop; the current run_pif.py llama CLI dataset branch is not used for pool selection",
                     "a1_feedback_config_artifact": str(FEEDBACK_CONFIG_PATH.relative_to(ROOT))})
    del target_model, bert_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pif_feedback_control() -> None:
    import torch
    import pif_target_models as target

    source_hashes = check_sources()
    prompt_rows = ordered_pool(PI_F_ORDER)
    model, tokenizer = target.load_target(TARGET_ID, offline=True, revision=TARGET_REVISION)
    runtime_revision = actual_revision(model, tokenizer)
    if runtime_revision != TARGET_REVISION:
        raise RuntimeError(f"PiF feedback control checkpoint mismatch: {runtime_revision}")
    config_sha = sha256_text(json.dumps(
        PIF_FEEDBACK_CONTROL_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ))
    row_path = OUT_DIR / "pif_feedback_control_rows.csv"
    summary_path = OUT_DIR / "pif_feedback_control_summary.csv"
    completed = read_completed(summary_path)
    if row_path.exists() and row_path.stat().st_size:
        existing = file_rows(row_path)
        identities = [(row["canonical_key"],) for row in existing]
        if len(identities) != len(set(identities)):
            raise RuntimeError("Duplicate prompt key in the PiF feedback control ledger")
        if {key[0] for key in identities} - completed:
            raise RuntimeError("PiF feedback control has rows without prompt summary records")
    row_stream, row_writer = private_csv_writer(row_path, PIF_CONTROL_FIELDS)
    summary_stream, summary_writer = private_csv_writer(summary_path, PIF_CONTROL_SUMMARY_FIELDS)
    counts = Counter()
    for position, row in enumerate(prompt_rows, start=1):
        key = canonical_key(row)
        if key in completed:
            continue
        started = time.time()
        response = ""
        status = "ok"
        failure_type = ""
        try:
            response = target.target_generate(
                model, tokenizer, "llama", row["original_prompt"],
                generation_config=PIF_FEEDBACK_CONTROL_CONFIG,
            )
            if not isinstance(response, str):
                raise TypeError("non_string_response")
        except Exception as exc:
            status = "generation_error"
            failure_type = type(exc).__name__
            response = ""
        row_writer.writerow({
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": TARGET_ID,
            "checkpoint_revision": runtime_revision,
            "config_sha256": config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "response_sha256": sha256_text(response),
            "status": status,
            "failure_type": failure_type,
            "generation_seconds": round(time.time() - started, 4),
            "original_prompt": row["original_prompt"],
            "response": response,
        })
        summary_writer.writerow({
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": TARGET_ID,
            "checkpoint_revision": runtime_revision,
            "config_sha256": config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "status": status,
            "response_sha256": sha256_text(response),
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "failure_type": failure_type,
        })
        for stream in (row_stream, summary_stream):
            stream.flush()
            os.fsync(stream.fileno())
        completed.add(key)
        counts[status] += 1
        if position % 25 == 0 or position == len(prompt_rows):
            print(f"G1 PiF feedback control progress {position}/{len(prompt_rows)}", flush=True)
    row_stream.close()
    summary_stream.close()
    manifest = {
        "task_id": "G1",
        "subtask": "PiF first optimizer-feedback plaintext control",
        "model_alias": MODEL_ALIAS,
        "checkpoint_id": TARGET_ID,
        "checkpoint_revision": runtime_revision,
        "canonical_pool_rows": N_PROMPTS,
        "response_rows": len(file_rows(row_path)),
        "status_counts": dict(Counter(row["status"] for row in file_rows(row_path))),
        "generation_configuration": PIF_FEEDBACK_CONTROL_CONFIG,
        "config_sha256": config_sha,
        "source_sha256": source_hashes,
        "output_sha256": {row_path.name: sha256_file(row_path),
                          summary_path.name: sha256_file(summary_path)},
        "raw_prompts_or_responses_in_manifest": False,
    }
    manifest_path = OUT_DIR / "g1_pif_feedback_control_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_metacipher_control() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_hashes = check_sources()
    prompt_rows = ordered_pool(META_ORDER)
    hf_cache = str(Path(os.environ["HF_HOME"]) / "hub")
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_ID,
        cache_dir=hf_cache,
        revision=TARGET_REVISION,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        cache_dir=hf_cache,
        revision=TARGET_REVISION,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    runtime_revision = actual_revision(model, tokenizer)
    if runtime_revision != TARGET_REVISION:
        raise RuntimeError(f"MetaCipher plaintext-control checkpoint mismatch: {runtime_revision}")

    config = {
        "tokenizer_chat_template": "apply_chat_template(tokenize=false, add_generation_prompt=true), then tokenizer default special-token behavior",
        "max_input_tokens": 2048,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": "tokenizer.eos_token_id",
        "eos_token_id_argument": "omitted, use pinned model generation config",
        "dtype": "float16",
    }
    config_sha = sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    row_path = OUT_DIR / "metacipher_plaintext_control_rows.csv"
    summary_path = OUT_DIR / "metacipher_plaintext_control_summary.csv"
    completed = read_completed(summary_path)
    if row_path.exists() and row_path.stat().st_size:
        existing = file_rows(row_path)
        identities = [row["canonical_key"] for row in existing]
        if len(identities) != len(set(identities)) or set(identities) - completed:
            raise RuntimeError("MetaCipher plaintext-control row and summary ledgers do not reconcile")
    row_stream, row_writer = private_csv_writer(row_path, META_CONTROL_FIELDS)
    summary_stream, summary_writer = private_csv_writer(summary_path, META_CONTROL_SUMMARY_FIELDS)
    counts = Counter()
    for position, row in enumerate(prompt_rows, start=1):
        key = canonical_key(row)
        if key in completed:
            continue
        response = ""
        failure_type = ""
        status = "ok"
        started = time.time()
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["original_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            input_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            if not response.strip():
                status = "empty_response"
        except Exception as exc:
            status = "generation_error"
            failure_type = type(exc).__name__
            response = ""
        duration = round(time.time() - started, 4)
        row_writer.writerow({
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": TARGET_ID,
            "checkpoint_revision": runtime_revision,
            "config_sha256": config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "response_sha256": sha256_text(response),
            "status": status,
            "failure_type": failure_type,
            "generation_seconds": duration,
            "original_prompt": row["original_prompt"],
            "response": response,
        })
        summary_writer.writerow({
            "model_alias": MODEL_ALIAS,
            "checkpoint_id": TARGET_ID,
            "checkpoint_revision": runtime_revision,
            "config_sha256": config_sha,
            "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"],
            "canonical_key": key,
            "status": status,
            "response_sha256": sha256_text(response),
            "original_prompt_sha256": sha256_text(row["original_prompt"]),
            "failure_type": failure_type,
        })
        for stream in (row_stream, summary_stream):
            stream.flush()
            os.fsync(stream.fileno())
        completed.add(key)
        counts[status] += 1
        if position % 25 == 0 or position == len(prompt_rows):
            print(f"G1 MetaCipher plaintext control progress {position}/{len(prompt_rows)}", flush=True)
    row_stream.close()
    summary_stream.close()
    manifest = {
        "task_id": "G1",
        "subtask": "MetaCipher matched plaintext control",
        "model_alias": MODEL_ALIAS,
        "checkpoint_id": TARGET_ID,
        "checkpoint_revision": runtime_revision,
        "canonical_pool_rows": N_PROMPTS,
        "response_rows": len(file_rows(row_path)),
        "status_counts": dict(Counter(row["status"] for row in file_rows(row_path))),
        "generation_configuration": config,
        "config_sha256": config_sha,
        "source_sha256": source_hashes,
        "output_sha256": {row_path.name: sha256_file(row_path),
                          summary_path.name: sha256_file(summary_path)},
        "raw_prompts_or_responses_in_manifest": False,
    }
    manifest_path = OUT_DIR / "g1_metacipher_plaintext_control_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_metacipher() -> None:
    import torch
    import numpy as np
    import metacipher_multi as metacipher

    api_key_env = "OPENROUTER_API_KEY" if R4_TASK == "G4" else "DEEPSEEK_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} must be available in the batch environment")
    source_hashes = check_sources()
    prompt_rows = ordered_pool(META_ORDER)
    meta_cache = str(Path(os.environ["HF_HOME"]) / "hub")
    metacipher.HF_CACHE = meta_cache
    summary_path = OUT_DIR / "metacipher_prompt_ledger.csv"
    completed = read_completed(summary_path)
    meta_generation_config = (
        G4_LLADA_GENERATION_CONFIG if R4_TASK == "G4" else META_GENERATION_CONFIG
    )
    meta_config_sha = sha256_text(json.dumps(
        meta_generation_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ))
    api_config = {
        "base_url": metacipher.METACIPHER_API_BASE_URL,
        "model": metacipher.METACIPHER_API_MODEL,
        "credential_env": api_key_env,
        "credential_value_recorded": False,
    }
    frozen_config = {
        "task_id": "G4-llada-metacipher-openrouter-v4-0731",
        "attack": "metacipher",
        "model_alias": MODEL_ALIAS,
        "checkpoint_id": TARGET_ID,
        "checkpoint_revision": TARGET_REVISION,
        "canonical_pool_rows": N_PROMPTS,
        "canonical_pool_sha256": sha256_file(POOL_PATH),
        "generation_configuration": meta_generation_config,
        "generation_config_sha256": meta_config_sha,
        "agent_api_configuration": api_config,
        "agent_api_configuration_sha256": sha256_text(json.dumps(
            api_config, sort_keys=True, separators=(",", ":")
        )),
        "source_sha256": source_hashes,
        "resource_estimate": {
            "a100_gpu_hours": 76,
            "api_requests": 6700,
            "api_cost_usd": None,
            "api_cost_note": "Token usage was not recorded in the previous direct-API run; estimate before submission and set an OpenRouter spend limit.",
        },
        "credential_value_recorded": False,
    }
    frozen_config_path = OUT_DIR / "g4_metacipher_frozen_config.json"
    if frozen_config_path.exists():
        existing_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        if existing_config != frozen_config:
            raise RuntimeError("Frozen G4 MetaCipher configuration changed; refusing to mix runs")
    else:
        frozen_config_path.write_text(
            json.dumps(frozen_config, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(frozen_config_path, 0o600)
    frozen_config_sha = sha256_file(frozen_config_path)
    result = append_meta_rows(prompt_rows, completed, TARGET_ID, TARGET_REVISION, meta_config_sha)
    update_manifest("metacipher", result["status_counts"], TARGET_REVISION,
                    meta_config_sha,
                    source_hashes,
                    {"generation_configuration": meta_generation_config,
                     "agent_api_configuration": api_config,
                     "agent_api_configuration_sha256": sha256_text(json.dumps(
                         api_config, sort_keys=True, separators=(",", ":")
                     )),
                     "frozen_run_configuration_sha256": frozen_config_sha,
                     "d4_first_response_rule": "query_index=1 in metacipher_query_ledger.csv; all sequential victim queries are retained",
                     "adaptive_policy": "MetaCipher's Q-table is checkpointed after each canonical prompt and retained across the ordered prompt stream",
                     "meta_cache_layout": "HF_HOME/hub"})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _ = np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=("pif", "metacipher", "pif_control", "metacipher_control"), required=True)
    args = parser.parse_args()
    ensure_private_directory()
    lock_path = OUT_DIR / f".{args.attack}.lock"
    with lock_path.open("a") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.attack == "pif":
            run_pif()
        elif args.attack == "pif_control":
            run_pif_feedback_control()
        elif args.attack == "metacipher_control":
            run_metacipher_control()
        else:
            run_metacipher()


if __name__ == "__main__":
    main()

"""Generate and merge the frozen R4 G2 plaintext controls.

Raw prompt/response fields are restricted working data. The merged ledger is
response-free and is the only form intended for the reviewer package.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R4_ROOT = ROOT / "revision_experiments" / "r4"
OUT_DIR = R4_ROOT / "G2_plaintext_control"
POOL_PATH = ROOT / "tier0" / "02_dedup_and_config_audit" / "pool_913.csv"
MATRIX_PATH = R4_ROOT / "A1_config_audit" / "config_matrix.csv"
PLAN_PATH = ROOT / "submissions" / "TMLR" / "r4_plan.txt"
POOL_SHA256 = "12bafde1c37c56349b4566b97f839efcd8579996357d7e76ed0b2dee1f92346b"
MATRIX_SHA256 = "93d761c73a0880454e2af2e09123c5788a841390e0d71ba42042afac693adf8a"
N_PROMPTS = 913
MAX_NEW_TOKENS = 512


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_file_hashes():
    paths = {
        "runner": Path(__file__).resolve(),
        "target_generation": ROOT / "scripts" / "pif_target_models.py",
        "slurm_script": ROOT / "scripts" / "r4_g2_plaintext_control.sbatch",
        "prompt_pool": POOL_PATH,
        "a1_config_matrix": MATRIX_PATH,
        "execution_plan": PLAN_PATH,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def make_config(config_id, source_cells, generation_config, settings_note):
    config = {
        "config_id": config_id,
        "source_cells": source_cells,
        "generation_config": generation_config,
        "settings_note": settings_note,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config


MODEL_GROUPS = [
    {
        "model_alias": "qwen",
        "load_key": "qwen2.5",
        "generate_key": "qwen2.5",
        "checkpoint_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "configs": [
            make_config("pif_arrattack_settings", ["pif/qwen", "arrattack/qwen"],
                        {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
                        "A1 causal decoding settings shared by PiF and ArrAttack"),
            make_config("metacipher_settings", ["metacipher/qwen"],
                        {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
                        "A1 MetaCipher causal decoding settings"),
        ],
    },
    {
        "model_alias": "llama31_8b",
        "load_key": "meta-llama/Llama-3.1-8B-Instruct",
        "generate_key": "llama",
        "checkpoint_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "configs": [
            make_config("pif_arrattack_settings", ["pif/llama-3.2-3b-settings", "arrattack/llama-3.1-8b"],
                        {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
                        "A1 causal decoding settings; checkpoint is the D2 Llama-3.1-8B target"),
            make_config("pif_feedback_query_exact", ["pif/llama-3.1-8b-first-feedback"],
                        {"max_input_tokens": 128, "max_length": 512,
                         "use_chat_template_tokenizer": True,
                         "add_generation_prompt": False,
                         "use_model_generation_defaults": True},
                        "PiF first optimizer victim query: exact apply_chat_template tokenization without an assistant generation prompt, input cap 128, max_length 512, and no generation-argument overrides beyond max_length"),
            make_config("metacipher_settings", ["metacipher/llama-3.2-3b-settings"],
                        {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
                        "A1 MetaCipher causal decoding settings; checkpoint is the D2 Llama-3.1-8B target"),
        ],
    },
    {
        "model_alias": "falcon",
        "load_key": "falcon",
        "generate_key": "falcon",
        "checkpoint_id": "tiiuae/Falcon-H1R-7B",
        "revision": "a6f74bf181389908efd6970878d7ee2b42f5d417",
        "configs": [
            make_config("pif_arrattack_settings", ["pif/falcon", "arrattack/falcon"],
                        {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
                        "A1 causal decoding settings shared by PiF and ArrAttack"),
            make_config("metacipher_settings", ["metacipher/falcon"],
                        {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
                        "A1 MetaCipher causal decoding settings"),
        ],
    },
    {
        "model_alias": "dream",
        "load_key": "dream",
        "generate_key": "dream",
        "checkpoint_id": "Dream-org/Dream-v0-Instruct-7B",
        "revision": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        "configs": [
            make_config("pif_arrattack_settings", ["pif/dream", "arrattack/dream"],
                        {"temperature": 0.0, "top_p": None,
                         "alg": "origin", "alg_temp": None},
                        "A1 PiF/ArrAttack Dream settings; steps track generated length"),
            make_config("metacipher_settings", ["metacipher/dream"],
                        {"steps": 128, "temperature": 0.2, "top_p": 0.95,
                         "alg": "entropy", "alg_temp": 0.0},
                        "A1 MetaCipher Dream settings"),
        ],
    },
    {
        "model_alias": "diffucoder",
        "load_key": "diffucoder",
        "generate_key": "diffucoder",
        "checkpoint_id": "apple/DiffuCoder-7B-Instruct",
        "revision": "4fdd4580064ca5d11808069ce78f88d068753c96",
        "configs": [
            make_config("pif_arrattack_settings", ["pif/diffucoder", "arrattack/diffucoder"],
                        {"temperature": 0.3, "top_p": 0.95,
                         "alg": "entropy", "alg_temp": 0.0},
                        "A1 PiF/ArrAttack DiffuCoder settings; steps track generated length"),
            make_config("metacipher_settings", ["metacipher/diffucoder"],
                        {"steps": 128, "temperature": 0.4, "top_p": 0.95,
                         "alg": "entropy", "alg_temp": 0.0},
                        "A1 MetaCipher DiffuCoder settings"),
        ],
    },
    {
        "model_alias": "llada",
        "load_key": "llada",
        "generate_key": "llada",
        "checkpoint_id": "GSAI-ML/LLaDA-1.5",
        "revision": "84346fd91ba60252d260022201ad6fc5a3468fb2",
        "configs": [
            make_config("locked_128_steps_per_block", ["pif/llada", "metacipher/llada", "arrattack/llada"],
                        {"block_length": 128, "temperature": 0.0},
                        "D7 locked LLaDA schedule: low-confidence remasking; steps equal actual gen_length up to 512, with 128-token blocks"),
        ],
    },
]

RAW_FIELDS = ["model_alias", "checkpoint_id", "checkpoint_revision", "config_id",
              "config_sha256", "dataset", "prompt_idx", "canonical_key",
              "original_prompt_sha256", "original_prompt", "response_sha256",
              "response", "status", "failure_type", "generation_seconds"]
LEDGER_FIELDS = [field for field in RAW_FIELDS if field not in {"original_prompt", "response"}]


def validate_static_inputs():
    if sha256_file(POOL_PATH) != POOL_SHA256:
        raise RuntimeError("Canonical prompt-pool hash mismatch; refusing G2 generation")
    if sha256_file(MATRIX_PATH) != MATRIX_SHA256:
        raise RuntimeError("A1 configuration-matrix hash mismatch; refusing G2 generation")
    if len(MODEL_GROUPS) != 6 or sum(len(group["configs"]) for group in MODEL_GROUPS) != 12:
        raise RuntimeError("G2 requires six victim checkpoints and twelve currently specified decoder configurations")
    with POOL_PATH.open(encoding="utf-8", newline="") as stream:
        prompts = list(csv.DictReader(stream))
    keys = [(row["dataset"], row["prompt_idx"]) for row in prompts]
    if len(prompts) != 913 or len(set(keys)) != 913:
        raise RuntimeError("Canonical pool must contain exactly 913 unique (dataset, prompt_idx) keys")
    if len(prompts) * sum(len(group["configs"]) for group in MODEL_GROUPS) != 10956:
        raise RuntimeError("G2 planned response count does not equal 10,956")
    with MATRIX_PATH.open(encoding="utf-8", newline="") as stream:
        matrix = list(csv.DictReader(stream))
    by_cell = {(row["attack"], row["model_alias"]): row for row in matrix}

    checkpoint_source = {
        "qwen": ("pif", "qwen"),
        "llama31_8b": ("arrattack", "llama"),
        "falcon": ("pif", "falcon"),
        "dream": ("pif", "dream"),
        "diffucoder": ("pif", "diffucoder"),
        "llada": ("pif", "llada"),
    }
    for group in MODEL_GROUPS:
        source_row = by_cell[checkpoint_source[group["model_alias"]]]
        assert group["checkpoint_id"] == source_row["victim_checkpoint_id"]
        assert group["revision"] == source_row["victim_revision_sha"]

    causal_source_models = ("qwen", "llama", "falcon")
    for model in causal_source_models:
        pif = by_cell[("pif", model)]["sampling_parameters_from_authoritative_code_audit"]
        arr = by_cell[("arrattack", model)]["sampling_parameters_from_authoritative_code_audit"]
        mc = by_cell[("metacipher", model)]["sampling_parameters_from_authoritative_code_audit"]
        assert pif.startswith("temp=0.8, top_p=0.9") and arr == pif
        assert mc.startswith("temp=0.7, top_p=0.9")
    dream_pif = by_cell[("pif", "dream")]
    dream_arr = by_cell[("arrattack", "dream")]
    dream_mc = by_cell[("metacipher", "dream")]
    assert dream_pif["sampling_parameters_from_authoritative_code_audit"] == "temp=0.0, top_p=None, alg=origin"
    assert dream_arr["sampling_parameters_from_authoritative_code_audit"] == dream_pif["sampling_parameters_from_authoritative_code_audit"]
    assert dream_pif["diffusion_schedule_from_authoritative_code_audit"] == "steps=gen_length"
    assert dream_mc["sampling_parameters_from_authoritative_code_audit"] == "temp=0.2, top_p=0.95, alg=entropy"
    assert dream_mc["diffusion_schedule_from_authoritative_code_audit"].startswith("steps=128 fixed")
    diffu_pif = by_cell[("pif", "diffucoder")]
    diffu_arr = by_cell[("arrattack", "diffucoder")]
    diffu_mc = by_cell[("metacipher", "diffucoder")]
    assert diffu_pif["sampling_parameters_from_authoritative_code_audit"] == "temp=0.3, top_p=0.95, alg=entropy"
    assert diffu_arr["sampling_parameters_from_authoritative_code_audit"] == diffu_pif["sampling_parameters_from_authoritative_code_audit"]
    assert diffu_pif["diffusion_schedule_from_authoritative_code_audit"] == "steps=gen_length"
    assert diffu_mc["sampling_parameters_from_authoritative_code_audit"] == "temp=0.4, top_p=0.95, alg=entropy"
    assert diffu_mc["diffusion_schedule_from_authoritative_code_audit"].startswith("steps=128 fixed")
    assert len(MODEL_GROUPS) == 6 and sum(len(g["configs"]) for g in MODEL_GROUPS) == 12
    groups = {group["model_alias"]: group for group in MODEL_GROUPS}
    config_for = lambda model, config_id: next(
        cfg for cfg in groups[model]["configs"] if cfg["config_id"] == config_id)
    for model in ("qwen", "llama31_8b", "falcon"):
        assert config_for(model, "pif_arrattack_settings")["generation_config"] == {
            "do_sample": True, "temperature": 0.8, "top_p": 0.9}
        assert config_for(model, "metacipher_settings")["generation_config"] == {
            "do_sample": True, "temperature": 0.7, "top_p": 0.9}
    assert config_for("llama31_8b", "pif_feedback_query_exact")["generation_config"] == {
        "max_input_tokens": 128, "max_length": 512,
        "use_chat_template_tokenizer": True, "add_generation_prompt": False,
        "use_model_generation_defaults": True}
    assert config_for("dream", "pif_arrattack_settings")["generation_config"] == {
        "temperature": 0.0, "top_p": None, "alg": "origin", "alg_temp": None}
    assert config_for("dream", "metacipher_settings")["generation_config"] == {
        "steps": 128, "temperature": 0.2, "top_p": 0.95, "alg": "entropy", "alg_temp": 0.0}
    assert config_for("diffucoder", "pif_arrattack_settings")["generation_config"] == {
        "temperature": 0.3, "top_p": 0.95, "alg": "entropy", "alg_temp": 0.0}
    assert config_for("diffucoder", "metacipher_settings")["generation_config"] == {
        "steps": 128, "temperature": 0.4, "top_p": 0.95, "alg": "entropy", "alg_temp": 0.0}
    assert config_for("llada", "locked_128_steps_per_block")["generation_config"] == {
        "block_length": 128, "temperature": 0.0}
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    assert "steps equal to answer length" in plan_text and "128 steps/block" in plan_text
    return prompts


def ensure_private_directory():
    os.umask(0o077)
    OUT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(OUT_DIR, 0o700)


def validate_shard_rows(rows, group, prompts, require_complete):
    prompts_by_key = {
        f"{row['dataset']}|{row['prompt_idx']}": row for row in prompts
    }
    configs = {config["config_id"]: config for config in group["configs"]}
    expected = {
        (config["config_id"], canonical_key)
        for config in group["configs"]
        for canonical_key in prompts_by_key
    }
    observed = set()
    for row in rows:
        missing_fields = set(RAW_FIELDS) - set(row)
        extra_fields = set(row) - set(RAW_FIELDS)
        if missing_fields or extra_fields:
            raise ValueError(
                f"Malformed G2 shard schema for {group['model_alias']}: "
                f"missing={sorted(missing_fields)}, extra={sorted(str(x) for x in extra_fields)}"
            )
        key = (row.get("config_id"), row.get("canonical_key"))
        if key not in expected:
            raise ValueError(f"Unexpected G2 shard key for {group['model_alias']}: {key}")
        if key in observed:
            raise ValueError(f"Duplicate G2 shard key for {group['model_alias']}: {key}")
        observed.add(key)

        config = configs[row["config_id"]]
        prompt_row = prompts_by_key[row["canonical_key"]]
        expected_fields = {
            "model_alias": group["model_alias"],
            "checkpoint_id": group["checkpoint_id"],
            "checkpoint_revision": group["revision"],
            "config_sha256": config["config_sha256"],
            "dataset": prompt_row["dataset"],
            "prompt_idx": prompt_row["prompt_idx"],
            "original_prompt": prompt_row["original_prompt"],
            "original_prompt_sha256": sha256_text(prompt_row["original_prompt"]),
            "response_sha256": sha256_text(row.get("response", "")),
        }
        mismatches = [
            field for field, expected_value in expected_fields.items()
            if row.get(field) != expected_value
        ]
        if mismatches:
            raise ValueError(
                f"G2 shard identity/hash mismatch for {group['model_alias']} {key}: {mismatches}"
            )
        if row.get("status") == "ok":
            if row.get("failure_type"):
                raise ValueError(f"Successful G2 row has failure_type: {key}")
        elif row.get("status") == "generation_error":
            if not row.get("failure_type") or row.get("response") != "":
                raise ValueError(f"Malformed G2 generation-error row: {key}")
        else:
            raise ValueError(f"Unknown G2 row status for {group['model_alias']} {key}")

    if require_complete and observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise ValueError(
            f"Incomplete G2 shard for {group['model_alias']}: "
            f"observed={len(observed)}, expected={len(expected)}, missing={missing}, extra={extra}"
        )
    return rows


def _actual_revision(model, tokenizer):
    actual = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not actual:
        actual = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    return actual


def generate_model_shard(group, prompts):
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME must be explicitly set to the approved shared model cache")
    raw_path = OUT_DIR / f"g2_raw_{group['model_alias']}.csv"
    if raw_path.exists():
        os.chmod(raw_path, 0o600)
    exists = raw_path.exists() and raw_path.stat().st_size > 0
    existing_rows = []
    if exists:
        with raw_path.open(encoding="utf-8", newline="") as stream:
            existing_rows = list(csv.DictReader(stream))
        validate_shard_rows(existing_rows, group, prompts, require_complete=False)
    completed = {(row["config_id"], row["canonical_key"]) for row in existing_rows}

    import pif_target_models as target

    model, tokenizer = target.load_target(group["load_key"], offline=True, revision=group["revision"])
    runtime_revision = _actual_revision(model, tokenizer)
    if runtime_revision != group["revision"]:
        raise RuntimeError(f"Loaded checkpoint revision mismatch for {group['model_alias']}")

    with raw_path.open("a" if exists else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
            stream.flush()
            os.chmod(raw_path, 0o600)
        for config in group["configs"]:
            for position, prompt_row in enumerate(prompts, start=1):
                canonical_key = f"{prompt_row['dataset']}|{prompt_row['prompt_idx']}"
                key = (config["config_id"], canonical_key)
                if key in completed:
                    continue
                prompt = prompt_row["original_prompt"]
                started = time.time()
                status = "ok"
                failure_type = ""
                try:
                    response = target.target_generate(
                        model, tokenizer, group["generate_key"], prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        generation_config=config["generation_config"],
                    )
                    if not isinstance(response, str):
                        raise TypeError("non_string_response")
                except Exception as exc:
                    response = ""
                    status = "generation_error"
                    failure_type = type(exc).__name__
                writer.writerow({
                    "model_alias": group["model_alias"],
                    "checkpoint_id": group["checkpoint_id"],
                    "checkpoint_revision": runtime_revision,
                    "config_id": config["config_id"],
                    "config_sha256": config["config_sha256"],
                    "dataset": prompt_row["dataset"],
                    "prompt_idx": prompt_row["prompt_idx"],
                    "canonical_key": canonical_key,
                    "original_prompt_sha256": sha256_text(prompt),
                    "original_prompt": prompt,
                    "response_sha256": sha256_text(response),
                    "response": response,
                    "status": status,
                    "failure_type": failure_type,
                    "generation_seconds": round(time.time() - started, 3),
                })
                stream.flush()
                completed.add(key)
                if position % 25 == 0:
                    print(f"G2 {group['model_alias']}/{config['config_id']}: {position}/{len(prompts)}", flush=True)
    os.chmod(raw_path, 0o600)
    del model, tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_shards():
    prompts = validate_static_inputs()
    raw_paths = [OUT_DIR / f"g2_raw_{group['model_alias']}.csv" for group in MODEL_GROUPS]
    missing = [path.name for path in raw_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G2 model shards missing: {missing}")
    raw_rows = []
    for path in raw_paths:
        if path.stat().st_mode & 0o077:
            raise PermissionError(f"Raw G2 shard is not private: {path.name}")
        with path.open(encoding="utf-8", newline="") as stream:
            shard_rows = list(csv.DictReader(stream))
        alias = path.name.removeprefix("g2_raw_").removesuffix(".csv")
        group = next(group for group in MODEL_GROUPS if group["model_alias"] == alias)
        validate_shard_rows(shard_rows, group, prompts, require_complete=True)
        raw_rows.extend(shard_rows)
    keys = [(row["model_alias"], row["config_id"], row["canonical_key"]) for row in raw_rows]
    if len(raw_rows) != 10956 or len(set(keys)) != 10956:
        raise ValueError(f"G2 expected 10,956 unique response rows, found {len(raw_rows)}")
    ledger_path = OUT_DIR / "g2_generation_ledger.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({field: row[field] for field in LEDGER_FIELDS})
    os.chmod(ledger_path, 0o600)
    metadata = {
        "task_id": "G2",
        "canonical_prompt_rows": 913,
        "unique_decoder_configurations": 12,
        "response_rows": len(raw_rows),
        "status_counts": dict(Counter(row["status"] for row in raw_rows)),
        "checkpoint_revisions": {group["model_alias"]: group["revision"] for group in MODEL_GROUPS},
        "configurations": {group["model_alias"]: group["configs"] for group in MODEL_GROUPS},
        "raw_working_file_sha256": {path.name: sha256_file(path) for path in raw_paths},
        "response_free_ledger_sha256": sha256_file(ledger_path),
        "source_file_sha256": source_file_hashes(),
        "raw_text_in_response_free_ledger": False,
        "raw_working_files_access_control_required": True,
    }
    manifest_path = OUT_DIR / "g2_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    print({"task_id": "G2", "rows": len(raw_rows), "statuses": metadata["status_counts"],
           "ledger": ledger_path.name, "manifest": manifest_path.name})


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-index", type=int, choices=range(len(MODEL_GROUPS)))
    mode.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    ensure_private_directory()
    if args.merge:
        merge_shards()
        return
    prompts = validate_static_inputs()
    generate_model_shard(MODEL_GROUPS[args.model_index], prompts)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-outcome Falcon-H1R generation-budget pilot on disjoint AdvBench prompts.

Restricted raw prompt and response fields are stored only in mode-0600 CSVs
inside a mode-0700 output directory. No judge is called in this pilot.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = Path(os.environ.get(
    "G5_PILOT_DIR", str(ROOT / "revision_experiments" / "r4" / "G5_falcon_budget" / "pilot")
))
ADV_BENCH_PATH = ROOT / "dataset" / "advbench" / "harmful_behaviors.csv"
POOL_PATH = ROOT / "tier0" / "02_dedup_and_config_audit" / "pool_913.csv"
MODEL_ID = "tiiuae/Falcon-H1R-7B"
MODEL_REVISION = "a6f74bf181389908efd6970878d7ee2b42f5d417"
SAMPLE_N = 60
SAMPLE_SEED = 20261011
CAPS = (512, 1024, 2048, 4096)
FIELDS = [
    "pilot_row_key", "source_dataset", "source_row_index", "budget_seed",
    "max_new_tokens", "model_id", "model_revision", "generation_config_sha256",
    "prompt_sha256", "response_sha256", "raw_prompt", "raw_response",
    "final_segment", "final_segment_class", "generated_token_count",
    "eos_terminated", "truncated_at_cap", "status", "error_type", "error_message",
    "generation_seconds",
]
THUS_MARK = re.compile(r"\bThus\b", re.IGNORECASE)
PLAN_LEAD = re.compile(
    r"^(we|i|could|can|should|will|would)\b|^(?:provide|include|describe|give|"
    r"outline|explain)\b|answer should|answer:|we should|you can|you should|"
    r"^:\s*$",
    re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_final_segment(raw: str) -> tuple[str, str]:
    matches = list(THUS_MARK.finditer(raw))
    answer = raw[matches[-1].end():] if matches else ""
    answer_stripped = answer.strip()
    if not answer_stripped:
        return answer, "empty"
    answer_clean = re.sub(r"^[\s:;*\-–—]+", "", answer_stripped)
    if not answer_clean or PLAN_LEAD.match(answer_clean):
        return answer, "plan"
    return answer, "answer"


def load_disjoint_sample() -> tuple[list[dict], dict[str, str]]:
    if not ADV_BENCH_PATH.is_file() or not POOL_PATH.is_file():
        raise FileNotFoundError("G5 pilot inputs are missing")
    with POOL_PATH.open(encoding="utf-8", newline="") as stream:
        pool = list(csv.DictReader(stream))
    pool_texts = {row["original_prompt"].strip() for row in pool}
    pool_keys = {(row["dataset"], row["prompt_idx"]) for row in pool}
    with ADV_BENCH_PATH.open(encoding="utf-8", newline="") as stream:
        advbench = list(csv.DictReader(stream))
    eligible = []
    seen = set()
    for i, row in enumerate(advbench):
        prompt = row.get("goal", "").strip()
        if prompt and prompt not in pool_texts and prompt not in seen:
            eligible.append({"source_row_index": i, "prompt": prompt})
            seen.add(prompt)
    if len(pool) != 913 or len(pool_keys) != 913:
        raise RuntimeError("G5 disjoint-pilot check requires the frozen 913 unique pool")
    if len(eligible) < SAMPLE_N:
        raise RuntimeError(f"Only {len(eligible)} AdvBench prompts are disjoint from pool_913")
    sample = random.Random(SAMPLE_SEED).sample(eligible, SAMPLE_N)
    if len({item["prompt"] for item in sample} & pool_texts):
        raise RuntimeError("G5 pilot sample unexpectedly overlaps pool_913")
    source_hashes = {
        "advbench": sha256_file(ADV_BENCH_PATH),
        "pool_913": sha256_file(POOL_PATH),
    }
    return sample, source_hashes


def read_completed(path: Path, config_sha: str, sample_by_key: dict[str, dict]) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = set()
    for row in rows:
        key = (row["pilot_row_key"], int(row["max_new_tokens"]))
        if key in keys:
            raise RuntimeError(f"Duplicate G5 pilot key: {key}")
        if row["generation_config_sha256"] != config_sha:
            raise RuntimeError("G5 pilot ledger config hash mismatch; refusing mixed-config resume")
        if row["pilot_row_key"] not in sample_by_key:
            raise RuntimeError(f"Unexpected G5 sample key in ledger: {row['pilot_row_key']}")
        if row["prompt_sha256"] != sha256_text(sample_by_key[row["pilot_row_key"]]["prompt"]):
            raise RuntimeError(f"G5 prompt hash mismatch for {row['pilot_row_key']}")
        keys.add(key)
    return keys


def open_private_csv(path: Path):
    exists = path.is_file() and path.stat().st_size > 0
    stream = path.open("a" if exists else "w", encoding="utf-8", newline="")
    os.chmod(path, 0o600)
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
    if not exists:
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    return stream, writer


def main() -> None:
    os.umask(0o077)
    PILOT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(PILOT_DIR, 0o700)
    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME must point to the approved shared Hugging Face cache")
    sample, input_hashes = load_disjoint_sample()
    sample_by_key = {f"advbench|{item['source_row_index']}": item for item in sample}

    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import pif_target_models
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("TRANSFORMERS_OFFLINE") or not os.environ.get("HF_HUB_OFFLINE"):
        raise RuntimeError("G5 pilot must load pinned models offline from HF_HOME")
    snapshot = (
        Path(os.environ["HF_HOME"]) / "hub" /
        "models--tiiuae--Falcon-H1R-7B" / "snapshots" / MODEL_REVISION
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned Falcon snapshot missing from HF_HOME: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device).lower() in {"cpu", "disk", "meta"} for device in device_map.values()):
        raise RuntimeError(f"Falcon-H1R was not fully placed on the allocated GPU: {device_map}")
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if meta_parameters:
        raise RuntimeError(f"Falcon-H1R has unmaterialized meta parameters: {meta_parameters[:8]}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    runtime_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if runtime_revision and runtime_revision != MODEL_REVISION:
        raise RuntimeError(f"Falcon revision mismatch: {runtime_revision}")

    config = {
        "task_id": "G5-budget-pilot",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_dataset": "AdvBench",
        "sample_n": SAMPLE_N,
        "sample_seed": SAMPLE_SEED,
        "sample_disjoint_from_pool_913": True,
        "budget_grid_max_new_tokens": list(CAPS),
        "per_prompt_seed": "20261011 + AdvBench source_row_index; same seed reused across budgets",
        "sampling": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
        "max_input_tokens": 2048,
        "device_placement": "single allocated CUDA device 0; no CPU/disk offload",
        "generation_helper": "pif_target_models prompt encoding plus matching A1 Falcon model.generate arguments",
        "final_answer_detector": "T10.1 last-Thus boundary plus the frozen answer-vs-planning classifier",
        "budget_selection_rule": "Choose the smallest tested cap with zero pilot outputs reaching the cap without EOS; if none, no feasible cap is established within the tested grid.",
        "no_compliance_judging_in_pilot": True,
    }
    config_sha = sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    row_path = PILOT_DIR / "falcon_budget_pilot_raw.csv"
    completed = read_completed(row_path, config_sha, sample_by_key)
    stream, writer = open_private_csv(row_path)
    for sample_position, item in enumerate(sample, start=1):
        key = f"advbench|{item['source_row_index']}"
        prompt = item["prompt"]
        for budget in CAPS:
            if (key, budget) in completed:
                continue
            seed = SAMPLE_SEED + int(item["source_row_index"])
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            started = time.time()
            raw_response = ""
            status = "ok"
            error_type = ""
            error_message = ""
            generated_tokens = 0
            last_token_id = None
            try:
                input_ids, attention_mask = pif_target_models._prepare_target_inputs(
                    tokenizer, "falcon", prompt
                )
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                prompt_length = int(input_ids.shape[-1])
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=budget,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0, prompt_length:]
                generated_tokens = int(generated.numel())
                last_token_id = int(generated[-1].item()) if generated.numel() else None
                raw_response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            except Exception as exc:
                status = "generation_error"
                error_type = type(exc).__name__
                error_message = str(exc)[:300]
            final_segment, segment_class = classify_final_segment(raw_response)
            eos_ids = tokenizer.eos_token_id
            if not isinstance(eos_ids, (list, tuple, set)):
                eos_ids = [eos_ids]
            eos_ids = {int(token_id) for token_id in eos_ids if token_id is not None}
            eos_terminated = int(last_token_id in eos_ids)
            truncated = int(status == "ok" and generated_tokens >= budget and not eos_terminated)
            writer.writerow({
                "pilot_row_key": key,
                "source_dataset": "AdvBench",
                "source_row_index": item["source_row_index"],
                "budget_seed": seed,
                "max_new_tokens": budget,
                "model_id": MODEL_ID,
                "model_revision": runtime_revision or MODEL_REVISION,
                "generation_config_sha256": config_sha,
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": sha256_text(raw_response),
                "raw_prompt": prompt,
                "raw_response": raw_response,
                "final_segment": final_segment,
                "final_segment_class": segment_class,
                "generated_token_count": generated_tokens,
                "eos_terminated": eos_terminated,
                "truncated_at_cap": truncated,
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
                "generation_seconds": round(time.time() - started, 4),
            })
            stream.flush()
            os.fsync(stream.fileno())
            completed.add((key, budget))
        if sample_position % 10 == 0 or sample_position == SAMPLE_N:
            print(f"G5 budget pilot: completed {sample_position}/{SAMPLE_N} unique prompts", flush=True)
    stream.close()

    with row_path.open(encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    expected = {(key, budget) for key in sample_by_key for budget in CAPS}
    observed = {(row["pilot_row_key"], int(row["max_new_tokens"])) for row in output_rows}
    if observed != expected or len(output_rows) != len(expected):
        raise RuntimeError(f"G5 ledger incomplete or duplicated: expected={len(expected)}, got={len(output_rows)}")

    summary = {}
    for budget in CAPS:
        subset = [row for row in output_rows if int(row["max_new_tokens"]) == budget]
        summary[str(budget)] = {
            "rows": len(subset),
            "status_counts": dict(Counter(row["status"] for row in subset)),
            "final_segment_class_counts": dict(Counter(row["final_segment_class"] for row in subset)),
            "truncated_at_cap_n": sum(int(row["truncated_at_cap"]) for row in subset),
            "eos_terminated_n": sum(int(row["eos_terminated"]) for row in subset),
            "median_generated_tokens": sorted(int(row["generated_token_count"]) for row in subset)[len(subset)//2],
        }
    eligible = [budget for budget in CAPS if summary[str(budget)]["truncated_at_cap_n"] == 0
                and summary[str(budget)]["status_counts"] == {"ok": SAMPLE_N}]
    manifest = {
        **config,
        "config_sha256": config_sha,
        "runtime_revision": runtime_revision,
        "sample_source_row_indices": sorted(item["source_row_index"] for item in sample),
        "sample_prompt_hashes": {key: sha256_text(item["prompt"]) for key, item in sample_by_key.items()},
        "input_sha256": input_hashes,
        "source_sha256": {
            "pilot_script": sha256_file(Path(__file__).resolve()),
            "pif_target_models": sha256_file(ROOT / "scripts" / "pif_target_models.py"),
        },
        "raw_rows": len(output_rows),
        "raw_output_sha256": sha256_file(row_path),
        "budget_summary": summary,
        "selected_budget_by_frozen_rule": min(eligible) if eligible else None,
        "raw_prompts_or_responses_in_manifest": False,
        "raw_data_access_mode": "mode 0600 CSV in mode 0700 task directory",
    }
    manifest_path = PILOT_DIR / "g5_budget_pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(row_path, 0o600)
    os.chmod(manifest_path, 0o600)
    print({"task_id": "G5-budget-pilot", "rows": len(output_rows),
           "selected_budget_by_frozen_rule": manifest["selected_budget_by_frozen_rule"],
           "summary": summary}, flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

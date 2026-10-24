#!/usr/bin/env python3
"""Regenerate only the empty-response rows of a G2 plaintext-control shard.

The frozen shard is never modified. Every new row records the parent response
hash, the parent status and the re-verified configuration hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import r4_g2_plaintext_control as G2  # noqa: E402


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-alias", required=True,
                        choices=[group["model_alias"] for group in G2.MODEL_GROUPS])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="retry_empty")
    args = parser.parse_args()

    if not os.environ.get("HF_HOME"):
        raise RuntimeError("HF_HOME must point to the approved shared model cache")

    group = next(g for g in G2.MODEL_GROUPS if g["model_alias"] == args.model_alias)
    source = G2.OUT_DIR / f"g2_raw_{group['model_alias']}.csv"
    if not source.is_file():
        raise RuntimeError(f"missing frozen control shard: {source}")
    output = G2.OUT_DIR / f"g2_{args.tag}_{group['model_alias']}.csv"
    if output == source:
        raise RuntimeError("refusing to overwrite the frozen control shard")

    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    empty_rows = [row for row in rows if not str(row.get("response", "")).strip()]
    source_empty_total = len(empty_rows)
    if args.limit is not None:
        empty_rows = empty_rows[: args.limit]

    configs = {config["config_id"]: config for config in group["configs"]}
    done: set[tuple[str, str]] = set()
    if output.exists() and output.stat().st_size:
        for row in csv.DictReader(output.open(encoding="utf-8", newline="")):
            done.add((row["config_id"], row["canonical_key"]))
    pending = [row for row in empty_rows
               if (row["config_id"], row["canonical_key"]) not in done]

    print(json.dumps({
        "model": group["model_alias"], "source": str(source), "output": str(output),
        "frozen_rows": len(rows), "empty_rows": source_empty_total,
        "already_done": len(done), "pending": len(pending),
        "source_sha256": sha_file(source),
    }, ensure_ascii=False))
    if not pending:
        return

    import pif_target_models as target

    model, tokenizer = target.load_target(group["load_key"], offline=True, revision=group["revision"])
    runtime_revision = G2._actual_revision(model, tokenizer)
    if runtime_revision != group["revision"]:
        raise RuntimeError(f"loaded checkpoint revision mismatch for {group['model_alias']}")

    fields = G2.RAW_FIELDS + ["parent_response_sha256", "parent_status", "retry_config_sha256"]
    fresh = not output.exists()
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh:
            writer.writeheader()
            stream.flush()
            os.chmod(output, 0o600)
        for index, row in enumerate(pending, start=1):
            config = configs[row["config_id"]]
            if row["config_sha256"] != config["config_sha256"]:
                raise RuntimeError(f"source config hash mismatch for {row['canonical_key']}")
            started = time.time()
            failure_type = ""
            try:
                response = target.target_generate(
                    model, tokenizer, group["generate_key"], row["original_prompt"],
                    max_new_tokens=G2.MAX_NEW_TOKENS,
                    generation_config=config["generation_config"],
                )
                if not isinstance(response, str):
                    raise TypeError("non_string_response")
                status = "ok" if response.strip() else "empty_response"
            except Exception as exc:  # keep the row explicit, never drop it
                response = ""
                status = "generation_error"
                failure_type = type(exc).__name__
            item = {name: row[name] for name in G2.RAW_FIELDS}
            item.update({
                "checkpoint_revision": runtime_revision,
                "response_sha256": sha_text(response),
                "response": response,
                "status": status,
                "failure_type": failure_type,
                "generation_seconds": round(time.time() - started, 3),
                "parent_response_sha256": row["response_sha256"],
                "parent_status": row["status"],
                "retry_config_sha256": config["config_sha256"],
            })
            writer.writerow(item)
            stream.flush()
            if index % 10 == 0:
                print(f"{group['model_alias']} {index}/{len(pending)}", flush=True)
    os.chmod(output, 0o600)

    written = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    manifest = {
        "task_id": "G2-empty-retry",
        "model_alias": group["model_alias"],
        "checkpoint_id": group["checkpoint_id"],
        "checkpoint_revision": group["revision"],
        "source_shard": source.name,
        "source_sha256": sha_file(source),
        "output_shard": output.name,
        "output_sha256": sha_file(output),
        "empty_rows_in_source": source_empty_total,
        "rows_written": len(written),
        "status_counts": {
            status: sum(1 for row in written if row["status"] == status)
            for status in sorted({row["status"] for row in written})
        },
        "config_sha256_verified": sorted({row["retry_config_sha256"] for row in written}),
        "raw_prompts_or_responses_in_manifest": False,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    manifest_path = G2.OUT_DIR / f"g2_{args.tag}_{group['model_alias']}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    print(json.dumps(manifest, indent=4, ensure_ascii=False))

    del model, tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

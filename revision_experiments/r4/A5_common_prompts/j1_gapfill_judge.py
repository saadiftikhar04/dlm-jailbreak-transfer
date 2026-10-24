"""Score regenerated gap-fill responses with the pinned J1 judge and rebuild the merged labels.

The frozen J1 ledger is never modified. Regenerated rows are judged once and
recorded separately, then folded into `j1_final_labels.csv` alongside any
earlier retry pass.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "A4_discrepancy_audit"))

import a5_common_prompt_analysis as A5  # noqa: E402
import a4_existing_evidence as A4  # noqa: E402

LEDGER = HERE / "j1_primary_d3_ledger.csv"
RETRY_LEDGER = HERE / "j1_retry_judge_ledger.csv"
GAPFILL_LEDGER = HERE / "j1_gapfill_judge_ledger.csv"
GAPFILL_MANIFEST = HERE / "j1_gapfill_judge_manifest.json"
FINAL_LABELS = HERE / "j1_final_labels.csv"

FIELDS = [
    "scope", "model", "dataset", "prompt_idx", "canonical_key", "response_sha256",
    "original_prompt_sha256", "request_sha256", "response_status", "status", "category",
    "success", "api_attempts", "provider", "served_model", "usage_cost", "judgment_source",
    "failure_detail",
]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", type=Path, action="append", required=True,
                        help="Regenerated shard CSV; repeatable")
    args = parser.parse_args()

    api_key, route_info = A5.d3_preflight()
    frozen = load_csv(LEDGER)
    if len(frozen) != 22825:
        raise RuntimeError(f"frozen ledger row count changed: {len(frozen)}")
    frozen_by_id = {(r["scope"], r["model"], r["canonical_key"]): r for r in frozen}

    candidates = []
    for path in args.fill:
        for row in load_csv(path):
            if row["status"] != "ok" or not str(row["response"]).strip():
                continue
            scope = f"control:{row['config_id']}"
            key = (scope, row["model_alias"], row["canonical_key"])
            if key not in frozen_by_id:
                raise RuntimeError(f"gap-fill row absent from the frozen ledger: {key}")
            candidates.append({"key": key, "row": row, "source": path.name})
    if not candidates:
        raise RuntimeError("no usable regenerated rows were supplied")

    done: dict[tuple[str, str, str], dict] = {}
    if GAPFILL_LEDGER.is_file() and GAPFILL_LEDGER.stat().st_size:
        for row in load_csv(GAPFILL_LEDGER):
            done[(row["scope"], row["model"], row["canonical_key"])] = row
    pending = [item for item in candidates if item["key"] not in done]

    print(json.dumps({
        "mode": "gapfill_preflight", "candidates": len(candidates),
        "already_scored": len(done), "pending": len(pending), "route": route_info,
    }, ensure_ascii=False, indent=4))

    if not GAPFILL_LEDGER.exists():
        with GAPFILL_LEDGER.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        os.chmod(GAPFILL_LEDGER, 0o600)

    session = A5.requests.Session()
    spent = 0.0
    last_call = 0.0
    for item in pending:
        key = item["key"]
        row = item["row"]
        base = frozen_by_id[key]
        body, request_hash = A5.build_d3_request(
            {"original_prompt": row["original_prompt"], "response": row["response"]}, A4
        )
        wait = A5.D3_ATTEMPT_INTERVAL_SECONDS - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()

        status, category, provider, served_model = "judge_error", "", "", ""
        success: int | str = ""
        cost: float | str = ""
        detail = ""
        try:
            response = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "TMLR R4 J1 gap-fill",
                },
                json=body,
                timeout=300,
            )
            payload = response.json() if response.content else {}
            if not isinstance(payload, dict):
                payload = {}
            if response.status_code != 200:
                detail = json.dumps({"http_status": response.status_code}, ensure_ascii=False)
            else:
                diagnostic = A5.audit_d3_response_metadata(payload, getattr(response, "headers", {}))
                provider = str(diagnostic["provider"] or "")
                served_model = str(diagnostic["served_model"] or "")
                cost = "" if diagnostic["usage_cost"] is None else diagnostic["usage_cost"]
                if diagnostic["usage_cost"]:
                    spent += float(diagnostic["usage_cost"])
                if diagnostic["issues"]:
                    detail = json.dumps({"issues": diagnostic["issues"]}, ensure_ascii=False)
                else:
                    content = ""
                    choices = payload.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        content = str((choices[0].get("message") or {}).get("content") or "")
                    cat, ok, parsed = A4.parse_d3_result(content)
                    if parsed:
                        status, category = "ok", cat
                        success = int(bool(ok))
                    else:
                        detail = json.dumps({"parse_status": "invalid_json"}, ensure_ascii=False)
        except Exception as exc:
            detail = json.dumps({"exception_type": type(exc).__name__}, ensure_ascii=False)

        new_item = {
            "scope": base["scope"], "model": base["model"], "dataset": base["dataset"],
            "prompt_idx": base["prompt_idx"], "canonical_key": base["canonical_key"],
            "response_sha256": sha_text(row["response"]),
            "original_prompt_sha256": base["original_prompt_sha256"],
            "request_sha256": request_hash, "response_status": "regenerated_gap_fill",
            "status": status, "category": category, "success": success, "api_attempts": 1,
            "provider": provider, "served_model": served_model, "usage_cost": cost,
            "judgment_source": f"gapfill_new_api:{item['source']}", "failure_detail": detail,
        }
        with GAPFILL_LEDGER.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(new_item)
            stream.flush()
        done[key] = new_item

    overrides: dict[tuple[str, str, str], dict] = {}
    for path in (RETRY_LEDGER, GAPFILL_LEDGER):
        if path.is_file() and path.stat().st_size:
            for row in load_csv(path):
                if row["status"] != "ok":
                    continue
                overrides[(row["scope"], row["model"], row["canonical_key"])] = row

    merged = []
    for row in frozen:
        hit = overrides.get((row["scope"], row["model"], row["canonical_key"]))
        if hit is None:
            merged.append(row)
            continue
        updated = dict(row)
        for field in ("status", "category", "success", "provider", "served_model",
                      "usage_cost", "judgment_source", "failure_detail",
                      "request_sha256", "response_sha256", "response_status"):
            if field in hit:
                updated[field] = hit[field]
        merged.append(updated)

    with FINAL_LABELS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    os.chmod(FINAL_LABELS, 0o600)

    status_counts = collections.Counter(row["status"] for row in merged)
    manifest = {
        "task_id": "J1-gapfill",
        "judge_route": route_info,
        "frozen_ledger_sha256": file_hash(LEDGER),
        "fill_sources": [path.name for path in args.fill],
        "gapfill_rows": len(load_csv(GAPFILL_LEDGER)),
        "gapfill_ok": sum(1 for row in load_csv(GAPFILL_LEDGER) if row["status"] == "ok"),
        "gapfill_reported_cost_usd": spent,
        "final_status_counts": dict(status_counts),
        "final_ok_rows": status_counts["ok"],
        "final_labelled_rows": status_counts["ok"] + status_counts["empty_response"],
        "output_sha256": {
            GAPFILL_LEDGER.name: file_hash(GAPFILL_LEDGER),
            FINAL_LABELS.name: file_hash(FINAL_LABELS),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    GAPFILL_MANIFEST.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(GAPFILL_MANIFEST, 0o600)
    print(json.dumps(manifest, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Re-judge the recorded J1 judge_error rows once under the pinned paid D3 route.

This pass never edits the frozen J1 ledger. It writes a retry ledger plus a
merged label file so the main ledger stays an immutable record of what happened.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "A4_discrepancy_audit"))

import a5_common_prompt_analysis as A5  # noqa: E402
import a4_existing_evidence as A4  # noqa: E402

LEDGER = HERE / "j1_primary_d3_ledger.csv"
RETRY_LEDGER = HERE / "j1_retry_judge_ledger.csv"
RETRY_MANIFEST = HERE / "j1_retry_judge_manifest.json"
FINAL_LABELS = HERE / "j1_final_labels.csv"
MAX_REQUESTS = 60

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


def main() -> None:
    api_key, route_info = A5.d3_preflight()
    inputs, _ = A5.build_studywide_j1_inputs()
    by_id = {(r["scope"], r["model"], r["canonical_key"]): r for r in inputs}

    frozen_rows = load_csv(LEDGER)
    if len(frozen_rows) != 22825:
        raise RuntimeError(f"frozen ledger row count changed: {len(frozen_rows)}")
    targets = [r for r in frozen_rows if r["status"] == "judge_error"]
    if len(targets) > MAX_REQUESTS:
        raise RuntimeError(f"retry set larger than the authorised cap: {len(targets)}")

    done: dict[tuple[str, str, str], dict] = {}
    if RETRY_LEDGER.is_file() and RETRY_LEDGER.stat().st_size:
        for row in load_csv(RETRY_LEDGER):
            done[(row["scope"], row["model"], row["canonical_key"])] = row

    pending = [r for r in targets if (r["scope"], r["model"], r["canonical_key"]) not in done]
    print(json.dumps({
        "mode": "retry_preflight", "targets": len(targets), "already_done": len(done),
        "pending": len(pending), "route": route_info,
    }, indent=4, ensure_ascii=False))

    if not RETRY_LEDGER.exists():
        with RETRY_LEDGER.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        os.chmod(RETRY_LEDGER, 0o600)

    session = A5.requests.Session()
    spent = 0.0
    last_call = 0.0
    for row in pending:
        row_id = (row["scope"], row["model"], row["canonical_key"])
        record = by_id.get(row_id)
        if record is None:
            raise RuntimeError(f"retry row absent from frozen inputs: {row_id}")
        if record["response_sha256"] != row["response_sha256"]:
            raise RuntimeError(f"retry row response hash changed: {row_id}")

        body, request_hash = A5.build_d3_request(
            {"original_prompt": record["original_prompt"], "response": record["response"]}, A4
        )
        wait = A5.D3_ATTEMPT_INTERVAL_SECONDS - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()

        attempts = 1
        status = "judge_error"
        category = provider = served_model = ""
        success: int | str = ""
        cost: float | str = ""
        detail = ""
        try:
            response = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "TMLR R4 J1 judge_error retry",
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
        except Exception as exc:  # transport failure keeps the row explicit
            detail = json.dumps({"exception_type": type(exc).__name__}, ensure_ascii=False)

        item = {
            "scope": row["scope"], "model": row["model"], "dataset": row["dataset"],
            "prompt_idx": row["prompt_idx"], "canonical_key": row["canonical_key"],
            "response_sha256": row["response_sha256"],
            "original_prompt_sha256": row["original_prompt_sha256"],
            "request_sha256": request_hash, "response_status": row["response_status"],
            "status": status, "category": category, "success": success, "api_attempts": attempts,
            "provider": provider, "served_model": served_model, "usage_cost": cost,
            "judgment_source": "retry_new_api", "failure_detail": detail,
        }
        with RETRY_LEDGER.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(item)
            stream.flush()
        done[row_id] = item

    retry_rows = load_csv(RETRY_LEDGER)
    merged = []
    for row in frozen_rows:
        row_id = (row["scope"], row["model"], row["canonical_key"])
        hit = done.get(row_id)
        if hit is not None and hit["status"] == "ok":
            updated = dict(row)
            for field in ("status", "category", "success", "api_attempts", "provider",
                          "served_model", "usage_cost", "judgment_source", "failure_detail"):
                updated[field] = hit[field]
            merged.append(updated)
        else:
            merged.append(row)

    with FINAL_LABELS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    os.chmod(FINAL_LABELS, 0o600)

    status_counts = collections.Counter(r["status"] for r in merged)
    manifest = {
        "task_id": "J1-judge_error-retry",
        "judge_route": route_info,
        "frozen_ledger_sha256": file_hash(LEDGER),
        "retry_targets": len(targets),
        "retry_recovered_ok": sum(1 for r in retry_rows if r["status"] == "ok"),
        "retry_still_error": sum(1 for r in retry_rows if r["status"] != "ok"),
        "retry_reported_cost_usd": spent,
        "final_status_counts": dict(status_counts),
        "final_ok_rows": status_counts["ok"],
        "total_requests_this_pass": len(pending),
        "output_sha256": {
            RETRY_LEDGER.name: file_hash(RETRY_LEDGER),
            FINAL_LABELS.name: file_hash(FINAL_LABELS),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    RETRY_MANIFEST.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(RETRY_MANIFEST, 0o600)
    print(json.dumps(manifest, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

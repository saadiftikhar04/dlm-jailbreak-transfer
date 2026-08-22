"""Backfill reproduced_response for all samples from the on-disk result CSVs.

The runner overwrites the same results.csv per cell (each sample re-runs the
script fresh), so the LAST run's CSV holds only the last sample. Earlier
samples' responses are recoverable from per-sample logs only for pif; but
metacipher writes metacipher_<model>_1entry_malicious_instruct_results.csv
which is also overwritten. HOWEVER: each run appends nothing - it's a fresh
file. So backfill from logs is needed for all but the final sample of each
cell. Strategy: parse each repro_XX log for the response text.
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent


def extract_from_log(attack: str, log: Path) -> str:
    txt = log.read_text(encoding="utf-8", errors="ignore")
    if attack == "metacipher":
        # response logged between "TARGET MODEL RESPONSE" header line and the following dashes line
        m = re.search(r"TARGET MODEL RESPONSE[^\n]*\n-{10,}\n([\s\S]*?)\n-{10,}", txt)
        if m:
            return m.group(1).strip()
    elif attack == "pif":
        # victim output is not fully logged; use results.csv if this was the last run
        return ""
    else:
        pass
    return ""


def main():
    data = json.loads((HERE / "repro_results.json").read_text())
    # group by cell to know which sample is last
    from collections import defaultdict
    cells = defaultdict(list)
    for i, r in enumerate(data):
        cells[(r["attack"], r["model"])].append(i)

    for (attack, model), idxs in cells.items():
        work = HERE / f"work_{attack}_{model}"
        # last sample: read the fresh results.csv
        if attack == "metacipher":
            f = work / f"metacipher_{model}_1entry_results_FINAL.csv"
            col = "victim_output"
            alt = work / f"metacipher_{model}_1entry_malicious_instruct_results.csv"
        elif attack == "pif":
            f = work / "slim_repo" / "results" / "pif" / model / "malicious_instruct" / "results.csv"
            col = "victim_output"
            alt = None
        else:
            f = work / "slim_repo" / "results" / model / "arrattack_results.csv"
            col = "target_response"
            alt = None
        src = f if f.exists() else alt
        rows = []
        if src and src.exists():
            with open(src, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        last_resp = (rows[-1].get(col, "") or "").strip() if rows else ""

        for pos, i in enumerate(idxs):
            r = data[i]
            if pos == len(idxs) - 1 and last_resp:
                r["reproduced_response"] = last_resp
                r["resp_source"] = "results.csv"
            else:
                log = Path(r["repro_log"])
                t = extract_from_log(attack, log) if log.exists() else ""
                if attack == "arrattack" and log.exists():
                    # arrattack logs "Target response: ..." lines? fallback: leave empty
                    t = ""
                r["reproduced_response"] = t
                r["resp_source"] = "log" if t else "missing"

    (HERE / "repro_results.json").write_text(json.dumps(data, indent=4, ensure_ascii=False))
    n_log = sum(1 for r in data if r.get("resp_source") == "log")
    n_csv = sum(1 for r in data if r.get("resp_source") == "results.csv")
    n_miss = sum(1 for r in data if r.get("resp_source") == "missing")
    print(f"backfilled: log={n_log} results.csv={n_csv} missing={n_miss}")


if __name__ == "__main__":
    main()

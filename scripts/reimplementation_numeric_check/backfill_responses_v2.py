"""Backfill v2: parse responses from per-sample logs for metacipher and pif,
results.csv for the last sample of each cell. Handles logging prefixes."""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()

LOG_TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (INFO|WARNING|ERROR) - ")


def strip_ts(line: str) -> str:
    return LOG_TS.sub("", line.rstrip())


def extract_metacipher(txt: str) -> str:
    # last TARGET MODEL RESPONSE block before its closing dash line
    blocks = re.split(r"TARGET MODEL RESPONSE[^\n]*\n", txt)
    if len(blocks) < 2:
        return ""
    best = ""
    for b in blocks[1:]:
        lines = b.splitlines()
        if not lines:
            continue
        body = [strip_ts(l) for l in lines[1:]]  # skip opening dashes
        out, done = [], False
        for l in body:
            s = l.strip()
            if set(s) <= {"─", "-"} and len(s) >= 20:
                done = True
                break
            out.append(l)
        text = "\n".join(out).strip()
        if text and not best:
            best = text  # first attempt's response is what we want (final attempt recorded)
            # keep scanning only if empty; use FIRST block (attempt1) — but final_cipher may differ.
            break
    return best


def extract_pif(txt: str) -> str:
    # final victim output logged as "  Response (Nc): <text>"
    matches = re.findall(r"Response \(\d+c\): (.*)", txt)
    return matches[-1].strip() if matches else ""


def main():
    data = json.loads((HERE / "repro_results.json").read_text())
    from collections import defaultdict
    cells = defaultdict(list)
    for i, r in enumerate(data):
        cells[(r["attack"], r["model"])].append(i)

    for (attack, model), idxs in cells.items():
        work = HERE / f"work_{attack}_{model}"
        src = None
        col = None
        if attack == "metacipher":
            src = work / f"metacipher_{model}_1entry_results_FINAL.csv"
            col = "victim_output"
        elif attack == "pif":
            src = work / "slim_repo" / "results" / "pif" / model / "malicious_instruct" / "results.csv"
            col = "victim_output"
        else:
            src = work / "slim_repo" / "results" / model / "arrattack_results.csv"
            col = "target_response"
        rows = []
        if src.exists():
            with open(src, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        last_resp = (rows[-1].get(col, "") or "").replace("\\n", " ").strip() if rows else ""

        for pos, i in enumerate(idxs):
            r = data[i]
            if r.get("resp_source") == "results.csv":
                continue
            log = Path(r["repro_log"])
            resp = ""
            if pos == len(idxs) - 1 and last_resp:
                resp, source = last_resp, "results.csv"
            elif log.exists():
                txt = log.read_text(encoding="utf-8", errors="ignore")
                if attack == "metacipher":
                    resp = extract_metacipher(txt)
                elif attack == "pif":
                    resp = extract_pif(txt)
                source = "log" if resp else "missing"
            else:
                source = "missing"
            r["reproduced_response"] = resp.replace("\n", "\\n")[:4000] if resp else ""
            r["resp_source"] = source

    (HERE / "repro_results.json").write_text(json.dumps(data, indent=4, ensure_ascii=False))
    from collections import Counter
    print(Counter(r.get("resp_source") for r in data))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
01_inventory.py — Enumerate and inventory every result CSV in the repo.

The student's `/results` tree is believed format-correct, but its VALUES are not
yet trusted. Before sampling any reproduction, we build a full machine-readable
inventory of every CSV file: its role (raw vs judged), attack, model, dataset,
row count, and exact column set. This gives the sampling step a complete,
version-stamped universe to draw from.

Re-run this after ANY `git pull`/`git checkout` of the results so the sampled
manifest always matches the on-disk universe.

Outputs
-------
scripts/cross-check/outputs/inventory.json
    - universe: list of {role, attack, model, dataset, path, rows, columns, sha256}
    - rollup  : per (attack, role) file counts + total row counts
    - git     : results/ git snapshot (HEAD sha + per-file log) for provenance
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import _paths
HERE = Path(__file__).resolve().parent
ROOT = _paths.ROOT
RESULTS = ROOT / "results"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RAW_SUBDIRS = {
    "pif": "pif",
    "metacipher": "metacipher",
    "arrattack": "arrattack",
    "arrattack_judged_special": "arrattack",  # handled below
}
JUDGED_DIRS = {
    "pif": Path("pif/PIF_JUDGED"),
    "metacipher": Path("metacipher/Metacipher_Judged"),
    "arrattack": Path("arrattack/Arrattack_Judged"),
}

MODELS = {"falcon", "qwen", "qwen2.5", "llama", "llada", "dream", "diffucoder"}


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_and_cols(p: Path):
    """Robust CSV read: handles quoted multiline fields; returns (rows, cols)."""
    with open(p, newline="", encoding="utf-8-sig") as f:
        rd = list(csv.reader(f))
    if not rd:
        return 0, []
    header = rd[0]
    n = 0
    for r in rd[1:]:
        # a data row must have at least as many fields as header (or be a full row)
        if r and any(x.strip() for x in r):
            n += 1
    return n, header


def git_snapshot():
    def g(*a):
        try:
            return subprocess.run(
                ["git", "-C", str(ROOT), *a],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except Exception:
            return None
    return {
        "head": g("rev-parse", "HEAD"),
        "head_short": g("rev-parse", "--short", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "last_results_commit": g("log", "-1", "--format=%h %ci %s", "--", "results"),
        "results_log": g("log", "--format=%h %ci %s", "--", "results").splitlines()[:8],
    }


def classify(p: Path):
    """Return role/attack/model/dataset for a results file, or None if unknown."""
    rel = p.relative_to(RESULTS)
    parts = rel.parts  # e.g. ('pif','falcon','harmbench','results.csv')

    # PIF raw: pif/<model>/<dataset>/results.csv
    if parts[0] == "pif" and parts[-1] == "results.csv" and len(parts) == 4:
        return ("raw", "pif", parts[1], parts[2])
    # PIF judged: pif/PIF_JUDGED/<model>_pif_final_judged.csv
    if len(parts) >= 2 and parts[1] == "PIF_JUDGED":
        m = parts[2].split("_pif_final_judged.csv")[0]
        return ("judged", "pif", m, None)
    # MC: metacipher/<model>/metacipher_results.csv ; metacipher/Metacipher_Judged/<model>.csv
    if parts[0] == "metacipher":
        if parts[-1] == "metacipher_results.csv" and len(parts) == 3:
            return ("raw", "metacipher", parts[1], None)
        if len(parts) >= 2 and parts[1] == "Metacipher_Judged":
            m = parts[2].rsplit(".csv", 1)[0]
            return ("judged", "metacipher", m, None)
    # AA: arrattack/<model>/arrattack_results.csv ; arrattack/Arrattack_Judged/arrattack_*_judged.csv
    if parts[0] == "arrattack":
        if parts[-1] == "arrattack_results.csv" and len(parts) == 3:
            return ("raw", "arrattack", parts[1], None)
        if len(parts) >= 2 and parts[1] == "Arrattack_Judged":
            fn = parts[2]
            m = fn.replace("arrattack_", "", 1).replace("_judged.csv", "", 1)
            return ("judged", "arrattack", m, None)
    return None


def main():
    universe = []
    for p in sorted(RESULTS.rglob("*.csv")):
        cls = classify(p)
        n, cols = rows_and_cols(p)
        entry = {
            "role": cls[0] if cls else "unknown",
            "attack": cls[1] if cls else "unknown",
            "model": cls[2] if cls else "unknown",
            "dataset": cls[3] if cls else None,
            "path": str(p.relative_to(ROOT)),
            "rows": n,
            "columns": cols,
            "sha256": file_sha256(p),
        }
        universe.append(entry)

    total_rows = sum(e["rows"] for e in universe)
    rollup = defaultdict(lambda: defaultdict(list))
    for e in universe:
        rollup[(e["role"], e["attack"])]["files"].append(
            {"model": e["model"], "dataset": e["dataset"], "rows": e["rows"]})

    inv = {
        "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "repo_root": str(ROOT),
        "git": git_snapshot(),
        "universe_count": len(universe),
        "total_rows": total_rows,
        "rollup": {
            f"{r}/{a}": {"n_files": len(v["files"]), "rows": sum(f["rows"] for f in v["files"])}
            for (r, a), v in rollup.items()
        },
        "files": universe,
    }
    out_f = OUT / "inventory.json"
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=4, ensure_ascii=False)

    print(f"Wrote {out_f}")
    print(f"Files: {len(universe)} | Total rows: {total_rows}")
    for key, val in sorted(inv["rollup"].items()):
        print(f"  {key:<22} {val['n_files']} files / {val['rows']} rows")
    unk = [e for e in universe if e["role"] == "unknown"]
    if unk:
        print(f"\n!!! {len(unk)} unclassified file(s):")
        for e in unk:
            print("   ", e["path"], e["columns"])


if __name__ == "__main__":
    sys.exit(main())
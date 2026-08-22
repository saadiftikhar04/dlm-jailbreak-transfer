#!/usr/bin/env python3
"""
02_sample.py — Draw a fixed, reproducible random sample from every result file.

Goal
----
The student's data is format-correct but not yet trusted. We will reproduce a
small RANDOM subset of every file and compare the newly generated responses
against what the CSV claims. This script draws that subset.

Design decisions
----------------
* One manifest covering ALL 54 files (not a hand-picked few), so no file is
  exempt from spot-checking.
* A fixed global seed (override with --seed) makes the same sample appear on
  every re-run, so the audit is reproducible and reviewable.
* Sampling is STRATIFIED at the granularity the file already stores:
    - pif raw      : 24 files, each = one (model, dataset). Sample per FILE.
    - pif judged   :  6 files, each = 913 rows across 4 datasets. Sample per FILE,
                      and additionally ensure every dataset is represented.
    - metacipher   :  6 raw + 6 judged = 12 files, 913 rows each. Sample per FILE.
    - arrattack raw:  6 files, 165 decision rows sampled from the multi-attempt table
                      (each (dataset,prompt_idx) block). Sample per FILE.
    - arrattack judged: 6 files, 165 rows. Sample per FILE.
* Each sample row stores enough provenance to re-run one attack: the row identity
  (dataset, prompt_idx, original_prompt), the file, and which response column to
  compare against after reproduction.

Default sample sizes (override via CLI):
    --n-per-file      : rows sampled per judged / decision file        (default 8)
    --n-raw-pif       : rows sampled per (model, dataset) pif raw      (default 4)
    --arrattack-raw-samples-per-model : decision blocks in AA raw      (default 6)

Outputs
-------
outputs/manifest.json       — full manifest (all files + all drawn rows)
outputs/manifest.csv        — flat table of drawn rows for review
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import _paths

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Column that holds the *victim model's raw response* for each attack+role.
# This is what a reproduction must re-derive and compare against.
RESPONSE_COL = {
    ("raw", "pif"): "victim_output",
    ("judged", "pif"): "victim_output",
    ("raw", "metacipher"): "target_response",
    ("judged", "metacipher"): "final_response",
    ("raw", "arrattack"): "target_response",
    ("judged", "arrattack"): "final_response",
}

# Column that identifies the prompt in each file.
PROMPT_COL = {
    ("raw", "pif"): "original_prompt",
    ("judged", "pif"): "original_prompt",
    ("raw", "metacipher"): "original_prompt",
    ("judged", "metacipher"): "original_prompt",
    ("raw", "arrattack"): "original_prompt",
    ("judged", "arrattack"): "original_prompt",
}


def read_csv_dicts(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-file", type=int, default=8,
                    help="rows per judged / decision file (PIF judged, MC raw+judged, AA judged)")
    ap.add_argument("--n-raw-pif", type=int, default=4,
                    help="rows per (model,dataset) PIF raw file")
    ap.add_argument("--n-raw-aa", type=int, default=6,
                    help="decision blocks sampled from each AA raw multi-attempt file")
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    inv = json.load(open(OUT / "inventory.json", encoding="utf-8"))
    files = inv["files"]

    manifest = {"seed": args.seed, "params": vars(args), "rows": []}
    by_att_model = defaultdict(list)

    for e in files:
        role, attack, model = e["role"], e["attack"], e["model"]
        if role == "unknown":
            continue
        p = _paths.ROOT / e["path"]
        rows = read_csv_dicts(p)
        resp_col = RESPONSE_COL.get((role, attack))
        prompt_col = PROMPT_COL.get((role, attack))

        # ---- choose which rows to draw ----
        if role == "raw" and attack == "pif":
            # pif raw: each file is one (model,dataset); uniform sample across rows
            drawn = _draw_uniform(rng, rows, args.n_raw_pif, keyfunc=lambda r: r.get(prompt_col, ""))

        elif role == "judged" and attack == "pif":
            # judged: 913 rows, 4 datasets stratified
            drawn = _draw_stratified(
                rng, rows, args.n_per_file, "dataset",
                keyfunc=lambda r: r.get(prompt_col, ""),
                min_per_group=1)

        elif attack == "metacipher":
            # raw + judged both 913 rows, stratified by dataset
            drawn = _draw_stratified(
                rng, rows, args.n_per_file, "dataset",
                min_per_group=1)

        elif role == "raw" and attack == "arrattack":
            # multi-attempt table: group by (dataset, prompt_idx), pick distinct decision blocks
            blocks = {}
            for i, r in enumerate(rows):
                blocks.setdefault((r.get("dataset"), r.get("prompt_idx")), i)
            block_ids = sorted(blocks.values(), key=lambda i: (rows[i].get("dataset", ""), int(rows[i].get("prompt_idx") or 0)))
            chosen_blocks = rng.sample(block_ids, min(args.n_raw_aa, len(block_ids)))
            drawn = [
                {"block_start_row": blocks.get((rows[i].get("dataset"), rows[i].get("prompt_idx"))),
                 "row": rows[i], "idx_in_file": i}
                for i in chosen_blocks
            ]

        elif role == "judged" and attack == "arrattack":
            drawn = _draw_stratified(
                rng, rows, args.n_per_file, "dataset", min_per_group=1)

        else:
            continue

        for d in drawn:
            row = d["row"]
            manifest["rows"].append({
                "attack": attack,
                "role": role,
                "model": model,
                "dataset": e["dataset"] if e["dataset"] is not None else row.get("dataset"),
                "file": e["path"],
                "row_idx_in_file": d.get("idx_in_file", -1),
                "block_start_row": d.get("block_start_row", None),
                "prompt_idx": row.get("prompt_idx"),
                "prompt": row.get(prompt_col, ""),
                "resp_col": resp_col,
                "recorded_response": row.get(resp_col) if resp_col else None,
                "asr_flags": {k: row.get(k) for k in
                              ("asr_success", "llm_judge", "gpt_fuzz",
                               "attack_success_internal", "attack_success_gptfuzz",
                               "success", "judge_gpt")
                              if k in row},
            })
        by_att_model[f"{attack}/{role}/{model}"].append(len(drawn))

    mf = OUT / "manifest.json"
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    # flat CSV for human review
    mc = OUT / "manifest.csv"
    cols = ["attack", "role", "model", "dataset", "file", "prompt_idx",
            "resp_col", "asr_flags", "prompt"]
    with open(mc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in manifest["rows"]:
            w.writerow({k: ("" if r.get(k) is None else r[k])
                        for k in cols})

    print(f"Seed {args.seed} | manifest: {len(manifest['rows'])} sample rows -> {mf}")
    for k, v in sorted(by_att_model.items()):
        print(f"  {k:<28} {v}")


def _draw_uniform(rng, rows, n, keyfunc=lambda r: ""):
    n = min(n, len(rows))
    idxs = rng.sample(range(len(rows)), n)
    return [{"idx_in_file": i, "row": rows[i]} for i in idxs]


def _draw_stratified(rng, rows, n, group_col, keyfunc=lambda r: "", min_per_group=0):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        v = r.get(group_col, "")
        groups[str(v if v is not None else "")].append(i)
    chosen = []
    used = set()
    if min_per_group:
        for g, idxs in groups.items():
            take = min(min_per_group, len(idxs))
            if take:
                s = rng.sample(idxs, take)
                chosen.extend(s)
                used.update(s)
    remaining = [i for i in range(len(rows)) if i not in used]
    want = max(0, n - len(chosen))
    if remaining and want:
        chosen.extend(rng.sample(remaining, min(want, len(remaining))))
    return [{"idx_in_file": i, "row": rows[i]} for i in chosen]


if __name__ == "__main__":
    sys.exit(main())
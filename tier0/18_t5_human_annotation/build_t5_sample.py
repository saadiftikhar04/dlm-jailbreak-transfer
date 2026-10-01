"""T5 (C2 path C): build the blinded human annotation sample concentrated on the
contested cells + a 50-row overlap for a second annotator (real kappa).
Outputs tier0/18_t5_human_annotation/t5_annotation_sample.csv (blinded: only
original_prompt + final_response; NO judge labels). Labels column blank for the
annotator. overlap=1 marks the 50-row subset for a second annotator.
"""
import json, os, csv
import numpy as np

CJ = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/11_common_judge_rescore/common_judge_all_results.jsonl"
OUT = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/18_t5_human_annotation/t5_annotation_sample.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
SEED = 20260822
rng = np.random.default_rng(SEED)

# Contested cells (high-per-label value per D1): Dream/DiffuCoder MetaCipher
# (Claude ~79% vs official 0-0.6%), LLaDA ArrAttack (Claude 60.6 vs 8.5),
# Falcon MetaCipher (Claude 9-17.5 vs 0). Plus a small non-contested calibrator set.
TARGETS = {
    ("metacipher", "dream"): 60,
    ("metacipher", "diffucoder"): 60,
    ("arrattack", "llada"): 40,
    ("metacipher", "falcon"): 40,
    ("metacipher", "qwen"): 15,     # high-ASR calibrator
    ("metacipher", "llada"): 15,    # high-ASR calibrator
}
N_TARGET = sum(TARGETS.values())   # ~230

rows = []
for line in open(CJ):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    rows.append(r)

# sample per cell
sel = []
for (atk, m), cnt in TARGETS.items():
    cell = [r for r in rows if r["attack"] == atk and r["model"] == m]
    idx = rng.choice(len(cell), size=min(cnt, len(cell)), replace=False)
    for i in idx:
        r = cell[i]
        sel.append(r)

# add prompt_idx for traceability, dedup
records = {}
for r in sel:
    k = (r["attack"], r["model"], r["dataset"], r["prompt_idx"])
    records[k] = r
all_r = list(records.values())
rng.shuffle(all_r)

# 50-row overlap subset
overlap = set(rng.choice(len(all_r), size=50, replace=False))

out = []
for i, r in enumerate(sorted(all_r, key=lambda x: (x["attack"], x["model"]))):
    out.append({
        "id": i + 1,
        "overlap": 1 if i in overlap else 0,
        "attack": r["attack"], "model": r["model"], "dataset": r["dataset"],
        "prompt_idx": r["prompt_idx"],
        "original_prompt": r["original_prompt"],
        "response": (r["final_response"] or "") if isinstance(r.get("final_response"), str) else "",
        "human_label": "",   # annotator fills: 1 = harmful compliance, 0 = not
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

import collections
print("total:", len(out), "| overlap rows:", sum(1 for r in out if r["overlap"]))
print("by cell:", collections.Counter((r["attack"], r["model"]) for r in out))
print("wrote", OUT)
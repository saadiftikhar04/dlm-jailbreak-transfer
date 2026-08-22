"""
prepare_dataset.py
==================
Merges HarmBench, AdvBench, JailbreakBench, MaliciousInstruct, StrongReject
into a single deduplicated dataset, then creates non-overlapping splits.

Paper scale: 150 (judgment) / 579 (generation) / 196 (test) = 925 total
We have ~1433 prompts across all datasets — more than enough.

Output files (PROJECT_DIR/dataset/combined/):
  all_prompts.txt         — full deduplicated prompt list
  judgment_prompts.txt    — 150 prompts for Stage 1
  generation_prompts.txt  — 579 prompts for Stage 3
  test_prompts.txt        — 196 prompts for Stage 5
"""

import csv
import json
import os
import random

PROJECT_DIR  = "/scratch/si2356/dlm-jailbreak-transfer"
DATASET_DIR  = PROJECT_DIR + "/dataset"
OUT_DIR      = DATASET_DIR + "/combined"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

# ── Paper split sizes ─────────────────────────────────────────────────────────
N_JUDGMENT   = 150
N_GENERATION = 579
N_TEST       = 196

prompts = []

# ── 1. HarmBench ──────────────────────────────────────────────────────────────
path = DATASET_DIR + "/harmbench/text_all.csv"
with open(path) as f:
    for row in csv.DictReader(f):
        p = (row.get("Behavior") or "").strip()
        if p:
            prompts.append({"text": p, "source": "harmbench"})
print(f"HarmBench:         {len([x for x in prompts if x['source']=='harmbench'])} prompts")

# ── 2. AdvBench ───────────────────────────────────────────────────────────────
path = DATASET_DIR + "/advbench/harmful_behaviors.csv"
n_before = len(prompts)
with open(path) as f:
    for row in csv.DictReader(f):
        p = (row.get("goal") or "").strip()
        if p:
            prompts.append({"text": p, "source": "advbench"})
print(f"AdvBench:          {len(prompts)-n_before} prompts")

# ── 3. JailbreakBench ─────────────────────────────────────────────────────────
n_before = len(prompts)
# Try jailbreakbench.csv first
path = DATASET_DIR + "/jailbreakbench/jailbreakbench.csv"
if os.path.exists(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        goal_col = next((c for c in cols if c.lower() in ["goal","behavior","prompt","forbidden_prompt"]), None)
        if goal_col:
            for row in reader:
                p = (row.get(goal_col) or "").strip()
                if p:
                    prompts.append({"text": p, "source": "jailbreakbench"})
# Also try processed.csv
path2 = DATASET_DIR + "/jailbreakbench/processed.csv"
if os.path.exists(path2) and len(prompts) - n_before == 0:
    with open(path2) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        goal_col = next((c for c in cols if c.lower() in ["goal","behavior","prompt","forbidden_prompt"]), None)
        if goal_col:
            for row in reader:
                p = (row.get(goal_col) or "").strip()
                if p:
                    prompts.append({"text": p, "source": "jailbreakbench"})
print(f"JailbreakBench:    {len(prompts)-n_before} prompts")

# ── 4. MaliciousInstruct ──────────────────────────────────────────────────────
n_before = len(prompts)
path = DATASET_DIR + "/malicious_instruct/processed.csv"
if os.path.exists(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        goal_col = next((c for c in cols if c.lower() in ["goal","behavior","instruction","prompt","text"]), None)
        if goal_col:
            for row in reader:
                p = (row.get(goal_col) or "").strip()
                if p:
                    prompts.append({"text": p, "source": "malicious_instruct"})
# Also try txt
path2 = DATASET_DIR + "/malicious_instruct/malicious_instruct.txt"
if os.path.exists(path2) and len(prompts) - n_before == 0:
    with open(path2) as f:
        for line in f:
            p = line.strip()
            if p:
                prompts.append({"text": p, "source": "malicious_instruct"})
print(f"MaliciousInstruct: {len(prompts)-n_before} prompts")

# ── 5. StrongReject ───────────────────────────────────────────────────────────
n_before = len(prompts)
path = DATASET_DIR + "/strongreject/strongreject.csv"
if os.path.exists(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        goal_col = next((c for c in cols if c.lower() in ["forbidden_prompt","goal","behavior","prompt","text"]), None)
        if goal_col:
            for row in reader:
                p = (row.get(goal_col) or "").strip()
                if p:
                    prompts.append({"text": p, "source": "strongreject"})
print(f"StrongReject:      {len(prompts)-n_before} prompts")

# ── Deduplicate ───────────────────────────────────────────────────────────────
seen = set()
deduped = []
for item in prompts:
    key = item["text"].lower().strip()
    if key not in seen and len(key) > 20:
        seen.add(key)
        deduped.append(item)

print(f"\nTotal before dedup: {len(prompts)}")
print(f"Total after dedup:  {len(deduped)}")

total_needed = N_JUDGMENT + N_GENERATION + N_TEST
if len(deduped) < total_needed:
    print(f"\nWARNING: Only {len(deduped)} prompts, need {total_needed}")
    print("Adjusting split sizes proportionally...")
    ratio = len(deduped) / total_needed
    N_JUDGMENT   = int(N_JUDGMENT   * ratio)
    N_GENERATION = int(N_GENERATION * ratio)
    N_TEST       = len(deduped) - N_JUDGMENT - N_GENERATION
    print(f"Adjusted: {N_JUDGMENT}/{N_GENERATION}/{N_TEST}")

# ── Shuffle and split ─────────────────────────────────────────────────────────
random.shuffle(deduped)

judgment_set   = deduped[:N_JUDGMENT]
generation_set = deduped[N_JUDGMENT:N_JUDGMENT+N_GENERATION]
test_set       = deduped[N_JUDGMENT+N_GENERATION:N_JUDGMENT+N_GENERATION+N_TEST]

# ── Write outputs ─────────────────────────────────────────────────────────────
with open(OUT_DIR + "/all_prompts.txt", "w") as f:
    for item in deduped:
        f.write(item["text"] + "\n")

with open(OUT_DIR + "/judgment_prompts.txt", "w") as f:
    for item in judgment_set:
        f.write(item["text"] + "\n")

with open(OUT_DIR + "/generation_prompts.txt", "w") as f:
    for item in generation_set:
        f.write(item["text"] + "\n")

with open(OUT_DIR + "/test_prompts.txt", "w") as f:
    for item in test_set:
        f.write(item["text"] + "\n")

# Write metadata
meta = {
    "seed": SEED,
    "total": len(deduped),
    "judgment": N_JUDGMENT,
    "generation": N_GENERATION,
    "test": N_TEST,
    "sources": {s: sum(1 for x in deduped if x["source"]==s)
                for s in set(x["source"] for x in deduped)},
}
with open(OUT_DIR + "/split_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n=== SPLIT COMPLETE ===")
print(f"Judgment  ({N_JUDGMENT}):   {OUT_DIR}/judgment_prompts.txt")
print(f"Generation({N_GENERATION}): {OUT_DIR}/generation_prompts.txt")
print(f"Test      ({N_TEST}):   {OUT_DIR}/test_prompts.txt")
print(f"Metadata:               {OUT_DIR}/split_metadata.json")

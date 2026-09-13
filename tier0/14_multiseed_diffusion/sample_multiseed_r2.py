"""R2 Step 14 sampler (FIXED): 6-cell multi-seed diffusion sample.

Cells: (model, attack), model in [dream, diffucoder, llada], attack in [pif, arrattack].
~300 prompts/cell stratified to pool proportions (harmbench 131, strongreject 103,
jailbreakbench 33, malicious_instruct 33).

PiF: from PIF_JUDGED pif_prompt (913 rows, complete).
ArrAttack: from the full-pool results. IMPORTANT: dream/diffucoder shards renumber
(dataset,dataset_idx) per shard, so that key is useless cross-shard. The correct
identity is the ORIGINAL PROMPT TEXT (norms to a unique key). For each unique prompt
text pick the jailbreak_prompt of its best_attempt row from that shard's results.
llada is unsharded and keyed by (dataset,dataset_idx) directly.

Output: multiseed_sample_r2.json (indent=4).
"""
import os, sys, json
import numpy as np, pandas as pd
from collections import defaultdict, Counter
import re

ROOT = os.environ.get("DATA_ROOT", "/home/bc3194/Desktop/dlm-jailbreak-transfer")
AA = os.environ.get("AA_ROOT", ROOT)
SEED = 20260910
rng = np.random.default_rng(SEED)
MODELS = ["dream", "diffucoder", "llada"]
PROPS = {"harmbench": 131, "strongreject": 103, "jailbreakbench": 33,
         "malicious_instruct": 33}
SAMPLE_DATASETS = set(PROPS.keys())


def norm(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.replace('""', '"').replace("''", "'")
    s = re.sub(r'["\'“”‘’]', '', s)
    return s


def load_pif_rows(model):
    p = os.path.join(ROOT, "results", "pif", "PIF_JUDGED", f"{model}_pif_final_judged.csv")
    df = pd.read_csv(p).dropna(subset=["pif_prompt"])
    out = []
    for _, r in df.iterrows():
        ap = r["pif_prompt"] if (isinstance(r["pif_prompt"], str) and r["pif_prompt"].strip()) else r["original_prompt"]
        out.append({"dataset": r["dataset"], "prompt_idx": int(r["prompt_idx"]),
                    "original_prompt": r["original_prompt"], "attacked_prompt": ap,
                    "uid": norm(r["original_prompt"])})
    return out


def load_arr_rows(model):
    """Return list of {dataset, original_prompt, jailbreak_prompt, uid}. unique by uid.

    Builds a one-time uid index (dict) instead of scanning the full results
    frame per prompt — dream/diffucoder results are ~28k rows/shard x 10.
    """
    acc = {}
    if model == "llada":
        prog = pd.read_csv(os.path.join(AA, "results", "arrattack", "llada", "arrattack_progress.csv"))
        res = pd.read_csv(os.path.join(AA, "results", "arrattack", "llada", "arrattack_results.csv"))
        prog["uid"] = prog["original_prompt"].map(norm)
        res["uid"] = res["original_prompt"].map(norm)
        # best index: uid -> rows (llada unsharded, unique prompts)
        res_by_uid = {}
        for _, rr in res.iterrows():
            res_by_uid.setdefault(rr["uid"], []).append(rr)
        for u in prog["uid"].unique():
            if not u or u in acc:
                continue
            rows = res_by_uid.get(u, [])
            if not rows:
                continue
            # best_attempt row wins, else first
            ba_set = set(prog[prog["uid"] == u]["best_attempt"])
            hit = next((r for r in rows if r["attempt"] in ba_set), rows[0])
            jp = hit["jailbreak_prompt"] if (isinstance(hit["jailbreak_prompt"], str) and hit["jailbreak_prompt"].strip()) else hit["original_prompt"]
            acc[u] = {"dataset": hit["dataset"], "original_prompt": hit["original_prompt"],
                      "attacked_prompt": jp, "uid": u}
        return list(acc.values())

    # sharded dream/diffucoder: uid -> best_attempt jailbreak_prompt across all shards
    for sh in range(1, 11):
        prog = pd.read_csv(os.path.join(AA, "arrattack_fullpool", "results_shard",
                                        f"{model}_{sh:02d}", "arrattack_progress.csv"))
        res = pd.read_csv(os.path.join(AA, "arrattack_fullpool", "results_shard",
                                       f"{model}_{sh:02d}", "arrattack_results.csv"))
        prog["uid"] = prog["original_prompt"].map(norm)
        prog["_ba"] = prog["best_attempt"]
        res["uid"] = res["original_prompt"].map(norm)
        # index res by uid
        res_by_uid = {}
        for _, rr in res.iterrows():
            res_by_uid.setdefault(rr["uid"], []).append(rr)
        for _, pr in prog.iterrows():
            u = pr["uid"]
            if not u or u in acc:
                continue
            rows = res_by_uid.get(u, [])
            if not rows:
                continue
            hit = next((r for r in rows if r["attempt"] == pr["_ba"]), rows[0])
            jp = hit["jailbreak_prompt"] if (isinstance(hit["jailbreak_prompt"], str) and hit["jailbreak_prompt"].strip()) else hit["original_prompt"]
            acc[u] = {"dataset": hit["dataset"], "original_prompt": hit["original_prompt"],
                      "attacked_prompt": jp, "uid": u}
    return list(acc.values())


def sample_cell(rows, key_field, seed):
    rng2 = np.random.default_rng(seed)
    byds = defaultdict(list)
    for r in rows:
        byds[r["dataset"]].append(r)
    chosen = []
    for ds, n in PROPS.items():
        pool = byds.get(ds, [])
        if len(pool) < n:
            print(f"  WARN {key_field}: {ds} only {len(pool)} < {n}", flush=True)
        picks = rng2.choice(pool, size=min(n, len(pool)), replace=False) if pool else []
        chosen.extend(picks)
    return chosen


rows_out = []
for model in MODELS:
    # PiF
    pif = load_pif_rows(model)
    picked = sample_cell(pif, f"{model}/pif", SEED)
    for r in picked:
        rows_out.append({"attack": "pif", "model": model, "model_family": "diffusion",
                         "dataset": r["dataset"], "prompt_idx": r["prompt_idx"],
                         "arm": "A", "original_prompt": r["original_prompt"],
                         "attacked_prompt": r["attacked_prompt"]})
    print(f"{model}/pif: {len(picked)}", flush=True)
    # ArrAttack
    ar = load_arr_rows(model)
    picked = sample_cell(ar, f"{model}/arrattack", SEED)
    for r in picked:
        ds_norm = norm(ar[0]["original_prompt"])
        # map back to pool prompt_idx via pool by uid
        rows_out.append({"attack": "arrattack", "model": model, "model_family": "diffusion",
                         "dataset": r["dataset"], "prompt_idx": -1,  # corrected below via pool
                         "arm": "A", "original_prompt": r["original_prompt"],
                         "attacked_prompt": r["attacked_prompt"]})
    print(f"{model}/arrattack: {len(picked)}", flush=True)

# join arrattack prompt_idx from pool by uid
pool = pd.read_csv(os.path.join(ROOT, "arrattack_fullpool", "pool_913.csv"))
pool["uid"] = pool["original_prompt"].map(norm)
uid2pid = dict(zip(pool.uid, pool.prompt_idx))
for r in rows_out:
    if r["attack"] == "arrattack":
        pid = uid2pid.get(norm(r["original_prompt"]))
        r["prompt_idx"] = pid if pid is not None else -1
    if r["prompt_idx"] == -1:
        print(f"  ! no pool idx for {r['model']}/{r['attack']}: {r['original_prompt'][:50]}")

out = os.path.join(ROOT, "tier0", "14_multiseed_diffusion", "multiseed_sample_r2.json")
with open(out, "w") as f:
    json.dump(rows_out, f, indent=4, ensure_ascii=False)
print("\nper (model,attack):", dict(Counter((r["model"], r["attack"]) for r in rows_out)))
print(f"wrote {out}: {len(rows_out)} rows, indent=4")
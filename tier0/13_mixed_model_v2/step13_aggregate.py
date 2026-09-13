"""
R2 Step 13 — aggregate full-pool ArrAttack results per victim.

stage5_fullpool writes per-victim results/<key>/arrattack_progress.csv (one row
per prompt: jailbroken_gptfuzz / jailbroken_llm = final-stage labels) and
results/<key>/arrattack_results.csv (per-attempt). The paper's ASR is
asr_success == (gpt_fuzz == compliance), i.e. jailbroken_gptfuzz.

This reads PROGRESS_CSV per victim, tallies ASR per (dataset) and overall, and
writes a summary table + Wilson intervals via common.wilson_ci.
"""
import os, sys, csv, glob, json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as CM

HPC_RESULTS = os.environ.get(
    "ARR_FULLPOOL_RESULTS",
    "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/results")
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step13_results")
os.makedirs(LOCAL, exist_ok=True)

KEYS = {"qwen2.5": "qwen", "llama": "llama", "falcon": "falcon",
        "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}


def _wilson(succ, total):
    lo, hi, p = CM.wilson_ci(succ, total)
    return f"[{100*lo:.1f},{100*hi:.1f}]"

rows = []
for key, short in KEYS.items():
    d = os.path.join(HPC_RESULTS, key)
    prog = os.path.join(d, "arrattack_progress.csv")
    if not os.path.exists(prog):
        print(f"[step13] {short}: NO progress csv at {prog}")
        continue
    df = pd.read_csv(prog)
    n = len(df)
    col = "jailbroken_gptfuzz"
    s = df[col].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "1.0": 1, "false": 0, "0": 0, "0.0": 0})
    succ = pd.to_numeric(s, errors="coerce").fillna(0)
    rows.append({"model": short, "n": n, "success": int(succ.sum()),
                 "asr_pct": round(100 * succ.mean(), 2)})
    # per-dataset
    if "dataset" in df.columns:
        for ds, g in df.groupby("dataset"):
            ss = pd.to_numeric(g[col].astype(str).str.strip().str.lower().map(
                {"true": 1, "1": 1, "1.0": 1, "false": 0, "0": 0, "0.0": 0}),
                errors="coerce").fillna(0)
            rows.append({"model": short, "n": len(g), "success": int(ss.sum()),
                         "asr_pct": round(100 * ss.mean(), 2), "dataset": ds})

out = pd.DataFrame(rows)
# add Wilson CI for model-level (n=913) rows
out["ci"] = out.apply(lambda r: "" if "dataset" in r and pd.notna(r.get("dataset"))
                      else _wilson(r["success"], r["n"]), axis=1)
out.to_csv(os.path.join(LOCAL, "step13_fullpool_summary.csv"), index=False)
print(out.to_string(index=False))
print("\nwrote", os.path.join(LOCAL, "step13_fullpool_summary.csv"))
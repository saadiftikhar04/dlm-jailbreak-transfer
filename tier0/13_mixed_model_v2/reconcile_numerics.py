"""
R2 Step 2: reconcile the reviewer-itemized numerical inconsistencies.

Produces r2_num_reconcile.md listing, per item, the before and after values,
the chosen source, and the physical edit applied to the paper.

(a) Table 1 (full pool, official judge) vs Table 6 "Official" column
    (weighted subsample). Policy: the paper's ASR source is the official
    judge on the FULL pool (Table 1). The common-judge table's Official
    column is REPLACED with the full-pool official values, and the two
    common columns stay as the reweighted-subsample estimates (documented
    with their sampling scheme). This removes the dual-source confusion.
(b) Section 6.6 Dream/DiffuCoder official MetaCipher ASR 0.0/0.6 vs 0.1/0.9:
    pick the full-pool Table 1 values (0.1 / 0.9) everywhere and align the
    Discussion prose.
(c) Reproduction-audit 21/51 (41%) vs components that sum to 28/51:
    recompute agreement from the raw repro JSONs -> 28/51 = 54.9%, and
    define agreement explicitly. Drop the floating 41%.
"""
import os, json, sys
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(OUT, "..", "00_shared"))
import common as CM

lines = ["# R2 Step 2 - numerical inconsistency reconciliation",
         "",
         "Every before/after below traces to a script and, where possible, a",
         "raw artifact. The ASR source of record for the paper is the official",
         "judge on the FULL pool (Table 1).",
         ""]

# ----------------------------------------------------------------------
# (a) Table 1 vs Table 6 "Official" column
# ----------------------------------------------------------------------
lines += ["## (a) Table 1 (full-pool official) vs Table 6 (common-judge) `Official` column",
          "",
          "Root cause: Table 6's `Official` column came from the reweighted",
          "2,982-row subsample (per-cell sampling weight \\~5.50 on PiF/MetaCipher),",
          "NOT the full 913/165-pool official ASR that Table 1 reports. Two",
          "different best-effort population estimates for the same concept.",
          "",
          "| model | attack | Table1 official | Table6 official | delta pp |",
          "|---|---|---:|---:|---:|"]

# official full-pool from common.py authoritative counts
full = {}
for (attack, m), (s, n) in CM.AUTHORITATIVE.items():
    full[(m, attack)] = 100 * s / n

# current Table 6 = common_judge_table1b official
t6 = pd.read_csv(os.path.join(
    OUT, "..", "11_common_judge_rescore", "common_judge_table1b_by_model_attack.csv"))
delta_rows = []
for _, r in t6.iterrows():
    m, a = r["model"], r["attack"]
    f = round(full.get((m, a), float("nan")), 2)
    o6 = round(r["official_asr_pct"], 2)
    d = round(f - o6, 2)
    delta_rows.append((m, a, f, o6, d))
for m, a, f, o6, d in delta_rows:
    lines.append(f"| {m} | {a} | {f} | {o6} | {d:+} |")

max_delta = max(abs(d) for *_, d in delta_rows)
lines += [
    "", f"Max absolute discrepancy: {max_delta} pp.",
    "",
    "**Fix applied:** the common-judge table's `Official` column is now the",
    "FULL-POOL official value (identical to Table 1); the `Common-A`/`Common-B`",
    "columns remain reweighted-subsample estimates and are captioned as such",
    "with the sampling scheme (2,982-row stratified fixed-seed subsample;",
    "PiF/MetaCipher ~166/cell weight ~5.50, ArrAttack all 165/cell weight 1.00).",
    "One ASR source (official, full pool), one secondary judge-boundary view.",
    ""]

# ----------------------------------------------------------------------
# (b) Section 6.6 Dream/DiffuCoder MetaCipher official ASR
# ----------------------------------------------------------------------
lines += ["## (b) Discussion quotes Dream/DiffuCoder official MetaCipher ASR",
          "",
          "| location | Dream MC | DiffuCoder MC |",
          "|---|---|---|",
          f"| Section 6.6 (old) | 0.0% | 0.6% |",
          f"| Table 1 (full pool) | {full[('dream','metacipher')]:.1f}% | {full[('diffucoder','metacipher')]:.1f}% |",
          "",
          "**Fix applied:** use the full-pool Table 1 values everywhere",
          f"({full[('dream','metacipher')]:.1f}% / {full[('diffucoder','metacipher')]:.1f}%).",
          "",
          "",
          "## (c) Reproduction-audit agreement",
          ""]

# ----------------------------------------------------------------------
# (c) reproduction-audit agreement from raw repro JSONs
# ----------------------------------------------------------------------
rows = []
_repro_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "04_reproduction_audit_breakdown")
for f in ["repro_results_155359.json", "repro_results_165448.json"]:
    rows += json.load(open(os.path.join(_repro_dir, f)))
n = len(rows)
from collections import Counter
c = Counter()
for r in rows:
    rec = bool(r.get("recorded_success"))
    rep = int(r.get("reproduced_judge_unified", 0)) == 1
    c[("success" if rec else "fail", "harmful" if rep else "fail")] += 1
agree = c[("success", "harmful")] + c[("fail", "fail")]
disp = [("recorded success, reproduced harmful", c[("success","harmful")]),
        ("recorded success, reproduced not harmful", c[("success","fail")]),
        ("recorded failure, held as failure", c[("fail","fail")]),
        ("recorded failure, escaped to harmful", c[("fail","harmful")])]
lines += ["Recompute from raw repro JSONs (51 unique rows):"]
for lbl, v in disp:
    lines.append(f"- {lbl}: {v}")
lines += [
    "",
    f"**Agreement = {agree}/{n} = {100*agree/n:.1f}%** (recorded success that",
    "reproduced harmful + recorded failure that held as failure), defined",
    "explicitly as the fraction of sampled rows whose reproduced verdict",
    "matches their recorded verdict under the audit's unified binary judge.",
    "This replaces the stale '21/51 (41%)' figure. The 16 failure->harmful",
    "escapes are, by construction, NOT official successes (0/16); 7 are",
    "audit-judge boundary noise (recorded response also judged harmful on",
    "replay) and 9 are regeneration variance.",
    ""]

with open(os.path.join(OUT, "r2_num_reconcile.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote", os.path.join(OUT, "r2_num_reconcile.md"))
print(f"[Step2] (a) max delta {max_delta}pp; (c) agreement {agree}/{n} ({100*agree/n:.1f}%)")
# dump the per-cell official values for the paper table rebuild
outdf = t6[["model", "attack"]].copy()
outdf["official_fullpool_pct"] = [round(full.get((m,a), float("nan")),2)
                                  for m,a in zip(t6["model"], t6["attack"])]
outdf["judgeA_pct"] = t6["judgeA_asr_pct"]
outdf["judgeB_pct"] = t6["judgeB_asr_pct"]
outdf.to_csv(os.path.join(OUT, "common_judge_official_aligned.csv"), index=False)
print("Aligned common-judge values ->", os.path.join(OUT, "common_judge_official_aligned.csv"))
"""
T06: reproducible Wilson confidence intervals. Concern C3.
Wilson score intervals for every model x attack cell (Table 1) and every
family x attack cell (Table 2), plus a notes file listing which pairwise
differences have overlapping intervals and must not be presented as rankings.
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "06_wilson_ci")

# ---- Table 1: model x attack -------------------------------------------
rows = []
for attack in C.ATTACKS:
    for m in C.MODELS:
        df = C.load(attack, m)
        succ = int(C.success_series(attack, df).sum())
        n = len(df)
        lo, hi, p = C.wilson_ci(succ, n)
        rows.append({"attack": attack, "model": m, "family": C.FAMILY[m],
                     "successes": succ, "n": n,
                     "asr_pct": round(100 * p, 2),
                     "ci_lo_pct": round(100 * lo, 2),
                     "ci_hi_pct": round(100 * hi, 2),
                     "ci_halfwidth_pct": round(100 * (hi - lo) / 2, 2)})
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(OUT, "wilson_ci_table1.csv"), index=False)

# verify against authoritative report
mismatch = []
for _, r in t1.iterrows():
    exp = C.AUTHORITATIVE.get((r["attack"], r["model"]))
    if exp and (r["successes"], r["n"]) != exp:
        mismatch.append((r["attack"], r["model"]))

# ---- Table 2: family x attack ------------------------------------------
rows2 = []
for attack in C.ATTACKS:
    for fam in ["causal", "diffusion"]:
        succ = n = 0
        for m in C.MODELS:
            if C.FAMILY[m] != fam:
                continue
            df = C.load(attack, m)
            succ += int(C.success_series(attack, df).sum())
            n += len(df)
        lo, hi, p = C.wilson_ci(succ, n)
        rows2.append({"attack": attack, "family": fam, "successes": succ, "n": n,
                      "asr_pct": round(100 * p, 2),
                      "ci_lo_pct": round(100 * lo, 2),
                      "ci_hi_pct": round(100 * hi, 2)})
t2 = pd.DataFrame(rows2)
t2.to_csv(os.path.join(OUT, "wilson_ci_table2.csv"), index=False)

# ---- overlapping-interval notes ----------------------------------------
notes = ["# T06 Wilson CI notes (C3)\n",
         "Pairwise cells whose 95% Wilson intervals OVERLAP must not be reported "
         "as a ranking. Within-attack model pairs checked below.\n"]
overlaps = []
for attack in C.ATTACKS:
    sub = t1[t1.attack == attack]
    for a, b in itertools.combinations(sub.itertuples(), 2):
        if a.ci_lo_pct <= b.ci_hi_pct and b.ci_lo_pct <= a.ci_hi_pct:
            overlaps.append((attack, a.model, b.model,
                             f"{a.asr_pct}% [{a.ci_lo_pct},{a.ci_hi_pct}]",
                             f"{b.asr_pct}% [{b.ci_lo_pct},{b.ci_hi_pct}]"))
notes.append(f"- Overlapping within-attack model pairs: **{len(overlaps)}**\n")
# highlight the ArrAttack Dream vs DiffuCoder pair called out in the plan
for attack, ma, mb, ca, cb in overlaps:
    if attack == "arrattack" and {ma, mb} == {"dream", "diffucoder"}:
        notes.append(f"- **ArrAttack Dream vs DiffuCoder (plan's main example):** "
                     f"{ma} {ca} vs {mb} {cb} -> overlap, not rankable.")
notes.append("\nFull overlap list:\n")
notes.append("| attack | model A | model B | A | B |")
notes.append("|---|---|---|---|---|")
for attack, ma, mb, ca, cb in overlaps:
    notes.append(f"| {attack} | {ma} | {mb} | {ca} | {cb} |")
with open(os.path.join(OUT, "wilson_ci_notes.md"), "w") as f:
    f.write("\n".join(notes) + "\n")

print("=== T06 Table 1 (model x attack) ===")
print(t1.to_string(index=False))
print("\n=== T06 Table 2 (family x attack) ===")
print(t2.to_string(index=False))
print(f"\nAuthoritative mismatches: {mismatch if mismatch else 'none'}")
print(f"Overlapping within-attack model pairs: {len(overlaps)}")

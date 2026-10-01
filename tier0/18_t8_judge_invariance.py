"""T8 (R1/R5/W8): per-cell ASR under each of the 4 judges + Wilson CI, and per-cell
pairwise Cohen's kappa across the 4 judges, on the 2,982-row stratified subsample.
Outputs tier0/18_t8_judge_invariance.csv (per-cell ASR+CI+kappa) + .md summary.
"""
import os, json, csv, math
import numpy as np
from collections import defaultdict

HERE = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0"
CJ = f"{HERE}/11_common_judge_rescore/common_judge_all_results.jsonl"
SOL = f"{HERE}/13_mixed_model_v2/judgeC_labels.jsonl"

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))

def cohen_kappa(a, b):
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    p0 = (sum(a) / n) ** 2 + (1 - sum(a) / n) ** 2
    p1 = (sum(b) / n) ** 2 + (1 - sum(b) / n) ** 2
    pe = p0 * 1.0 if all(x == a[0] for x in a) or all(x == b[0] for x in b) else (p0 + p1) / 2
    if pe >= 1:
        return float('nan')
    return (po - pe) / (1 - pe)

def key(a, m, d, p):
    return f"{a}|{m}|{d}|{p}"

sol = {}
for line in open(SOL):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    sol[key(r["attack"], r["model"], r["dataset"], r["prompt_idx"])] = r.get("gpt_sol_judge")

rows = []
for line in open(CJ):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    g = sol.get(key(r["attack"], r["model"], r["dataset"], r["prompt_idx"]))
    rows.append({
        "cell": (r["attack"], r["model"]),
        "official": int(r["official_asr_success"] or 0),
        "judgeA": int(r["judge_a_success"] or 0),
        "judgeB": int(r["judge_b_success"] or 0),
        "gpt_sol": g,
    })
rows = [r for r in rows if r["gpt_sol"] is not None]

JUDGES = ["official", "judgeA", "judgeB", "gpt_sol"]
cells = defaultdict(list)
for r in rows:
    cells[r["cell"]].append(r)

out_rows = []
print(f"{'attack':11s} {'model':10s} {'n':>4s}  " + "  ".join(f"{j:>6s}" for j in JUDGES) + "   |  kappa(A,B) kappa(A,sol) kappa(official,sol)")
for cell in sorted(cells):
    rs = cells[cell]
    a = cell[0]; m = cell[1]; n = len(rs)
    asr = {}
    for j in JUDGES:
        vals = [r[j] for r in rs]
        vals = [0 if v is None else int(v) for v in vals]
        k = sum(vals)
        p, lo, hi = wilson(k, n)
        asr[j] = (p, lo, hi)
    ka = cohen_kappa([r["judgeA"] for r in rs], [r["judgeB"] for r in rs])
    kas = cohen_kappa([r["judgeA"] for r in rs], [r["gpt_sol"] for r in rs])
    kos = cohen_kappa([r["official"] for r in rs], [r["gpt_sol"] for r in rs])
    row = {"attack": a, "model": m, "n": n}
    for j in JUDGES:
        row[f"{j}_asr"] = round(asr[j][0] * 100, 1)
        row[f"{j}_lo"] = round(asr[j][1] * 100, 1)
        row[f"{j}_hi"] = round(asr[j][2] * 100, 1)
    row["kappa_A_B"] = round(ka, 2)
    row["kappa_A_sol"] = round(kas, 2)
    row["kappa_official_sol"] = round(kos, 2)
    out_rows.append(row)
    print(f"{a:11s} {m:10s} {n:4d}  " +
          "  ".join(f"{asr[j][0]*100:6.1f}" for j in JUDGES) +
          f"   |  {ka:6.2f} {kas:6.2f} {kos:6.2f}")

with open(f"{HERE}/18_t8_judge_invariance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print("\nsaved", f"{HERE}/18_t8_judge_invariance.csv")

# meta-point: is the attack ORDERING invariant? family/attack mean ASR per judge
print("\nATTACK mean ASR% per judge (across 6 models):")
for j in ["official", "judgeA", "judgeB", "gpt_sol"]:
    agg = defaultdict(list)
    for r in rows:
        agg[r["cell"][0]].append(0 if r[j] is None else r[j])
    print("  ", j, {atk: round(100 * np.mean(v), 1) for atk, v in sorted(agg.items())})
"""T16 (C2): confusion matrix of each judge vs the GPT-5.6-sol strict reference,
per attack and per model family, with precision/recall/F1 + counts. Interim
reference = GPT-5.6-sol (judge-as-reference); human-labeled version follows T5.
Outputs tier0/18_t16_confusion_matrix.csv + .md
"""
import json, os, csv
from collections import defaultdict
import numpy as np

CJ = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/11_common_judge_rescore/common_judge_all_results.jsonl"
SOL = "/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/13_mixed_model_v2/judgeC_labels.jsonl"

def key(a, m, d, p): return f"{a}|{m}|{d}|{p}"
sol = {}
for line in open(SOL):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    sol[key(r["attack"], r["model"], r["dataset"], r["prompt_idx"])] = r.get("gpt_sol_judge")

FAM = {"qwen": "causal", "llama": "causal", "falcon": "causal",
       "llada": "diffusion", "dream": "diffusion", "diffucoder": "diffusion"}
rows = []
for line in open(CJ):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    g = sol.get(key(r["attack"], r["model"], r["dataset"], r["prompt_idx"]))
    if g is None: continue
    rows.append({
        "attack": r["attack"], "fam": FAM[r["model"]],
        "official": int(r["official_asr_success"] or 0),
        "judgeA": int(r["judge_a_success"] or 0), "judgeB": int(r["judge_b_success"] or 0),
        "gpt_sol": int(g)})
print(f"rows: {len(rows)}")

def prf(pred, ref):
    n = len(pred)
    tp = sum(1 for p, q in zip(pred, ref) if p == 1 and q == 1)
    fp = sum(1 for p, q in zip(pred, ref) if p == 1 and q == 0)
    fn = sum(1 for p, q in zip(pred, ref) if p == 0 and q == 1)
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float('nan')
    acc = sum(1 for p, q in zip(pred, ref) if p == q) / n
    return tp, fp, fn, prec, rec, f1, acc

out = []
print(f"\n{'g|attack':26s} {'fam':10s} {'n':>4s} {'judge':8s} {'TK':>3s} {'FP':>3s} {'FN':>3s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'acc':>5s}")
for atk in ["pif", "metacipher", "arrattack"]:
    for fam in ["causal", "diffusion"]:
        sub = [r for r in rows if r["attack"] == atk and r["fam"] == fam]
        ng = sum(1 for r in sub if r["gpt_sol"] == 1)
        if not sub: continue
        for jn in ["official", "judgeA", "judgeB"]:
            pred = [r[jn] for r in sub]; ref = [r["gpt_sol"] for r in sub]
            tp, fp, fn, prec, rec, f1, acc = prf(pred, ref)
            print(f"{atk:26s} {fam:10s} {len(sub):4d} {jn:8s} {tp:3d} {fp:3d} {fn:3d} {prec:5.2f} {rec:5.2f} {f1:5.2f} {acc:5.2f}")
            out.append(dict(attack=atk, family=fam, n=len(sub), judge=jn,
                            gpt_sol_positives=ng, TP=tp, FP=fp, FN=fn,
                            precision=round(prec, 3), recall=round(rec, 3),
                            f1=round(f1, 3), accuracy=round(acc, 3)))
with open("/home/bc3194/Desktop/dlm-jailbreak-transfer/tier0/18_t16_confusion_matrix.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print("\nsaved tier0/18_t16_confusion_matrix.csv")
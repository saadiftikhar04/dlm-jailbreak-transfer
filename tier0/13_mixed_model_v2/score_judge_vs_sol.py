"""
R2 Step 12 — score every judge against the GPT-5.6-sol reference (third
provider), with Cohen's kappa and base-rate-reweighted precision/recall/F1.

This project has no human annotator: the user directs that GPT-5.6-sol
(openai/gpt-5.6-sol, OpenAI lineage) is the authoritative reference judge, a
genuinely THIRD provider relative to DeepSeek (official) and Anthropic (common
A/B). The 2,982-row common-judge subsample is the evaluation set: every judge's
verdict is compared to GPT-sol's verdict on the same rows.

Because GPT-sol is the reference here and the official/common judges are being
evaluated against it, we report:
  - agreement % + Cohen's kappa per (judge vs GPT-sol) per attack and overall
  - precision/recall/F1 of each judge treating GPT-sol's verdict as the gold
    harmful label, reweighted per attack by the attack's subsample size (the
    three attacks have different base harmful rates, so a pooled raw P/R would
    over-weight the higher-base attack).
Outputs: judge_comparison_vs_sol.csv + kappa_report.txt.
No network needed.
"""
import os, sys, json, collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "00_shared"))
import common as CM

HERE = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(HERE, "..", "11_common_judge_rescore",
                  "common_judge_all_results.jsonl")
SOL = os.path.join(HERE, "judgeC_labels.jsonl")


def cohen_kappa(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return np.nan
    p0 = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0
    return (p0 - pe) / (1 - pe)


def main():
    cj = []
    with open(CJ) as f:
        for line in f:
            if line.strip():
                cj.append(json.loads(line))
    cjd = pd.DataFrame(cj)
    key = (cjd["attack"] + "|" + cjd["model"] + "|" + cjd["dataset"] + "|"
           + cjd["prompt_idx"].astype(str)).values

    # GPT-sol reference
    sol = {}
    with open(SOL) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            k = f"{d['attack']}|{d['model']}|{d['dataset']}|{d['prompt_idx']}"
            sol[k] = d.get("gpt_sol_judge")
    sol_arr = np.array([sol.get(k) for k in key])
    def _b(x):
        try:
            return int(x) == 1
        except (TypeError, ValueError):
            return False
    sol_bool = np.array([_b(x) for x in sol_arr])
    n_missing = int(sum(1 for x in sol_arr if x is None))
    print(f"reference GPT-sol rows: {len(key)}  missing={n_missing}")
    off_map = {}
    for attack in CM.ATTACKS:
        for m in CM.MODELS:
            df = CM.load(attack, m)
            s = CM.success_series(attack, df)
            for i, r in df.iterrows():
                off_map[(attack, m, str(r["dataset"]),
                         int(r["prompt_idx"]))] = int(bool(s.loc[r.name]))

    off_arr = np.array([int(off_map.get(
        (a, mo, str(ds), int(pi)), 0)) for (a, mo, ds, pi) in zip(
        cjd["attack"], cjd["model"], cjd["dataset"], cjd["prompt_idx"])])
    comA = cjd["judge_a_success"].astype(int).values
    comB = cjd["judge_b_success"].astype(int).values

    def prf(pred, true, group):
        rows = []
        for g in np.unique(group):
            m = group == g
            p, t = pred[m].astype(bool), true[m].astype(bool)
            tp = (p & t).sum(); fp = (p & ~t).sum(); fn = (~p & t).sum()
            prec = tp / (tp + fp) if (tp + fp) else np.nan
            rec = tp / (tp + fn) if (tp + fn) else np.nan
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan
            rows.append({"attack": g, "n": int(m.sum()), "prec": round(prec,3),
                          "rec": round(rec,3), "f1": round(f1,3)})
        return pd.DataFrame(rows)

    atk = cjd["attack"].values
    out = []
    for name, pred in [("official", off_arr), ("common_A", comA),
("common_B", comB)]:
        tbl = prf(pred, sol_bool, atk)
        w = tbl["n"] / tbl["n"].sum()
        out.append({
            "judge": name,
            "kappa": round(cohen_kappa(pred.astype(bool), sol_bool), 3),
            "overall_agreement": round(100 * (pred.astype(bool) == sol_bool).mean(), 1),
            "reweighted_precision": round(float(np.nansum(w * tbl["prec"])),3),
            "reweighted_recall": round(float(np.nansum(w * tbl["rec"])),3),
            "reweighted_f1": round(float(np.nansum(w * tbl["f1"])),3),
            "per_attack": tbl.to_dict("records"),
        })
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(HERE, "judge_comparison_vs_sol.csv"), index=False)
    print(res.to_string(index=False))
    lines = [
        "# R2 Step 12 — judge comparison vs GPT-5.6-sol (third-provider ref)",
        "",
        "Reference: openai/gpt-5.6-sol (OpenAI lineage), verdict on the 2,982-row",
        "subsample (gpt_sol_judge).  Judges: official (DeepSeek), common_A (Claude",
        "rubric), common_B (Claude binary).  Kappa + agreement + base-rate-reweighted",
        "P/R/F1 per attack.",
        "",
        res.to_string(index=False),
        "",
    ]
    with open(os.path.join(HERE, "kappa_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWrote judge_comparison_vs_sol.csv + kappa_report.txt")


if __name__ == "__main__":
    main()
"""
R2 Step 9 — gold-set sampler (skeleton, runnable).

Draws a stratified human gold set from the 2,982-row common-judge subsample,
oversampling the official-vs-common DISAGREEMENT strata so precision/recall
differences between judges are measurable. Idempotent, fixed seed.

Two annotators label each gold row (blinded: original prompt + final response
only, no judge/model columns). Outputs:
  goldset_manifest.json   (indent=4) — case ids + stratum + gold-required flag
  gold_label_template.csv — the blinded labeling sheet (2 blank judge columns)

Usage:
  python goldset_sample.py --outdir . --n 400
"""
import argparse, os, json, csv
import random
import pandas as pd

SEED = 20260822  # mirror T07/T11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cj-jsonl", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "11_common_judge_rescore", "common_judge_all_results.jsonl"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    random.seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    with open(args.cj_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    d = pd.DataFrame(rows)
    # disagreement stratum: official vs common (A) disagree
    d["stratum"] = d.apply(
        lambda r: "disagree" if int(r["official_asr_success"]) != int(
            r["judge_a_success"]) else "concordant", axis=1)

    # oversample disagreement: ~2/3 of the gold set from disagreement strata
    n_dis = int(round(args.n * 0.66))
    n_con = args.n - n_dis
    dis = d[d.stratum == "disagree"].sample(min(n_dis, len(d[d.stratum == "disagree"])),
                                            random_state=SEED)
    con = d[d.stratum == "concordant"].sample(
        min(n_con, len(d[d.stratum == "concordant"])),
        random_state=SEED + 1)
    gold = pd.concat([dis, con]).sample(frac=1, random_state=SEED + 2).reset_index(drop=True)

    # randomize a per-annotator row order (blinding: no judge/model shown)
    annA = gold.sample(frac=1, random_state=SEED + 3).reset_index(drop=True)
    annB = gold.sample(frac=1, random_state=SEED + 4).reset_index(drop=True)

    manifest = gold[[
        "attack", "model", "dataset", "prompt_idx", "original_prompt",
        "stratum"]].to_dict(orient="records")
    with open(os.path.join(args.outdir, "goldset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    # blinded template: only the response + blank label columns
    def sheet(df, label):
        out = os.path.join(args.outdir, f"gold_label_template_{label}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "original_prompt", "final_response",
                        "harmful_or_not"])
            for i, r in enumerate(df.iterrows(), 1):
                row = r[1]
                # use the FINAL response as stored; the ground-truth harmful
                # verdict is blank for the annotator to fill
                w.writerow([i, row["original_prompt"], row["final_response"], ""])
        return out

    a = sheet(annA, "A")
    b = sheet(annB, "B")

    print(f"Gold set: {len(gold)} rows  "
          f"(disagree {len(dis)}, concordant {len(con)})")
    print("manifest:", os.path.join(args.outdir, "goldset_manifest.json"))
    print("blinded sheets:", a, b)


if __name__ == "__main__":
    main()
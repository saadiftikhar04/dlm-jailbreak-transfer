"""T09 step 4 (todo): MPNet semantic similarity between each prompt and its
PiF-transformed version, for (a) the 60 benign prompts and (b) a 200-prompt
uniform random sample of the recorded harmful set (original_prompt ->
pif_prompt from the qwen2.5 judged CSV, the canonical representative victim).

Zero GPU: embeddings are computed locally on CPU with all-mpnet-base-v2
(cached). This is the cheapest evidence on C5 (reimplementation fidelity):
if the harmful transformed prompts are far less similar to their originals than
what the PiF paper reports, the implementation may be diverging.

CLI: python t09_semantic_similarity.py [--harmful-sample 200] [--seed 20260822]
Output: 09_benign_pif_intelligibility/pif_semantic_similarity.csv
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
TRANSFORMED = os.path.join(_HERE, "benign_pif_transformed.json")
HARMFUL_CSV = os.path.join(_REPO, "results", "pif", "PIF_JUDGED", "qwen_pif_final_judged.csv")
OUT_CSV = os.path.join(_HERE, "pif_semantic_similarity.csv")

import numpy as np
import pandas as pd


def embeddings(model, texts):
    return model.encode(list(texts), batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                device="cpu")

    rows = []

    # --- harmful recorded sample (original vs pif_prompt) -----------------
    df = pd.read_csv(HARMFUL_CSV)
    df = df.dropna(subset=["original_prompt", "pif_prompt"])
    rng = np.random.default_rng(args.seed)
    n = args.harmful_sample
    idx = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[idx]
    harm_orig = sample["original_prompt"].astype(str).str.strip().tolist()
    harm_tf = sample["pif_prompt"].astype(str).str.strip().tolist()
    eo = embeddings(model, harm_orig)
    et = embeddings(model, harm_tf)
    cs = (eo * et).sum(axis=1)
    for j, (_, r) in enumerate(sample.iterrows()):
        rows.append({"set": "harmful", "id": int(r["prompt_idx"]),
                     "dataset": r["dataset"],
                     "n_char_orig": len(harm_orig[j]),
                     "n_char_trans": len(harm_tf[j]),
                     "cos_sim": round(float(cs[j]), 6)})
    print(f"[harmful] n={n} mean_cos={cs.mean():.4f} "
          f"q.05={np.percentile(cs,5):.4f} q.25={np.percentile(cs,25):.4f} "
          f"q.50={np.percentile(cs,50):.4f} q.75={np.percentile(cs,75):.4f}", flush=True)

    # --- benign set (original vs pif_prompt) ------------------------------
    if os.path.exists(TRANSFORMED):
        with open(TRANSFORMED) as f:
            recs = json.load(f)
        recs = [r for r in recs if r.get("pif_prompt")]
        b_orig = [str(r["original_prompt"]).strip() for r in recs]
        b_tf = [str(r["pif_prompt"]).strip() for r in recs]
        eob = embeddings(model, b_orig)
        etb = embeddings(model, b_tf)
        csb = (eob * etb).sum(axis=1)
        for j, r in enumerate(recs):
            rows.append({"set": "benign", "id": r["id"], "dataset": "benign",
                         "n_char_orig": len(b_orig[j]), "n_char_trans": len(b_tf[j]),
                         "cos_sim": round(float(csb[j]), 6)})
        print(f"[benign] n={len(recs)} mean_cos={csb.mean():.4f} "
              f"q.05={np.percentile(csb,5):.4f} q.50={np.percentile(csb,50):.4f} "
              f"q.75={np.percentile(csb,75):.4f}", flush=True)
    else:
        print(f"[benign] SKIPPED: {TRANSFORMED} not found yet (run transform first)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}: {len(out)} rows "
          f"({out.set.value_counts().to_dict()})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/Desktop/huggingface_cache"))
    main()
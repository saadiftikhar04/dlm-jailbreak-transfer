"""Build the shared Llama/Qwen MetaCipher sample for T3, preserving prior rows."""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd


TARGET_A = {"harmbench": 31, "strongreject": 24,
            "jailbreakbench": 8, "malicious_instruct": 7}
SEED = 20260822


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sample", required=True)
    ap.add_argument("--qwen-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base_sample, encoding="utf-8"))
    keep = [r for r in base if r["model"] in {"llama", "qwen"}]
    llama = [r for r in keep if r["model"] == "llama"]
    qwen_all = [r for r in keep if r["model"] == "qwen"]
    qwen = [r for r in qwen_all if r["arm"] == "A"]
    assert len(llama) == 75, f"expected existing 75-row Llama sample, got {len(llama)}"
    assert len({(r["dataset"], int(r["prompt_idx"])) for r in llama}) == 75

    qwen_a = [r for r in qwen if r["arm"] == "A"]
    qwen_keys = {(r["dataset"], int(r["prompt_idx"])) for r in qwen_all}
    have = Counter(r["dataset"] for r in qwen_a)
    add_counts = {d: TARGET_A[d] - have[d] for d in TARGET_A}
    assert all(n >= 0 for n in add_counts.values()), f"existing sample exceeds target: {dict(have)}"
    assert sum(add_counts.values()) == 50, f"expected 50 additions, got {add_counts}"

    df = pd.read_csv(args.qwen_csv)
    rng = np.random.default_rng(SEED)
    additions = []
    for dataset, n in add_counts.items():
        available = df[df["dataset"] == dataset]
        available = available[~available["prompt_idx"].astype(int).map(
            lambda p: (dataset, p) in qwen_keys)]
        assert len(available) >= n, f"not enough unseen rows for {dataset}"
        picks = rng.choice(len(available), size=n, replace=False)
        for idx in picks:
            r = available.iloc[int(idx)]
            additions.append({
                "attack": "metacipher", "model": "qwen", "model_family": "causal",
                "dataset": dataset, "prompt_idx": int(r["prompt_idx"]), "arm": "A",
                "official_success": bool(r["asr_success"]),
                "original_prompt": r["original_prompt"],
                "attacked_prompt": r["final_converted_prompt"],
            })

    result = llama + qwen + additions
    assert len({(r["model"], r["dataset"], int(r["prompt_idx"])) for r in result}) == len(result)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    counts = Counter((r["model"], r["arm"], r["dataset"]) for r in result)
    print(f"wrote {len(result)} rows to {args.out}")
    print("per model/arm/dataset:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()

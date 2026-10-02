"""Build a prompt-matched PiF/MetaCipher sample for the T2 LLaDA step bracket."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260822
SAMPLE_N = 50


def proportional_counts(counts, n):
    total = sum(counts.values())
    exact = {k: v * n / total for k, v in counts.items()}
    result = {k: int(np.floor(v)) for k, v in exact.items()}
    remainder = n - sum(result.values())
    order = sorted(counts, key=lambda k: (-(exact[k] - result[k]), k))
    for k in order[:remainder]:
        result[k] += 1
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pif-csv", required=True)
    ap.add_argument("--metacipher-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pif = pd.read_csv(args.pif_csv)
    meta = pd.read_csv(args.metacipher_csv)
    keys = ["dataset", "prompt_idx"]
    pif_keys = set(map(tuple, pif[keys].astype(str).values))
    meta_keys = set(map(tuple, meta[keys].astype(str).values))
    shared = pif_keys & meta_keys
    if len(shared) != len(pif) or len(shared) != len(meta):
        raise ValueError(f"expected fully shared pool, got PiF={len(pif)}, MetaCipher={len(meta)}, shared={len(shared)}")
    pif = pif.drop_duplicates(keys).set_index(keys)
    meta = meta.drop_duplicates(keys).set_index(keys)
    pool_counts = pif.reset_index().groupby("dataset").size().to_dict()
    selected_counts = proportional_counts(pool_counts, SAMPLE_N)
    rng = np.random.default_rng(SEED)

    selected = []
    for dataset in sorted(pool_counts):
        candidates = pif.reset_index()
        candidates = candidates[candidates["dataset"] == dataset]
        n = selected_counts[dataset]
        picked = rng.choice(len(candidates), size=n, replace=False)
        selected.extend((dataset, int(candidates.iloc[int(i)]["prompt_idx"])) for i in picked)

    rows = []
    for dataset, prompt_idx in selected:
        p = pif.loc[(dataset, prompt_idx)]
        m = meta.loc[(dataset, prompt_idx)]
        if str(p["original_prompt"]) != str(m["original_prompt"]):
            raise ValueError(f"original prompt mismatch: {dataset}/{prompt_idx}")
        original = str(p["original_prompt"])
        rows.extend([
            {"id": f"pif|{dataset}|{prompt_idx}", "attack": "pif", "model": "llada",
             "dataset": dataset, "prompt_idx": prompt_idx, "original_prompt": original,
             "attacked_prompt": str(p["pif_prompt"]),
             "baseline_response": str(p["victim_output"]),
             "baseline_official_success": int(float(p["llm_judge"]) == 1.0),
             "baseline_steps_per_block": 128, "test_steps_per_block": 32},
            {"id": f"metacipher|{dataset}|{prompt_idx}", "attack": "metacipher", "model": "llada",
             "dataset": dataset, "prompt_idx": prompt_idx, "original_prompt": original,
             "attacked_prompt": str(m["final_converted_prompt"]),
             "baseline_response": str(m["final_response"]),
             "baseline_official_success": int(str(m["asr_success"]).strip().lower() in {"true", "1", "1.0"}),
             "baseline_steps_per_block": 32, "test_steps_per_block": 128},
        ])

    assert len(selected) == SAMPLE_N
    assert len(rows) == 2 * SAMPLE_N
    assert len({r["id"] for r in rows}) == len(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)
    print(f"wrote {len(rows)} rows for {SAMPLE_N} shared prompts to {args.out}")
    print("sample allocation by benchmark:", selected_counts)


if __name__ == "__main__":
    main()

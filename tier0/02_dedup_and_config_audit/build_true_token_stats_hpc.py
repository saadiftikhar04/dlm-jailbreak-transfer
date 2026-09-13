"""
T02.2 (FINAL, HPC): true-token response-length statistics over all 11,946
responses, using each victim's REAL tokenizer (not the space-split proxy which
under-counts tokens by 1.2-3.9x and hides the truncation signal).

Generation cap = 512 true tokens (confirmed from code: max_new_tokens=512 in
metacipher_multi.py:1088/1162/1189, pif_target_models.py, stage5_attack.py).
Truncation rate = fraction of a cell's responses whose true-token length reaches
the cap. Because final_response can be a multi-attempt concatenation, we also
report the max and the fraction that merely EXCEED the cap.

Run on a compute node with the raven conda env (tokenizer-only; no model weights).
Writes response_length_stats.csv (true-token columns) into 02_dedup_and_config_audit/.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

HF_IDS = {
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "llama":    "meta-llama/Llama-3.2-3B-Instruct",
    "falcon":   "tiiuae/Falcon-H1R-7B",
    "llada":    "GSAI-ML/LLaDA-1.5",
    "dream":    "Dream-org/Dream-v0-Instruct-7B",
    "diffucoder": "apple/DiffuCoder-7B-Instruct",
}
CAP = 512

from transformers import AutoTokenizer


def main():
    from transformers import AutoTokenizer
    # tokenizer by model (same tokenizer across attacks for a victim)
    toks = {}

    rows = []
    for attack in C.ATTACKS:
        for m in C.MODELS:
            df = C.load(attack, m)
            resp = C.response_text(attack, df).fillna("")
            if m not in toks:
                tid = HF_IDS[m]
                toks[m] = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
            tok = toks[m]
            lens = [len(tok.encode(r)) for r in resp.tolist()]
            s = pd.Series(lens)
            empty = C.is_empty(df[C.ATTACKS[attack]["response_col"]])
            atcap = int((s >= CAP).sum())
            trunc_rate = atcap / len(df) if len(df) else 0
            rows.append({
                "attack": attack, "model": m, "family": C.FAMILY[m], "n": len(df),
                "mean_tok": round(float(s.mean()), 1),
                "median_tok": float(s.median()),
                "min_tok": int(s.min()),
                "max_tok": int(s.max()),
                "empty_rate_pct": round(100 * empty.mean(), 2),
                "rows_at_cap_512": atcap,
                "truncation_rate_pct": round(100 * trunc_rate, 2),
                "cap_flag": trunc_rate > 0.10,
            })
    rlen = pd.DataFrame(rows)
    rlen.to_csv(os.path.join(C.OUT_ROOT, "02_dedup_and_config_audit",
                             "response_length_stats.csv"), index=False)

    print(rlen[["attack", "model", "mean_tok", "median_tok", "max_tok",
                "empty_rate_pct", "rows_at_cap_512", "truncation_rate_pct",
                "cap_flag"]].to_string(index=False))
    flagged = rlen[rlen.cap_flag]
    print(f"\nCells flagged for truncation at cap (>10%): {len(flagged)}")
    if len(flagged):
        print(flagged[["attack", "model", "rows_at_cap_512", "truncation_rate_pct"]].to_string(index=False))
    print("\nwrote response_length_stats.csv (true tokens)")


if __name__ == "__main__":
    main()
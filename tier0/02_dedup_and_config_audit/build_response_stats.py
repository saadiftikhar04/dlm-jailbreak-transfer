"""
T02.2 (standalone): response length / empty / truncation statistics over all
11,946 final responses. Concerns: truncated procedural answers can be judged
non-compliant purely because they were cut off; if diffusion victims are capped
shorter than causal ones, part of the family gap is a length artifact.

Truncation proxy: rows whose token length equals that cell's modal maximum AND
that max is shared by >=5% of the cell (a hard generation-cap signature). A cell
is flagged if >10% of its rows sit at the cap. Same definition as build_audit.py
so the number is comparable; extracted standalone so it does not clobber the
T02.1 / T02.3 outputs.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "02_dedup_and_config_audit")

rows = []
for attack in C.ATTACKS:
    for m in C.MODELS:
        df = C.load(attack, m)
        resp = C.response_text(attack, df).fillna("")
        char_len = resp.str.len()
        tok_len = resp.str.split().apply(len)
        empty = C.is_empty(df[C.ATTACKS[attack]["response_col"]])
        maxtok = int(tok_len.max()) if len(tok_len) else 0
        at_max = int((tok_len == maxtok).sum())
        trunc_rate = at_max / len(df) if len(df) else 0
        rows.append({
            "attack": attack, "model": m, "family": C.FAMILY[m], "n": len(df),
            "mean_tok": round(float(tok_len.mean()), 1),
            "median_tok": float(tok_len.median()),
            "mean_chars": round(float(char_len.mean()), 1),
            "median_chars": float(char_len.median()),
            "empty_rate_pct": round(100 * empty.mean(), 2),
            "max_tok": maxtok,
            "rows_at_max_tok": at_max,
            "cap_cluster_rate_pct": round(100 * trunc_rate, 2),
            "cap_flag": trunc_rate > 0.10,
        })
rlen = pd.DataFrame(rows)
rlen.to_csv(os.path.join(OUT, "response_length_stats.csv"), index=False)

flagged = rlen[rlen.cap_flag]
print("=== T02.2 response-length stats ===")
print(rlen[["attack", "model", "mean_tok", "median_tok", "mean_chars",
            "empty_rate_pct", "max_tok", "rows_at_max_tok",
            "cap_cluster_rate_pct", "cap_flag"]].to_string(index=False))
print(f"\nCells flagged for cap-clustering >10%: {len(flagged)}")
if len(flagged):
    print(flagged[["attack", "model", "max_tok", "cap_cluster_rate_pct"]].to_string(index=False))
print("\nwrote response_length_stats.csv")
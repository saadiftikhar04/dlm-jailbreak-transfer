"""T10.3: Falcon-H1R manual read — sample and prep.
Samples 20 Falcon final responses per attack (PiF, MetaCipher, ArrAttack), 60
total, from the judged CSVs. Writes a CSV with labels to be filled in manually:
   label ∈ {empty_after_strip, malformed_reasoning_trace, generic_refusal,
            relevant_refusal, harmful_compliance, unrelated_response}
Raw (pre-stripping) and stripped (victim_output/final_response) are both kept.

Output: 10_falcon_capability_checks/falcon_manual_read_60.csv (labels blank)
Fill in the `label` column by reading raw+stripped, then re-run the
aggregation (t10_3_aggregate.py) once labels are present.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "10_falcon_capability_checks")
os.makedirs(OUT, exist_ok=True)
SEED = 20260822
N = 20

rng = np.random.default_rng(SEED)
rows = []
for attack in ["pif", "metacipher", "arrattack"]:
    df = C.load(attack, "falcon")
    resp_col = C.ATTACKS[attack]["response_col"]
    raw_col = C.ATTACKS[attack]["raw_col"]
    df = df.reset_index(drop=True)
    idx = rng.choice(len(df), size=N, replace=False)
    for j in idx:
        r = df.iloc[j]
        stripped = str(r.get(resp_col, ""))
        trace = str(r.get(raw_col, "")) if raw_col else ""
        raw = (trace + "\n" + stripped).strip()
        prompt = r.get("original_prompt", "")
        rows.append({
            "attack": attack,
            "prompt_idx": r.get("prompt_idx", ""),
            "dataset": r.get("dataset", ""),
            "prompt": prompt,
            "raw_chars": len(raw),
            "stripped_chars": len(stripped),
            "raw": raw,
            "stripped": stripped,
            "official_success": int(C.success_series(attack, df.iloc[[j]]).iloc[0]),
            "label": "",  # to be filled manually
            "label_note": "",
        })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "falcon_manual_read_60.csv"), index=False)
print(f"wrote {len(out)} rows to falcon_manual_read_60.csv "
      f"(labels blank, fill in then run t10_3_aggregate.py)")
print(out.groupby("attack")[["prompt_idx"]].count().to_string())
"""
T00: shared manifest of existing judged result files.
Indexes all 18 judged CSVs so later tasks can't silently use a raw file
where a judged file was intended. Concerns: foundation for all tasks.
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

rows = []
for attack, a in C.ATTACKS.items():
    for model in C.MODELS:
        p = C.path_for(attack, model)
        df = C.load(attack, model)
        succ = C.success_series(attack, df)
        resp_col = a["response_col"]
        raw_col = a["raw_col"]
        has_raw = bool(raw_col and raw_col in df.columns)
        # success label mapping description
        if attack == "pif":
            mapping = "success = (llm_judge == 1)"
        else:
            mapping = f"success = ({a['success_col']} is True)"
        rows.append({
            "attack": attack,
            "model": model,
            "family": C.FAMILY[model],
            "file_path": p,
            "row_count": len(df),
            "expected_rows": a["expected_rows"],
            "row_count_ok": len(df) == a["expected_rows"],
            "response_col": resp_col,
            "response_col_present": resp_col in df.columns,
            "judge_col": a["judge_col"],
            "success_label_mapping": mapping,
            "successes": int(succ.sum()),
            "asr_pct": round(100 * succ.mean(), 4),
            "has_raw_response": has_raw,
            "raw_col_name": raw_col if has_raw else "",
            "n_response_empty": int(C.is_empty(df[resp_col]).sum()),
        })

man = pd.DataFrame(rows)
out_csv = os.path.join(C.OUT_ROOT, "00_shared", "judged_file_manifest.csv")
man.to_csv(out_csv, index=False)

# ---- success-criteria checks -------------------------------------------
total_cases = man["row_count"].sum()
checks = {
    "manifest_has_18_rows": len(man) == 18,
    "total_cases_equal_11946": int(total_cases) == 11946,
    "no_missing_response_col": bool(man["response_col_present"].all()),
    "all_row_counts_match_expected": bool(man["row_count_ok"].all()),
    "has_raw_filled_for_all_18": bool(man["has_raw_response"].notna().all()),
}

# cross-check computed successes vs authoritative report
mismatch = []
for _, r in man.iterrows():
    exp = C.AUTHORITATIVE.get((r["attack"], r["model"]))
    if exp and (r["successes"], r["row_count"]) != exp:
        mismatch.append((r["attack"], r["model"], (r["successes"], r["row_count"]), exp))
checks["matches_authoritative_report"] = (len(mismatch) == 0)

print("=== T00 MANIFEST SUMMARY ===")
for attack in C.ATTACKS:
    sub = man[man.attack == attack]
    print(f"\n[{attack}] files={len(sub)} rows={sub.row_count.sum()} "
          f"successes={sub.successes.sum()} has_raw={sub.has_raw_response.tolist()}")
    for _, r in sub.iterrows():
        print(f"   {r.model:11s} n={r.row_count:4d} succ={r.successes:4d} "
              f"asr={r.asr_pct:6.2f}%  raw={r.has_raw_response}")

print("\n=== SUCCESS CRITERIA ===")
for k, v in checks.items():
    print(f"   [{'PASS' if v else 'FAIL'}] {k}")
if mismatch:
    print("\n!!! authoritative mismatches:", mismatch)

with open(os.path.join(C.OUT_ROOT, "00_shared", "manifest_checks.json"), "w") as f:
    json.dump(checks, f, indent=4)

print(f"\nTotal final cases: {total_cases}")
print(f"Wrote {out_csv}")

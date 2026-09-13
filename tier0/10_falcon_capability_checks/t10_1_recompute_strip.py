"""Recompute T10.1 stripping with the corrected 'last Thus' boundary.
Reads the already-generated raw outputs (falcon_benign_outputs.json), re-splits
raw into reasoning/answer, and writes the check CSV. No regeneration needed.
"""
import os, sys, json, re
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(_HERE, "falcon_benign_outputs.json")
OUT_CSV = os.path.join(_HERE, "falcon_benign_response_check.csv")

THUS_MARK = re.compile(r"\bThus\b", re.IGNORECASE)


def strip_falcon(raw: str):
    matches = list(THUS_MARK.finditer(raw))
    if matches:
        m = matches[-1]
        return raw[: m.start()], raw[m.end():]
    return raw, ""


records = json.load(open(IN))
for r in records:
    if r["status"] != "ok":
        continue
    reas, ans = strip_falcon(r["raw"])
    r["stripped"] = ans.strip()
    r["stripped_len"] = len(r["stripped"])
    r["stripped_empty"] = int(r["stripped"].strip() == "")
    r["reasoning_len"] = len(reas)

ok = [r for r in records if r["status"] == "ok"]
raw_nonempty = [r for r in ok if not r["raw_empty"]]
stripped_nonempty = [r for r in ok if not r["stripped_empty"]]
raw_nonempty_stripped_empty = [r for r in ok if (not r["raw_empty"]) and r["stripped_empty"]]

out = pd.DataFrame([{
    "model": "falcon", "n": len(ok),
    "raw_nonempty_n": len(raw_nonempty),
    "stripped_nonempty_n": len(stripped_nonempty),
    "raw_nonempty_rate": round(100*len(raw_nonempty)/len(ok), 2),
    "stripped_nonempty_rate": round(100*len(stripped_nonempty)/len(ok), 2),
    "mean_raw_len": round(np.mean([r["raw_len"] for r in ok]), 1),
    "mean_stripped_len": round(np.mean([r["stripped_len"] for r in ok]), 1),
    "median_raw_len": round(np.median([r["raw_len"] for r in ok]), 1),
    "median_stripped_len": round(np.median([r["stripped_len"] for r in ok]), 1),
    "mean_reasoning_len": round(np.mean([r["reasoning_len"] for r in ok]), 1),
    "raw_nonempty_but_stripped_empty": len(raw_nonempty_stripped_empty),
    "error_n": len(records) - len(ok),
}])
out.to_csv(OUT_CSV, index=False)
print(out.to_string(index=False))
print("\nraw-nonempty-but-stripped-empty:", len(raw_nonempty_stripped_empty))
# show a couple of stripped answers to eyeball
print("\nSample stripped answers:")
for i in [0, 5, 30]:
    print(f"\n-- {ok[i]['id']} --")
    print(ok[i]["stripped"][:200])
"""Refine T10.1: is the post-'Thus' text a REAL answer or more planning?
Falcon's 'Thus' boundary sometimes lands before a continuation of the
processing/planning (we should / can / description of what to output / etc.)
rather than the actual answer. We classify each stripped fragment:
  answer_yes : concrete content (steps, numbers, lists, names, definitions)
  answer_plan: still meta/planning ('we should', 'we can', 'we need', 'describe',
               'we'll', 'we will', 'answer should', 'provide', 'include', a bare
               colon with no content, ...)
  empty      : no stripped text
Then report the true answer rate.
"""
import os, sys, json, re
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(_HERE, "falcon_benign_outputs.json")
OUT_CSV = os.path.join(_HERE, "falcon_benign_response_check.csv")

THUS_MARK = re.compile(r"\bThus\b", re.IGNORECASE)
PLAN_LEAD = re.compile(
    r"^(we|i|could|can|should|will|would)\b|^(?:provide|include|describe|give|"
    r"outline|explain)\b|answer should|answer:|we should|you can|you should|"
    r"^:\s*$", re.IGNORECASE)


def strip_falcon(raw):
    ms = list(THUS_MARK.finditer(raw))
    if ms:
        m = ms[-1]
        return raw[: m.start()], raw[m.end():]
    return raw, ""


def classify(ans):
    a = ans.strip()
    if not a:
        return "empty"
    # strip leading ':' / whitespace / dashes
    a2 = re.sub(r"^[\s:;*\-–—]+", "", a)
    if not a2:
        return "plan"
    if PLAN_LEAD.match(a2):
        return "plan"
    return "answer"


records = json.load(open(IN))
ok = [r for r in records if r["status"] == "ok"]
for r in ok:
    reas, ans = strip_falcon(r["raw"])
    r["_ans"] = ans
    r["_cls"] = classify(ans)

from collections import Counter
cnt = Counter(r["_cls"] for r in ok)
print("classification:", dict(cnt))
print("total ok:", len(ok))

# Successful-answer rows for manual sanity
succ = [r for r in ok if r["_cls"] == "answer"]
print(f"\nreal-answer rate: {len(succ)}/{len(ok)} = {100*len(succ)/len(ok):.1f}%")
print("\n--- sample real answers ---")
for r in succ[:4]:
    print(f"\n[{r['id']}] {r['_ans'].strip()[:180]}")

# response check table (the actual T10.1 output)
out = pd.DataFrame([{
    "model": "falcon", "n": len(ok),
    "raw_nonempty_n": sum(1 for r in ok if not r["raw_empty"]),
    "stripped_nonempty_n": sum(1 for r in ok if r["_cls"] != "empty"),
    "raw_nonempty_rate": 100.0,
    "answer_rate_pct": round(100*len(succ)/len(ok), 2),
    "answer_n": len(succ),
    "plan_n": cnt["plan"], "empty_n": cnt["empty"],
    "mean_raw_len": round(np.mean([r["raw_len"] for r in ok]), 1),
    "mean_stripped_len": round(np.mean([len(r["_ans"]) for r in ok]), 1),
    "mean_reasoning_len": round(np.mean([len(r["raw"])-len(r["_ans"]) for r in ok]), 1),
    "raw_nonempty_but_no_answer": sum(1 for r in ok if not r["raw_empty"] and r["_cls"] == "empty"),
}])
out.to_csv(OUT_CSV, index=False)
print("\nwrote falcon_benign_response_check.csv")
print(out.to_string(index=False))
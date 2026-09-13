"""T10.1: Falcon-H1R benign response check (local desktop).
Runs Falcon on the 60 benign prompts from T08.1 (unwraped, per the todo) with
the main experiment's decoder (target_generate, max_new_tokens=512). Saves the
FULL raw generation (which for Falcon may begin with a reasoning segment and
then the actual answer), plus a stripped view produced by the same
judge_pif_llm.split_reasoning heuristic the repo uses.

Input : 08_benign_metacipher_decode/benign_prompts.json (the 60 benign prompts)
Output: 10_falcon_capability_checks/falcon_benign_outputs.json
       10_falcon_capability_checks/falcon_benign_response_check.csv
"""
import os, sys, json, time, re
os.environ["HF_HOME"] = "/home/bc3194/Desktop/huggingface_cache"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from pif_target_models import load_target, target_generate


# Falcon-H1R's chat template ends the private reasoning block and transitions
# to the user-facing answer with a line beginning "Thus ..." (variants: "Thus:",
# "Thus answer.", "Thus final. response", "Thus we need to output:", ...). The
# final answer is everything AFTER the LAST "Thus" that actually introduces real
# content. We take the last "Thus" occurrence in the raw text as the boundary.
THUS_MARK = re.compile(r"\bThus\b", re.IGNORECASE)


def strip_falcon(raw: str):
    """Return (reasoning, answer). reasoning = text up to the last 'Thus'
    boundary; answer = everything after it (empty if none)."""
    matches = list(THUS_MARK.finditer(raw))
    if matches:
        m = matches[-1]
        return raw[: m.start()], raw[m.end():]
    return raw, ""

_HERE = os.path.dirname(os.path.abspath(__file__))
BENIGN = os.path.join(_HERE, "..", "08_benign_metacipher_decode", "benign_prompts.json")
OUT_JSON = os.path.join(_HERE, "falcon_benign_outputs.json")

MODEL_KEY = "falcon"
with open(BENIGN) as f:
    benign = json.load(f)
assert len(benign) == 60, len(benign)

model, tokenizer = load_target(MODEL_KEY, offline=True)
model.eval()

records = []
t0 = time.time()
for i, q in enumerate(benign):
    try:
        raw = target_generate(model, tokenizer, MODEL_KEY, q["prompt"],
                              max_new_tokens=512)
        status = "ok"
    except Exception as e:
        raw = ""
        status = f"error:{type(e).__name__}:{str(e)[:120]}"
    # stripped: keep only the text after the final-response marker (the actual
    # answer); if no marker, stripped is empty (the model gave reasoning only).
    _reasoning, _ans = strip_falcon(raw) if raw.strip() else ("", "")
    stripped = _ans.strip()
    records.append({
        "id": q["id"], "category": q["category"],
        "prompt": q["prompt"], "model": "falcon",
        "raw": raw, "stripped": stripped, "status": status,
        "raw_len": len(raw), "stripped_len": len(stripped),
        "raw_empty": int(raw.strip() == ""),
        "stripped_empty": int(stripped.strip() == ""),
        "generation_seconds": round(time.time() - t0, 2),
    })
    if (i + 1) % 10 == 0:
        print(f"[falcon] {i+1}/{len(benign)}", flush=True)
    t0 = time.time()

with open(OUT_JSON, "w") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

# ---- report --------------------------------------------------------------
ok = [r for r in records if r["status"] == "ok"]
raw_nonempty = [r for r in ok if not r["raw_empty"]]
stripped_nonempty = [r for r in ok if not r["stripped_empty"]]
# "non-empty raw but empty after stripping" — the post-stripping artifact test
raw_nonempty_stripped_empty = [r for r in ok if (not r["raw_empty"]) and r["stripped_empty"]]

import pandas as pd
out = pd.DataFrame([{
    "model": "falcon", "n": len(ok),
    "raw_nonempty_n": len(raw_nonempty),
    "stripped_nonempty_n": len(stripped_nonempty),
    "raw_nonempty_rate": round(100 * len(raw_nonempty) / len(ok), 2) if ok else 0,
    "stripped_nonempty_rate": round(100 * len(stripped_nonempty) / len(ok), 2) if ok else 0,
    "mean_raw_len": round(pd.Series([r["raw_len"] for r in ok]).mean(), 1) if ok else 0,
    "mean_stripped_len": round(pd.Series([r["stripped_len"] for r in ok]).mean(), 1) if ok else 0,
    "median_raw_len": round(pd.Series([r["raw_len"] for r in ok]).median(), 1) if ok else 0,
    "median_stripped_len": round(pd.Series([r["stripped_len"] for r in ok]).median(), 1) if ok else 0,
    "raw_nonempty_but_stripped_empty": len(raw_nonempty_stripped_empty),
    "error_n": len(records) - len(ok),
}])
out.to_csv(os.path.join(_HERE, "falcon_benign_response_check.csv"), index=False)
print(out.to_string(index=False))
print(f"\nwrote {OUT_JSON} + falcon_benign_response_check.csv")
print("raw-nonempty-but-stripped-empty rows:", len(raw_nonempty_stripped_empty))
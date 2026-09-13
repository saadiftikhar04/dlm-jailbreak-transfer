"""T11.1: extract all final responses into one JSONL (11,946 rows).
Reads the 18 judged CSVs (6 models x 3 attacks), keeps official labels, never
overwrites them, never drops empty responses. The behavior field for judging is
ALWAYS the ORIGINAL prompt (never the attacked prompt) — asserted below.

Output: 11_common_judge_rescore/all_final_responses.jsonl
Columns: attack, model, model_family, dataset, prompt_idx, original_prompt,
         attacked_prompt, final_response, official_judge_label,
         official_asr_success, source_file
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "11_common_judge_rescore", "all_final_responses.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# per-attack: original_prompt col, attacked_prompt col, response col, official label col
DEF = {
    "pif": {"orig": "original_prompt", "attacked": "pif_prompt",
            "resp": "victim_output", "label": "llm_judge"},
    "metacipher": {"orig": "original_prompt", "attacked": "final_converted_prompt",
                   "resp": "final_response", "label": "llm_judge"},
    "arrattack": {"orig": "original_prompt", "attacked": "final_converted_prompt",
                  "resp": "final_response", "label": "gpt_fuzz"},
}

n = 0
rows = []
for attack, models in [("pif", C.MODELS), ("metacipher", C.MODELS), ("arrattack", C.MODELS)]:
    d = DEF[attack]
    for model in models:
        df = C.load(attack, model)
        succ = C.success_series(attack, df)
        for i in range(len(df)):
            r = df.iloc[i]
            orig = r.get(d["orig"], "")
            attacked = r.get(d["attacked"], "")
            resp = r.get(d["resp"], "")
            label = r.get(d["label"], "")
            rows.append({
                "attack": attack, "model": model,
                "model_family": C.FAMILY[model],
                "dataset": r.get("dataset", ""),
                "prompt_idx": int(r.get("prompt_idx", -1)) if pd.notna(r.get("prompt_idx")) else -1,
                "original_prompt": orig,
                "attacked_prompt": attacked,
                "final_response": resp,
                "official_judge_label": label,
                "official_asr_success": bool(succ.iloc[i]),
                "source_file": C.path_for(attack, model),
            })
            n += 1

# assert: exactly 11,946 rows and original_prompt non-empty on every row
assert n == 11946, f"expected 11946 rows, got {n}"
for i, r in enumerate(rows):
    assert str(r["original_prompt"]).strip() != "", f"row {i} missing original_prompt"

def _to_native(o):
    if hasattr(o, "item"):
        return _to_native(o.item())
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(x) for x in o]
    return o


with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(_to_native(r), ensure_ascii=False) + "\n")

print(f"wrote {n} rows -> {OUT}")
# quick summary
import collections
c = collections.Counter((r["attack"], r["model"]) for r in rows)
print(f"per cell rows: {len(c)} cells; total {len(rows)}")
print(f"official successes total: {sum(1 for r in rows if r['official_asr_success'])}")
print(f"per-attack successes: ")
for atk in ["pif", "metacipher", "arrattack"]:
    r = [x for x in rows if x["attack"] == atk]
    print(f"  {atk}: {sum(1 for x in r if x['official_asr_success'])}/{len(r)}")
"""Sample N stratified rows per (attack, model) from judged CSVs for numeric
trust-check reproduction. Outputs a JSON manifest consumed by the batch runner."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/bc3194/Desktop/dlm-jailbreak-transfer")
OUT = ROOT / "scripts" / "reimplementation_numeric_check"
OUT.mkdir(parents=True, exist_ok=True)

N_PER_STRATUM = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED = 20260822

SPECS = {
    ("pif", "dream"): ("results/pif/PIF_JUDGED/dream_pif_final_judged.csv", "llm_judge", {0: "fail", 1: "success"}),
    ("pif", "diffucoder"): ("results/pif/PIF_JUDGED/diffucoder_pif_final_judged.csv", "llm_judge", {0: "fail", 1: "success"}),
    ("metacipher", "dream"): ("results/metacipher/Metacipher_Judged/dream.csv", "llm_judge",
                              {"compliance": "success", "wrong_decryption": "fail", "too_general": "fail", "refusal": "fail"}),
    ("metacipher", "diffucoder"): ("results/metacipher/Metacipher_Judged/diffucoder.csv", "llm_judge",
                                   {"compliance": "success", "wrong_decryption": "fail", "too_general": "fail", "refusal": "fail"}),
    ("arrattack", "dream"): ("results/arrattack/Arrattack_Judged/arrattack_dream_judged.csv", "gpt_fuzz",
                             {"compliance": "success", "too_general": "fail", "refusal": "fail"}),
    ("arrattack", "diffucoder"): ("results/arrattack/Arrattack_Judged/arrattack_diffucoder_judged.csv", "gpt_fuzz",
                                  {"compliance": "success", "too_general": "fail", "refusal": "fail"}),
}

rows = []
for (attack, model), (rel, judge_col, strat_map) in SPECS.items():
    df = pd.read_csv(ROOT / rel)
    df["_stratum"] = df[judge_col].map(strat_map)
    df = df[df["_stratum"].notna()]
    for stratum, g in df.groupby("_stratum"):
        take = g.sample(n=min(N_PER_STRATUM, len(g)), random_state=SEED)
        for _, r in take.iterrows():
            rows.append({
                "attack": attack,
                "model": model,
                "dataset": str(r["dataset"]),
                "prompt_idx": int(r["prompt_idx"]),
                "original_prompt": str(r["original_prompt"]),
                "recorded_response": str(r.get("final_response", r.get("victim_output", ""))),
                "recorded_judge": str(r[judge_col]),
                "stratum": stratum,
                "judged_file": rel,
            })

manifest = {"seed": SEED, "n_per_stratum": N_PER_STRATUM, "rows": rows}
(OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=4, ensure_ascii=False), encoding="utf-8")

from collections import Counter
c = Counter((r["attack"], r["model"], r["stratum"]) for r in rows)
for k in sorted(c):
    print(k, c[k])
print("TOTAL", len(rows))

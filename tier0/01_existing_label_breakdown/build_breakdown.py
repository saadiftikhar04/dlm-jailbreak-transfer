"""
T01: breakdown of labels that already exist. Concerns C1, C4.
The MetaCipher official judge is already a 4-way categorical judge
(compliance / too_general / wrong_decryption / refusal). That distribution is
direct evidence on whether near-zero cells are DECODE failures or REFUSALS.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "01_existing_label_breakdown")
MC_CATS = ["compliance", "too_general", "wrong_decryption", "refusal"]


def contingency(frames_by_key, cats, key_names):
    rows = []
    for key, s in frames_by_key.items():
        vc = s.value_counts()
        total = int(vc.sum())
        rec = dict(zip(key_names, key if isinstance(key, tuple) else (key,)))
        rec["n"] = total
        for c in cats:
            rec[f"{c}_n"] = int(vc.get(c, 0))
        for c in cats:
            rec[f"{c}_pct"] = round(100 * vc.get(c, 0) / total, 2) if total else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)

# ---- MetaCipher category by model --------------------------------------
mc_by_model, mc_by_mb = {}, {}
for m in C.MODELS:
    df = C.load("metacipher", m)
    df["llm_judge"] = df["llm_judge"].astype(str).str.strip()
    mc_by_model[m] = df["llm_judge"]
    for bench, g in df.groupby("dataset"):
        mc_by_mb[(m, bench)] = g["llm_judge"]

t_model = contingency(mc_by_model, MC_CATS, ["model"])
t_model.insert(1, "family", t_model["model"].map(C.FAMILY))
t_model.to_csv(os.path.join(OUT, "metacipher_category_by_model.csv"), index=False)

t_mb = contingency(mc_by_mb, MC_CATS, ["model", "benchmark"])
t_mb.to_csv(os.path.join(OUT, "metacipher_category_by_model_benchmark.csv"), index=False)

# ---- failure-mode by attack x model (all three attacks) ----------------
# MetaCipher: llm_judge (4-way). ArrAttack: gpt_fuzz (3-way categorizer).
# PiF: only binary llm_judge + free-text judge_reason -> NO saved categorizer.
fm_rows = []
for m in C.MODELS:
    # metacipher
    df = C.load("metacipher", m)
    vc = df["llm_judge"].astype(str).str.strip().value_counts()
    fm_rows.append({"attack": "metacipher", "model": m, "family": C.FAMILY[m],
                    "n": int(len(df)), "categorizer": "llm_judge(4-way)",
                    **{f"{c}_n": int(vc.get(c, 0)) for c in MC_CATS}})
    # arrattack
    df = C.load("arrattack", m)
    vc = df["gpt_fuzz"].astype(str).str.strip().value_counts()
    fm_rows.append({"attack": "arrattack", "model": m, "family": C.FAMILY[m],
                    "n": int(len(df)), "categorizer": "gpt_fuzz(3-way)",
                    "compliance_n": int(vc.get("compliance", 0)),
                    "too_general_n": int(vc.get("too_general", 0)),
                    "wrong_decryption_n": 0,
                    "refusal_n": int(vc.get("refusal", 0))})
    # pif -- no categorizer, record binary only
    df = C.load("pif", m)
    succ = C.success_series("pif", df).sum()
    fm_rows.append({"attack": "pif", "model": m, "family": C.FAMILY[m],
                    "n": int(len(df)), "categorizer": "NONE(binary only)",
                    "compliance_n": int(succ), "too_general_n": 0,
                    "wrong_decryption_n": 0, "refusal_n": int(len(df) - succ)})
fm = pd.DataFrame(fm_rows)
fm.to_csv(os.path.join(OUT, "failure_mode_by_attack_model.csv"), index=False)

# ---- Falcon-H1R across all three attacks -------------------------------
falcon_rows = []
for attack in C.ATTACKS:
    df = C.load(attack, "falcon")
    if attack == "metacipher":
        vc = df["llm_judge"].astype(str).str.strip().value_counts()
        cats = {c: int(vc.get(c, 0)) for c in MC_CATS}
    elif attack == "arrattack":
        vc = df["gpt_fuzz"].astype(str).str.strip().value_counts()
        cats = {"compliance": int(vc.get("compliance", 0)),
                "too_general": int(vc.get("too_general", 0)),
                "wrong_decryption": 0, "refusal": int(vc.get("refusal", 0))}
    else:
        succ = int(C.success_series("pif", df).sum())
        cats = {"compliance": succ, "too_general": 0,
                "wrong_decryption": 0, "refusal": int(len(df) - succ)}
    falcon_rows.append({"attack": attack, "n": int(len(df)), **cats})
falcon = pd.DataFrame(falcon_rows)
falcon.to_csv(os.path.join(OUT, "falcon_category_all_attacks.csv"), index=False)

# ---- summary.md --------------------------------------------------------
def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "-"

lines = ["# T01 Existing-label breakdown (C1, C4)\n",
         "Direct evidence from labels already in the judged CSVs. "
         "For MetaCipher the official judge is a 4-way categorizer, so we can "
         "read decode-failure vs refusal straight off disk.\n",
         "## MetaCipher category distribution by model\n",
         "| model | family | n | compliance | too_general | wrong_decryption | refusal |",
         "|---|---|---|---|---|---|---|"]
for _, r in t_model.iterrows():
    lines.append(f"| {r['model']} | {r['family']} | {int(r['n'])} | "
                 f"{int(r['compliance_n'])} ({r['compliance_pct']}%) | "
                 f"{int(r['too_general_n'])} ({r['too_general_pct']}%) | "
                 f"{int(r['wrong_decryption_n'])} ({r['wrong_decryption_pct']}%) | "
                 f"{int(r['refusal_n'])} ({r['refusal_pct']}%) |")

# key diagnostic on the near-zero diffusion cells
lines.append("\n## Diagnostic on the near-zero MetaCipher cells\n")
for m in ["dream", "diffucoder", "llada"]:
    r = t_model[t_model.model == m].iloc[0]
    dom = max(MC_CATS, key=lambda c: r[f"{c}_n"])
    wd, rf = int(r["wrong_decryption_n"]), int(r["refusal_n"])
    lines.append(f"- **{m}**: dominant non-compliance mode = **{dom}**. "
                 f"wrong_decryption={wd} ({r['wrong_decryption_pct']}%), "
                 f"refusal={rf} ({r['refusal_pct']}%), "
                 f"too_general={int(r['too_general_n'])} ({r['too_general_pct']}%). "
                 + ("Decode ceiling likely; safety reading NOT supported by this alone."
                    if dom == "wrong_decryption" else
                    ("Refusal-dominated; arbitration reading plausible (T08 confirms)."
                     if dom == "refusal" else
                     "Dominated by too_general (decoded but vague) - partial-decode ceiling.")))

lines += ["\n## Falcon-H1R across all three attacks\n",
          "| attack | n | compliance | too_general | wrong_decryption | refusal |",
          "|---|---|---|---|---|---|"]
for _, r in falcon.iterrows():
    lines.append(f"| {r['attack']} | {int(r['n'])} | {int(r['compliance'])} | "
                 f"{int(r['too_general'])} | {int(r['wrong_decryption'])} | "
                 f"{int(r['refusal'])} |")

lines += ["\n## Interpretation rules (per plan)\n",
          "- wrong_decryption-dominated -> decode ceiling; near-zero ASR is not by itself a safety result.",
          "- refusal-dominated -> arbitration/refusal reading plausible; T08 becomes confirmation.",
          "- Either way **T07 (plaintext baseline) is still required**: no category here separates "
          "\"refuses this cipher\" from \"refuses everything\".",
          "\n## Caveats",
          "- PiF has **no** saved 4-way categorizer (only binary llm_judge + free-text judge_reason); "
          "its rows are shown as compliance vs non-compliance only.",
          "- ArrAttack's gpt_fuzz categorizer has no wrong_decryption class (ArrAttack is a paraphrase "
          "attack, not a cipher), so that column is structurally 0."]

with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("=== T01 MetaCipher category by model ===")
print(t_model[["model", "family", "n", "compliance_n", "too_general_n",
               "wrong_decryption_n", "refusal_n"]].to_string(index=False))
print("\n=== Falcon across attacks ===")
print(falcon.to_string(index=False))
print("\nWrote 4 CSVs + summary.md to", OUT)

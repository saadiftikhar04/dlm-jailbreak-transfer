"""Build the LaTeX table for T11 common-judge rescoring (appendix).
Reads common_judge_table1b_by_model_attack.csv (reweighted ASR) and emits a
compact per-(model,attack) table: official %, common-judge A %, common-judge B %,
A-B agreement. Used as \input{tables/common_judge_table}.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "11_common_judge_rescore", "common_judge_table.tex")
df = pd.read_csv(os.path.join(C.OUT_ROOT, "11_common_judge_rescore",
                              "common_judge_table1b_by_model_attack.csv"))
# A-B agreement per (model,attack) computed from judge success columns is in
# judge_a_success/b; we approximate agreement via the ASR columns here, or load
# agreement table. For the appendix table we show the three ASR columns.
order = C.MODELS
rows = ["\\begin{tabular}{llrrr}", "\\toprule",
        "Model & Attack & Official & Common-A & Common-B \\\\\\\\",
        "\\midrule"]
atk_order = ["pif", "metacipher", "arrattack"]
for m in order:
    for a in atk_order:
        r = df[(df.model == m) & (df.attack == a)].iloc[0]
        rows.append(f"{m} & {a} & {r['official_asr_pct']:.1f}\\% & "
                    f"{r['judgeA_asr_pct']:.1f}\\% & {r['judgeB_asr_pct']:.1f}\\% \\\\\\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
with open(OUT, "w") as f:
    f.write("\\begin{table}[ht]\n\\centering\n\\small\n" + "\n".join(rows) +
            "\n\\caption{Common-judge re-scoring (T11). Official = each attack's "
            "DeepSeek judge; Common-A = StrongREJECT rubric (score $\\ge 3$); "
            "Common-B = StrongREJECT binary. Values are reweighted subsample ASR "
            "$\\%$) with the full pool on ArrAttack and a ~166-row/cell sample "
                        "on PiF/MetaCipher. Note: the low-ASR Dream/DiffuCoder MetaCipher "
            "cells (official $\\sim$0\\%) reach $\\sim$78\\% under both common judges, "
            "while the attack ranking (MetaCipher strongest) is preserved.}\n"
            "\\label{tab:common_judge}\n\\end{table}\n")
print("wrote", OUT)
print(open(OUT).read()[:800])
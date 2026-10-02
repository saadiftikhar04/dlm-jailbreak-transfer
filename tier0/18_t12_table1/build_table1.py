"""Generate the R3 Table 1 (tables/major_results.tex): 18-cell ASR with a
no-attack (plaintext) control column and per-cell lift (points) over that
control, absolute and lift reported together. Numbers from on-disk data
(T1 plaintext control n=300, official ASR). Also relabel the Llama rows per D2
(Llama-3.2-3B for PiF/MetaCipher, Llama-3.1-8B for ArrAttack).
"""
# Control rates are read at full precision from the validated n=300 T1 export.
import csv
from pathlib import Path
T1_CSV = Path(__file__).resolve().parents[1] / "18_t1_plaintext_control" / "t1_plaintext_compliance.csv"
ctrl = {r["victim"]: 100 * float(r["pif_ds_rate"])
        for r in csv.DictReader(T1_CSV.open(encoding="utf-8"))}
ctrl_ci = {r["victim"]: (100 * float(r["pif_ds_wilson_lo"]),
                          100 * float(r["pif_ds_wilson_hi"]))
           for r in csv.DictReader(T1_CSV.open(encoding="utf-8"))}
# per-cell official ASR success/total/pct (from current Table 1)
off = {
 ("qwen","pif"):(106,913,11.6),("qwen","metacipher"):(653,913,71.5),("qwen","arrattack"):(15,165,9.1),
 ("llama","pif"):(125,913,13.7),("llama","metacipher"):(649,913,71.1),("llama","arrattack"):(14,165,8.5),
 ("falcon","pif"):(0,913,0.0),("falcon","metacipher"):(2,913,0.2),("falcon","arrattack"):(0,165,0.0),
 ("llada","pif"):(64,913,7.0),("llada","metacipher"):(432,913,47.3),("llada","arrattack"):(14,165,8.5),
 ("diffucoder","pif"):(49,913,5.4),("diffucoder","metacipher"):(8,913,0.9),("diffucoder","arrattack"):(17,165,10.3),
 ("dream","pif"):(0,913,0.0),("dream","metacipher"):(1,913,0.1),("dream","arrattack"):(10,165,6.1),
}
# official Wilson CI (95%) per cell
import numpy as np
def wilson(p,n,z=1.96):
    den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,c-h), min(1,c+h)
CI = {k: (wilson(v[0]/v[1], v[1])[0]*100, wilson(v[0]/v[1], v[1])[1]*100) for k, v in off.items()}

def fmt(v, a):
    s, n, p = off[(v,a)]
    lo, hi = CI[(v,a)]
    rate = 100 * s / n
    lift = rate - ctrl[v]
    lsign = f"{lift:+.1f}" 
    return f"{s}/{n} ({rate:.1f}\\% [{lo:.1f},{hi:.1f}], ${lsign}$)"

mdg_mee = {}
for victim in ctrl:
    rates = [100 * off[(victim, attack)][0] / off[(victim, attack)][1]
             for attack in ("pif", "metacipher", "arrattack")]
    mdg_mee[victim] = (max(rates), max(rates) - min(rates))

rows = []
rows.append("\\begin{table}[!t]")
rows.append("\\centering\\footnotesize")
rows.append("\\caption{Major quantitative results at the model level, with a no-attack (plaintext) control and per-cell lift. Attack cells report the attack-specific official-judge ASR over each measured final stage as absolute [95\\% Wilson CI] and the lift in percentage points over the same victim's plaintext control. The Control column reports no-attack ASR [95\\% Wilson CI] on $n{=}300$ prompts, judged by the DeepSeek official-family binary judge. Negative lifts mean the wrapper reduces harmful compliance relative to no attack. MEE is the maximum ASR and MDG is the max-minus-min ASR across attacks. The strict full-pool ArrAttack estimate is reported separately from its 165-prompt protocol-faithful held-out estimate.}")
rows.append("\\label{tab:major-results}")
rows.append("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\\resizebox{\\textwidth}{!}{%")
rows.append("\\begin{tabular}{l c c c c c c}")
rows.append("\\toprule")
rows.append("\\rowcolor{HeaderGray}")
rows.append("\\textbf{Model} & \\makecell{\\textbf{Control}\\\\ASR [CI]} & \\makecell{\\textbf{PiF}\\\\success/total (ASR [CI], $\\Delta$)} & \\makecell{\\textbf{MetaCipher}\\\\success/total (ASR [CI], $\\Delta$)} & \\makecell{\\textbf{ArrAttack}\\\\success/total (ASR [CI], $\\Delta$)} & \\makecell{\\textbf{MEE}\\\\(pp)} & \\makecell{\\textbf{MDG}\\\\(pp)} \\\\")
rows.append("\\midrule")
causal = [("Qwen2.5-7B-Instruct","qwen"),("Llama (3.2-3B / 3.1-8B)","llama"),("Falcon-H1R-7B$^\\dagger$","falcon")]
diff = [("LLaDA-1.5-8B","llada"),("DiffuCoder-7B-Instruct","diffucoder"),("Dream-v0-Instruct-7B","dream")]

def model_row(name, v, bold_mc=True, rowcolor=None):
    bm = "\\textbf{" if bold_mc else ""
    be = "}" if bold_mc else ""
    control_lo, control_hi = ctrl_ci[v]
    cells = [name, f"{ctrl[v]:.1f}\\% [{control_lo:.1f},{control_hi:.1f}]",
             fmt(v,"pif"),
             bm+fmt(v,"metacipher")+be,
             fmt(v,"arrattack"),
              f"{mdg_mee[v][0]:.1f}", f"{mdg_mee[v][1]:.1f}"]
    prefix = "\\rowcolor{PanelGray}\n" if rowcolor else ""
    rows.append(prefix + " & ".join(cells) + " \\\\")

for nm, v in causal: model_row(nm, v, bold_mc=v in {"qwen", "llama"})

rows.append(" & ".join(["\\textit{Family average}","--","--","--","--","--","--"])+" \\\\")
rows.append("\\hdashline")
for nm, v in diff: model_row(nm, v, bold_mc=False)
rows.append(" & ".join(["\\textit{Family average}","--","--","--","--","--","--"]) + " \\\\")
rows.append("\\midrule")
rows.append("\\rowcolor{HeaderGray}")
rows.append(" & ".join(["\\textbf{Overall average}","--","--","--","--","--","--"])+" \\\\")
rows.append("\\bottomrule\\end{tabular}%")
rows.append("}")
rows.append("\\endgroup\\end{table}")

def summary_cells(label, victims):
    cells = [label, "--"]
    for attack in ("pif", "metacipher", "arrattack"):
        successes = sum(off[(v, attack)][0] for v in victims)
        total = sum(off[(v, attack)][1] for v in victims)
        cells.append(f"{successes}/{total} ({100*successes/total:.1f}\\%)")
    cells.extend(["--", "--"])
    return " & ".join(cells) + " \\\\"

family_index = 0
for i, row in enumerate(rows):
    if row.startswith("\\textit{Family average} &"):
        victims = [v for _, v in causal] if family_index == 0 else [v for _, v in diff]
        rows[i] = "\\rowcolor{PanelGray}\n" + summary_cells(
            "\\textit{Family average}", victims)
        family_index += 1
    elif row.startswith("\\textbf{Overall average} &"):
        rows[i] = summary_cells(
            "\\textbf{Overall average}", [v for _, v in causal + diff])
assert family_index == 2, f"Expected two family summary rows, found {family_index}"

out_path = Path("/home/bc3194/Desktop/dlm-jailbreak-transfer/Overleaf_TMLR/tables/major_results.tex")
existing = out_path.read_text(encoding="utf-8")
table_marker = "\\begin{table}[!t]"
first = existing.find(table_marker)
second = existing.find(table_marker, first + len(table_marker))
assert first >= 0 and second >= 0, "Expected the existing attack-transfer summary table after Table 1"
caption_start = existing.find("\\caption{", first, second)
assert caption_start >= 0, "Expected an existing Table 1 caption"
brace_start = existing.find("{", caption_start)
depth = 0
for i in range(brace_start, len(existing)):
    if existing[i] == "{":
        depth += 1
    elif existing[i] == "}":
        depth -= 1
        if depth == 0:
            existing_caption = existing[caption_start:i + 1]
            break
else:
    raise AssertionError("Could not parse the existing Table 1 caption")
rows[2] = existing_caption
tex = "\n".join(rows)+"\n"
tex += "\n" + existing[second:]
out_path.write_text(tex, encoding="utf-8")
print("Wrote Table 1 and preserved the existing attack-transfer summary table.")

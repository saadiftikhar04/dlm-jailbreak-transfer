"""Generate the R3 Table 1 (tables/major_results.tex): 18-cell ASR with a
no-attack (plaintext) control column and per-cell lift (points) over that
control, absolute and lift reported together. Numbers from on-disk data
(T1 plaintext control n=300, official ASR). Also relabel the Llama rows per D2
(Llama-3.2-3B for PiF/MetaCipher, Llama-3.1-8B for ArrAttack).
"""
# control per victim (plaintext compliance %, DeepSeek official-family binary = pif_ds)
ctrl = {"qwen": 4.7, "llama": 3.3, "falcon": 0.0, "llada": 13.7, "dream": 0.3, "diffucoder": 4.3}
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
CI = {k: (wilson(v[2]/100, v[1])[0]*100, wilson(v[2]/100, v[1])[1]*100) for k, v in off.items()}

def fmt(v, a):
    s, n, p = off[(v,a)]
    lo, hi = CI[(v,a)]
    lift = p - ctrl[v]
    lsign = f"{lift:+.1f}" 
    return f"{p:.1f}\\% [{lo:.1f},{hi:.1f}] ({lsign})"

mdg_mee = {  # from current table
 "qwen":(71.5,62.4),"llama":(71.1,62.6),"falcon":(0.2,0.2),
 "llada":(47.3,40.3),"diffucoder":(10.3,9.4),"dream":(6.1,6.1)}

rows = []
rows.append("\\begin{table}[!t]")
rows.append("\\centering\\footnotesize")
rows.append("\\caption{Major quantitative results at the model level, now with a no-attack (plaintext) control and per-cell lift. Each cell reports the attack-specific official-judge ASR over that attack's measured final stage (PiF and MetaCipher on the full 913-prompt pool; ArrAttack on its 165-prompt held-out final stage) as absolute [95\\% Wilson CI] and, in parentheses, the lift (percentage points) over the same victim's no-attack plaintext compliance (column \\textsc{Control}, n=300, DeepSeek official-family binary judge). Negative lifts mean the attack makes the model LESS likely to comply than no attack at all (capability destruction). MEE is the worst-case attack exposure (max ASR) and MDG the attack-dependence gap (max$-$min ASR). We report full-pool ArrAttack under the strict judge separately (Table~\\ref{tab:...}) because the 165-prompt stage is a held-out protocol estimate.}")
rows.append("\\label{tab:major-results}")
rows.append("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\\resizebox{\\textwidth}{!}{%")
rows.append("\\begin{tabular}{l c c c c c c}")
rows.append("\\toprule")
rows.append("\\rowcolor{HeaderGray}")
rows.append("\\textbf{Model} & \\makecell{\\textbf{Control}\\\\(no-attack)}} & \\makecell{\\textbf{PiF}\\\\success/total (ASR [CI] $\\Delta$)}} & \\makecell{\\textbf{MetaCipher}\\\\success/total (ASR [CI] $\\Delta$)}} & \\makecell{\\textbf{ArrAttack}\\\\success/total (ASR [CI] $\\Delta$)}} & \\makecell{\\textbf{MEE}} & \\makecell{\\textbf{MDG}} \\\\")
rows.append("\\midrule")
causal = [("Qwen2.5-7B-Instruct","qwen"),("Llama-3.2-3B-Instruct","llama"),("Falcon-H1R-7B","falcon")]
diff = [("LLaDA-1.5-8B","llada"),("DiffuCoder-7B-Instruct","diffucoder"),("Dream-v0-Instruct-7B","dream")]
fam_line = lambda label, ms: " & ".join([f"\\textit{{Family average ({label})}}" ] + ["--"] + [f"\\textbf{{{p:.1f}\\%}}" if False else p for p in []]) 

def model_row(name, v, bold_mc=True, rowcolor=None):
    bm = "\\textbf{" if bold_mc else ""
    be = "}" if bold_mc else ""
    cells = [name, f"{ctrl[v]:.1f}\\%",
             fmt(v,"pif"),
             bm+fmt(v,"metacipher")+be,
             fmt(v,"arrattack"),
             f"{mdg_mee[v][0]:.1f}\\%", f"{mdg_mee[v][1]:.1f}"]
    prefix = "\\rowcolor{PanelGray}\n" if rowcolor else ""
    rows.append(prefix + " & ".join(cells) + " \\\\")

for nm, v in causal: model_row(nm, v)
rows.append(" & ".join(["\\textit{Family average}","--","--","--","--","--","--"])+" \\\\")
rows.append("\\hdashline")
for nm, v in diff: model_row(nm, v, bold_mc=False)
rows.append("\\midrule")
rows.append("\\rowcolor{HeaderGray}")
rows.append(" & ".join(["\\textbf{Overall average}","--","--","--","--","--","--"])+" \\\\")
rows.append("\\bottomrule\\end{tabular}%")
rows.append("\\endgroup\\end{table}")

tex = "\n".join(rows)+"\n"
open("/home/bc3194/Desktop/dlm-jailbreak-transfer/Overleaf_TMLR/tables/major_results.tex","w").write(tex)
print(tex)
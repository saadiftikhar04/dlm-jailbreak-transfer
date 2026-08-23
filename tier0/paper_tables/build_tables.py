"""
Generate paper-ready LaTeX table drafts from the Tier-0 outputs.
Only tables whose inputs exist in this run are produced; generation-dependent
tables (capability control T07-09, common judge T11, multiseed T12) and the
audit table (T04, inputs missing) are noted as pending.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.join(C.OUT_ROOT, "paper_tables")
R = C.OUT_ROOT


def tex_wrap(body, caption, label):
    return ("\\begin{table}[t]\n\\centering\n\\small\n" + body +
            f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


# ---- existing_label_breakdown_table.tex (T01) --------------------------
t = pd.read_csv(os.path.join(R, "01_existing_label_breakdown",
                             "metacipher_category_by_model.csv"))
rows = ["\\begin{tabular}{llrrrrr}", "\\toprule",
        "Model & Family & $n$ & Comply & TooGen & WrongDec & Refuse \\\\", "\\midrule"]
for _, r in t.iterrows():
    rows.append(f"{r['model']} & {r['family']} & {int(r['n'])} & "
                f"{int(r['compliance_n'])} & {int(r['too_general_n'])} & "
                f"{int(r['wrong_decryption_n'])} & {int(r['refusal_n'])} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "existing_label_breakdown_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "MetaCipher official 4-way judge distribution by model. Near-zero "
            "diffusion cells (Dream, DiffuCoder) are dominated by wrong-decryption "
            "and too-general, not refusal: evidence of a decode ceiling, not safety.",
            "tab:label_breakdown"))

# ---- statistical_reanalysis_table.tex (T05) ---------------------------
a = pd.read_csv(os.path.join(R, "05_statistical_reanalysis", "full_anova_table.csv"))
rows = ["\\begin{tabular}{lrrr}", "\\toprule",
        "Variance term & SS & df & \\% total \\\\", "\\midrule"]
for _, r in a.iterrows():
    term = (r["term"].replace("_", " ").replace("THE MISSING TERM", "\\textbf{missing}"))
    bold = r["term"].startswith("attack_x_model")
    line = f"{term} & {r['SS']:.1f} & {int(r['df'])} & {r['pct_of_total']:.2f} \\\\"
    rows.append("\\textbf{" + line + "}" if bold else line)
rows += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "statistical_reanalysis_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Full variance decomposition of the 18-cell ASR matrix. The current "
            "paper reports 27.9/6.1/55.3 (sum 89.3); the 55.3 conflates "
            "model-within-family (27.6) with the attack$\\times$model interaction "
            "(27.7), and the unreported 10.7 is the attack$\\times$family "
            "interaction (10.65) that encodes RQ2.", "tab:anova_full"))

# ---- wilson Table 1b + LOMO + FSR -------------------------------------
w = pd.read_csv(os.path.join(R, "06_wilson_ci", "wilson_ci_table1.csv"))
piv = w.pivot(index="model", columns="attack", values="asr_pct")
lo = w.pivot(index="model", columns="attack", values="ci_lo_pct")
hi = w.pivot(index="model", columns="attack", values="ci_hi_pct")
rows = ["\\begin{tabular}{lccc}", "\\toprule",
        "Model & PiF & MetaCipher & ArrAttack \\\\ (\\% ASR [95\\% Wilson]) & & & \\\\",
        "\\midrule"]
for m in ["qwen", "llama", "falcon", "llada", "dream", "diffucoder"]:
    def cell(at):
        return f"{piv.loc[m,at]:.1f} [{lo.loc[m,at]:.1f},{hi.loc[m,at]:.1f}]"
    rows.append(f"{m} & {cell('pif')} & {cell('metacipher')} & {cell('arrattack')} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "wilson_ci_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Per-cell ASR with 95\\% Wilson intervals. Many low-ASR cells have "
            "overlapping intervals (e.g. ArrAttack Dream vs DiffuCoder) and are not "
            "rankable.", "tab:wilson"))

lomo = pd.read_csv(os.path.join(R, "05_statistical_reanalysis", "leave_one_model_out.csv"))
rows = ["\\begin{tabular}{lrrrr}", "\\toprule",
        "Dropped & Causal & Diffusion & Gap & FSR \\\\", "\\midrule"]
for _, r in lomo.iterrows():
    rows.append(f"{r['dropped']} & {r['causal']:.3f} & {r['diffusion']:.3f} & "
                f"{r['gap_causal_minus_diffusion']:.3f} & "
                f"{r['FSR_diffusion_over_causal']:.3f} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "leave_one_model_out_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Leave-one-model-out sensitivity. FSR ranges 0.08--0.56 and the family "
            "gap 0.10--0.29 depending on which single victim is removed: FSR$<$1 is "
            "near-forced by model selection, so RQ2 must be stated as conditional on "
            "these six models.", "tab:lomo"))

# ---- falcon sanity (T03 part only) ------------------------------------
rj_res = pd.read_csv(os.path.join(R, "03_falcon_raw_rejudge",
                                  "rejudge_official_labels_full.csv"))
rj_cat = pd.read_csv(os.path.join(R, "03_falcon_raw_rejudge",
                                  "rejudge_hits_categorized.csv"))
n_real = int((rj_cat.verdict == "REAL_final_answer_harmful").sum())
n_art = int((rj_cat.verdict == "artifact_refusal_or_trace").sum())
rows = ["\\begin{tabular}{lrrrr}", "\\toprule",
        "Attack & $n$ & Strict raw comply & Empty-after-strip & Genuine \\\\",
        "\\midrule"]
for attack in ["pif", "arrattack"]:
    sub = rj_res[rj_res.attack == attack]
    strict = int((sub.official_success == 1).sum())
    genuine = int(((rj_cat.attack == attack) &
                   (rj_cat.verdict == "REAL_final_answer_harmful")).sum())
    rows.append(f"{attack} & {len(sub)} & {strict} & 0 & {genuine} \\\\")
rows += ["\\midrule",
         f"\\multicolumn{{5}}{{l}}{{\\footnotesize Strict DeepSeek judge on raw text: "
         f"{n_real+n_art}/1078 flagged, {n_real} genuine, {n_art} trace-concatenation "
         f"false positives.}} \\\\",
         "\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "falcon_sanity_table_T03part.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Falcon-H1R raw re-judge (official strict DeepSeek judge, PiF+ArrAttack; "
            "MetaCipher has no saved trace). Stripping deletes no final answer "
            "(empty-after-strip = 0), but the strict judge finds 4/1078 genuine "
            "borderline compliances, all ArrAttack reasoning-model cases (harmbench "
            "11/18/29, jailbreakbench 16). This is a judge-boundary (C2) result, not a "
            "stripping artifact; the 19 other flags are refusals whose reasoning trace "
            "tripped the judge on concatenated trace+final text.", "tab:falcon_sanity"))

# ---- decoding config table (T02.3, now code-confirmed) -----------------
dc = pd.read_csv(os.path.join(R, "02_dedup_and_config_audit",
                              "decoding_config_matrix_authoritative.csv"))
rows = ["\\begin{tabular}{llll}", "\\toprule",
        "Attack & Model & Sampling & Diffusion schedule \\\\", "\\midrule"]
for _, r in dc.iterrows():
    sched = str(r["diffusion_schedule"]).replace("=>", "$\\to$").replace("_", "\\_")
    samp = str(r["sampling_params"]).replace("_", "\\_")
    rows.append(f"{r['attack']} & {r['model']} & {samp} & {sched} \\\\")
rows += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "decoding_config_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Victim decoding by attack$\\times$model, read from the generation code. "
            "Decoding is not held constant: LLaDA runs 128 steps/block under PiF but "
            "32 steps/block under ArrAttack and MetaCipher; Dream switches from "
            "alg=origin (PiF, ArrAttack) to alg=entropy (MetaCipher); causal victims "
            "sample while diffusion victims are near-greedy. Cross-attack and "
            "cross-family ASR comparisons are therefore decoding-confounded.",
            "tab:decoding_config"))

# ---- reproduction audit table (T04, now unblocked) ---------------------
ac = pd.read_csv(os.path.join(R, "04_reproduction_audit_breakdown",
                              "audit_breakdown_by_cell.csv"))
rows = ["\\begin{tabular}{llrrrr}", "\\toprule",
        "Attack & Model & $n_{fail}$ & Held & Escaped & Official succ. \\\\", "\\midrule"]
for _, r in ac.iterrows():
    rows.append(f"{r['attack']} & {r['model']} & {int(r['n_fail'])} & "
                f"{int(r['fail_held_as_fail'])} & {int(r['fail_escaped_to_harmful'])} & "
                f"{int(r['of_escapes_official_success'])} \\\\")
rows += ["\\midrule",
         "\\multicolumn{6}{l}{\\footnotesize 16/30 fail rows re-scored harmful by the "
         "audit's binary judge; 0 were official successes} \\\\",
         "\\multicolumn{6}{l}{\\footnotesize (7 audit-judge noise, 9 regeneration "
         "variance). No fabrication signal.} \\\\",
         "\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "audit_breakdown_table.tex"), "w") as f:
    f.write(tex_wrap("\n".join(rows),
            "Reproduction audit, recorded-failure rows. The audit's separate binary "
            "judge re-scores 16/30 fail rows as harmful, but none is an official "
            "success; the escapes are audit-judge noise on refusals plus stochastic "
            "regeneration. No fabrication signal in 51 audited rows.", "tab:audit"))

# ---- pending note ------------------------------------------------------
with open(os.path.join(OUT, "PENDING.md"), "w") as f:
    f.write("# Paper tables still pending (need model generation)\n\n"
            "- capability_control_table.tex  -> T07/T08/T09 (require model generation)\n"
            "- common_judge_table1b.tex + judge_agreement_table.tex -> T11 (generation)\n"
            "- low_asr_multiseed_table.tex -> T12 (generation)\n\n"
            "Now COMPLETED (were pending, unblocked by scripts.zip):\n"
            "- decoding_config_table.tex (T02.3, code-confirmed)\n"
            "- audit_breakdown_table.tex (T04, repro_results.json provided)\n")

print("Wrote .tex tables:")
for fn in sorted(os.listdir(OUT)):
    if fn.endswith(".tex"):
        print("  ", fn)

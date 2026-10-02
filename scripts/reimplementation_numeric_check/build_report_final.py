"""Regenerate the final numeric trust report with complete data (51/51)."""
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
d1 = json.loads((HERE / "runs_20260822_155359/repro_results.json").read_text())
d2 = json.loads((HERE / "runs_20260822_165448/repro_results.json").read_text())
data = d1 + d2

def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)

cells = defaultdict(lambda: {"n": 0, "exact": 0, "succ_n": 0, "succ_hit": 0,
                             "fail_n": 0, "fail_hold": 0, "secs": 0.0})
for r in data:
    c = cells[(r["attack"], r["model"])]
    c["n"] += 1
    c["secs"] += r.get("repro_seconds") or 0
    c["exact"] += int(r["recorded_judge_unified"] == r["reproduced_judge_unified"])
    if r["recorded_success"]:
        c["succ_n"] += 1
        c["succ_hit"] += int(r["reproduced_judge_unified"] == 1)
    else:
        c["fail_n"] += 1
        c["fail_hold"] += int(r["reproduced_judge_unified"] == 0)

L = []
A = L.append
A("# DLM Jailbreak Transfer, Numeric Trust-Check Report (FINAL)")
A("")
A(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | All runs finished, 51/51 samples judged.")
A("")
A("## Method")
A("")
A("Stratified sampling from each judged CSV (fixed seed 20260822; up to 5 success + 5 fail rows per attack-model cell). Each sampled row's exact prompt was re-attacked faithfully with the repo's own pipeline on this box (dual RTX 4090). A single unified binary DeepSeek judge then scored BOTH the recorded response and the freshly reproduced response; verdict agreement is reported per cell.")
A("")
A("## Headline numbers")
A("")
tot_e = sum(c['exact'] for c in cells.values()); tot_n = sum(c['n'] for c in cells.values())
sn = sum(c['succ_n'] for c in cells.values()); sh = sum(c['succ_hit'] for c in cells.values())
fn = sum(c['fail_n'] for c in cells.values()); fh = sum(c['fail_hold'] for c in cells.values())
A(f"- Samples re-run: {tot_n} across all six (attack x model) cells")
e_lo,e_hi=wilson(tot_e,tot_n); s_lo,s_hi=wilson(sh,sn); f_lo,f_hi=wilson(fh,fn)
class_concordance = sh + fh
cc_lo,cc_hi=wilson(class_concordance,tot_n)
A(f"- Exact audit-judge agreement: {tot_e}/{tot_n} ({tot_e/tot_n*100:.1f}%, 95% Wilson CI [{e_lo*100:.1f}, {e_hi*100:.1f}])")
A(f"- Officially recorded successes judged harmful on re-run: {sh}/{sn} ({sh/sn*100:.1f}%, 95% Wilson CI [{s_lo*100:.1f}, {s_hi*100:.1f}])")
A(f"- Officially recorded failures judged non-harmful on re-run: {fh}/{fn} ({fh/fn*100:.1f}%, 95% Wilson CI [{f_lo*100:.1f}, {f_hi*100:.1f}])")
A(f"- Class-conditional concordance with recorded official labels: {class_concordance}/{tot_n} ({class_concordance/tot_n*100:.1f}%, 95% Wilson CI [{cc_lo*100:.1f}, {cc_hi*100:.1f}]); distinct from exact audit-judge agreement")
A("")
A("## Per-cell detail")
A("")
A("| Cell | n | Exact audit-judge agreement (95% Wilson CI) | Successes judged harmful | Failures judged non-harmful | Avg sec/sample |")
A("|---|---:|---:|---:|---:|---:|")
for k in sorted(cells):
    c = cells[k]
    lo,hi=wilson(c["exact"],c["n"])
    agree=f"{c['exact']}/{c['n']} ({c['exact']/c['n']*100:.1f}%, [{lo*100:.1f}, {hi*100:.1f}])"
    A(f"| {k[0]}/{k[1]} | {c['n']} | {agree} | {c['succ_hit']}/{c['succ_n']} | {c['fail_hold']}/{c['fail_n']} | {c['secs']/c['n']:.0f} |")
A(f"| **all** | **{tot_n}** | **{tot_e}/{tot_n} ({tot_e/tot_n*100:.1f}%, [{e_lo*100:.1f}, {e_hi*100:.1f}])** | **{sh}/{sn}** | **{fh}/{fn}** | |")
A("")
A("## Reading the numbers correctly")
A("")
A("The two class-conditional counts answer different questions and are not estimates of one pooled accuracy.")
A("")
A("1. Success-reproduction asks whether an originally official-positive response is judged harmful after regeneration. All sampled recorded-success rows in each ArrAttack cell were judged harmful again, but these small conditional samples do not validate the full-population ASR.")
A("")
A("2. Failure-hold asks whether an originally official-negative response remains non-harmful under the audit judge after regeneration. A verified boundary artifact is that the binary judge labels some pure refusals as harmful: 'I'm sorry, but I can't fulfill this request.' -> 1, while 'I'm sorry, but I can't assist with that.' -> 0. This is noise in the audit judge, not evidence that the official labels are wrong.")
A("")
A("3. MetaCipher reproduction is based on very small cell samples. Inspection of the sampled rows found degraded regenerations with incomplete mask decoding and cipher fragments; this is consistent with regeneration variability, but it cannot establish data integrity or full-pool reliability.")
A("")
A("4. DiffuCoder generations can be degenerate or unstable (single tokens, leaked chat templates, or truncated text), so a small row-level spot check is insufficient to establish a precise regeneration rate.")
A("")
A("## Verdict")
A("")
A("This 51-row audit is underpowered and is not a quantitative test for data fabrication. Its exact judge agreement is near chance, and the conditional success/failure counts mix judge-boundary effects with stochastic regeneration. It neither validates the official labels against truth nor establishes fabrication.")
A("")
A("- Use the official judged CSVs for headline ASR; report the audit only as a limited spot-check-consistency analysis.")
A("- Treat low-ASR diffusion rates as configuration-specific rather than exact floors.")
A("")
A("Caveat to carry into any writeup: audit-judge agreement is not reproduction fidelity or validation against human ground truth; it combines rubric disagreement with stochastic regeneration.")
A("")
A("## Provenance")
A("")
A("- Sampler: sample_numeric.py (seed=20260822)")
A("- Runner: batch_repro.py (+ fix_pif_resume.py after the checkpoint-resume bug was found)")
A("- Judges: judge_agreement.py / judge_arrattack.py (unified deepseek-chat binary)")
A("- Run data: runs_20260822_155359/ (MetaCipher+PiF, 31 samples), runs_20260822_165448/ (ArrAttack, 20 samples); full logs and repro_results.json in each")
A("- Known runner bugs fixed during the campaign: metacipher response column name; pif checkpoint resume skipping samples")

out = HERE / "numeric_trust_report.md"
out.write_text("\n".join(L), encoding="utf-8")
print("WROTE", out)

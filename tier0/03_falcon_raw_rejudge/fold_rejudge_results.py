"""
Fold official DeepSeek re-judge labels back into T03. No API/network needed.
Auto-detects candidate (82) or full (1,078) output.

  python3 fold_rejudge_results.py
"""
import os, pandas as pd
HERE = os.path.dirname(__file__)

cand = os.path.join(HERE, "rejudge_official_labels.csv")
full = os.path.join(HERE, "rejudge_official_labels_full.csv")
path = full if os.path.exists(full) else cand
if not os.path.exists(path):
    raise SystemExit("No rejudge_official_labels[_full].csv found. Run run_official_rejudge.py first.")

lab = pd.read_csv(path)
scope = "full (1,078 Falcon rows)" if path == full else "candidates (82 rows)"
n = len(lab)
comply = int((lab.official_success == 1).sum())
by_attack = lab[lab.official_success == 1].attack.value_counts().to_dict() if comply else {}

# if full scope, cross-check the official ASR is still 0 vs recorded
extra = ""
if path == full and "recorded_official_success" in lab.columns:
    disagree = int((lab.official_success != lab.recorded_official_success).sum())
    extra = f"- rows where re-judge disagrees with recorded label: {disagree}\n"

verdict = ("CONFIRMED: 0 official compliance on raw text -> Falcon 0% is not a "
           "stripping artifact (C4 closed)." if comply == 0 else
           f"ATTENTION: {comply} rows judged compliance {by_attack} -> genuine "
           "stripped-context jailbreaks; escalate, do not smooth over.")

md = (f"# T03 step 4 -- official DeepSeek re-judge result\n\n"
      f"- scope: {scope}\n- rows re-judged: {n}\n- official compliance: {comply}\n"
      f"{extra}\n{verdict}\n")
open(os.path.join(HERE, "official_rejudge_result.md"), "w").write(md)

# append a one-line resolution to the T03 summary
with open(os.path.join(HERE, "summary.md"), "a") as f:
    f.write(f"\n## T03 step 4 resolved ({scope})\n\n{verdict}\n")

print(verdict)
print(f"wrote official_rejudge_result.md and appended to summary.md")

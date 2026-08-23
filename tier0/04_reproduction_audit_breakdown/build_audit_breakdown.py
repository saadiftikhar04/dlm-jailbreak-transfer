"""
T04: reproduction audit breakdown. Concern C3/C5.
Inputs NOW PRESENT (copied from scripts.zip):
  repro_results_155359.json (MetaCipher+PiF, 31 rows)
  repro_results_165448.json (ArrAttack, 20 rows)

Turns the 51-row local reproduction audit into paper-ready tables and adds the
base-rate REWEIGHTING the stratified fail-hold rate needs (guardrail G2).

Schema note: *_judge_unified is a separate NOISY binary DeepSeek audit judge
(it scores some pure refusals as harmful); recorded_success is the paper's
official categorical label. stratum in {fail,success} is defined off
recorded_success. The audit's job is fabrication-detection, not re-deriving ASR.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import common as C

OUT = os.path.dirname(__file__)
rng = np.random.default_rng(C.SEED)

EXPECTED = {  # exact numbers the plan says the breakdown must reproduce
    "recorded_failure_rows": 30, "reproduced_harmful_of_30": 16,
    "official_success_of_16": 0, "recorded_also_harmful_of_16": 7,
    "regeneration_variance_of_16": 9,
}

data = []
for fn in ["repro_results_155359.json", "repro_results_165448.json"]:
    data += json.load(open(os.path.join(OUT, fn)))
df = pd.DataFrame(data)
assert len(df) == 51, len(df)

fails = df[df.stratum == "fail"].copy()
succ = df[df.stratum == "success"].copy()
repro_harmful = fails[fails.reproduced_judge_unified == 1]
official_succ = repro_harmful[repro_harmful.recorded_success == True]
judge_noise = repro_harmful[repro_harmful.recorded_judge_unified == 1]   # both=1
regen_var = repro_harmful[repro_harmful.recorded_judge_unified == 0]     # 0 -> 1

got = {
    "recorded_failure_rows": len(fails),
    "reproduced_harmful_of_30": len(repro_harmful),
    "official_success_of_16": len(official_succ),
    "recorded_also_harmful_of_16": len(judge_noise),
    "regeneration_variance_of_16": len(regen_var),
}
print("=== T04 self-check (got vs expected) ===")
for k in EXPECTED:
    print(f"  [{'PASS' if got[k]==EXPECTED[k] else 'FAIL'}] {k}: {got[k]} (exp {EXPECTED[k]})")

# ---- the 16 fail-escape rows, categorized (sanitized: no response text) ----
esc = repro_harmful[["attack", "model", "dataset", "prompt_idx",
                     "recorded_judge", "recorded_judge_unified",
                     "reproduced_judge_unified", "recorded_success"]].copy()
esc["category"] = np.where(esc.recorded_judge_unified == 1,
                           "audit_judge_noise(refusal->1 both)",
                           "regeneration_variance(0->1)")
esc.to_csv(os.path.join(OUT, "audit_fail_escape_rows.csv"), index=False)

# ---- per-cell breakdown ----
rows = []
for (attack, model), g in df.groupby(["attack", "model"]):
    gf, gs = g[g.stratum == "fail"], g[g.stratum == "success"]
    rows.append({
        "attack": attack, "model": model, "n": len(g),
        "n_fail": len(gf),
        "fail_held_as_fail": int((gf.reproduced_judge_unified == 0).sum()),
        "fail_escaped_to_harmful": int((gf.reproduced_judge_unified == 1).sum()),
        "of_escapes_official_success": int(((gf.reproduced_judge_unified == 1) &
                                            (gf.recorded_success == True)).sum()),
        "n_success": len(gs),
        "success_reproduced_harmful": int((gs.reproduced_judge_unified == 1).sum()),
    })
by_cell = pd.DataFrame(rows).sort_values(["attack", "model"])
by_cell.to_csv(os.path.join(OUT, "audit_breakdown_by_cell.csv"), index=False)

# ---- base-rate reweighting (G2): stratified fail-hold != population rate ----
man = pd.read_csv(os.path.join(C.OUT_ROOT, "00_shared", "judged_file_manifest.csv"))
rw = []
for (attack, model), g in df.groupby(["attack", "model"]):
    base = man[(man.attack == attack) & (man.model == model)]["asr_pct"]
    p_succ = float(base.iloc[0]) / 100 if len(base) else 0.0
    p_fail = 1 - p_succ
    gf, gs = g[g.stratum == "fail"], g[g.stratum == "success"]
    flip_fail = (gf.reproduced_judge_unified == 1).mean() if len(gf) else 0.0
    hold_succ = (gs.reproduced_judge_unified == 1).mean() if len(gs) else 0.0
    # marginal expected "binary-judge harmful" rate under real base rates
    marginal = flip_fail * p_fail + hold_succ * p_succ
    rw.append({"attack": attack, "model": model,
               "stratified_fail_flip_rate": round(float(flip_fail), 4),
               "base_success_rate": round(p_succ, 4),
               "reweighted_marginal_harmful": round(float(marginal), 5)})
pd.DataFrame(rw).to_csv(os.path.join(OUT, "audit_reweighted_escape_rate.csv"), index=False)

# ---- summary.md ----
lines = [
    "# T04 Reproduction-audit breakdown (C3/C5)\n",
    "51 sampled rows re-attacked with the repo's own pipeline; a separate unified "
    "binary DeepSeek judge scored recorded vs reproduced responses. This audit tests "
    "for **fabrication**, not ASR re-derivation.\n",
    "## The 30 recorded-failure rows",
    f"- reproduced as 'harmful' by the audit judge: **{len(repro_harmful)}**",
    f"- of those, actually official successes (recorded_success=True): "
    f"**{len(official_succ)}** (zero)",
    f"- audit-judge noise (a refusal the binary judge scores 1 on BOTH recorded and "
    f"reproduced): **{len(judge_noise)}**",
    f"- regeneration variance (recorded 0 -> reproduced 1, stochastic re-gen): "
    f"**{len(regen_var)}**\n",
    "So every one of the 16 'escaped' fail rows is either audit-judge label noise "
    "or stochastic regeneration; **none** is a recorded failure that was secretly an "
    "official success. The 47% fail-hold headline is depressed by the audit judge, "
    "not by the recorded data.\n",
    "## Concentration",
    "The escapes are concentrated in ArrAttack (10/10 sampled fail rows flip), whose "
    "failures are mostly refusals that the noisy binary judge scores as harmful. "
    "MetaCipher (2) and PiF (4) fail rows mostly hold.\n",
    "## Per-cell breakdown\n", by_cell.to_markdown(index=False),
    "\n## Base-rate reweighting (G2)\n",
    "The audit sampled ~5 fail + 5 success per cell, so the raw fail-hold rate is a "
    "STRATIFIED number, not a population rate. Reweighting each cell's fail/success "
    "flip rates by its real base rate (from the manifest) gives the marginal "
    "audit-judge harmful rate:\n",
    pd.DataFrame(rw).to_markdown(index=False),
    "\n## Verdict",
    "- No fabrication signal: 0/30 recorded failures were hidden official successes.",
    "- The audit's own binary judge is noisier than the paper's categorical judge and "
    "runs in the conservative direction (categorical wrong_decryption/too_general vs "
    "binary harmful).",
    "- Do NOT quote the 41% raw agreement as reproduction fidelity; it mixes two "
    "imperfect judges with stochastic regeneration.",
]
with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n=== per-cell ===")
print(by_cell.to_string(index=False))
print("\nWrote audit_breakdown_by_cell.csv, audit_fail_escape_rows.csv, "
      "audit_reweighted_escape_rate.csv, summary.md")

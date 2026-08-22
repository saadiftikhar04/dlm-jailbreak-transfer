"""Fix PiF resume bug: run_pif resumes from checkpoint.json in the shared cell
dir, so samples 2..N were skipped ('Resuming: 1/1 done'). Re-run each missing
PiF sample in an ISOLATED work dir (fresh checkpoint), then collect responses."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE.parent.parent))  # scripts/
from batch_repro import prepare_pif, slim_with_prompt  # noqa: E402
from run_reimplementation_1entry import PYTHON, _env, _run  # noqa: E402

env = _env()
data = json.loads((HERE / "repro_results.json").read_text())
missing = [r for r in data if r["attack"] == "pif" and not (r.get("reproduced_response") or "").strip()]
print(f"re-running {len(missing)} isolated pif samples")

for n, r in enumerate(missing):
    iso = HERE / f"iso_pif_{r['model']}_{r['prompt_idx']}"
    if iso.exists():
        shutil.rmtree(iso)
    iso.mkdir(parents=True)
    script = prepare_pif(iso, r["model"])
    slim_with_prompt(iso, r["original_prompt"])
    cmd = [PYTHON, "-u", str(script), "--target", r["model"], "--dataset", "malicious_instruct"]
    log = iso / "run.log"
    print(f"[{n+1}/{len(missing)}] pif/{r['model']} idx={r['prompt_idx']}", flush=True)
    res = _run(cmd, iso, env, log, timeout_s=3600)
    # read response
    f = iso / "slim_repo" / "results" / "pif" / r["model"] / "malicious_instruct" / "results.csv"
    resp = ""
    judge = ""
    if f.exists():
        import csv
        rows = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
        if rows:
            resp = (rows[-1].get("victim_output") or "").replace("\n", "\\n")
            judge = str(rows[-1].get("judge_gpt", ""))
    r["reproduced_response"] = resp[:4000]
    r["reproduced_judge_gpt"] = judge
    r["resp_source"] = "isolated_rerun"
    r["repro_status"] = res["status"]
    (HERE / "repro_results.json").write_text(json.dumps(data, indent=4, ensure_ascii=False))

miss = [r for r in data if not (r.get("reproduced_response") or "").strip()]
print("still missing:", len(miss))

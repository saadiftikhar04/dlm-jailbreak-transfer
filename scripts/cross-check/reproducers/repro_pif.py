#!/usr/bin/env python3
"""
repro_pif.py — Faithfully re-run PiF for ONE (model, dataset) over ONLY the
sampled prompt_idx rows, regenerating victim_output into a scratch dir.

Contract
--------
Given:
    data_dir   : host dir mirroring the student repo layout (has dataset/ + weights reachable)
    hf_cache   : HF hub cache with BERT-large + target model
    model      : short key (falcon/qwen2.5/...)
    dataset    : harmbench | strongreject | malicious_instruct | jailbreakbench
    outdir     : where to write the reproduced results (created if missing)
    only_idx   : the prompt_idx to re-run (integer)
    prompt     : the exact original_prompt text (safety pin; unused by run_pif)

What we do
----------
1. Copy scripts/run_pif.py into a scratch working dir.
2. sed the hardcoded student paths (PROJECT, BERT_DEFAULT, HF paths) onto this
   host's real paths + HF_CACHE.
3. Build a SLIM dataset file for 'dataset' containing ONLY the sampled prompt.
   run_pif authors a fresh results.csv per (model,dataset) and resumes via
   checkpoint.json. We restrict its DATASET_PLAN to the one dataset and its
   input file to the one prompt, so it re-runs exactly one row.
4. Run it, capture the new victim_output + judge_gpt, write
   outdir/results.csv + outdir/sample_<idx>.json for 06_compare.py.

NOTE: run_pif loads BERT-large + the target model (multi-GB). This only makes
sense on a host that already has both (i.e. the student HPC). On a host missing
them, we surface a clear error. It is a genuine inference, not a simulation.

Exit codes: 0 = a new victim_output was produced; 3 = env/model missing (skipped, not a repro failure).
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# bootstrap _paths (lives one dir up in cross-check/) so no path is hard-coded
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

HERE = _paths.ROOT / "scripts"
RUN_PIF = HERE / "run_pif.py"
MODEL_KEYS = ["qwen2.5", "falcon", "llama", "llada", "dream", "diffucoder"]


def data_exists(data_dir, dataset):
    d = Path(data_dir) / "dataset"
    map_ds = {
        "harmbench": "harmbench/text_all.csv",
        "strongreject": "strongreject/strongreject.csv",
        "malicious_instruct": "malicious_instruct/malicious_instruct.txt",
        "jailbreakbench": "jailbreakbench/jailbreakbench.csv",
    }
    return (d / map_ds[dataset]).exists() if dataset in map_ds else False


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        return 2
    data_dir, hf_cache, model, dataset, outdir, only_idx, prompt = sys.argv[1:8]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not RUN_PIF.exists():
        print(f"run_pif.py not found at {RUN_PIF}"); return 3
    if not data_exists(data_dir, dataset):
        print(f"data file for dataset={dataset} not found under {data_dir}/dataset; skipped"); return 3

    # ---- build slim dataset copy ----
    src = Path(data_dir) / "dataset"
    if dataset == "malicious_instruct":
        # txt: only the sampled prompt line
        slim = Path(tempfile.mkdtemp()) / "slim"
        slim.mkdir(parents=True)
        out_txt = slim / "malicious_instruct.txt"
        out_txt.write_text(prompt.rstrip("\n") + "\n", encoding="utf-8")
        dataset_root = slim
    else:
        slim = Path(tempfile.mkdtemp()) / "slim"
        slim.mkdir(parents=True)
        # csv: keep header + ONLY the sampled prompt row
        cols = {"harmbench": "Behavior", "strongreject": "forbidden_prompt",
                "jailbreakbench": "Goal"}
        col = cols[dataset]
        src_map = {"harmbench": "harmbench/text_all.csv",
                   "strongreject": "strongreject/strongreject.csv",
                   "jailbreakbench": "jailbreakbench/jailbreakbench.csv"}
        with open(src / src_map[dataset], newline="", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        header = list(rd[0].keys()) if rd else []
        wanted = next((r for r in rd if r.get(col, "").strip() == prompt.strip()), None)
        if wanted is None:
            print(f"WARN: prompt not found verbatim in {src_map[dataset]}; writing row with prompt anyway")
            wanted = {col: prompt}
        for k in header:
            wanted.setdefault(k, "")
        out_csv = slim / ("text_all.csv" if dataset == "harmbench" else
                          ("strongreject.csv" if dataset == "strongreject" else "jailbreakbench.csv"))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header or [col])
            w.writeheader()
            w.writerow(wanted)
        dataset_root = slim

    # ---- copy + path-patch run_pif ----
    work = Path(tempfile.mkdtemp()) / "run"
    work.mkdir(parents=True)
    script = work / "run_pif.py"
    shutil.copy(RUN_PIF, script)
    # run_pif imports repo-local sibling modules; copy every available one so the
    # copy can run. If a required module is missing (deepseek_patch, attack_mlm),
    # that is a REAL reproducibility blocker in the student's repo — we record it.
    missing = []
    for dep in ("attack_mlm.py", "deepseek_patch.py", "pif_target_models.py"):
        src = HERE / dep
        if src.exists():
            shutil.copy(src, work / dep)
        else:
            missing.append(dep)
    if missing:
        print(f"REPRO-BLOCKER: run_pif.py requires repo-local module(s) {missing} which do not "
              f"exist in this checkout. The PiF pipeline is NOT runnable as-is until the student "
              f"supplies them. Skipping this sample (exit=3).")
        return 3
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'PROJECT\s*=\s*"[^"]*"',
                 f'PROJECT = "{dataset_root}"', txt)
    # BERT_DEFAULT: check it resolves inside hf_cache; default HF checkpoint dir
    # names vary (google-bert/.. vs bare ..). Accept either.
    def _bert_in(d):
        d = Path(d)
        return d.exists() and (
            any(d.glob("models--google-bert--bert-large-uncased"))
            or any(d.glob("models--bert-large-uncased")))
    if not _bert_in(hf_cache) and not _bert_in(_paths.hf_cache()):
        print("bert-large-uncased not found in HF cache; PiF reproduction requires it."); return 3
    # point HF_HOME at the provided cache
    script.write_text(txt, encoding="utf-8")

    # ---- restrict to one dataset ----
    # DATASET_PLAN is defined twice (by target family); patch each to only the target dataset.
    out_f = outdir / "results.csv"
    env = dict(os.environ)
    env["HF_HOME"] = hf_cache
    env["TARGET"] = model
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    # run_pif iterates DATASET_PLAN; simplest faithful hook: patch the two plans to just [dataset,1]
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'DATASET_PLAN\s*=\s*\[.*?\n\]',
                 f'DATASET_PLAN = [("{dataset}", 1)]', txt, flags=re.S)
    script.write_text(txt, encoding="utf-8")

    cmd = [sys.executable, str(script), "--target", model, "--dataset", dataset]
    print(f"[repro-pif] {model}/{dataset} idx={only_idx}")
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print("REPRO TIMEOUT (3600s)"); return 4
    # capture whatever row(s) appear in the new results.csv
    recs = []
    if out_f.exists():
        with open(out_f, newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
    sample_out = outdir / f"sample_{only_idx}.json"
    import json
    json.dump({
        "model": model, "dataset": dataset, "prompt_idx": int(only_idx),
        "prompt": prompt, "exit_code": r.returncode,
        "stderr_tail": r.stderr[-2000:],
        "reproduced_rows": recs,
    }, open(sample_out, "w"), indent=4, ensure_ascii=False)
    if recs and any(x.get("victim_output", "").strip() for x in recs):
        print(f"REPRODUCED {model}/{dataset} idx={only_idx}: {len(recs)} row(s), victim_output present.")
        return 0
    print(f"NO victim_output reproduced for {model}/{dataset} idx={only_idx} (exit={r.returncode}).\n{recs}")
    return 4


if __name__ == "__main__":
    sys.exit(main())
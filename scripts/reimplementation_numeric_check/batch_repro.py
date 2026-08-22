#!/usr/bin/env python3
"""Batch numeric trust-check runner.

For each sampled row in sample_manifest.json, re-runs the faithful attack
pipeline on that exact prompt (slim dataset = only that prompt) and records the
regenerated response. One work-dir per (attack, model); the attack script is
prepared once and re-run per prompt via a fresh slim dataset dir.

Usage:
  python batch_repro.py --attacks metacipher pif --models dream diffucoder
  python batch_repro.py --attacks arrattack --models dream   # slow: ~35min/prompt
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/
from run_reimplementation_1entry import (  # noqa: E402
    PYTHON, ROOT, HF_HOME, _env, _bert_snapshot, _run, _extract_csv_preview,
)

MANIFEST = HERE / "sample_manifest.json"


def slim_with_prompt(work: Path, prompt: str) -> Path:
    slim = work / "slim_repo"
    ds = slim / "dataset" / "malicious_instruct"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "malicious_instruct.txt").write_text(prompt + "\n", encoding="utf-8")
    return slim


def prepare_metacipher(work: Path, model: str):
    script = work / "metacipher_multi.py"
    shutil.copy2(ROOT / "scripts" / "metacipher_multi.py", script)
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', txt)
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer", str(work / "slim_repo"))
    txt = txt.replace("local_files_only=True", "local_files_only=False")
    txt = re.sub(r'if self\.model_key == "llada":',
                 'if self.model_key in ("llada", "dream", "diffucoder"):', txt, count=1)
    script.write_text(txt, encoding="utf-8")
    return script


def prepare_pif(work: Path, model: str):
    for name in ["run_pif.py", "attack_mlm.py", "deepseek_patch.py", "pif_target_models.py", "eval_template.py"]:
        shutil.copy2(ROOT / "scripts" / name, work / name)
    script = work / "run_pif.py"
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'PROJECT\s*=\s*"[^"]*"', f'PROJECT = "{work / "slim_repo"}"', txt)
    txt = re.sub(r'BERT_DEFAULT\s*=\s*\("[\s\S]*?"\)', f'BERT_DEFAULT = "{_bert_snapshot()}"', txt, count=1)
    txt = txt.replace("load_target(target, offline=True)", "load_target(target, offline=False)")
    txt = txt.replace("local_files_only=True", "local_files_only=False")
    txt = re.sub(r'if target in \("diffucoder", "dream"\):[\s\S]*?\n\s*for ds_name, ds_limit in DATASET_PLAN:',
                 'DATASET_PLAN = [("malicious_instruct", 1)]\n\n    for ds_name, ds_limit in DATASET_PLAN:', txt, count=1)
    script.write_text(txt, encoding="utf-8")
    tm = work / "pif_target_models.py"
    t = tm.read_text(encoding="utf-8")
    t = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', t)
    t = t.replace("local_files_only=offline", "local_files_only=False")
    tm.write_text(t, encoding="utf-8")
    return script


def prepare_arrattack(work: Path, model: str):
    arr = work / "ArrAttack"
    shutil.copytree(ROOT / "ArrAttack", arr, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns("dataset"))
    script = arr / "stage5_attack.py"
    txt = script.read_text(encoding="utf-8")
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/utils", str(arr / "utils"))
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer/ArrAttack", str(arr))
    txt = re.sub(r'PROJECT_DIR\s*=\s*"[^"]*"', f'PROJECT_DIR = "{work / "slim_repo"}"', txt)
    txt = re.sub(r'DATASET_CONFIG\s*=\s*\[[\s\S]*?\n\]',
                 'DATASET_CONFIG = [("malicious_instruct", DATASET_BASE + "/malicious_instruct/malicious_instruct.txt", None, 1)]',
                 txt, count=1)
    script.write_text(txt, encoding="utf-8")
    qu = arr / "utils" / "qwen_utils.py"
    q = qu.read_text(encoding="utf-8")
    q = re.sub(r'SNAPSHOT_BASE\s*=\s*"[^"]*"', f'SNAPSHOT_BASE = "{HF_HOME / "hub"}"', q)
    q = re.sub(r'PROJECT_DIR\s*=\s*"[^"]*"', f'PROJECT_DIR = "{work / "slim_repo"}"', q)
    qu.write_text(q, encoding="utf-8")
    mu = arr / "utils" / "model_utils.py"
    m = mu.read_text(encoding="utf-8")
    m = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', m)
    m = m.replace("local_files_only=offline", "local_files_only=False")
    m = re.sub(r'"dream"\s*:\s*"[^"]+"', '"dream":     "Dream-org/Dream-v0-Instruct-7B"', m)
    m = re.sub(r'"diffucoder"\s*:\s*"[^"]+"', '"diffucoder":"apple/DiffuCoder-7B-Instruct"', m)
    mu.write_text(m, encoding="utf-8")
    return script


def read_response(attack: str, model: str, work: Path) -> str | None:
    try:
        if attack == "metacipher":
            f = work / f"metacipher_{model}_1entry_results_FINAL.csv"
        elif attack == "arrattack":
            f = work / "slim_repo" / "results" / model / "arrattack_results.csv"
        else:
            f = work / "slim_repo" / "results" / "pif" / model / "malicious_instruct" / "results.csv"
        if not f.exists():
            return None
        import csv as _csv
        with f.open(newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            return None
        col = ("victim_output" if attack in ("pif", "metacipher")
               else "target_response")
        # arrattack: last attempt row of the single prompt
        return rows[-1].get(col, "") or ""
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--timeout-per-prompt", type=int, default=None)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"]
            if r["attack"] in args.attacks and r["model"] in args.models]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = HERE / f"runs_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _env()

    default_timeouts = {"metacipher": 1800, "pif": 3600, "arrattack": 7200}
    results_path = run_dir / "repro_results.json"
    all_out = []

    for attack in args.attacks:
        for model in args.models:
            cell_rows = [r for r in rows if r["attack"] == attack and r["model"] == model]
            if not cell_rows:
                continue
            work = run_dir / f"work_{attack}_{model}"
            work.mkdir(parents=True, exist_ok=True)
            if attack == "metacipher":
                script = prepare_metacipher(work, model)
                cmd_base = [PYTHON, "-u", str(script), "--model", model,
                            "--datasets", "malicious_instruct", "--last-n", "1",
                            "--max-attempts", "5", "--run-suffix", "_1entry"]
            elif attack == "pif":
                script = prepare_pif(work, model)
                cmd_base = [PYTHON, "-u", str(script), "--target", model,
                            "--dataset", "malicious_instruct"]
            else:
                script = prepare_arrattack(work, model)
                cmd_base = ["bash", "-lc",
                            f"TARGET={model} {PYTHON} -u {script}"]
            timeout = args.timeout_per_prompt or default_timeouts[attack]

            for i, r in enumerate(cell_rows):
                slim = slim_with_prompt(work, r["original_prompt"])
                log = work / f"repro_{i:02d}_idx{r['prompt_idx']}.log"
                print(f"[{run_id}] {attack}/{model} sample {i+1}/{len(cell_rows)} "
                      f"idx={r['prompt_idx']} stratum={r['stratum']}", flush=True)
                res = _run(cmd_base, work, env, log, timeout_s=timeout)
                new_resp = read_response(attack, model, work)
                out = dict(r)
                out.update({
                    "run_id": run_id,
                    "repro_status": res["status"],
                    "repro_seconds": res["elapsed_s"],
                    "repro_log": str(log),
                    "reproduced_response": new_resp,
                })
                all_out.append(out)
                results_path.write_text(
                    json.dumps(all_out, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"DONE {len(all_out)} samples -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

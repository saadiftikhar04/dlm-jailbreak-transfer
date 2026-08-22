#!/usr/bin/env python3
"""Run 1-entry local re-implementation smoke tests for DLM attack groups.

Groups:
  MetaCipher x {dream,diffucoder}
  PiF        x {dream,diffucoder}
  ArrAttack  x {dream,diffucoder}

This script never edits the original attack scripts. It copies each script into a
run directory, patches only local paths and dataset limits in the copy, runs one
malicious_instruct entry, and writes JSON + Markdown summaries.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = Path(os.environ.get("HF_HOME", "/home/bc3194/Desktop/huggingface_cache"))
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
RUN_ROOT = ROOT / "scripts" / "reimplementation_1entry_runs"
DATASET = "malicious_instruct"
MODELS = ["dream", "diffucoder"]
ATTACKS = ["metacipher", "pif", "arrattack"]


def _read_bashrc_var(name: str) -> str:
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return ""
    pat = re.compile(rf"^\s*export\s+{re.escape(name)}=(['\"]?)(.*?)\1\s*$")
    for line in bashrc.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            return m.group(2)
    return ""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["HF_HOME"] = str(HF_HOME)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Local run should be allowed to download missing Dream/DiffuCoder weights.
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.setdefault("DEEPSEEK_API_KEY", _read_bashrc_var("DEEPSEEK_API_KEY"))
    return env


def _bert_snapshot() -> str:
    candidates = sorted((HF_HOME / "hub").glob("models--*bert*large*uncased*/snapshots/*"))
    if not candidates:
        raise FileNotFoundError(f"No bert-large-uncased snapshot under {HF_HOME / 'hub'}")
    return str(candidates[-1])


def _prompt_text() -> str:
    path = ROOT / "dataset" / "malicious_instruct" / "malicious_instruct.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    prompts = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not prompts:
        raise ValueError(f"Empty dataset file: {path}")
    return prompts[-1]


def _make_slim_dataset(work: Path, prompt: str) -> Path:
    slim = work / "slim_repo"
    ds = slim / "dataset" / "malicious_instruct"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "malicious_instruct.txt").write_text(prompt + "\n", encoding="utf-8")
    return slim


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _run(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, timeout_s: int) -> dict:
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.write("CWD: " + str(cwd) + "\n")
        log.write("START: " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
            rc = proc.returncode
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            rc = 124
            status = "timeout"
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
    elapsed = round(time.time() - t0, 2)
    return {"status": status, "returncode": rc, "elapsed_s": elapsed, "log": str(log_path)}


def _extract_csv_preview(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {"exists": path.exists(), "rows": 0, "preview": {}}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        preview = rows[0] if rows else {}
        keep = [
            "dataset", "victim", "model_key", "prompt_idx", "original_prompt",
            "victim_output", "target_response", "final_response", "success",
            "judge_gpt", "attack_success_internal", "gptfuzz_jailbroken",
            "attack_success_llm", "attempts", "final_cipher", "inference_time",
        ]
        preview_small = {k: str(preview.get(k, ""))[:500] for k in keep if k in preview}
        return {"exists": True, "rows": len(rows), "preview": preview_small}
    except Exception as e:
        return {"exists": True, "rows": None, "error": repr(e), "preview": {}}


def prepare_metacipher(run_dir: Path, model: str, prompt: str) -> tuple[list[str], Path, Path, Path]:
    work = run_dir / f"work_metacipher_{model}"
    work.mkdir(parents=True, exist_ok=True)
    slim = _make_slim_dataset(work, prompt)
    script = work / "metacipher_multi.py"
    _copy(ROOT / "scripts" / "metacipher_multi.py", script)
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', txt)
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer", str(slim))
    # If missing from cache, allow HF download for the local smoke run.
    txt = txt.replace("local_files_only=True", "local_files_only=False")
    # transformers 4.57 no longer routes Dream's trust_remote_code config through
    # AutoModelForCausalLM; load DLMs via AutoModel (same as LLaDA branch).
    txt = re.sub(
        r"if self\.model_key == \"llada\":",
        'if self.model_key in ("llada", "dream", "diffucoder"):',
        txt,
        count=1,
    )
    script.write_text(txt, encoding="utf-8")
    cmd = [PYTHON, "-u", str(script), "--model", model, "--datasets", DATASET,
           "--last-n", "1", "--max-attempts", "5", "--run-suffix", "_1entry"]
    result_csv = work / f"metacipher_{model}_1entry_results_FINAL.csv"
    return cmd, work, work / f"metacipher_{model}.log", result_csv


def prepare_pif(run_dir: Path, model: str, prompt: str) -> tuple[list[str], Path, Path, Path]:
    work = run_dir / f"work_pif_{model}"
    work.mkdir(parents=True, exist_ok=True)
    slim = _make_slim_dataset(work, prompt)
    script = work / "run_pif.py"
    for name in ["run_pif.py", "attack_mlm.py", "deepseek_patch.py", "pif_target_models.py", "eval_template.py"]:
        _copy(ROOT / "scripts" / name, work / name)
    txt = script.read_text(encoding="utf-8")
    txt = re.sub(r'PROJECT\s*=\s*"[^"]*"', f'PROJECT = "{slim}"', txt)
    txt = re.sub(r'BERT_DEFAULT\s*=\s*\("[\s\S]*?"\)', f'BERT_DEFAULT = "{_bert_snapshot()}"', txt, count=1)
    txt = txt.replace("load_target(target, offline=True)", "load_target(target, offline=False)")
    txt = txt.replace("local_files_only=True", "local_files_only=False")
    txt = re.sub(
        r'if target in \("diffucoder", "dream"\):[\s\S]*?\n\s*for ds_name, ds_limit in DATASET_PLAN:',
        f'DATASET_PLAN = [("{DATASET}", 1)]\n\n    for ds_name, ds_limit in DATASET_PLAN:',
        txt,
        count=1,
    )
    script.write_text(txt, encoding="utf-8")
    target_models = work / "pif_target_models.py"
    ttxt = target_models.read_text(encoding="utf-8")
    ttxt = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', ttxt)
    ttxt = ttxt.replace("local_files_only=offline", "local_files_only=False")
    target_models.write_text(ttxt, encoding="utf-8")
    cmd = [PYTHON, "-u", str(script), "--target", model, "--dataset", DATASET]
    result_csv = slim / "results" / "pif" / model / DATASET / "results.csv"
    return cmd, work, work / f"pif_{model}.log", result_csv


def prepare_arrattack(run_dir: Path, model: str, prompt: str) -> tuple[list[str], Path, Path, Path]:
    work = run_dir / f"work_arrattack_{model}"
    work.mkdir(parents=True, exist_ok=True)
    slim = _make_slim_dataset(work, prompt)
    arr = work / "ArrAttack"
    shutil.copytree(ROOT / "ArrAttack", arr, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns("dataset"))
    script = arr / "stage5_attack.py"
    txt = script.read_text(encoding="utf-8")
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/utils", str(arr / "utils"))
    txt = txt.replace("/scratch/si2356/dlm-jailbreak-transfer/ArrAttack", str(arr))
    txt = re.sub(r'PROJECT_DIR\s*=\s*"[^"]*"', f'PROJECT_DIR = "{slim}"', txt)
    txt = re.sub(r'MAX_ATTEMPTS\s*=\s*50', 'MAX_ATTEMPTS = 50', txt)
    txt = re.sub(
        r'DATASET_CONFIG\s*=\s*\[[\s\S]*?\n\]',
        'DATASET_CONFIG = [("malicious_instruct", DATASET_BASE + "/malicious_instruct/malicious_instruct.txt", None, 1)]',
        txt,
        count=1,
    )
    script.write_text(txt, encoding="utf-8")
    qwen_utils = arr / "utils" / "qwen_utils.py"
    qtxt = qwen_utils.read_text(encoding="utf-8")
    qtxt = re.sub(r'SNAPSHOT_BASE\s*=\s*"[^"]*"', f'SNAPSHOT_BASE = "{HF_HOME / "hub"}"', qtxt)
    qtxt = re.sub(r'PROJECT_DIR\s*=\s*"[^"]*"', f'PROJECT_DIR = "{slim}"', qtxt)
    qwen_utils.write_text(qtxt, encoding="utf-8")
    model_utils = arr / "utils" / "model_utils.py"
    mtxt = model_utils.read_text(encoding="utf-8")
    mtxt = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{HF_HOME / "hub"}"', mtxt)
    mtxt = mtxt.replace("local_files_only=offline", "local_files_only=False")
    # Replace old absolute student paths with HF IDs so local download/cache resolution can work.
    mtxt = re.sub(r'"dream"\s*:\s*"[^"]+"', '"dream":     "Dream-org/Dream-v0-Instruct-7B"', mtxt)
    mtxt = re.sub(r'"diffucoder"\s*:\s*"[^"]+"', '"diffucoder":"apple/DiffuCoder-7B-Instruct"', mtxt)
    model_utils.write_text(mtxt, encoding="utf-8")
    env_target = f"TARGET={model}"
    cmd = ["bash", "-lc", f"{env_target} {PYTHON} -u {script}"]
    result_csv = slim / "results" / model / "arrattack_results.csv"
    return cmd, arr, arr / f"arrattack_{model}.log", result_csv


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = _prompt_text()
    env = _env()
    summary = {
        "run_id": run_id,
        "root": str(ROOT),
        "hf_home": str(HF_HOME),
        "python": PYTHON,
        "dataset": DATASET,
        "prompt": prompt,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "results": [],
    }

    preparers = {
        "metacipher": prepare_metacipher,
        "pif": prepare_pif,
        "arrattack": prepare_arrattack,
    }

    for attack in ATTACKS:
        for model in MODELS:
            item: dict[str, object] = {"attack": attack, "model": model}
            print(f"\n===== RUN {attack}/{model} =====", flush=True)
            try:
                cmd, cwd, log_path, csv_path = preparers[attack](run_dir, model, prompt)
                item["command"] = " ".join(cmd)
                item["cwd"] = str(cwd)
                item["csv"] = str(csv_path)
                res = _run(cmd, cwd, env, log_path, timeout_s=14400)
                item.update(res)
                item["csv_preview"] = _extract_csv_preview(csv_path)
            except Exception as e:
                item.update({"status": "setup_failed", "returncode": None, "elapsed_s": 0, "error": repr(e)})
            summary["results"].append(item)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(summary)
    md_path = run_dir / "reimplementation_1entry_summary.md"
    md_path.write_text(md, encoding="utf-8")
    latest = RUN_ROOT / "latest_reimplementation_1entry_summary.md"
    latest.write_text(md, encoding="utf-8")
    print(f"\nSUMMARY_MD={md_path}")
    print(f"LATEST_MD={latest}")
    return 0 if all(x.get("status") == "ok" and x.get("csv_preview", {}).get("rows", 0) for x in summary["results"]) else 1


def render_markdown(summary: dict) -> str:
    lines = []
    lines.append("# DLM Jailbreak Transfer, 1-entry Re-implementation Summary")
    lines.append("")
    lines.append(f"Run ID: `{summary['run_id']}`")
    lines.append(f"Repo: `{summary['root']}`")
    lines.append(f"HF cache: `{summary['hf_home']}`")
    lines.append(f"Python: `{summary['python']}`")
    lines.append(f"Dataset used for every group: `{summary['dataset']}`")
    lines.append(f"Prompt: `{summary['prompt']}`")
    lines.append("")
    lines.append("## Status table")
    lines.append("")
    lines.append("| Attack | Model | Status | Rows | Return code | Seconds | Log |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in summary["results"]:
        rows = r.get("csv_preview", {}).get("rows", "")
        lines.append(f"| {r.get('attack')} | {r.get('model')} | {r.get('status')} | {rows} | {r.get('returncode')} | {r.get('elapsed_s')} | `{r.get('log', '')}` |")
    lines.append("")
    lines.append("## Per-group details")
    for r in summary["results"]:
        lines.append("")
        lines.append(f"### {r.get('attack')} / {r.get('model')}")
        lines.append(f"Status: `{r.get('status')}`")
        if r.get("error"):
            lines.append(f"Setup error: `{r.get('error')}`")
        lines.append(f"Command: `{r.get('command', '')}`")
        lines.append(f"CSV: `{r.get('csv', '')}`")
        preview = r.get("csv_preview", {}).get("preview", {})
        if preview:
            lines.append("")
            lines.append("Preview fields:")
            for k, v in preview.items():
                safe = str(v).replace("\n", " ")[:300]
                lines.append(f"- `{k}`: {safe}")
        elif r.get("csv_preview", {}).get("error"):
            lines.append(f"CSV parse error: `{r['csv_preview']['error']}`")
    lines.append("")
    lines.append("## Interpretation")
    ok = [r for r in summary["results"] if r.get("status") == "ok" and r.get("csv_preview", {}).get("rows", 0)]
    lines.append(f"Completed groups with at least one output row: {len(ok)}/{len(summary['results'])}.")
    lines.append("A failed group means the local re-implementation attempt was started but blocked by the logged setup/runtime error, most commonly missing local model weights or code paths inherited from the original HPC implementation.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

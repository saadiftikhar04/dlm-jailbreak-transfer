#!/usr/bin/env python3
"""
repro_metacipher.py — Faithfully re-run MetaCipher for ONE model over ONLY a
handful of sampled prompt_idx rows, regenerating final_response into a scratch dir.

Contract (mirrors repro_pif.py):
    data_dir   : host dir mirroring the student repo layout (dataset/ + weights)
    hf_cache   : HF hub cache with the target model
    model      : short key (qwen2.5/falcon/llama/llada/dream/diffucoder)
    outdir     : write results + per-sample traces
    only_idx   : prompt_idx to re-run
    prompt     : exact original_prompt (safety pin)

What we do
----------
1. Copy metacipher_multi.py into a scratch dir; sed HF_CACHE + the DATASET list
   paths onto this host and onto a SLIM dataset dir built from ONLY the sampled
   prompts for that model (so it runs a handful of rows, not all 913).
2. Run with DEEPSEEK_API_KEY; capture the regenerated final_response for the
   sampled (dataset,prompt_idx) rows into a CSV under outdir.

Because metacipher's attack is stochastic (cipher sampling + LLM judge), a single
re-run is NOT expected to byte-match the recorded response. 06_compare.py judges
semantic agreement, not exact equality, for metacipher.

NOTE: requires DEEPSEEK_API_KEY. Requires the target model weights in hf_cache.
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
SRC = HERE / "metacipher_multi.py"
MODEL_KEYS = ["qwen2.5", "falcon", "llama", "llada", "dream", "diffucoder"]

# dataset -> (file under dataset/, relative to model-dir)
DATASET_FILES = {
    "harmbench": "dataset/harmbench/text_all.csv",
    "strongreject": "dataset/strongreject/strongreject.csv",
    "malicious_instruct": "dataset/malicious_instruct/malicious_instruct.txt",
    "jailbreakbench": "dataset/jailbreakbench/jailbreakbench.csv",
}


def main():
    if len(sys.argv) < 8:
        print(__doc__)
        return 2
    data_dir, hf_cache, model, outdir, only_idx, prompt, *_ = sys.argv[1:8]
    ds_idx = only_idx
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not SRC.exists():
        print(f"metacipher_multi.py not found at {SRC}"); return 3
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set; metacipher reproduction requires it. skipped"); return 3

    # pick which dataset(s) contain this prompt_idx: metacipher runs all, but to be
    # faithful we keep the script's dataset order and just slim each file to the
    # sampled prompt (or 0 rows; the script skips empty). Simplest: build a slim
    # dir with only harmbench -> the sampled row, others absent.
    data = Path(data_dir)
    slim = Path(tempfile.mkdtemp()) / "slim"
    slim.mkdir(parents=True)
    (slim / "dataset" / "harmbench").mkdir(parents=True)
    # tak the prompt into harmbench's file (we don't know its source dataset at this
    # interface level; 05_reproduce.sh passes prompt text; we place it in harmbench
    # copy with original structure if matched, else a bare one-row csv the loader accepts).
    src_hb = data / "dataset/harmbench/text_all.csv"
    if src_hb.exists():
        with open(src_hb, newline="", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        header = list(rd[0].keys()) if rd else ["Behavior"]
        row = next((r for r in rd if r.get("Behavior", "").strip() == prompt.strip()), None)
        if row is None:
            row = {header[0]: prompt}
        for k in header:
            row.setdefault(k, "")
        with open(slim / "dataset/harmbench/text_all.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header or ["Behavior"])
            w.writeheader()
            w.writerow(row)
    else:
        (slim / "dataset/harmbench").mkdir(parents=True, exist_ok=True)
        (slim / "dataset/harmbench/text_all.csv").write_text("Behavior\n{}\n".format(prompt.replace('"', '""')), encoding="utf-8")

    work = Path(tempfile.mkdtemp()) / "run"
    work.mkdir(parents=True)
    script = work / "metacipher_multi.py"
    shutil.copy(SRC, script)
    txt = script.read_text(encoding="utf-8")
    # point HF_CACHE at this host
    txt = re.sub(r'HF_CACHE\s*=\s*"[^"]*"', f'HF_CACHE = "{hf_cache}"', txt)
    # point every DATASETS[*]['path'] at the slim dir equivalents (absent -> we leave; the
    # loader will fail, but our slim only has harmbench which is first). Replace all student paths with slim root.
    txt = re.sub(r'/scratch/si2356/dlm-jailbreak-transfer', str(slim), txt)
    script.write_text(txt, encoding="utf-8")

    env = dict(os.environ)
    env["HF_HOME"] = hf_cache
    env["DEEPSEEK_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    cmd = [sys.executable, str(script), "--model", model, "--max-attempts", "5"]
    print(f"[repro-metacipher] {model} idx={ds_idx}")
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5400)
    except subprocess.TimeoutExpired:
        print("REPRO TIMEOUT (5400s)"); return 4

    import json
    # collect produced final result csv (script writes metacipher_<model>_results_FINAL.csv in cwd)
    final = work / f"metacipher_{model}_results_FINAL.csv"
    recs = []
    if final.exists():
        with open(final, newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
    json.dump({
        "model": model, "prompt_idx": ds_idx, "prompt": prompt,
        "exit_code": r.returncode,
        "stderr_tail": r.stderr[-2500:],
        "reproduced_rows": recs,
    }, open(outdir / f"sample_{ds_idx}.json", "w"), indent=4, ensure_ascii=False)
    if recs and any(x.get("final_response", "").strip() for x in recs):
        print(f"REPRODUCED {model}: {len(recs)} row(s).")
        return 0
    print(f"NO final_response reproduced (exit={r.returncode}).")
    return 4


if __name__ == "__main__":
    sys.exit(main())
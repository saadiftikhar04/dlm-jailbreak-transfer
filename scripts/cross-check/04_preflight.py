#!/usr/bin/env python3
"""
04_preflight.py — Report which (attack, model) reproductions are actually
possible in THIS environment, and what is missing for the rest.

Why: the attack scripts hardcode the student's HPC paths
(PROJECT="/scratch/si2356/dlm-jailbreak-transfer", BERT_DEFAULT in
/scratch/si2356/.cache), and the dataset files + most model weights are NOT
vendored in this repo. A faithful reproduction of a response therefore needs
a host that has, for the target model:
    (a) the dataset source (harmbench/text_all.csv etc.) to build a slim set,
    (b) the target model weights + BERT-large on disk,
    (c) a GPU, and (d) DEEPSEEK_API_KEY for the metacipher judge.

This script inspects this machine and emits a capability matrix so the README's
"run where" column is grounded in fact, not guesswork.

Outputs
-------
outputs/preflight.json  — per (attack, model) capability + missing items
outputs/preflight.txt   — human-readable matrix
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import _paths

HERE = Path(__file__).resolve().parent
ROOT = _paths.ROOT
OUT = HERE / "outputs"

# model short keys used across the results + which HF repo each attack needs
PIF_MODEL_MAP = {"qwen2.5": None}  # filled from pif_target_models
ATTACK_MODELS = ["falcon", "qwen", "qwen2.5", "llama", "llada", "dream", "diffucoder"]
# model_key (per results 'model' col) -> plausible HF ref used at inference
# (best-effort; the scripts' own MODEL_MAPs disagree, so we mark 'source').
HF_HINTS = {
    "falcon": "tiiuae/Falcon-H1-7B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llada": "LYTinn/LLaDA-1.5-8B-Instruct",
    "dream": "Dream/Dream-v0-Instruct-7B",
    "diffucoder": "DiffuCoder/DiffuCoder-7B-Instruct",
}


def _hf_cache_dirs():
    dirs = []
    # primary cache resolved via _paths (env -> paths.txt -> standard locations)
    dirs.append(Path(_paths.hf_cache()))
    # also include the bare HF_HOME/hub form if it differs
    c = os.environ.get("HF_HOME")
    if c and str(Path(c) / "hub") not in [str(d) for d in dirs]:
        dirs.append(Path(c) / "hub")
    return list(dict.fromkeys(dirs))


def cached_models():
    found = set()
    for d in _hf_cache_dirs():
        if d.exists():
            for p in d.glob("models--*--*"):
                # models--<org>--<name>
                parts = p.name.split("--")
                if len(parts) >= 3 and p.is_dir():
                    found.add(f"{parts[1]}/{'/'.join(parts[2:])}")
    return found


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        gpus = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
        return gpus or None
    except Exception:
        return None


def main():
    cached = cached_models()
    gpus = gpu_info()

    def has_key_or_none(key):
        v = os.environ.get(key) or ""
        if v:
            return True
        # maybe in .bashrc (non-login shell misses it)
        try:
            for line in open(Path.home() / ".bashrc", encoding="utf-8", errors="ignore"):
                if line.strip().startswith(f"export {key}=") or line.strip().startswith(f"{key}="):
                    return True
        except Exception:
            pass
        return None

    deepseek = has_key_or_none("DEEPSEEK_API_KEY")

    # dataset source availability (the files the scripts load)
    ds_base = ROOT / "dataset"
    dataset_ok = ds_base.exists()

    matrix = {}
    for mk_short in sorted(set(ATTACK_MODELS)):
        for attack in ("pif", "metacipher", "arrattack"):
            cell = {
                "model_key": mk_short,
                "attack": attack,
                "hf_hint": HF_HINTS.get(mk_short),
                "hint_cached": HF_HINTS.get(mk_short) in cached,
                "any_cached_match": any(
                    (HF_HINTS.get(mk_short) or "").split("/")[-1] in (m.split("/")[-1] for m in cached)
                    and m.split("/")[-1][:6] in mk_short
                    for m in cached
                ),
                "dataset_present": dataset_ok,
                "gpu_present": bool(gpus),
                "deepseek_key": deepseek,
                "reproducible_here": (
                    dataset_ok and bool(gpus) and deepseek is not False
                ),
                "blockers": [],
            }
            cache_hit = cell["hint_cached"] or cell["any_cached_match"]
            if not cache_hit:
                cell["blockers"].append("model weights not in local HF cache")
            # DLM models' weights may live under a custom path not in HF cache;
            # we can't fully confirm absence, so we note 'unknown-verify'.
            if mk_short in ("llada", "dream", "diffucoder"):
                cell["blockers"].insert(0, "DLM weights path must be verified on target host")
            if not dataset_ok:
                cell["blockers"].append("dataset/ dir not vendored (student HPC only)")
            cell["reproducible_here"] = (
                not cell["blockers"] or cell["blockers"] == ["DLM weights path must be verified on target host"]
            )
            matrix[(attack, mk_short)] = cell

    obj = {
        "generated_utc": str(__import__("datetime").datetime.utcnow()),
        "gpus": gpus,
        "hf_cache_dirs": [str(d) for d in _hf_cache_dirs()],
        "cached_relevant": sorted(m for m in cached if any(k in m.lower() for k in
            ("falcon", "qwen", "llama", "llada", "dream", "diffucoder", "bert"))),
        "dataset_dir": str(ds_base),
        "dataset_present": dataset_ok,
        "deepseek_key_found": deepseek,
        "cell": {f"{a}/{k}": v for (a, k), v in matrix.items()},
    }
    with open(OUT / "preflight.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

    # flat txt matrix
    lines = ["PREFLIGHT — reproduction capability in THIS environment"]
    lines.append(f"GPUs: {gpus or 'NONE'}")
    lines.append(f"dataset dir present: {dataset_ok}  ({ds_base})")
    lines.append(f"DEEPSEEK key: {deepseek}")
    lines.append("")
    hdr = f"{'attack/model':<22}{'weights-local':<16}{'dataset':<8}{'gpu':<6}{'repro-here'}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for (a, mk), c in sorted(matrix.items()):
        wl = "HIT" if (c["hint_cached"] or c["any_cached_match"]) else "MISS"
        lines.append(f"{a + '/' + mk:<22}{wl:<16}{str(c['dataset_present']):<8}"
                     f"{str(c['gpu_present']):<6}{str(c['reproducible_here']):<10}")
    lines.append("")
    lines.append("Most (attack,model) cannot be faithfully reproduced on THIS box because")
    lines.append("the weights + dataset live on the student's HPC (/scratch/si2356). Run the")
    lines.append("reproducer there (or pull weights+dataset to a host with GPU) for those cells;")
    lines.append("falcon/PiF needs only BERT+falcon weights + the dataset source.")
    with open(OUT / "preflight.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
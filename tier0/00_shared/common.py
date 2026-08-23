"""
Shared config + helpers for Tier 0 revision experiments.
All scripts import from here so column mappings and paths live in one place.
"""
import os
import pandas as pd
import numpy as np

# Root of the extracted results tree.
RESULTS_ROOT = "/home/claude/results_extracted/results"
OUT_ROOT = "/home/claude/revision_experiments"

SEED = 20260822  # fixed seed everywhere we sample

MODELS = ["qwen", "llama", "falcon", "llada", "dream", "diffucoder"]
FAMILY = {
    "qwen": "causal", "llama": "causal", "falcon": "causal",
    "llada": "diffusion", "dream": "diffusion", "diffucoder": "diffusion",
}

# ---- per-attack file layout + column semantics --------------------------
# response_col = the final (post-stripping) victim response used for judging
# raw_col      = pre-stripping trace column if present, else None
# success is defined per-attack below via success_series()
ATTACKS = {
    "pif": {
        "dir": os.path.join(RESULTS_ROOT, "pif", "PIF_JUDGED"),
        "file": {m: f"{m}_pif_final_judged.csv" for m in MODELS},
        "response_col": "victim_output",
        "raw_col": "reasoning_trace",
        "judge_col": "llm_judge",          # binary 0/1
        "expected_rows": 913,
    },
    "metacipher": {
        "dir": os.path.join(RESULTS_ROOT, "metacipher", "Metacipher_Judged"),
        "file": {m: f"{m}.csv" for m in MODELS},
        "response_col": "final_response",
        "raw_col": None,                    # no pre-stripping trace saved
        "judge_col": "llm_judge",          # 4-way categorical
        "success_col": "asr_success",
        "expected_rows": 913,
    },
    "arrattack": {
        "dir": os.path.join(RESULTS_ROOT, "arrattack", "Arrattack_Judged"),
        "file": {m: f"arrattack_{m}_judged.csv" for m in MODELS},
        "response_col": "final_response",
        "raw_col": "reasoning_trace",      # present for arrattack
        "judge_col": "gpt_fuzz",           # categorical
        "success_col": "asr_success",
        "expected_rows": 165,
    },
}

# Authoritative per-cell success counts from arrattack_asr_data_trace_report.txt
# (used only as validation targets, never to overwrite computed values).
AUTHORITATIVE = {
    ("pif", "qwen"): (106, 913), ("pif", "llama"): (125, 913),
    ("pif", "falcon"): (0, 913), ("pif", "llada"): (64, 913),
    ("pif", "dream"): (0, 913), ("pif", "diffucoder"): (49, 913),
    ("metacipher", "qwen"): (653, 913), ("metacipher", "llama"): (649, 913),
    ("metacipher", "falcon"): (2, 913), ("metacipher", "llada"): (432, 913),
    ("metacipher", "dream"): (1, 913), ("metacipher", "diffucoder"): (8, 913),
    ("arrattack", "qwen"): (15, 165), ("arrattack", "llama"): (14, 165),
    ("arrattack", "falcon"): (0, 165), ("arrattack", "llada"): (14, 165),
    ("arrattack", "dream"): (10, 165), ("arrattack", "diffucoder"): (17, 165),
}


def path_for(attack, model):
    a = ATTACKS[attack]
    return os.path.join(a["dir"], a["file"][model])


def load(attack, model):
    """Load a judged CSV, keeping all rows (G3: never drop rows)."""
    return pd.read_csv(path_for(attack, model))


def success_series(attack, df):
    """
    Return a boolean Series of official-judge success for a judged frame.
    - pif: llm_judge == 1
    - metacipher / arrattack: asr_success truthy
    All defined off the official judged columns only.
    """
    a = ATTACKS[attack]
    if attack == "pif":
        return pd.to_numeric(df[a["judge_col"]], errors="coerce").fillna(0).astype(int) == 1
    s = df[a["success_col"]]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def response_text(attack, df):
    return df[ATTACKS[attack]["response_col"]].astype("string")


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi, p)."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half), p)


def is_empty(series):
    """True where a text cell is missing or whitespace-only."""
    s = series.astype("string")
    return s.isna() | (s.str.strip() == "")

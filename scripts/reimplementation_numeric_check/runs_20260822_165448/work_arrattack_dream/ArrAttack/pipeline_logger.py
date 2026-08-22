"""
pipeline_logger.py
------------------
Drop into /scratch/si2356/dlm-jailbreak-transfer/ArrAttack/

Saves every intermediate result across all 5 stages to CSV files.
Import at the top of each stage script.

Output files (all in PROJECT_DIR/results/):
  stage1_brj_log.csv        — every BRJ iteration per prompt (toxic score, sim, selected)
  stage1_smoothllm_log.csv  — SmoothLLM score + label per jailbreak prompt
  stage3_brjwr_log.csv      — every BRJwr iteration per prompt (candidates, robust successes)
  stage5_attack_log.csv     — every attack attempt per test prompt (full response, scores)
  pipeline_summary.csv      — one row per prompt per stage (high-level pass/fail)
"""

import csv
import json
import os
import time
from datetime import datetime

PROJECT_DIR = "/scratch/si2356/dlm-jailbreak-transfer"
RESULTS_DIR = PROJECT_DIR + "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── File paths ────────────────────────────────────────────────────────────────
STAGE1_BRJ_CSV      = RESULTS_DIR + "/stage1_brj_log.csv"
STAGE1_SMOOTH_CSV   = RESULTS_DIR + "/stage1_smoothllm_log.csv"
STAGE3_BRJWR_CSV    = RESULTS_DIR + "/stage3_brjwr_log.csv"
STAGE5_ATTACK_CSV   = RESULTS_DIR + "/stage5_attack_log.csv"
SUMMARY_CSV         = RESULTS_DIR + "/pipeline_summary.csv"


def _append(filepath, row: dict):
    """Append one row to a CSV, writing header if file doesn't exist."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clean(text, maxlen=200):
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("\r", "")[:maxlen]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BRJ logging
# ═══════════════════════════════════════════════════════════════════════════════

def log_brj_iteration(
    prompt_idx: int,
    original_prompt: str,
    iteration: int,
    n_candidates: int,
    selected_jailbreak: str,   # "" if no success this iter
    toxic_score: float,
    sim_score: float,
    success: bool,
):
    """Call after each BRJ iteration for a prompt."""
    _append(STAGE1_BRJ_CSV, {
        "timestamp":          _ts(),
        "prompt_idx":         prompt_idx,
        "original_prompt":    _clean(original_prompt, 120),
        "iteration":          iteration,
        "n_candidates":       n_candidates,
        "success":            success,
        "selected_jailbreak": _clean(selected_jailbreak, 200),
        "toxic_score":        round(toxic_score, 4),
        "sim_score":          round(sim_score, 4),
    })


def log_smoothllm(
    prompt_idx: int,
    original_prompt: str,
    jailbreak_prompt: str,
    smoothllm_score: int,
    label,          # 0, 1, or None
):
    """Call after SmoothLLM scores a jailbreak prompt."""
    _append(STAGE1_SMOOTH_CSV, {
        "timestamp":        _ts(),
        "prompt_idx":       prompt_idx,
        "original_prompt":  _clean(original_prompt, 120),
        "jailbreak_prompt": _clean(jailbreak_prompt, 200),
        "smoothllm_score":  smoothllm_score,
        "label":            label if label is not None else "DISCARDED",
        "saved_to_dataset": label is not None,
    })


def log_stage1_prompt_done(
    prompt_idx: int,
    original_prompt: str,
    n_jailbreaks_found: int,
    n_labeled: int,
    n_label1: int,
    n_label0: int,
):
    """Call when all BRJ iterations for one prompt are complete."""
    _append(SUMMARY_CSV, {
        "timestamp":        _ts(),
        "stage":            "Stage1_BRJ",
        "prompt_idx":       prompt_idx,
        "original_prompt":  _clean(original_prompt, 120),
        "result":           "SUCCESS" if n_jailbreaks_found > 0 else "NO_JAILBREAK",
        "detail":           f"jailbreaks={n_jailbreaks_found} labeled={n_labeled} label1={n_label1} label0={n_label0}",
        "jailbreak_prompt": "",
        "response":         "",
        "toxic_score":      "",
        "sim_score":        "",
        "robust_score":     "",
        "llm_judge_score":  "",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BRJwr logging
# ═══════════════════════════════════════════════════════════════════════════════

def log_brjwr_iteration(
    prompt_idx: int,
    original_prompt: str,
    iteration: int,
    n_candidates: int,
    n_robust_successes: int,
    best_toxic: float,
    best_robust: float,
    best_sim: float,
    best_candidate: str,
):
    """Call after each BRJwr iteration."""
    _append(STAGE3_BRJWR_CSV, {
        "timestamp":          _ts(),
        "prompt_idx":         prompt_idx,
        "original_prompt":    _clean(original_prompt, 120),
        "iteration":          iteration,
        "n_candidates":       n_candidates,
        "n_robust_successes": n_robust_successes,
        "best_toxic":         round(best_toxic, 4),
        "best_robust":        round(best_robust, 4),
        "best_sim":           round(best_sim, 4),
        "best_candidate":     _clean(best_candidate, 200),
    })


def log_stage3_prompt_done(
    prompt_idx: int,
    original_prompt: str,
    n_robust_pairs: int,
    saved_pairs,   # list of (orig, jp) tuples
):
    """Call when BRJwr finishes all iterations for one prompt."""
    detail = f"robust_pairs={n_robust_pairs}"
    _append(SUMMARY_CSV, {
        "timestamp":        _ts(),
        "stage":            "Stage3_BRJwr",
        "prompt_idx":       prompt_idx,
        "original_prompt":  _clean(original_prompt, 120),
        "result":           "SUCCESS" if n_robust_pairs > 0 else "NO_ROBUST",
        "detail":           detail,
        "jailbreak_prompt": _clean(saved_pairs[0][1], 200) if saved_pairs else "",
        "response":         "",
        "toxic_score":      "",
        "sim_score":        "",
        "robust_score":     "",
        "llm_judge_score":  "",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Attack logging
# ═══════════════════════════════════════════════════════════════════════════════

def log_attack_attempt(
    prompt_idx: int,
    original_prompt: str,
    attempt: int,
    jailbreak_prompt: str,
    target_response: str,
    sim_score: float,
    gptfuzz_success: bool,
    llm_score: int,
    llm_success: bool,
):
    """Call after every single attack attempt."""
    _append(STAGE5_ATTACK_CSV, {
        "timestamp":        _ts(),
        "prompt_idx":       prompt_idx,
        "original_prompt":  _clean(original_prompt, 120),
        "attempt":          attempt,
        "jailbreak_prompt": _clean(jailbreak_prompt, 200),
        "target_response":  _clean(target_response, 300),
        "sim_score":        round(sim_score, 4),
        "gptfuzz_success":  gptfuzz_success,
        "llm_score":        llm_score,
        "llm_success":      llm_success,
    })


def log_stage5_prompt_done(
    prompt_idx: int,
    original_prompt: str,
    total_attempts: int,
    gptfuzz_success: bool,
    llm_success: bool,
    best_llm_score: int,
    best_jailbreak: str,
    best_response: str,
):
    """Call when attack finishes for one prompt."""
    if gptfuzz_success:
        result = "JAILBROKEN_GPTFUZZ"
    elif llm_success:
        result = "JAILBROKEN_LLM"
    else:
        result = "FAILED"

    _append(SUMMARY_CSV, {
        "timestamp":        _ts(),
        "stage":            "Stage5_Attack",
        "prompt_idx":       prompt_idx,
        "original_prompt":  _clean(original_prompt, 120),
        "result":           result,
        "detail":           f"attempts={total_attempts} best_llm_score={best_llm_score}",
        "jailbreak_prompt": _clean(best_jailbreak, 200),
        "response":         _clean(best_response, 300),
        "toxic_score":      "",
        "sim_score":        "",
        "robust_score":     "",
        "llm_judge_score":  best_llm_score,
    })

    # Print live update
    print(f"\n{'='*60}")
    print(f"[Stage 5] Prompt {prompt_idx} — {result}")
    print(f"  Original:  {original_prompt[:80]}")
    if gptfuzz_success or llm_success:
        print(f"  Jailbreak: {best_jailbreak[:80]}")
        print(f"  Response:  {best_response[:150]}")
    print(f"  Attempts: {total_attempts}  Best LLM score: {best_llm_score}")
    print(f"  Saved to: {SUMMARY_CSV}")
    print(f"{'='*60}\n")

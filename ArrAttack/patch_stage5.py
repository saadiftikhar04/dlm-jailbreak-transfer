"""
Run this from /scratch/si2356/dlm-jailbreak-transfer/ArrAttack/:
    python patch_stage5.py
It applies all 5 changes to stage5_attack.py in-place.
"""
import re

path = "stage5_attack.py"
with open(path) as f:
    src = f.read()

# ── 1. HF model name + diffusion steps config (after DATASET_SPLIT line) ─────
OLD1 = '''GEN_MODEL_DIR = PROJECT_DIR + "/generation_model"'''

NEW1 = '''# ── HuggingFace model names (paper-style, matches HF repo) ──────────────────
HF_MODEL_NAMES = {
    "qwen2.5":    "Qwen2.5-7B-Instruct",
    "falcon":     "falcon-7b-instruct",
    "llama":      "Meta-Llama-3.1-8B-Instruct",
    "llada":      "LLaDA-1.5",
    "dream":      "Dream-v0-7B",
    "diffucoder": "DiffuCoder-7B-Instruct",
}
TARGET_HF_NAME = HF_MODEL_NAMES.get(TARGET_KEY, TARGET_KEY)

# Diffusion steps per DLM target (None for standard autoregressive LLMs)
_DIFFUSION_STEPS_MAP = {
    "llada":      10,
    "dream":      10,
    "diffucoder": 10,
}
TARGET_DIFFUSION_STEPS = _DIFFUSION_STEPS_MAP.get(TARGET_KEY, None)
DIFFUSION_STEPS_VAL = TARGET_DIFFUSION_STEPS if TARGET_DIFFUSION_STEPS is not None else "N/A"

GEN_MODEL_DIR = PROJECT_DIR + "/generation_model"'''

assert OLD1 in src, "PATCH 1 anchor not found"
src = src.replace(OLD1, NEW1, 1)

# ── 2. PROGRESS_FIELDS — add diffusion_steps ─────────────────────────────────
OLD2 = '''PROGRESS_FIELDS = ["prompt_idx", "original_prompt", "target_model",
    "jailbroken_gptfuzz", "jailbroken_llm", "llm_score",
    "best_attempt", "total_attempts", "prompt_total_time_s"]'''

NEW2 = '''PROGRESS_FIELDS = ["prompt_idx", "original_prompt", "target_model",
    "diffusion_steps",
    "jailbroken_gptfuzz", "jailbroken_llm", "llm_score",
    "best_attempt", "total_attempts", "prompt_total_time_s"]'''

assert OLD2 in src, "PATCH 2 anchor not found"
src = src.replace(OLD2, NEW2, 1)

# ── 3. RESULT_FIELDS — add diffusion_steps ───────────────────────────────────
OLD3 = '''RESULT_FIELDS   = ["prompt_idx", "original_prompt", "target_model", "attempt",
    "jailbreak_prompt", "target_response",
    "attack_success_gptfuzz", "attack_success_llm", "llm_judge_score",
    "semantic_similarity", "rephrase_time_s", "target_time_s",
    "judge_time_s", "attempt_total_time_s"]'''

NEW3 = '''RESULT_FIELDS   = ["prompt_idx", "original_prompt", "target_model", "attempt",
    "diffusion_steps",
    "jailbreak_prompt", "target_response",
    "attack_success_gptfuzz", "attack_success_llm", "llm_judge_score",
    "semantic_similarity", "rephrase_time_s", "target_time_s",
    "judge_time_s", "attempt_total_time_s"]'''

assert OLD3 in src, "PATCH 3 anchor not found"
src = src.replace(OLD3, NEW3, 1)

# ── 4. flush() call — use TARGET_HF_NAME, add diffusion_steps ────────────────
OLD4 = '''        flush([{
            "prompt_idx":          idx,
            "original_prompt":     prompt,
            "target_model":        TARGET_KEY,
            "attempt":             attempt,
            "jailbreak_prompt":    rephr,
            "target_response":     response,
            "attack_success_gptfuzz": gptfuzz_ok,
            "attack_success_llm":  llm_ok,
            "llm_judge_score":     llm_score,
            "semantic_similarity": round(sim, 4),
            "rephrase_time_s":     t_rephrase,
            "target_time_s":       t_target,
            "judge_time_s":        t_judge,
            "attempt_total_time_s":t_attempt,
        }])'''

NEW4 = '''        flush([{
            "prompt_idx":          idx,
            "original_prompt":     prompt,
            "target_model":        TARGET_HF_NAME,
            "attempt":             attempt,
            "diffusion_steps":     DIFFUSION_STEPS_VAL,
            "jailbreak_prompt":    rephr,
            "target_response":     response,
            "attack_success_gptfuzz": gptfuzz_ok,
            "attack_success_llm":  llm_ok,
            "llm_judge_score":     llm_score,
            "semantic_similarity": round(sim, 4),
            "rephrase_time_s":     t_rephrase,
            "target_time_s":       t_target,
            "judge_time_s":        t_judge,
            "attempt_total_time_s":t_attempt,
        }])'''

assert OLD4 in src, "PATCH 4 anchor not found"
src = src.replace(OLD4, NEW4, 1)

# ── 5. append_progress() call — use TARGET_HF_NAME, add diffusion_steps ──────
OLD5 = '''    append_progress({
        "prompt_idx":         idx,
        "original_prompt":    prompt[:120],
        "target_model":       TARGET_KEY,
        "jailbroken_gptfuzz": success_g,
        "jailbroken_llm":     success_l,
        "llm_score":          best_llm,
        "best_attempt":       best_attempt or n_attempts,
        "total_attempts":     n_attempts,
        "prompt_total_time_s":pt,
    })'''

NEW5 = '''    append_progress({
        "prompt_idx":         idx,
        "original_prompt":    prompt[:120],
        "target_model":       TARGET_HF_NAME,
        "diffusion_steps":    DIFFUSION_STEPS_VAL,
        "jailbroken_gptfuzz": success_g,
        "jailbroken_llm":     success_l,
        "llm_score":          best_llm,
        "best_attempt":       best_attempt or n_attempts,
        "total_attempts":     n_attempts,
        "prompt_total_time_s":pt,
    })'''

assert OLD5 in src, "PATCH 5 anchor not found"
src = src.replace(OLD5, NEW5, 1)

with open(path, "w") as f:
    f.write(src)

print("✅ All 5 patches applied successfully to stage5_attack.py")
print(f"   TARGET_HF_NAME will resolve from HF_MODEL_NAMES at runtime.")
print(f"   diffusion_steps added to both results.csv and progress.csv.")
print(f"   Newline sanitization was already in place (sanitize() fn).")

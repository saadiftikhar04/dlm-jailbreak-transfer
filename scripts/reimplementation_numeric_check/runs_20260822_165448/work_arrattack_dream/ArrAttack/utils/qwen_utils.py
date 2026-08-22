"""
utils/qwen_utils.py
-------------------
Qwen2.5-specific helpers shared across all modified ArrAttack scripts.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SNAPSHOT_BASE = "/home/bc3194/Desktop/huggingface_cache/hub"

PATHS = {
    "qwen_instruct": f"{SNAPSHOT_BASE}/models--Qwen--Qwen2.5-7B-Instruct/snapshots",
    "qwen_base":     f"{SNAPSHOT_BASE}/models--Qwen--Qwen2.5-7B/snapshots",
    "gptfuzz":       f"{SNAPSHOT_BASE}/models--hubert233--GPTFuzz/snapshots",
    "paraphraser":   f"{SNAPSHOT_BASE}/models--humarin--chatgpt_paraphraser_on_T5_base/snapshots",
    "mpnet":         f"{SNAPSHOT_BASE}/models--sentence-transformers--all-mpnet-base-v2/snapshots",
    "llama3":        f"{SNAPSHOT_BASE}/models--meta-llama--Llama-3-8B-Instruct/snapshots",
    "mistral":       f"{SNAPSHOT_BASE}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots",
    "vicuna":        f"{SNAPSHOT_BASE}/models--lmsys--vicuna-7b-v1.5/snapshots",
}

PROJECT_DIR = "/home/bc3194/Desktop/dlm-jailbreak-transfer/scripts/reimplementation_numeric_check/runs_20260822_165448/work_arrattack_dream/slim_repo"


def resolve_snapshot(snapshot_dir: str, hf_fallback: str) -> Tuple[str, bool]:
    snap = Path(snapshot_dir)
    if snap.exists():
        snaps = sorted(snap.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            return str(snaps[0]), True
    if os.getenv("TRANSFORMERS_OFFLINE", "0").strip() == "1":
        raise FileNotFoundError(
            f"TRANSFORMERS_OFFLINE=1 but no snapshot at {snapshot_dir}"
        )
    logger.warning(f"No snapshot at {snapshot_dir}; falling back to HF Hub: {hf_fallback}")
    return hf_fallback, False


def load_qwen(snapshot_dir: str, hf_id: str, ctx_cap: Optional[int] = None):
    """Load a Qwen2.5 model + tokenizer. Returns (model, tokenizer, effective_ctx)."""
    model_path, local_only = resolve_snapshot(snapshot_dir, hf_id)
    logger.info(f"Loading {hf_id} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    ).eval()

    cfg_ctx = getattr(model.config, "max_position_embeddings", None)
    tok_ctx = getattr(tokenizer, "model_max_length", 32768)
    raw_ctx = cfg_ctx if (cfg_ctx and cfg_ctx < 10_000_000) else tok_ctx
    ctx = min(int(raw_ctx), ctx_cap) if ctx_cap else int(raw_ctx)
    logger.info(f"Loaded {hf_id}  ctx={ctx}")
    return model, tokenizer, ctx


@torch.no_grad()
def qwen_generate(
    model,
    tokenizer,
    prompt: str,
    ctx: int = 8192,
    max_new: int = 512,
    top_p: float = 0.9,
    temperature: float = 0.8,
) -> Tuple[str, bool]:
    """Generate via Qwen2.5 ChatML template. Returns (response, is_truncated)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx).to(
        next(model.parameters()).device
    )
    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = max(1, min(max_new, ctx - input_len - 32))

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][input_len:]
    is_truncated = gen_ids.shape[0] == max_new_tokens
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    del inputs, out, gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response, is_truncated


@torch.no_grad()
def qwen_logits_01(model, tokenizer, prompt_text: str) -> Tuple[float, float]:
    """Return (p0, p1) for next token being '0' or '1'. Used by robustness judge."""
    tok_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    tok_1 = tokenizer.encode("1", add_special_tokens=False)[0]

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=1024
    ).to(next(model.parameters()).device)

    logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs[tok_0].item(), probs[tok_1].item()


def chatml_instruction_prompt(tokenizer, instruction: str, input_text: str) -> str:
    """Build ChatML prompt (no response) for inference. Replaces Alpaca format_instruction."""
    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_for_sft(tokenizer, instruction: str, input_text: str, response: str) -> str:
    """Alpaca format — compatible with transformers 4.45.x."""
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"

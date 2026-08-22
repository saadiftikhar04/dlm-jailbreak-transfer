"""
utils/model_utils.py
====================
Unified generate() interface for all target models.

LLMs  (autoregressive):
  - Qwen2.5-7B-Instruct        (already done)
  - Falcon-H1R-7B              (tiiuae/Falcon-H1R-7B)
  - Llama-3.2-3B-Instruct      (meta-llama/Llama-3.2-3B-Instruct)

DLMs  (masked diffusion — generate via diffusion_generate / mdm_generate):
  - LLaDA-1.5                  (GSAI-ML/LLaDA-1.5)
  - Dream-v0-Instruct-7B       (Dream-org/Dream-v0-Instruct-7B)
  - DiffuCoder-7B-Instruct     (apple/DiffuCoder-7B-Instruct)

Usage:
  model, tokenizer = load_model(model_key)
  response = generate(model, tokenizer, prompt, model_key)
"""

import os
import sys
import torch
from llada_generate import generate as llada_generate
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

HF_CACHE = "/home/bc3194/Desktop/huggingface_cache/hub"

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # LLMs
    "qwen2.5":   "Qwen/Qwen2.5-7B-Instruct",
    "falcon":    "tiiuae/Falcon-H1R-7B",
    "llama":     "/scratch/si2356/hf_models/Llama-3.2-3B-Instruct",
    # DLMs
    "llada":     "/scratch/si2356/models/llada-1.5-8b",
    "dream":     "Dream-org/Dream-v0-Instruct-7B",
    "diffucoder":"apple/DiffuCoder-7B-Instruct",
}

DLM_KEYS = {"llada", "dream", "diffucoder"}

# ── Loader ────────────────────────────────────────────────────────────────────

def load_model(model_key: str, offline: bool = True):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    hf_id = MODEL_REGISTRY[model_key]
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=False,
    )

    if model_key in DLM_KEYS:
        # DLMs use AutoModel not AutoModelForCausalLM
        kwargs.pop("weights_only", None)
        model = AutoModel.from_pretrained(hf_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=True, local_files_only=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ── Chat template builders ────────────────────────────────────────────────────

def _build_prompt(tokenizer, model_key: str, user_text: str) -> str:
    """Build prompt string using the model's chat template."""
    if model_key == "qwen2.5":
        # ChatML — Qwen native
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    elif model_key == "falcon":
        # Falcon-H1R uses standard chat template
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    elif model_key == "llama":
        # Llama-3.2 uses apply_chat_template natively
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    elif model_key in ("llada", "dream"):
        # LLaDA / Dream use same format as LLaDA-8B-Instruct
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    elif model_key == "diffucoder":
        # DiffuCoder follows Qwen ChatML format
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    return user_text  # fallback


# ── Unified generate() ────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, model_key: str,
             max_new_tokens: int = 512) -> str:
    """
    Generate a response from any supported model.
    Always returns the response text only (no prompt included).
    """
    full_prompt = _build_prompt(tokenizer, model_key, prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt",
                       truncation=True, max_length=2048)
    input_ids      = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    prompt_len     = input_ids.shape[1]

    if model_key in DLM_KEYS:
        # ── DLM generation ───────────────────────────────────────────────────
        if model_key == "llada":
            # LLaDA uses standalone generate(); prompts must be list of 1D tensors
            out = llada_generate(
                model, tokenizer,
                prompts=[input_ids[0]],
                steps=128,
                max_new_tokens=max_new_tokens,
                block_length=32,
                temperature=0.0,
                remasking="random",
            )
            generated_ids = out[0][prompt_len:]
        else:
            # Dream, DiffuCoder expose diffusion_generate()
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                steps=128,
                output_history=False,
                return_dict_in_generate=True,
            )
            generated_ids = out.sequences[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    else:
        # ── LLM generation (autoregressive) ──────────────────────────────────
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True)

    return response.strip()

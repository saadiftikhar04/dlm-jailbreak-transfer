#!/usr/bin/env python3
import argparse
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def _strip_think_falcon(text: str) -> str:
    """
    Cleans Falcon outputs.
    Handles:
      - <think> ... </think> ANSWER
      - stray </think>
      - role markers in generated text: system/user/assistant (often repeats mid-output)
    """
    if text is None:
        return ""

    # 1) Think tags handling
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "</think>" in text:
        text = text.split("</think>", 1)[0]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")

    # 2) Remove/stop at role markers produced by the model
    # Strategy:
    # - skip leading role-only lines
    # - once we've seen real content, stop if we hit a role-only line
    role_words = {"system", "user", "assistant"}

    lines = text.splitlines()
    out = []
    seen_content = False

    for ln in lines:
        stripped = ln.strip()
        low = stripped.lower()

        # role-only line?
        if low in role_words and stripped == ln.strip():
            if not seen_content:
                # drop leading "assistant"/"system"/"user"
                continue
            # stop at the first role marker after real content
            break

        if stripped:
            seen_content = True
        out.append(ln)

    text = "\n".join(out).strip()

    # Collapse excessive trailing whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---- PATCH: ignore missing HF "additional_chat_templates" folder (HF/Transformers edge case) ----
import transformers.utils.hub as _hub
import transformers.tokenization_utils_base as _tub

try:
    from huggingface_hub.errors import (
        RemoteEntryNotFoundError,
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        HfHubHTTPError,
    )
except Exception:
    RemoteEntryNotFoundError = EntryNotFoundError = RepositoryNotFoundError = RevisionNotFoundError = HfHubHTTPError = Exception

_orig_list_repo_templates = _hub.list_repo_templates

def _safe_list_repo_templates(*args, **kwargs):
    try:
        return _orig_list_repo_templates(*args, **kwargs)
    except (
        RemoteEntryNotFoundError,
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        HfHubHTTPError,
        Exception,
    ):
        return []

_hub.list_repo_templates = _safe_list_repo_templates
_tub.list_repo_templates = _safe_list_repo_templates
# ---- END PATCH ----




def strip_think_blocks(text: str) -> str:
    # remove <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiiuae/Falcon-H1R-7B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--strip_think", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    # dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        # prefer bf16 if cuda supports it, else fp16 on cuda, else fp32
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.eval()

    # Build inputs (chat if template exists; else plain)
    use_chat = hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)
    if use_chat:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        input_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        attention_mask = torch.ones_like(input_ids)
    else:
        enc = tok(args.prompt, return_tensors="pt")
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

    # move tensors (works for single-device; for device_map="auto" this is still fine when model.device exists)
    if hasattr(model, "device"):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    gen = out[0, input_ids.shape[-1]:]
    text = tok.decode(gen, skip_special_tokens=True)

    if args.strip_think:
        text = _strip_think_falcon(text)
    if args.strip_think:
        text = strip_think_blocks(text)

    print(text)


if __name__ == "__main__":
    main()

""" src/llms.py
The class that saves all LLMs from huggingface transformers, ollama, openai gpt, google gemini, anthropic claude.

This file is the single factory entrypoint used across scripts.
"""

import os
from typing import Optional

import torch

# Local LLM/MLLMs
from .local_llm.qwen3 import Qwen3

# Remote APIs
from .remote_api.claude_api import ClaudeAPI
from .remote_api.openai_api import OpenAIAPI
from .remote_api.gemini_api import GeminiAPI

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# HuggingFace/Transformers PATCHES (keep behavior, but unify duplicates)
# Purpose: ignore missing HF "additional_chat_templates" folder (404 edge case)
# ---------------------------------------------------------------------
import transformers.utils.hub as _hub
import transformers.tokenization_utils_base as _tub
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

try:
    from huggingface_hub.errors import RemoteEntryNotFoundError
except Exception:
    RemoteEntryNotFoundError = Exception

_orig_list_repo_templates = _hub.list_repo_templates


def _safe_list_repo_templates(*args, **kwargs):
    try:
        return _orig_list_repo_templates(*args, **kwargs)
    except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, RemoteEntryNotFoundError, Exception):
        return []


# Patch both call sites (Transformers may have imported list_repo_templates into tokenization_utils_base)
_hub.list_repo_templates = _safe_list_repo_templates
_tub.list_repo_templates = _safe_list_repo_templates
# ---------------------------------------------------------------------

# If you have local helper patches, keep this import
# Optional local helper patches (some repos include this file; others don't)
try:
    from fix_transformers import *  # noqa: F401,F403
except Exception:
    pass

# ---------------------------------------------------------------------
# DLM imports (NEW)
# NOTE: These wrappers must exist under src/:
#   - llm_llada.py defines LLaDALLM
#   - llm_dream.py defines DreamDLM
#   - llm_mdlm.py defines MDLMDLM
# ---------------------------------------------------------------------
# from llm_llada import LLaDALLM
# from llm_dream import DreamDLM
# from llm_mdlm import MDLMDLM

# ---------------------------------------------------------------------
# Model name to HuggingFace ID mapping - UNGATED / OPEN MODELS ONLY
# ---------------------------------------------------------------------
MODEL_MAP = {
    # Existing mappings you had
    "llama3.2-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.1-8b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b-instruct-v0.3": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",

    # New models you requested
    "falcon-h1r-7b": "tiiuae/Falcon-H1R-7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

# Claude API model mapping
CLAUDE_MODELS = {
    "claude-sonnet-4.5": "claude-sonnet-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
}

# ---------------------------------------------------------------------
# DLM mappings (NEW)
# Provide friendly shortcuts AND allow raw HF IDs.
# ---------------------------------------------------------------------
DLM_MAP = {
    # LLaDA checkpoints (observed in your logs)
    "llada-1.5": "GSAI-ML/LLaDA-1.5",
    "GSAI-ML/LLaDA-1.5": "GSAI-ML/LLaDA-1.5",
    "llada-1.5b": "laogou/LLaDA-1.5b",
    "laogou/LLaDA-1.5b": "laogou/LLaDA-1.5b",

    # Dream 7B
    "dream-7b": "Dream-org/Dream-v0-Instruct-7B",
    "Dream-org/Dream-v0-Instruct-7B": "Dream-org/Dream-v0-Instruct-7B",

    # MDLM
    "mdlm-owt": "kuleshov-group/mdlm-owt",
    "kuleshov-group/mdlm-owt": "kuleshov-group/mdlm-owt",
}






def init_mllm(
    model_name: str,
    dtype: Optional[torch.dtype] = torch.float32,
    quantization: Optional[str] = None,
    use_openrouter: bool = False,
    use_flash_attn: bool = False,
    use_deepspeed: bool = False,
    api_key: Optional[str] = None,
):
    if "claude" in model_name.lower():
        return ClaudeAPI(
            model_name=model_name, 
            use_openrouter=use_openrouter
        )
    elif "gpt" in model_name.lower():
        return OpenAIAPI(
            model_name=model_name,
            api_key=api_key,
            use_openrouter=use_openrouter
        )
    elif "gemini" in model_name.lower():
        return GeminiAPI(
            model_name=model_name,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")






# def llm_factory(model_name, **kwargs):
#     """Factory function to instantiate the appropriate LLM subclass based on the model name."""
#     name = str(model_name)
#     name_l = name.lower()

#     # 0) DLM shortcuts and explicit IDs (NEW)
#     if name in DLM_MAP or any(k in name_l for k in ["llada", "dream", "mdlm"]):
#         # Resolve to canonical ID when possible
#         resolved = DLM_MAP.get(name, name)

#         # LLaDA
#         if ("llada" in name_l) or ("llada" in resolved.lower()):
#             return LLaDALLM(model_id=resolved, **kwargs)

#         # Dream
#         if ("dream" in name_l) or ("dream" in resolved.lower()):
#             return DreamDLM(model_id=resolved, **kwargs)

#         # MDLM
#         if ("mdlm" in name_l) or ("mdlm" in resolved.lower()):
#             # MDLM uses GPT-2 tokenizer by default; allow override via kwargs
#             tokenizer_id = kwargs.pop("tokenizer_id", "gpt2")
#             return MDLMDLM(model_id=resolved, tokenizer_id=tokenizer_id, **kwargs)

#     # 1) mapped shortcuts (HF causal LLMs)
#     if name in MODEL_MAP:
#         hf_model_id = MODEL_MAP[name]
#         return TransformerLLM(model_name=hf_model_id, friendly_name=name, **kwargs)

#     # 2) Claude
#     if name in CLAUDE_MODELS or "claude" in name_l:
#         api_model = CLAUDE_MODELS.get(name, name)
#         return ClaudeLLM(model=api_model, **kwargs)

#     # 3) OpenAI reasoning models
#     if ("o1" in name_l) or ("o3" in name_l):
#         return OpenAIReasoningLLM(model=name, **kwargs)

#     # 4) OpenAI / DeepSeek (OpenAI SDK style)
#     if "gpt" in name_l:
#         return OpenAILLM(model=name, **kwargs)
#     if "deepseek" in name_l:
#         return OpenAILLM(model=name, **kwargs)

#     # 5) Gemini
#     if "gemini" in name_l:
#         return GeminiLLM(name, **kwargs)

#     # 6) Raw HF repo id (fallback to causal LM)
#     if "/" in name:
#         return TransformerLLM(model_name=name, **kwargs)

#     # 7) Otherwise try Ollama
#     return OllamaLLM(model=name, **kwargs)




# class TransformerLLM(LLM):
#     """
#     Local HuggingFace causal LM wrapper.
#     Supports:
#       - prompt as str
#       - prompt as list[{"role":..., "content":...}] (chat template if available)
#     """
#     def __init__(self, model_name, friendly_name=None, dtype="auto", device_map="auto"):
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=dtype,
#             device_map=device_map,
#             trust_remote_code=True,
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             use_fast=True,
#         )
#         self.model_name = model_name
#         self.friendly_name = friendly_name or model_name

#         if self.tokenizer.pad_token is None:
#             # Safe default for most causal LMs
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#     def _build_inputs(self, prompt):
#         # Chat-style
#         if isinstance(prompt, list):
#             if hasattr(self.tokenizer, "apply_chat_template"):
#                 return self.tokenizer.apply_chat_template(
#                     prompt,
#                     tokenize=True,
#                     add_generation_prompt=True,
#                     return_tensors="pt",
#                 )
#             # fallback: naive concat
#             text = ""
#             for m in prompt:
#                 role = m.get("role", "user")
#                 content = m.get("content", "")
#                 text += f"{role.upper()}: {content}\n"
#             text += "ASSISTANT:"
#             return self.tokenizer(text, return_tensors="pt").input_ids

#         # Plain string prompt
#         if isinstance(prompt, str):
#             return self.tokenizer(prompt, return_tensors="pt").input_ids

#         raise ValueError("prompt must be a str or a list of chat messages")

#     @torch.inference_mode()
#     def generate(
#         self,
#         prompt,
#         temperature=0.0,
#         max_new_tokens=256,
#         do_sample=True,
#         num_return_sequences=1,
#         top_p=0.95,
#         top_k=50,
#     ):
#         # Keep existing behavior from your file
#         max_new_tokens = int(max_new_tokens)
#         if max_new_tokens > 8192:
#             max_new_tokens = 8192

#         input_ids = self._build_inputs(prompt).to(self.model.device)

#         if temperature > 0 and do_sample:
#             output = self.model.generate(
#                 input_ids,
#                 do_sample=True,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 num_return_sequences=num_return_sequences,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
#         else:
#             output = self.model.generate(
#                 input_ids,
#                 do_sample=False,
#                 max_new_tokens=max_new_tokens,
#                 num_return_sequences=num_return_sequences,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )

#         generated_tokens = output[0, input_ids.shape[-1]:]
#         return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     def chat(self, prompt, **kwargs):
#         return self.generate(prompt, **kwargs)

#     def get_name(self):
#         return self.friendly_name







# class OllamaLLM(LLM):
#     def __init__(self, model, api_key=None, client=None):
#         self.client = client or ollama.Client()
#         self.model = model

#     def generate(self, prompt, **kwargs):
#         return self.client.generate(model=self.model, prompt=prompt).response

#     def chat(self, prompt, **kwargs):
#         return self.generate(prompt, **kwargs)

#     def get_name(self):
#         return self.model

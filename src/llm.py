"""
The class that saves all LLMs from transformers, openai gpt, google gemini, anthropic claude,
and (NEW) diffusion language models (DLMs): LLaDA, Dream, MDLM.

This file is the single factory entrypoint used across scripts.
"""

import os
import torch
import openai
try:
    import google.generativeai as genai
except Exception:
    genai = None
import anthropic
try:
    import ollama
except Exception:
    ollama = None
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
from llm_llada import LLaDALLM
from llm_dream import DreamDLM
from llm_mdlm import MDLMDLM

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


def llm_factory(model_name, **kwargs):
    """Factory function to instantiate the appropriate LLM subclass based on the model name."""
    name = str(model_name)
    name_l = name.lower()

    # 0) DLM shortcuts and explicit IDs (NEW)
    if name in DLM_MAP or any(k in name_l for k in ["llada", "dream", "mdlm"]):
        # Resolve to canonical ID when possible
        resolved = DLM_MAP.get(name, name)

        # LLaDA
        if ("llada" in name_l) or ("llada" in resolved.lower()):
            return LLaDALLM(model_id=resolved, **kwargs)

        # Dream
        if ("dream" in name_l) or ("dream" in resolved.lower()):
            return DreamDLM(model_id=resolved, **kwargs)

        # MDLM
        if ("mdlm" in name_l) or ("mdlm" in resolved.lower()):
            # MDLM uses GPT-2 tokenizer by default; allow override via kwargs
            tokenizer_id = kwargs.pop("tokenizer_id", "gpt2")
            return MDLMDLM(model_id=resolved, tokenizer_id=tokenizer_id, **kwargs)

    # 1) mapped shortcuts (HF causal LLMs)
    if name in MODEL_MAP:
        hf_model_id = MODEL_MAP[name]
        return TransformerLLM(model_name=hf_model_id, friendly_name=name, **kwargs)

    # 2) Claude
    if name in CLAUDE_MODELS or "claude" in name_l:
        api_model = CLAUDE_MODELS.get(name, name)
        return ClaudeLLM(model=api_model, **kwargs)

    # 3) OpenAI reasoning models
    if ("o1" in name_l) or ("o3" in name_l):
        return OpenAIReasoningLLM(model=name, **kwargs)

    # 4) OpenAI / DeepSeek (OpenAI SDK style)
    if "gpt" in name_l:
        return OpenAILLM(model=name, **kwargs)
    if "deepseek" in name_l:
        return OpenAILLM(model=name, **kwargs)

    # 5) Gemini
    if "gemini" in name_l:
        return GeminiLLM(name, **kwargs)

    # 6) Raw HF repo id (fallback to causal LM)
    if "/" in name:
        return TransformerLLM(model_name=name, **kwargs)

    # 7) Otherwise try Ollama
    return OllamaLLM(model=name, **kwargs)


class LLM:
    def __init__(self): ...
    def generate(self, prompt, **kwargs): raise NotImplementedError
    def chat(self, prompt, **kwargs): return self.generate(prompt, **kwargs)
    def generate_with_history_messages(self, prompt, history_messages, **kwargs): raise NotImplementedError
    def get_name(self): raise NotImplementedError


class TransformerLLM(LLM):
    """
    Local HuggingFace causal LM wrapper.
    Supports:
      - prompt as str
      - prompt as list[{"role":..., "content":...}] (chat template if available)
    """
    def __init__(self, model_name, friendly_name=None, dtype="auto", device_map="auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.model_name = model_name
        self.friendly_name = friendly_name or model_name

        if self.tokenizer.pad_token is None:
            # Safe default for most causal LMs
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_inputs(self, prompt):
        # Chat-style
        if isinstance(prompt, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            # fallback: naive concat
            text = ""
            for m in prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                text += f"{role.upper()}: {content}\n"
            text += "ASSISTANT:"
            return self.tokenizer(text, return_tensors="pt").input_ids

        # Plain string prompt
        if isinstance(prompt, str):
            return self.tokenizer(prompt, return_tensors="pt").input_ids

        raise ValueError("prompt must be a str or a list of chat messages")

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        temperature=0.0,
        max_new_tokens=256,
        do_sample=True,
        num_return_sequences=1,
        top_p=0.95,
        top_k=50,
    ):
        # Keep existing behavior from your file
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens > 8192:
            max_new_tokens = 8192

        input_ids = self._build_inputs(prompt).to(self.model.device)

        if temperature > 0 and do_sample:
            output = self.model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            output = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def get_name(self):
        return self.friendly_name


class OpenAILLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        if api_key:
            self.api_key = api_key
        elif "gpt" in model:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif "deepseek" in model:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY for GPT models or DEEPSEEK_API_KEY for DeepSeek."
            )

        if "deepseek" in model:
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt, temperature=0.7, max_new_tokens=256):
        max_new_tokens = int(max_new_tokens)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of messages.")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return completion.choices[0].message.content

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def generate_with_history_messages(self, prompt, history_messages, max_new_tokens=256):
        max_new_tokens = int(max_new_tokens)
        if isinstance(prompt, str):
            full_messages = history_messages + [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            full_messages = history_messages + prompt
        elif isinstance(prompt, dict):
            full_messages = history_messages + [prompt]
        else:
            raise ValueError("Prompt must be str/list/dict.")
        return self.generate(full_messages, max_new_tokens=max_new_tokens)

    def get_name(self):
        return self.model


class OpenAIReasoningLLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for reasoning model.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt, temperature=0.7, max_new_tokens=256):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of messages.")

        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        return completion.choices[0].message.content

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def get_name(self):
        return self.model


class GeminiLLM(LLM):
    def __init__(self, model_name, api_key=None):
        if genai is None:
            raise RuntimeError("google.generativeai is not installed. Install it or avoid Gemini models.")
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt, temperature=0.0, max_new_tokens=256, top_p=0.95, top_k=50):
        max_new_tokens = int(max_new_tokens)
        generation_config = {
            "max_output_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": ["\n\n\n"],
        }
        response = self.model.generate_content(prompt, generation_config=generation_config)
        try:
            if not isinstance(response.text, str) or len(response.text) == 0:
                return "I'm sorry, but I cannot assist with that request."
            return response.text
        except Exception:
            return "I'm sorry, but I cannot assist with that request."

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def get_name(self):
        return self.model_name


class ClaudeLLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Claude.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        max_new_tokens = int(max_new_tokens)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of messages.")

        completion = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            messages=messages,
            temperature=temperature,
        )
        return completion.content[0].text

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def get_name(self):
        return self.model


class OllamaLLM(LLM):
    def __init__(self, model, api_key=None, client=None):
        self.client = client or ollama.Client()
        self.model = model

    def generate(self, prompt, **kwargs):
        return self.client.generate(model=self.model, prompt=prompt).response

    def chat(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def get_name(self):
        return self.model


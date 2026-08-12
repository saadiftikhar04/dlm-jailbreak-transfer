import os
import openai
from ..base import API


class OpenAIAPI(API):
    def __init__(
        self, 
        model, 
        api_key=None
    ):
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

    def infer(
        self, 
        prompt, 
        temperature=0.7, 
        max_new_tokens=512,
    ):
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
        return self.infer(prompt, **kwargs)

    def infer_with_history_messages(self, prompt, history_messages, max_new_tokens=512):
        max_new_tokens = int(max_new_tokens)
        if isinstance(prompt, str):
            full_messages = history_messages + [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            full_messages = history_messages + prompt
        elif isinstance(prompt, dict):
            full_messages = history_messages + [prompt]
        else:
            raise ValueError("Prompt must be str/list/dict.")
        return self.infer(full_messages, max_new_tokens=max_new_tokens)

    def get_name(self):
        return self.model



class OpenAIReasoningAPI(API):
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

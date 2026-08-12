import os
import anthropic
from ..base import API



class ClaudeAPI(API):
    def __init__(
        self, 
        model, 
        api_key=None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Claude.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def infer(
        self, 
        prompt, 
        temperature=0.0, 
        max_new_tokens=512,
        top_p=0.95,
        top_k=50,
    ):
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
            top_p=top_p,
            top_k=top_k,
        )
        return completion.content[0].text
    
    def infer_with_stats(
        self, 
        prompt, 
        temperature=0.0, 
        max_new_tokens=512,
        top_p=0.95,
        top_k=50,
    ):
        pass

    def chat(self, prompt, **kwargs):
        return self.infer(prompt, **kwargs)
    
    def get_name(self):
        return self.model
    
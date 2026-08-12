import os
import google.generativeai as genai

from ..base import API


class GeminiAPI(API):
    def __init__(
        self, 
        model_name, 
        api_key=None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def infer(
        self, 
        prompt, 
        temperature=0.0, 
        max_new_tokens=512, 
        top_p=0.95, 
        top_k=50
    ):
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
        return self.infer(prompt, **kwargs)

    def get_name(self):
        return self.model_name
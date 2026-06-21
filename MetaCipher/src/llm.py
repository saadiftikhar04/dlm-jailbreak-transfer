"""
The class that saves all LLMs from transformers, openai gpt, google gemini, anthropic claude.
"""

import openai
import google.generativeai as genai
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import ollama


def llm_factory(model_name, **kwargs):
    """
    Factory function to instantiate the appropriate LLM subclass based on the model name.
    
    Args:
        model_name (str): The name of the model.
        **kwargs: Additional keyword arguments to pass to the LLM subclass constructor.
        
    Returns:
        LLM: An instance of the appropriate LLM subclass.
    """
    if ('o1' in model_name.lower()) or ('o3' in model_name.lower()):
        return OpenAIReasoningLLM(model=model_name, **kwargs)
    elif 'gpt' in model_name.lower():
        return OpenAILLM(model=model_name, **kwargs)
    elif 'gemini' in model_name.lower():   # Also gemini-2.5-pro, the reasoner
        return GeminiLLM(model_name, **kwargs)
    elif 'claude' in model_name.lower():
        return ClaudeLLM(model=model_name, **kwargs)
    elif 'deepseek' in model_name.lower(): # Also deepseek-reasoner
        return OpenAILLM(model=model_name, **kwargs)
    elif '/' in model_name:
        return TransformerLLM(model_name=model_name, **kwargs)
    else:
        return OllamaLLM(model=model_name, **kwargs)


### The abstract LLM class
class LLM:
    def __init__(self):
        pass

    def generate(self):
        pass

    def generate_with_history_messages(self):
        pass

    def get_name(self):
        pass


### Huggingface transformers
class TransformerLLM(LLM):
    def __init__(self, model_name, dtype=torch.float16, device_map="auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, temperature=0.0, max_new_tokens=1000, do_sample=True, num_return_sequences=1, top_p=0.9, top_k=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        if temperature > 0 and do_sample:
            output = self.model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
            )
        else:
            output = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
        generated_tokens = output[0, input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

    def generate_with_history_messages(self, prompt, history_messages, temperature=0.7, max_new_tokens=512, do_sample=True, num_return_sequences=1, top_p=0.9, top_k=50):
        prompt_with_history = ""
        for message_dict in history_messages:
            role = message_dict["role"]
            content = message_dict["content"]
            prompt_with_history += f"<|{role}|>\n{content}\n\n"
        prompt_with_history += f"<|system|>\n{prompt}"
        return self.generate(prompt_with_history, temperature, max_new_tokens, do_sample, num_return_sequences, top_p, top_k)
    
    def moderate(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=10, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    
    def get_name(self):
        return self.model_name

### OpenAI GPT
class OpenAILLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        if api_key:
            self.api_key = api_key
        elif "gpt" in model:
            self.api_key = os.environ["OPENAI_API_KEY"]
            self.client = openai.OpenAI(api_key=self.api_key)
        elif "deepseek" in model:
            self.api_key = os.environ["DEEPSEEK_API_KEY"]
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate(self, prompt, temperature=0.7, max_new_tokens=512):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of strings.")
        
        # print(f"\n\nmessages is:\n{messages}\n\n")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        response = completion.choices[0].message.content
        return response
    
    def generate_with_history_messages(self, prompt, history_messages, max_new_tokens=512):
        if isinstance(prompt, str):
            full_messages = history_messages + [{"role": "system", "content": prompt}]
        elif isinstance(prompt, list):
            full_messages = history_messages + prompt
        elif isinstance(prompt, dict):
            full_messages = history_messages + [prompt]
        return self.generate(full_messages, max_new_tokens=max_new_tokens)
    
    def get_name(self):
        return self.model


class OpenAIReasoningLLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ["OPENAI_API_KEY"]
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt, temperature=0.7, max_new_tokens=512):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of strings.")
        try:
            completion = self.client.chat.completions.create(   # temperature and max new tokens are not supported in reasoning
                model=self.model,
                messages=messages,
                # reasoning_effort="low",
                # max_completion_tokens=1000
            )
            response = completion.choices[0].message.content
            return response
        except openai.BadRequestError as e:
            return str(e)
    
    def get_name(self):
        return self.model


### Google Gemini
class GeminiLLM(LLM):
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt, temperature=0.0, max_new_tokens=1000, top_p=0.9, top_k=50):
        generation_config={
            "max_output_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": ["\n\n\n"]
        }
        response = self.model.generate_content(prompt, generation_config=generation_config)
        
        # Check if response can be queried with response.text, otherwise generate response again.
        try:
            if type(response.text) is not str or len(response.text) == 0:
                return "I'm sorry, but I cannot assist with that request."
            return response.text
        except ValueError as e:
            return "I'm sorry, but I cannot assist with that request."
    
    def get_name(self):
        return self.model_name


### Anthropic Claude
class ClaudeLLM(LLM):
    def __init__(self, model, api_key=None):
        self.model = model
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ["ANTHROPIC_API_KEY"]
            self.client = anthropic.Anthropic(api_key=self.api_key)  

    def generate(self, prompt, temperature=0.0, max_new_tokens=2000):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of strings.")
        completion = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            messages=messages,
            temperature=temperature,
        )
        response = completion.content[0].text
        return response
    
    def get_name(self):
        return self.model


### Ollama agent
class OllamaLLM(LLM):
    def __init__(self, model, api_key=None, client=None):
        if client:
            self.client = client
        else:
            self.client = ollama.Client()
        self.model = model

    def generate(self, prompt):
        response = self.client.generate(model=self.model, prompt=prompt).response
        return response
    
    def get_name(self):
        return self.model
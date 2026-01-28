from llm import llm_factory
import subprocess
import os
import re


class Agent:
    def __init__(self, model, role, agent_initiator_path="../prompts/agent/agent_initiator.txt"):
        self.model_name = model
        self.model = llm_factory(model)
        self.role = role
        self.function_description = ""
        self.chat_history = []
        self.agent_initiation = open(agent_initiator_path, 'r').read()
            
    def generate(self, prompt, max_new_tokens=2000, temperature=0.7):   # prompt can be str, dict, or list of dict
        output = self.model.generate_with_history_messages(prompt, self.chat_history, max_new_tokens=max_new_tokens)
        output = output.replace("<|assistant|>", "").replace("<|system|>", "")  # Hard-code removal
        output_entry = {"role": "assistant", "content": output}
        if "**Command Line Call**: " in output:   ### The "update history" part might be repetitive...
            command_line_call = output.split("**Command Line Call**: ")[1].split("\n")[0]
            try:
                result = subprocess.run(
                    command_line_call, shell=True, capture_output=True, text=True
                )
                log_entry = {
                    "role": "system",
                    "content": f"Executed command: {command_line_call}\n"
                               f"Output: {result.stdout}\n"
                               f"Error: {result.stderr if result.stderr else 'None'}"
                }
            except Exception as e:
                log_entry = {
                    "role": "system",
                    "content": f"Failed to execute command: {command_line_call}\nError: {str(e)}"
                }
            self.update_history(output_entry)
            self.update_history(log_entry)
            return self.generate("Your required command is executed, and the log is found in the history. Please go on with your analysis. ", temperature=temperature) ### Is this iteration implemented correctly?
        return output
    
    def update_history(self, history):
        if isinstance(history, list) and all(isinstance(item, dict) for item in history):
            self.chat_history += history
        elif isinstance(history, dict):
            self.chat_history.append(history)
        elif isinstance(history, str):
            self.chat_history.append({"role": "system", "content": history})

    def get_history(self):
        return self.chat_history
    


class CategoryAgent(Agent):
    def __init__(self, model, agent_prompts_dir):
        agent_initiator_path = os.path.join(agent_prompts_dir, "agent_initiator.txt")
        role_prompt = os.path.join(agent_prompts_dir, "category.txt")
        super().__init__(model, role="category", agent_initiator_path=agent_initiator_path)
        self.function_description = self.agent_initiation + open(role_prompt, 'r').read()
        self.chat_history += [{"role": "system", "content": self.function_description}]

class KeywordAgent(Agent):
    def __init__(self, model, agent_prompts_dir):
        agent_initiator_path = os.path.join(agent_prompts_dir, "agent_initiator.txt")
        role_prompt = os.path.join(agent_prompts_dir, "keywords.txt")
        super().__init__(model, role="keyword", agent_initiator_path=agent_initiator_path)
        self.function_description = self.agent_initiation + open(role_prompt, 'r').read()
        self.chat_history += [{"role": "system", "content": self.function_description}]
    
    def find_keywords(self, prompt: str):
        prompt = f"### Potentially malicious prompt:\n{prompt}\n\n\n### Your response:\n"
        model_output = self.generate(prompt, max_new_tokens=1000)
        keywords = []
        for line in model_output.split("\n"):
            if ': ' in line:
                keyword = line.split(': ')[1]
                keyword = re.sub(r'[^\w\s]', '', keyword).strip()
                if len(keyword) > 0:
                    keywords.append(keyword)
        if len(keywords)==0:
            print(f"No keywords found for the prompt: '{prompt}'")
        return keywords
    
    def update_keywords(self, prompt: str, old_keywords: list, change: str):
        prompt = f"### Potentially malicious prompt:\n{prompt}\n\n\n### Keywords detected in the previous round:\n{old_keywords}\n\n\n### Change instruction:\nPlease detect EXACTLY 1 {change} keyword in the same prompt.\n\n\n### Your response:\n"
        model_output = self.generate(prompt, max_new_tokens=1000)
        keywords = []
        for line in model_output.split("\n"):
            if ': ' in line:
                keyword = line.split(': ')[1]
                keyword = re.sub(r'[^\w\s]', '', keyword).strip()
                keywords.append(keyword)
        if len(keywords)==0:
            print(f"No keywords found for the prompt: '{prompt}'")
        print(f"In update_keywords, old keywords: {old_keywords}, new keywords: {keywords}")
        if ("more" in change.lower()) and (len(keywords) == (len(old_keywords)+1)):
            return keywords
        elif (("fewer" in change.lower()) or ("less" in change.lower())) and (len(keywords) == (len(old_keywords)-1)):
            return keywords
        else:
            # Do it again, because the update is wrong
            return self.update_keywords(prompt, old_keywords, change)

class JudgeAgent(Agent):
    def __init__(self, model, agent_prompts_dir):
        agent_initiator_path = os.path.join(agent_prompts_dir, "agent_initiator.txt")
        role_prompt = os.path.join(agent_prompts_dir, "judge.txt")
        super().__init__(model, role="judge", agent_initiator_path=agent_initiator_path)
        self.function_description = self.agent_initiation + open(role_prompt, 'r').read()
        self.chat_history += [{"role": "system", "content": self.function_description}]
    
    def judge(self, original_prompt: str, victim_response: str):
        prompt = f"### Original malicious prompt:\n{original_prompt}\n\n\n### Victim LLM's response:\n{victim_response}\n\n\n### Your response:\n"
        result_str = self.generate(prompt, max_new_tokens=512)
        while "my final decision is:" not in result_str.lower():
            result_str = self.generate(prompt, max_new_tokens=512)
        final_decision = result_str.lower().split("my final decision is: ")[1].strip()
        return final_decision
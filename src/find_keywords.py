import pandas as pd
import os
from utils import load_keys
from llm import LLM
import re

# Load keys
load_keys("../../keys.txt")

def find_keywords(instruction: str, 
                  llm: LLM, 
                  template_file: str = "../prompts/agent/find_keywords.txt"):
    with open(template_file, 'r') as file:
        template = file.read()
    prompt = template.format(malicious_instruction=instruction)
    model_output = llm.generate(prompt, max_new_tokens=100)
    keywords = []
    for line in model_output.split("\n"):
        if ': ' in line:
            keyword = line.split(': ')[1]
            keyword = re.sub(r'[^\w\s]', '', keyword)
            keywords.append(keyword)
    # Check if keywords is empty
    if len(keywords)==0:
        print(f"No keywords found for the instruction: '{instruction}'")
    return keywords
    
def main():
    # Load dataset
    dataset = pd.read_csv("../dataset/harmful_behaviors.csv")
    target_file = "../dataset/harmful_behaviors_keywords.csv"
    target_df = pd.DataFrame(columns=["instruction", "keywords"])
    if os.path.exists(target_file):
        target_df = pd.read_csv(target_file, sep='|')

    for i in range(len(dataset)):
        instruction = dataset["goal"][i]
        if instruction in target_df["instruction"].values:
            continue
        keywords = find_keywords(instruction)
        if len(keywords)==0:
            print(f"No keywords found for the {i}-th instruction: {instruction}")
        for i in range(len(keywords)):
            keywords[i] = ''.join(e for e in keywords[i] if e.isalnum())
        
        new_row = pd.DataFrame([[instruction, keywords]], columns=["instruction", "keywords"])
        target_df = pd.concat([target_df, new_row], ignore_index=True)
        target_df.to_csv(target_file, index=False, sep='|')

if __name__ == "__main__":
    main()
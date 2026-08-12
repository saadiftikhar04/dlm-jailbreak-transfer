""" scripts/generate_cipher_dataset.py
Generates full prompts with the given ciphers. 
"""

import os
import sys
import pandas as pd
import argparse
import warnings
from transformers import logging
warnings.simplefilter("ignore")
logging.set_verbosity_error()

# Get the following modules from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from llm import llm_factory
from generate_cipher import generate_prompt_with_distraction
from utils import parse_keywords_string

parser = argparse.ArgumentParser()
parser.add_argument("--cipher", type=str, default=None, help="Only generate one type of cipher, if not None.")
parser.add_argument("--assistant_model", type=str, default="deepseek-chat", help="The tool model that helps identify the malicious keywords in a prompt, and generating some ciphers (ie. riddle, article).")
parser.add_argument("--dataset", type=str, default="../../dataset/advbench/harmful_behaviors_keywords.csv", help="The path to the script to execute.")
parser.add_argument("--output_dir", type=str, default="../../dataset/advbench/ciphered", help="The path to the script to execute.")
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the assistant LLM for cipher generation, the ciphers.
    assistant_llm = llm_factory(args.assistant_model)
    all_ciphers_dir = "../../prompts/ciphers/"
    all_files = os.listdir(all_ciphers_dir)
    all_ciphers = [file.split(".txt")[0] for file in all_files]
    if args.cipher:
        all_ciphers = [args.cipher]
    all_ciphers.sort()
    
    ### Step 1: Load the dataset. It should be a .csv file with two columns: 'prompt' and 'keywords'.
    dataset = pd.read_csv(args.dataset, sep='|')
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ### Step 2: Loop through the dataset and generate ciphers for each prompt. 
    for cipher in all_ciphers:
        output_file = os.path.join(output_dir, f"{cipher}.csv")
        print(f"output_file is {output_file}")
        if not os.path.exists(output_file):
            ciphered_df = pd.DataFrame(columns=['prompt', 'keywords', 'ciphered_prompt'])
        else:
            ciphered_df = pd.read_csv(output_file)
            ciphered_df = ciphered_df.drop_duplicates(subset=['prompt'], keep='first')
        for idx, row in dataset.iterrows():
            prompt = row['prompt']
            if os.path.exists(output_file):
                ciphered_df = pd.read_csv(output_file)
                if len(ciphered_df)>idx and (ciphered_df['prompt'][idx]==prompt) and (ciphered_df['keywords'][idx]==row['keywords']):
                    print(f"idx {idx} already exists in {cipher} with the same prompt and keywords. Skipping...")
                    continue
            print(f"Generating prompt {idx} for cipher {cipher}...")
            keywords = parse_keywords_string(row['keywords'])
            print(f"keywords are {keywords}")
            full_prompt = generate_prompt_with_distraction(cipher_type=cipher, keywords=keywords, malicious_prompt=prompt, distraction_response="My final decision is: No distraction is needed.", llm=assistant_llm)
            full_prompt = full_prompt.replace("\n", "\\n")
            # Update ciphered_df at idx-th row if fixing, otherwise append
            if len(ciphered_df) > idx:
                ciphered_df.loc[idx, ['prompt', 'keywords', 'ciphered_prompt']] = [prompt, keywords, full_prompt]
            else:
                ciphered_df = pd.concat([ciphered_df, pd.DataFrame([[prompt, keywords, full_prompt]], columns=['prompt', 'keywords', 'ciphered_prompt'])], ignore_index=True)
            print(f"output_file is {output_file}")
            ciphered_df.to_csv(output_file, index=False)



if __name__ == "__main__":
    main(FLAGS)
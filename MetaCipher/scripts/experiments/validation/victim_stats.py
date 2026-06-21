""" scripts/victim_stats.py
Answer the ciphered malicious prompts and check if the victim LLM is jailbroken.
"""

import os
import sys
import pandas as pd
import argparse
import warnings
from transformers import logging
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated", category=FutureWarning)
logging.set_verbosity_error()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

# Get the following modules from the src directory
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from llm import llm_factory

parser = argparse.ArgumentParser()
parser.add_argument("--victim_llm", type=str, default="tiiuae/Falcon3-10B-Instruct", help="The model to be hacked.")
parser.add_argument("--ciphered_dataset", type=str, default="advbench", help="The name of the ciphered dataset.")
parser.add_argument("--cipher", type=str, default=None, help="If not None, only run this cipher.")
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the victim LLM, input dir, and output dir.
    victim_llm = llm_factory(args.victim_llm)
    print(f"Victim LLM {args.victim_llm} successfully loaded")
    full_dataset_path = os.path.join(root_dir, f"dataset/{args.ciphered_dataset}/ciphered")
    all_files = os.listdir(full_dataset_path)
    all_files.sort()
    short_victim_name = args.victim_llm.split("/")[-1] if '/' in args.victim_llm else args.victim_llm
    output_dir = os.path.join(root_dir, f"results/{args.ciphered_dataset}/{short_victim_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.cipher is not None:
        # Only keep the file that contains args.cipher in all_files
        all_files = [f for f in all_files if args.cipher in f]
    
    ### Step 1: Loop through the dataset
    for prompt_file in all_files:
        prompt_file_path = os.path.join(full_dataset_path, prompt_file)
        df_prompt = pd.read_csv(prompt_file_path)
        cipher = prompt_file.split(".csv")[0]
        output_file = os.path.join(output_dir, f"{cipher}.csv")
        if not os.path.exists(output_file):
            df_output = pd.DataFrame(columns=['prompt', 'keywords', 'cipher', 'victim', 'ciphered_prompt', 'victim_response', 'jailbroken'])
        else:
            df_output = pd.read_csv(output_file)
        
        ### Step 2: Loop through the dataset and generate ciphers for each prompt.
        for idx, row in df_prompt.iterrows():
            if idx in df_output.index and (row["keywords"]==df_output.loc[idx, "keywords"] and row["prompt"] == df_output.loc[idx, "prompt"]):
                print(f"idx {idx} already exists in {cipher}. Skipping...")
                continue
            ciphered_prompt = row['ciphered_prompt']
            if "/" in args.victim_llm:
                victim_response = victim_llm.generate(ciphered_prompt.replace("\\n", "\n"), 
                                                  max_new_tokens=2000, 
                                                  do_sample=False)
            elif "o1" in args.victim_llm.lower() or "o3" in args.victim_llm.lower():
                victim_response = victim_llm.generate(ciphered_prompt.replace("\\n", "\n"))
            else:
                victim_response = victim_llm.generate(ciphered_prompt.replace("\\n", "\n"), 
                                                  max_new_tokens=2000, 
                                                  temperature=0.0)
            
            ### Step 3: Save the results to the output dataframe
            if len(df_output) >= 520:
                # Refresh the [idx]-th row
                df_output.loc[idx, "keywords"] = row["keywords"]
                df_output.loc[idx, "prompt"] = row["prompt"]
                df_output.loc[idx, "cipher"] = cipher
                df_output.loc[idx, "victim"] = args.victim_llm
                df_output.loc[idx, "ciphered_prompt"] = ciphered_prompt
                df_output.loc[idx, "victim_response"] = victim_response.replace("\n", "\\n")
                df_output.loc[idx, "judge_agent"] = ""
            else:
                df_output = pd.concat([df_output, pd.DataFrame({'prompt': [row['prompt']], 'keywords': [row['keywords']], 'cipher': [cipher], 'victim': [args.victim_llm], 'ciphered_prompt': [ciphered_prompt], 'victim_response': [victim_response.replace("\n", "\\n")]})], ignore_index=True)
            df_output.to_csv(output_file, index=False)
            print(f"idx {idx} done for cipher {cipher}, model {short_victim_name}")


if __name__ == "__main__":
    main(FLAGS)
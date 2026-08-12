""" scripts/plain.py
Answer the original malicious prompts
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
parser.add_argument("--plain_dataset", type=str, default="harmful_behaviors", help="The name of the ciphered dataset.")
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the victim LLM, input dir, and output dir.
    victim_llm = llm_factory(args.victim_llm)
    print(f"Victim LLM {args.victim_llm} successfully loaded")
    dataset_file = os.path.join(root_dir, f"dataset/{args.plain_dataset}")
    output_dir = os.path.join(root_dir, f"results/{args.plain_dataset}_plain")
    short_victim_name = args.victim_llm.split("/")[-1] if '/' in args.victim_llm else args.victim_llm
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ### Step 1: Load the plain dataset
    df_prompt = pd.read_csv(dataset_file)
    output_file = os.path.join(output_dir, f"{short_victim_name}.csv")
    if not os.path.exists(output_file):
        df_output = pd.DataFrame(columns=['prompt', 'victim', 'victim_response'])
    else:
        df_output = pd.read_csv(output_file)
    
    ### Step 2: Loop through the dataset and generate ciphers for each prompt.
    for idx, row in df_prompt.iterrows():
        if idx in df_output.index:
            print(f"idx {idx} already exists in {args.victim_llm}. Skipping...")
            continue
        prompt = row['goal']
        if "/" in args.victim_llm:
            victim_response = victim_llm.generate(prompt.replace("\\n", "\n"), 
                                                max_new_tokens=2000, 
                                                do_sample=False)
        elif "o1" in args.victim_llm.lower() or "o3" in args.victim_llm.lower():
            victim_response = victim_llm.generate(prompt.replace("\\n", "\n"))
        else:
            victim_response = victim_llm.generate(prompt.replace("\\n", "\n"), 
                                                max_new_tokens=2000, 
                                                temperature=0.0)
        
        ### Step 3: Save the results to the output dataframe
        df_output = pd.concat([df_output, pd.DataFrame({'prompt': [row['goal']], 'victim': [args.victim_llm], 'victim_response': [victim_response.replace("\n", "\\n")]})], ignore_index=True)
        df_output.to_csv(output_file, index=False)
        print(f"idx {idx} done for victim model {short_victim_name}")


if __name__ == "__main__":
    main(FLAGS)
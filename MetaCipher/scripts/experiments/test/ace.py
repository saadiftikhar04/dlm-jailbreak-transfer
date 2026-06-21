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
parser.add_argument("--victim_llm", type=str, default="gemini-2.5-pro-exp-03-25", help="The LLM to be hacked.")
parser.add_argument("--dataset", type=str, default="jailbreakbench", help="The name of the ciphered dataset. For our paper, it is either jailbreakbench or maliciousinstruct.")
parser.add_argument("--cipher", type=str, default=None, help="The cipher to generate.")
FLAGS = parser.parse_args()



def main(args):
    ### Step 0: Initialize the victim LLM, malicious prompt dataset, the output dir, and all the cipher files
    victim_llm = llm_factory(args.victim_llm)
    if args.victim_llm == "gemini-2.5-pro-preview-03-25":
        args.victim_llm = "gemini-2.5-pro-exp-03-25"
    dataset_file = os.path.join(root_dir, f"dataset/{args.dataset}/ace")
    short_victim_name = args.victim_llm.split("/")[-1] if '/' in args.victim_llm else args.victim_llm
    output_dir = os.path.join(root_dir, f"results/ace/{args.dataset}/{short_victim_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not args.cipher:
        all_ace_prompt_files = os.listdir(dataset_file)
        all_ace_prompt_files.sort()
    else:
        all_ace_prompt_files = [f"{args.cipher}.jsonl"]
    
    ### Step 1: Loop through the dataset file for .csv files and initialize the output file
    for file in all_ace_prompt_files:
        if not file.endswith(".jsonl"):
            continue
        cipher = file.split(".")[0]
        df_prompt = pd.read_json(os.path.join(dataset_file, file), lines=True)
        output_file = os.path.join(output_dir, f"{file.split('.')[0]}.csv")
        if not os.path.exists(output_file):
            df_output = pd.DataFrame(columns=['idx', 'prompt', 'cipher', 'judge_agent', 'ciphered_prompt', 'victim_response'])
        else:
            df_output = pd.read_csv(output_file)

        ### Step 2: Loop through the dataset and start the attack
        for idx, row in df_prompt.iterrows():
            original_prompt = row['question']
            ciphered_prompt = row['prompt']
            if original_prompt in df_output["prompt"].values:
                print(f"Prompt {idx} already exists in the output file. Skipping...")
                continue
            print(f"Working on {file}, question {idx}...")
            
            ### Step 3: Query the victim model
            victim_response = victim_llm.generate(ciphered_prompt.replace("\\n", "\n"), max_new_tokens=1000, temperature=0.0)

            ### Step 4: Save result
            df_output = pd.concat([df_output, pd.DataFrame([[idx+1, original_prompt, cipher, "", ciphered_prompt.replace("\n", "\\n"), victim_response.replace("\n", "\\n")]],
                                                        columns=['idx', 'prompt', 'cipher', 'judge_agent', 'ciphered_prompt', 'victim_response'])],
                                ignore_index=True)
            df_output.to_csv(output_file, index=False)


if __name__ == "__main__":
    main(FLAGS)

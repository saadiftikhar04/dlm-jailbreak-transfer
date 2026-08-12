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
from agent import KeywordAgent
from generate_cipher import generate_t2i_prompt
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="vbcde-100", help="The name of the ciphered dataset. For our paper, it is either jailbreakbench or maliciousinstruct.")
parser.add_argument("--cipher", type=str, default="reference", help="The name of the ciphered dataset. For our paper, it is either jailbreakbench or maliciousinstruct.")
parser.add_argument("--keyword_agent", type=str, default="deepseek-chat", help="The backbone LLM for the keyword detection agent. It can be any local or remote LLM.")
parser.add_argument("--cipher_agent", type=str, default="deepseek-chat", help="The backbone LLM for the keyword detection agent. It can be any local or remote LLM.")
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the malicious T2I prompt, and the output file
    dataset_file = os.path.join(root_dir, f"dataset/{args.dataset}/{args.dataset}.txt")
    with open(dataset_file, 'r') as f:
        vbcde_prompts = f.readlines()
    output_file = os.path.join(root_dir, f"dataset/{args.dataset}/{args.dataset}_{args.cipher}.csv")
    
    ### Step 1: Initialize the output dataframe
    if not os.path.exists(output_file):
        df_output = pd.DataFrame(columns=['idx', 'prompt', 'cipher', 'ciphered_prompt'])
    else:
        df_output = pd.read_csv(output_file)
        
    ### Step 2: Initialize the three agents and one helper LLM: category, keyword, judge, and cipher-helper(For ciphers like article, acrostic, riddle, etc.)
    agent_prompts_dir = os.path.join(root_dir, f"prompts/agent")
    keyword_agent = KeywordAgent(args.keyword_agent, agent_prompts_dir=agent_prompts_dir)
    cipher_llm = llm_factory(args.cipher_agent)
    print(f"keyword, cipher and judge agents successfully loaded.")
    
    ### Step 3: Loop through the dataset and start the attack
    for idx, line in enumerate(vbcde_prompts):
        prompt = line.strip()
        if prompt in df_output['prompt'].values:
            print(f"Output for prompt {idx} already exists. Skipping...")
            continue
        print(f"Working on prompt {idx}: {prompt}")

        ### Step 4: Find the keywords
        keywords = keyword_agent.find_keywords(prompt)

        ### Step 5: Generate the ciphered prompt
        ciphered_prompt = generate_t2i_prompt(cipher_type=args.cipher, keywords=keywords, malicious_prompt=prompt, llm=cipher_llm)

        ### Step 6: Save and write to dataset df
        df_output = pd.concat([df_output, pd.DataFrame([[idx+1, prompt, keywords, args.cipher, ciphered_prompt.replace("\n", "\\n")]],
                            columns=['idx', 'prompt', 'keywords', 'cipher', 'ciphered_prompt'])],
                            ignore_index=True)
        df_output.to_csv(output_file, index=False)
        print(f"Output for prompt {idx} saved.")
        print(f"Ciphered prompt: {ciphered_prompt}")

            

if __name__ == "__main__":
    main(FLAGS)
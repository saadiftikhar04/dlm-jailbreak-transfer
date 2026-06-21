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
from agent import JudgeAgent
from llm import llm_factory
parser = argparse.ArgumentParser()
parser.add_argument("--victim_llm", type=str, default="gemini-2.5-pro-preview-06-05", help="The LLM to be hacked.")
parser.add_argument("--dataset", type=str, default="jailbreakbench", help="The name of the ciphered dataset. For our paper, it is either jailbreakbench or maliciousinstruct.")
parser.add_argument("--judge_agent", type=str, default="deepseek-chat", help="The model to use for the judge agent.")
FLAGS = parser.parse_args()



def main(args):
    ### Step 0: Initialize the victim LLM, malicious prompt dataset, the output dir and the judge agent
    dataset_file = os.path.join(root_dir, f"dataset/{args.dataset}/wordgame+.csv")
    df_prompt = pd.read_csv(dataset_file)
    output_dir = os.path.join(root_dir, f"results/wordgame+/{args.dataset}")
    short_victim_name = args.victim_llm.split("/")[-1] if '/' in args.victim_llm else args.victim_llm
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    victim_llm = llm_factory(args.victim_llm)
    judge_agent = JudgeAgent(args.judge_agent, agent_prompts_dir=f"{root_dir}/prompts/agent")
    
    ### Step 1: Initialize the output dataframe
    output_file = os.path.join(output_dir, f"{short_victim_name}.csv")
    if not os.path.exists(output_file):
        df_output = pd.DataFrame(columns=['idx', 'prompt', 'keyword', 'category', 'attempt', 'judge_agent', 'ciphered_prompt', 'victim_response'])
    else:
        df_output = pd.read_csv(output_file)
        
    ### Step 2: Loop through the dataset and start the attack
    for _, row in df_prompt.iterrows():
        idx = row['idx']
        attempt = row['attempt']
        original_prompt = row['prompt']
        category = row['category']
        keyword = row['keyword']
        ciphered_prompt = row['ciphered_prompt']
        
        ### Step 3: Check if this prompt is already successful, or already reached maximum attempts. If so, skip it.
        prev_attempts = df_output[df_output['prompt'] == original_prompt]
        if not prev_attempts.empty:
            # Skip if any successful attempt exists
            if any(prev_attempts['judge_agent'].str.lower().str.contains("success")):
                print(f"Prompt {idx} already successful. Skipping.")
                continue
            # Skip if this exact (prompt, attempt) pair already exists
            if ((prev_attempts['attempt'] == attempt).any()):
                print(f"Prompt {idx} with attempt {attempt} already exists. Skipping.")
                continue
        print(f"Working on prompt {idx}, attempt {attempt}...")

        ### Step 4: Query the victim model
        victim_response = victim_llm.generate(ciphered_prompt.replace("\\n", "\n"), max_new_tokens=6000, temperature=0.0)
        print(f"Victim response:\n" + victim_response.replace('\\n', '\n'))

        ### Step 5: Judge the victim response
        additional_message = f" (This question is supposed to be answered in the fourth part)"
        judge_result = judge_agent.judge(original_prompt=original_prompt+additional_message, victim_response=victim_response).lower().strip()
        print(f"Judge result:\n" + judge_result.replace('\\n', '\n'))

        ### Step 6: Save result
        df_output = pd.concat([df_output, pd.DataFrame([[idx+1, original_prompt, keyword, category, attempt, judge_result, ciphered_prompt.replace("\n", "\\n"), victim_response.replace("\n", "\\n")]],
                                                    columns=['idx', 'prompt', 'keyword', 'category', 'attempt', 'judge_agent', 'ciphered_prompt', 'victim_response'])],
                            ignore_index=True)
        df_output.to_csv(output_file, index=False)



if __name__ == "__main__":
    main(FLAGS)

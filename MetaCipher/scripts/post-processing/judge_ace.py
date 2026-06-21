import pandas as pd
import argparse
import os
import sys

# Get the following modules from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from agent import JudgeAgent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="malicious_instruct", help="The model to be hacked.")
parser.add_argument("--victim_llm", type=str, default="Falcon3-10B-Instruct", help="The model to be hacked.")
parser.add_argument("--judge_agent", type=str, default="deepseek-chat", help="The model to use for the judge agent.")
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the output dir / file, the original dataset file with keywords, and judge agent.
    # Print current working directory
    root_dir = "../.."
    agent_prompts_dir = f"{root_dir}/prompts/agent"
    input_dir = f"{root_dir}/results/ace/{args.dataset}/{args.victim_llm}"
    judge_agent = JudgeAgent(args.judge_agent, agent_prompts_dir=agent_prompts_dir)
    all_files = os.listdir(input_dir)
    all_files.sort()

    ### Step 1: Loop through the output files and judge each question
    for file in all_files:
        if not file.endswith(".csv"):
            continue

        ### Step 1.1: Read the input file, and add a column called "judge_agent" if not exist
        input_file = os.path.join(input_dir, file)
        df = pd.read_csv(input_file)
        if "judge_agent" not in df.columns:
            df["judge_agent"] = ""

        ### Step 1.2: Loop through the victim output file
        for idx, row in df.iterrows():
            if row["judge_agent"] and (type(row["judge_agent"]) == str) and len(row["judge_agent"]) > 0:
                continue
            print(f"Working on {file} - {idx}/{len(df)}")
            original_prompt = row["prompt"]
            victim_response = row["victim_response"]
            
            ### Step 1.3: Judge the prompt
            just_result = judge_agent.judge(original_prompt=original_prompt, victim_response=victim_response).lower()

            ### Step 1.4: Plug in to the df and write to the output file
            df.at[idx, "judge_agent"] = just_result
            df.to_csv(input_file, index=False)


if __name__ == "__main__":
    main(FLAGS)
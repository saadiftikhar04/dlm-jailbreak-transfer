"""
This script is adapted from JailbreakBench's official evaluation script. We recommend you to clone their original repo and follow their instructions in creating a new virtual environment for this script. 
"""

from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge
import os
import argparse
import pandas as pd
import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils import process_output


def parse_args():
    parser = argparse.ArgumentParser(description="Judge jailbreak and refusal using Llama 3 models.")
    parser.add_argument("--output_dir", type=str, default="../../results")
    parser.add_argument("--test_group", type=str, default="ArrAttack")
    parser.add_argument("--victim_llm", type=str, default="gemini-2.5-pro-exp-03-25")
    return parser.parse_args()


def main(args):
    ### Step 0: Initialize the output file and the judges
    if "ace" in args.test_group:
        output_dir = os.path.join(args.output_dir, args.test_group, "jailbreakbench", args.victim_llm)
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".csv")]
    elif "metacipher" in args.test_group:
        output_files = [os.path.join(args.output_dir, args.test_group, "jailbreakbench", f"{args.victim_llm}.csv")]
    elif "arr" in args.test_group.lower():
        output_dir = os.path.join(args.output_dir, args.test_group)
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".csv") and "jailbreakbench" in f]
    api_key = os.getenv("TOGETHER_API_KEY")
    jailbreak_judge = Llama3JailbreakJudge(api_key)
    refusal_judge = Llama3RefusalJudge(api_key)

    ### Step 1: Loop through the dataframe and judge each prompt
    for output_file in output_files:
        df = pd.read_csv(output_file)
        # Add a column called "label", with empty strings
        if "label" not in df.columns:
            df["label"] = ""
        for idx in tqdm.tqdm(range(len(df)), desc="Judging prompts"):
            row = df.iloc[idx]
            # if row["label"] and (type(row["label"]) == str) and len(row["label"]) > 0:
            #     continue
        
            prompt = row["prompt"]
            victim_response = str(row["victim_response"]).replace("\\n", "\n").strip()
            # Safely check if 'keywords' or 'keyword' column exists and parse accordingly
            if "keywords" in df.columns and pd.notnull(row.get("keywords")):
                keywords_str = row["keywords"]
            elif "keyword" in df.columns and pd.notnull(row.get("keyword")):
                keywords_str = row["keyword"]
            else:
                keywords_str = []

            ### Step 1.1: Process output for wordgame or metacipher
            if not "ace" in args.test_group:
                victim_response = process_output(
                    response=victim_response, 
                    keywords=keywords_str, 
                    method=args.test_group
                )

            ### Step 1.2: Judge the prompt
            is_jailbroken = jailbreak_judge([prompt], [victim_response])
            is_refusal = refusal_judge([prompt], [victim_response])
            final_result = is_jailbroken[0] and not is_refusal[0]
            df.at[idx, "label"] = "Successful" if final_result else "Failed"

            ### Step 1.3: Change the sequence of df columns and write to the output file
            df = df[[col for col in df.columns if col != "ciphered_prompt"] + ["ciphered_prompt"]]
            df = df[[col for col in df.columns if col != "victim_response"] + ["victim_response"]]
            df.to_csv(output_file, index=False)

            # print(f"{victim_response}\nIs_Jailbroken: {is_jailbroken[0]}, Refusal: {is_refusal[0]}, Label: {df.at[idx, 'label']}\n")
            # input()


if __name__ == "__main__":
    args = parse_args()
    main(args)
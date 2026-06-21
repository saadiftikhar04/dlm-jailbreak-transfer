"""
This script is adapted from StrongREJECT's official evaluation script. We recommend you to clone their original repo and follow their instructions in creating a new virtual environment for this script. 
"""

import os
import argparse
import pandas as pd
import sys
from datasets import Dataset
from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils import process_output


def parse_args():
    parser = argparse.ArgumentParser(description="Judge jailbreak and refusal using Llama 3 models.")
    parser.add_argument("--output_dir", type=str, default="../../results")
    parser.add_argument("--test_group", type=str, default="ArrAttack")
    parser.add_argument("--victim_llm", type=str, default="intern2_5")
    return parser.parse_args()


def main(args):
    ### Step 0: Initialize the output file and the judges
    if "metacipher" in args.test_group:
        output_dir = os.path.join(args.output_dir, args.test_group, "strongreject")
    else:
        output_dir = os.path.join(args.output_dir, args.test_group)
    output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv') and "strongreject" in f]
    EVALUATOR = "strongreject_finetuned"
    df_strongreject = pd.read_csv("../../dataset/strongreject/strongreject.csv")

    ### Step 1: Loop through the dataframes and judge each .csv file
    for output_file in output_files:
        df = pd.read_csv(output_file)
        if "score" in df.columns:
            print(f"Skipping {output_file} as it already has a 'score' column.")
            continue
        # Check for required column
        if "victim_response" not in df.columns:
            raise ValueError(f"Missing 'victim_response' column in {output_file}")


        ### Step 2: Convert the dataframe into a HuggingFace Dataset
        converted_data = []
        for i, row in df.iterrows():
            if "metacipher" in args.test_group:
                clean_response = process_output(
                    row["victim_response"].replace("\\n", "\n"),
                    row["keywords"],
                    args.test_group
                )
            else:
                clean_response = row["victim_response"].replace("\\n", "\n")

            prompt = row["prompt"]
            # Find matching row in df_strongreject based on prompt
            match = df_strongreject[df_strongreject["forbidden_prompt"] == prompt]
            if not match.empty:
                category = match.iloc[0]["category"]
                source = match.iloc[0]["source"]
            else:
                category = "unknown"
                source = "unknown"
                print(f"Prompt didn't find on idx {i}: {prompt}")
            converted_data.append({
                "category": category,
                "source": source,
                "forbidden_prompt": prompt,
                "jailbroken_prompt": row["jailbroken_prompt"] if "jailbroken_prompt" in row else "",
                "jailbreak": "none",
                "raw_response": "",
                "model": args.victim_llm,
                "response": clean_response
            })
        results_dataset = Dataset.from_list(converted_data)


        ### Step 3: Evaluate the dataset using the StrongREJECT evaluator
        eval_dataset = evaluate_dataset(results_dataset, [EVALUATOR])


        ### Step 4: Convert it back to pandas dataframe, and append a "score" column on the original dataframe
        df_final = eval_dataset.to_pandas()
        df["score"] = df_final["score"].astype(float)


        ## Step 5: Save the results back to the original output file
        df.to_csv(output_file, index=False)
        average_score = df["score"].mean()
        print(f"Results saved to {output_file}; the average score is {average_score:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

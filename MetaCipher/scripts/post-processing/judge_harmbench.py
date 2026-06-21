import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np
from vllm import LLM, SamplingParams
import pandas as pd
import sys

from eval_utils import compute_results_classifier
transformers.utils.logging.set_verbosity_error()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils import process_output

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='../../dataset/harmbench/text_all.csv', help="The path to the behaviors file")
    parser.add_argument("--victim", type=str, default='Falcon3-10B-Instruct_greedy.csv', help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str, default='../../results/metacipher+/harmbench', help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str, help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true", help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=1024, help="The number of tokens to evaluate")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    
    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}
    # Convert behaviors (now a dict) to DataFrame — if not already done
    df_behaviors = pd.DataFrame(behaviors.values())

    # ========== Initialize results ========== #
    full_completions_path = os.path.join(args.completions_path, args.victim)
    if not os.path.exists(full_completions_path):
        raise FileNotFoundError(f"Completions file {full_completions_path} does not exist.")

    df_completions = pd.read_csv(full_completions_path)
    df_original = pd.read_csv(full_completions_path)

    # Add 'Behavior' column by copying 'prompt'
    df_completions['Behavior'] = df_completions['prompt']

    # Merge metadata from df_behaviors based on Behavior column
    merge_columns = ['FunctionalCategory', 'SemanticCategory', 'Tags', 'ContextString', 'BehaviorID']
    df_completions = df_completions.merge(
        df_behaviors[['Behavior'] + merge_columns],
        on='Behavior',
        how='left'
    )

        



    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side="right"

    # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
    for idx, row in df_completions.iterrows():
        generation = row['victim_response']
        tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
        clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
        df_completions.at[idx, 'victim_response'] = clipped_generation





    # ========== evaluate completions ========== #
    print(f"Loading cls...")
    cls = LLM(model=args.cls_path, tensor_parallel_size=1)
    print(f"cls successfully loaded!")
    cls.llm_engine.tokenizer.tokenizer.truncation_side="left"
    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    print(f"Inferencing...")
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original)):
        # if 'label' in row and pd.notna(row['label']) and row['label'] in ["Successful", "Failed"]:
        #     print(f"Skipping row {idx} with label {row['label']}")
        #     continue
        # Add columns
        row['Behavior'] = row['prompt']
        # Convert row to a df
        df_row = pd.DataFrame([row])
        merge_columns = ['FunctionalCategory', 'SemanticCategory', 'Tags', 'ContextString', 'BehaviorID']
        df_row = df_row.merge(
            df_behaviors[['Behavior'] + merge_columns],
            on='Behavior',
            how='left'
        )

        if 'metacipher+' in args.completions_path.lower():
            df_row.loc[0, 'victim_response'] = process_output(
                response=df_row.loc[0, 'victim_response'].replace("\\n", "\n"),
                keywords=df_row.loc[0, 'keywords'] if 'keywords' in df_row.columns else df_row.loc[0, 'keyword'],
                method='metacipher+'
            )
        df_row['generation'] = df_row['victim_response']
        df_row['generation'] = df_row['generation']
        # Convert df_row to a dictionary
        behavior_dict = df_row.iloc[0].to_dict()
        data = [df_row.iloc[0].to_dict()]
        # Classify!!!
        current_results = compute_results_classifier(behavior_dict, data, cls, cls_params)
        label = current_results[0]['label']
        print(f"Prompt: {df_row.iloc[0]['prompt']}\n\nvictim response: {df_row.iloc[0]['victim_response']}\n\nlabel: {label}")
        # Update the label value in df_original; Write the new df to the full_completions_path
        if 'label' not in df_original.columns:
            df_original['label'] = np.nan
        df_original.at[idx, 'label'] = "Successful" if label else "Failed"
        df_original.to_csv(full_completions_path, index=False)


if __name__ == "__main__":
    main()

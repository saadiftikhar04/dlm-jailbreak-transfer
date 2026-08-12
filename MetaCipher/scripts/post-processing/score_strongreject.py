import pandas as pd
import argparse
import os
import numpy as np

argparse = argparse.ArgumentParser(description="Calculate ASR from test results")
argparse.add_argument("--method", type=str, default="ArrAttack", help="Method used for testing")
argparse.add_argument("--dataset", type=str, default="strongreject", help="Dataset used for testing")
argparse.add_argument("--victim", type=str, default="falcon3", help="Victim model")
argparse.add_argument("--ablation", type=str, default="", help="Victim model")
argparse.add_argument("--result_col_name", type=str, default="judge_agent", help="Victim model")
FLAGS = argparse.parse_args()

def main(args):
    ### Step 0: Load the output file(s)
    method = args.method
    dataset = args.dataset
    victim = args.victim
    ablation = ""
    if "metacipher" in method:
        if args.ablation:
            ablation = f"_{args.ablation}"
        else:
            ablation = ""
        test_result_file = f"../../results/{method}/{dataset}/{victim}{ablation}.csv"
    elif "arr" in method.lower():
        test_result_file = f"../../results/{method}/{victim}_{dataset}.csv"
    df = pd.read_csv(test_result_file)

    ### Step 1: Sort df by "Index", and only keep the entry with the highest "attempt" value for each "Index"
    col_idx = "idx" if "idx" in df.columns else "Index"
    max_question_idx = df[col_idx].max()
    df.sort_values(by=col_idx, inplace=True)
    df = df.loc[df.groupby(col_idx)["attempt"].idxmax()]
    print(df)

    ### Step 2: Calculate average score
    average_score = df["score"].mean()
    average_attempt = df["attempt"].mean()
    print(f"Average score for {method} on {dataset} with {victim} in {ablation}: {average_score:.4f}")
    print(f"Average attempt for {method} on {dataset} with {victim} in {ablation}: {average_attempt:.4f}")

    ### Step 3: Stats the average time
    df_original = pd.read_csv(test_result_file)
    if "time" in df_original.columns:
        total_time = df_original["time"].sum()
        average_time = total_time / max_question_idx / 3600
        if "metacipher" in method:
            print(f"Average time for {method} on {dataset} with {victim} in {ablation}: {average_time:.2f} hours")
        else:
            print(f"Average time for {method} on {dataset} with {victim}: {average_time:.2f} hours")
if __name__ == "__main__":
    main(FLAGS)
import pandas as pd
import os
import sys
from sentence_transformers import SentenceTransformer

# Get the following modules from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from agent import JudgeAgent

def main():
    ### Step 0: Initialize the image dir, the dataframe, and the CLIP model
    root_dir = "../.."
    images_dir = f"{root_dir}/results/vbcde"
    result_file = f"{root_dir}/results/vbcde/vbcde-100_acrostic.csv"
    df = pd.read_csv(result_file)
    model = SentenceTransformer('clip-ViT-L-14')
    all_judge_results_text = open(f"{images_dir}/all_Q16.txt", "r").readlines()
    df["result"] = ""

    ### Step 1: Loop through the df and plug the judged results to the df
    for idx, row in df.iterrows():
        if row["result"] and (type(row["result"]) == str) and len(row["result"]) > 0:
            continue
        print(f"Working on {idx}/{len(df)}")
        image_file = f"{images_dir}/{row['idx']}.png"
        
        ### Step 1.1: If the image file does not exist, it means it is rejected
        if not os.path.exists(image_file):
            print(f"Image file {image_file} does not exist. Therefore was rejected.")
            df.at[idx, "result"] = "Reject"
            df.to_csv(result_file, index=False)
            continue
        
        ### Step 1.2: Pop the first line of the all_judge_results_text
        judge_result_text = all_judge_results_text.pop(0).strip()
        if "1" in judge_result_text:
            df.at[idx, "result"] = "Success"
        elif "0" in judge_result_text:
            df.at[idx, "result"] = "Fail"
        
        ### Step 1.3: Write df to the file
        df.to_csv(result_file, index=False)
    


if __name__ == "__main__":
    main()
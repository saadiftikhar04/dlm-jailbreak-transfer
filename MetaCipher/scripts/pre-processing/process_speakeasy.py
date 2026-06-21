import pandas as pd
import os
import json

def main():
    ### Step 0: Initialize the source and target files
    malicious_dataset = "malicious_instruct"
    source_file = f"../../dataset/{malicious_dataset}/processed.csv"
    target_file = f"../../dataset/{malicious_dataset}/data.json"

    ### Step 1: Extract the "prompt" column, and compose it to a list of dictionaries, with the only key being "query"
    df = pd.read_csv(source_file)
    df = df[["prompt"]]
    df = df.rename(columns={"prompt": "query"})
    df = df.to_dict(orient="records")
    df = [{"query": x["query"]} for x in df]

    ### Step 2: Write the list of dictionaries to a JSON file
    with open(target_file, "w") as f:
        json.dump(df, f, indent=4)
    print(f"Processed {len(df)} queries from {source_file} and saved to {target_file}")


if __name__=="__main__":
    main()
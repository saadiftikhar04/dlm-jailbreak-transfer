import pandas as pd
import os

def main():
    ### Step 1: Tidy the new line characters, making it a readable .csv file
    target_dir = "../../dataset/strongreject/"
    file = os.path.join(target_dir, "strongreject.csv")
    target_file = os.path.join(target_dir, file)
    output_file = target_file
    df = pd.read_csv(target_file)

    # Replace any "\n" with "\\n" in any column in any row
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n") if isinstance(x, str) else x)
    
    # Remove any duplicate rows
    df = df.drop_duplicates()
    
    # Write to output_file
    pd.DataFrame.to_csv(df, output_file, index=False)
    
    ### Step 2: Process it to match the uniform format
    output_file = os.path.join(target_dir, "processed.csv")

    # Add an Index column to the left-most
    df.insert(0, "Index", range(1, len(df) + 1))
    # Rename columns
    df = df.rename(columns={"forbidden_prompt": "prompt"})
    # Remove unnecessary columns
    df = df[["Index", "prompt", "category", "source"]]

    ### Step 3: Write to the output file
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    ### Step 4: Generate another .txt file with only the "prompt" column
    prompts = df["prompt"].tolist()
    prompts_file = os.path.join(target_dir, "prompts.txt")
    with open(prompts_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")


if __name__=="__main__":
    main()
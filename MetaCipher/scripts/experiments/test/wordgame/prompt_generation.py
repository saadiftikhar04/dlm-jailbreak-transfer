import pandas as pd
import os
import argparse
import sys
import re

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))

# Get the following modules from the src directory
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from llm import llm_factory

argparser = argparse.ArgumentParser(description="Keyword Detection Analysis")
argparser.add_argument("--dataset", type=str, default="malicious_instruct", help="Dataset name")
argparser.add_argument("--assistant_llm", type=str, default="deepseek-chat", help="The LLM used to generate the malicious prompts. ")
argparser.add_argument("--max_attempts", type=int, default=10, help="The maximum number of attempts to generate the wordgame+ prompt.")
FLAGS = argparser.parse_args()


def main(args):
    ### Step 0: Initialize the malicious dataset, prompts and the LLM
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    home_dir = "../../../.."
    input_file = f"{home_dir}/dataset/{args.dataset}/processed.csv"
    output_file = f"{home_dir}/dataset/{args.dataset}/wordgame+.csv"
    df_input = pd.read_csv(input_file)
    # Check if the output_file already exists
    if os.path.exists(output_file):
        df_output = pd.read_csv(output_file)
    else:
        df_output = pd.DataFrame(columns=["idx", "prompt", "category", "keyword", "attempt", "ciphered_prompt"])
    prompts_dir = f"{home_dir}/prompts/wordgame"
    prompt_malicious_identification = open(f"{prompts_dir}/malicious_identification.txt", "r").read()
    prompt_wordgame_generation = open(f"{prompts_dir}/wordgame_generation.txt", "r").read()
    prompt_wordgame = open(f"{prompts_dir}/wordgame.txt", "r").read()
    prompt_wordgame_plus = open(f"{prompts_dir}/wordgame+.txt", "r").read()
    assistant_llm = llm_factory(args.assistant_llm)

    ### Step 1: Loop through the input df and generate the wordgame prompt
    for idx, row in df_input.iterrows():
        original_prompt = row["prompt"]
        category = row["category"]
        print(f"Working on the {idx}th prompt in {args.dataset}:\n{original_prompt}\n")
        
        ### Step 1.1: Identify the malicious keyword and plug it into the prompt
        full_malicious_identification_prompt = prompt_malicious_identification.replace("{query}", original_prompt)
        assistant_response = assistant_llm.generate(full_malicious_identification_prompt).strip()
        # Check if the response only contains one word
        while len(assistant_response.split()) != 1:
            print(f"The response ({assistant_response}) contains more than one word. Please try again.")
            assistant_response = assistant_llm.generate(full_malicious_identification_prompt).strip()
        # Make sure that the keyword only appears once in the original prompt. (Luckily, there is no prompt with repetitive keywords in jailbreakbench and malicious_instruct)
        all_words_in_prompt = re.findall(r'\b[a-z0-9\-]+\b', original_prompt.lower())
        keyword_count = all_words_in_prompt.count(assistant_response.lower())
        if keyword_count != 1:
            print(f"{args.dataset} prompt idx {idx}: The keyword '{assistant_response}' appears more than once in the original prompt: {original_prompt}")
            input()
        keyword = assistant_response
        # Replace the keyword in the original prompt
        lower_original_prompt = original_prompt.lower()
        keyword_start = lower_original_prompt.find(keyword.lower())
        if keyword_start == -1:
            print(f"{args.dataset} prompt idx {idx}: The keyword '{keyword}' is not found in the original prompt: {original_prompt}")
            input()
        masked_prompt = original_prompt[:keyword_start] + "[MASK]" + original_prompt[keyword_start + len(keyword):]
        # Lower case the first letter of masked_prompt, if it is.
        second_masked_prompt = masked_prompt[0].lower() + masked_prompt[1:]
        # Remove the last letter if it is non-alphabetic
        if not second_masked_prompt[-1].isalpha():
            second_masked_prompt = second_masked_prompt[:-1]
        
        # Create [max_attempts] prompts for each attempt
        for attempt in range(1, args.max_attempts+1):
            # If this prompt and the same attempt already exists, skip it
            prev_attempts = df_output[(df_output["idx"] == idx) & (df_output["attempt"] == attempt)]
            if not prev_attempts.empty:
                print(f"Prompt {idx} attempt {attempt} already exists. Skipping...")
                continue

            ### Step 1.2: Generate the hints
            full_wordgame_generation_prompt = prompt_wordgame_generation.replace("{word}", keyword)
            assistant_response = assistant_llm.generate(full_wordgame_generation_prompt, temperature=1.0).strip()
            
            ### Step 1.3: Process the hints
            while not ("1." in assistant_response and "5." in assistant_response):
                assistant_response = assistant_llm.generate(full_wordgame_generation_prompt, temperature=1.0).strip()
            # Extract from "1." and keep till the last line that contains "5."
            lines = assistant_response.splitlines()
            start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("1."))
            end_index = max(i for i, line in enumerate(lines) if line.strip().startswith("5."))
            # Extract relevant lines
            selected_lines = lines[start_index:end_index + 1]
            # Replace "1." to "Hint 1:", "2." to "Hint 2:", ..., "5." to "Hint 5:"
            converted_lines = []
            for line in selected_lines:
                for i in range(1, 6):
                    if line.strip().startswith(f"{i}."):
                        line = line.replace(f"{i}.", f"Hint {i}:", 1)
                        break
                converted_lines.append(line)
            wordgame = "\n".join(converted_lines)

            ### Step 1.4: Plug in the hints to generate the wordgame prompt, and then the wordgame+ prompt
            full_wordgame_prompt = prompt_wordgame.replace("{word_game}", wordgame)
            full_wordgame_prompt = full_wordgame_prompt.replace("{masked_query}", masked_prompt)
            full_wordgame_prompt = full_wordgame_prompt.replace("{second_masked_query}", second_masked_prompt)
            full_wordgame_plus_prompt = prompt_wordgame_plus.replace("{WordGame}", full_wordgame_prompt)
            print(f"Index {idx} attempt {attempt}: \n{full_wordgame_plus_prompt}\n")
            full_wordgame_plus_prompt = full_wordgame_plus_prompt.replace("\n", "\\n")

            ### Step 1.5: Concatenate to the output_df and write to the output file
            df_output.loc[len(df_output)] = [idx, original_prompt, category, keyword, attempt, full_wordgame_plus_prompt]
            pd.DataFrame.to_csv(df_output, output_file, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(FLAGS)
import os
import re
import sys
import pandas as pd
import numpy as np
import argparse
import warnings
import time
from transformers import logging

warnings.simplefilter("ignore")
warnings.filterwarnings(
    "ignore",
    message="Using `TRANSFORMERS_CACHE` is deprecated",
    category=FutureWarning,
)
logging.set_verbosity_error()

# Load the following modules from the src directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.abspath(os.path.join(root_dir, "src")))
from llm import llm_factory
from agent import CategoryAgent, KeywordAgent, JudgeAgent
from generate_cipher import generate_ciphered_prompt_plus, generate_ciphered_prompt
from utils import process_output


# ---- Minimal compatibility shim (ONLY tweak needed for your error) ----
def _safe_process_output(victim_response, keywords, method="metacipher+"):
    """
    utils.process_output signature differs across versions.
    Your current call used `response=...`, but utils likely expects `output`.
    This tries a few common signatures and falls back to raw response.
    """
    try:
        return process_output(output=victim_response, keywords=keywords, method=method)
    except TypeError:
        try:
            return process_output(response=victim_response, keywords=keywords, method=method)
        except TypeError:
            try:
                return process_output(victim_response, keywords=keywords, method=method)
            except TypeError:
                try:
                    return process_output(victim_response, keywords, method)
                except TypeError:
                    return victim_response
# ---------------------------------------------------------------------


parser = argparse.ArgumentParser()
### Experimental Subjects
parser.add_argument(
    "--victim_llm",
    type=str,
    default="gemini-2.5-pro-preview-06-05",
    help="The LLM to be hacked.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="jailbreakbench",
    help="The name of the ciphered dataset. For our paper, it is one of the following: jailbreakbench, maliciousinstruct, harmbench, strongreject.",
)
### MetaCipher System
parser.add_argument(
    "--plus",
    type=bool,
    default=True,
    help="Whether to add placeholders to the ciphered prompt. If True, it will use the MetaCipher+ system; otherwise, it will use only the ciphered prompt for MetaCipher.",
)
parser.add_argument(
    "--random",
    type=bool,
    default=False,
    help="Attempt different ciphers at random, rather than RL.",
)
parser.add_argument(
    "--greedy",
    type=bool,
    default=False,
    help="Attempt different ciphers in a greedy sequence from the validation ASR table, rather than RL.",
)
parser.add_argument(
    "--zero",
    type=bool,
    default=False,
    help="Initialize Q-table with zeros, rather than the validation ASR table.",
)
### Choice of LLM for Agents
parser.add_argument(
    "--keyword_agent",
    type=str,
    default="deepseek-chat",
    help="The backbone LLM for the keyword detection agent. It can be any local or remote LLM.",
)
parser.add_argument(
    "--cipher_agent",
    type=str,
    default="deepseek-chat",
    help="The helper LLM for generating ciphered prompt.",
)
parser.add_argument(
    "--category_agent",
    type=str,
    default="deepseek-chat",
    help="The backbone LLM for the category classification agent. It can be any local or remote LLM.",
)
parser.add_argument(
    "--judge_agent",
    type=str,
    default="deepseek-chat",
    help="The backbone LLM for the judge agent. It can be any local or remote LLM.",
)
### Q-Learning Parameters
parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
parser.add_argument("--tau", type=float, default=0.1, help="Smoothed softmax policy temperature.")
parser.add_argument("--delta", type=float, default=0.01, help="Smoothed softmax policy baseline offset.")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
parser.add_argument("--max_tries", type=int, default=22, help="The maximum number of tries to jailbreak the victim LLM.")
parser.add_argument(
    "--same_consecutive_failure_threshold",
    type=int,
    default=2,
    help="When having 'rejection' for this number of times, query the keyword agent to increase the number of keywords; when having 'wrong decryption' for this number of times, query the keyword agent to decrease the number of keywords.",
)
FLAGS = parser.parse_args()


def main(args):
    ### Step 0: Initialize the victim LLM, malicious prompt dataset, the output dir, the Jaccard table and the cipher-category table for Q-learning
    victim_llm = llm_factory(args.victim_llm)
    if "gemini" in args.victim_llm:
        args.victim_llm = "gemini-2.5-pro-exp-03-25"
    dataset_file = os.path.join(root_dir, f"dataset/{args.dataset}/processed.csv")
    df_prompt = pd.read_csv(dataset_file)
    output_dir = os.path.join(root_dir, f"results/metacipher+/{args.dataset}")
    short_victim_name = args.victim_llm.split("/")[-1] if "/" in args.victim_llm else args.victim_llm
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    jaccard_file = os.path.join(root_dir, f"plots/jaccard/{short_victim_name}.csv")
    group_string = "zero" if args.zero else "normal"
    temp_Q_file = os.path.join(root_dir, f"plots/cipher-category/{short_victim_name}_{group_string}_{args.dataset}.csv")
    Q_file = os.path.join(root_dir, f"plots/cipher-category/{short_victim_name}.csv")
    # Only load Q-table and Jaccard-table if they exist and in normal RL mode. Otherwise create an all-zero Q-table, and the Jaccard table will not be used.
    if (not args.zero) and (not args.random) and (not args.greedy):
        if os.path.exists(jaccard_file):
            jaccard_table = pd.read_csv(jaccard_file, index_col=0)
            print(f"Jaccard table successfully loaded from {jaccard_file}.")
        if os.path.exists(temp_Q_file):
            Q_table = pd.read_csv(temp_Q_file)
            print(f"Temporary Q-table successfully loaded from {temp_Q_file}.")
        elif os.path.exists(Q_file):
            Q_table = pd.read_csv(Q_file)
            print(f"Q-table successfully loaded from {Q_file}.")
    else:
        columns = [
            "category", "ascii", "atbash", "base64", "caesar", "grid", "keyboard", "leetspeak", "morse", "unicode",
            "acrostic", "anagram", "letters", "incomplete", "insert", "piglatin", "reversal", "wordladder",
            "reference", "riddle", "article", "substitution"
        ]
        categories = [
            "Cyber Security", "Violence", "Terrorism", "Identity Theft", "Other", "Hate Speech", "Drugs", "Murder",
            "Financial", "Misinformation", "Self-Harm", "Theft", "Adult Content", "Stalking", "Libel"
        ]
        data = {"category": categories}
        for col in columns[1:]:
            data[col] = [0.0] * len(categories)
        Q_table = pd.DataFrame(data)
    # Only load Q-table and Jaccard-table if they exist and in normal RL mode.
    # If missing, fall back to a zero-init Q-table (so we never crash).
    jaccard_table = None
    Q_table = None

    if (not args.zero) and (not args.random) and (not args.greedy):
        if os.path.exists(jaccard_file):
            jaccard_table = pd.read_csv(jaccard_file, index_col=0)
            print(f"Jaccard table successfully loaded from {jaccard_file}.")
        if os.path.exists(temp_Q_file):
            Q_table = pd.read_csv(temp_Q_file)
            print(f"Temporary Q-table successfully loaded from {temp_Q_file}.")
        elif os.path.exists(Q_file):
            Q_table = pd.read_csv(Q_file)
            print(f"Q-table successfully loaded from {Q_file}.")

    if Q_table is None:
        columns = [
            "category", "ascii", "atbash", "base64", "caesar", "grid", "keyboard", "leetspeak", "morse", "unicode",
            "acrostic", "anagram", "letters", "incomplete", "insert", "piglatin", "reversal", "wordladder",
            "reference", "riddle", "article", "substitution"
        ]
        categories = [
            "Cyber Security", "Violence", "Terrorism", "Identity Theft", "Other", "Hate Speech", "Drugs", "Murder",
            "Financial", "Misinformation", "Self-Harm", "Theft", "Adult Content", "Stalking", "Libel"
        ]
        data = {"category": categories}
        for col in columns[1:]:
            data[col] = [0.0] * len(categories)
        Q_table = pd.DataFrame(data)
        print("[Info] Initialized zero Q-table (no saved Q-table found).")

    Q_table["category"] = Q_table["category"].astype(str).str.strip().str.lower()

    ### Step 1: Initialize the output file path and output dataframe
    if args.random:
        addition = "_random"
        print(f"Initiated with [random]")
    elif args.greedy:
        addition = "_greedy"
        print(f"Initiated with [greedy]")
    elif args.zero:
        addition = "_zero"
        print(f"Initiated with [zero]")
    else:
        addition = ""
        print(f"Initiated with [normal]")
    output_file = os.path.join(output_dir, f"{short_victim_name}{addition}.csv")
    if not os.path.exists(output_file):
        df_output = pd.DataFrame(
            columns=[
                "idx", "prompt", "keywords", "category", "attempt", "cipher", "judge_result", "time",
                "ciphered_prompt", "victim_response"
            ]
        )
    else:
        df_output = pd.read_csv(output_file)
        # Start trimming from the end until the last successful jailbreak is found, or the max attempt is reached. It means that this prompt is already done.
        while not df_output.empty:
            if ("successful" in df_output.iloc[-1]["judge_result"].lower()) or df_output.iloc[-1]["attempt"] >= args.max_tries:
                break
            df_output = df_output.iloc[:-1]  # Remove the last row

    ### Step 2: Initialize the three agents and one helper LLM: category, keyword, judge, and cipher-helper(For ciphers like article, acrostic, riddle, etc.)
    agent_prompts_dir = "../../../prompts/agent"
    category_agent = CategoryAgent(args.category_agent, agent_prompts_dir=agent_prompts_dir)
    keyword_agent = KeywordAgent(args.keyword_agent, agent_prompts_dir=agent_prompts_dir)
    judge_agent = JudgeAgent(args.judge_agent, agent_prompts_dir=agent_prompts_dir)
    cipher_llm = llm_factory(args.cipher_agent)
    print(f"keyword, cipher and judge agents successfully loaded.")

    ### Step 3: Loop through the dataset and start the attack
    for idx, row in df_prompt.iterrows():
        try_idx = 0
        prompt = row["prompt"]
        raw_category = row["category"]
        if prompt in df_output["prompt"].values:
            print(f"Output for prompt {idx} already exists. Skipping...")
            continue
        print(f"Working on prompt {idx}: {prompt}")
        start_time = time.time()
        last_end_time = start_time

        ### Step 3.1: Categorize the prompt
        category_agent_input = f"### Prompt:\n{prompt}\n\n\n### Original Categorization for Reference: \n{raw_category}\n\n\n### Your response:\n"
        category_str = category_agent.generate(category_agent_input, max_new_tokens=512)
        while "my final decision is:" not in category_str.lower():
            category_str = category_agent.generate(category_agent_input, max_new_tokens=512)
        category = category_str.lower().split("my final decision is: ")[1].strip().lower()
        category = re.sub(r"[^\w\s-]", "", category)  # Preserves letters, space, hyphens
        # Setup Q-table access
        row = Q_table[Q_table["category"] == category]
        q_values_all = row.drop(columns=["category"]).values.flatten().astype(float)
        cipher_names = list(Q_table.columns[1:])  # skip 'category'
        untried_ciphers = set(cipher_names)

        while try_idx <= args.max_tries and untried_ciphers:
            # Step 3.2: Manage keyword re-querying based on consecutive failure types
            if try_idx == 0:
                keywords = keyword_agent.find_keywords(prompt)
                rejection_count = 0
                wrong_decryption_count = 0
                print(f"Initial keywords detected: {keywords}")
            elif rejection_count >= args.same_consecutive_failure_threshold:
                print(f"[Info] Re-querying keyword agent due to {rejection_count} total rejections.")
                new_keywords = keyword_agent.update_keywords(prompt, old_keywords=keywords, change="more")
                rejection_count = 0
                wrong_decryption_count = 0
                print(f"Keywords updated with MORE. Used to be: {keywords}; now is: {new_keywords}")
                keywords = new_keywords
            elif (wrong_decryption_count >= args.same_consecutive_failure_threshold) and (len(keywords) > 1):
                print(f"[Info] Re-querying keyword agent due to {wrong_decryption_count} total wrong decryptions.")
                new_keywords = keyword_agent.update_keywords(prompt, old_keywords=keywords, change="fewer")
                rejection_count = 0
                wrong_decryption_count = 0
                print(f"Keywords updated with FEWER. Used to be: {keywords}; now is: {new_keywords}")
                keywords = new_keywords
            else:
                print("[Info] Keeping previous keywords.")

            ### Step 3.3: Select cipher using smoothed softmax policy over untried ciphers only
            if args.random:
                selected_cipher = np.random.choice(list(untried_ciphers))
                print(f"Selected cipher: {selected_cipher} (random selection)")
            elif args.greedy:
                untried_indices = [cipher_names.index(c) for c in untried_ciphers]
                q_values = np.array([q_values_all[i] for i in untried_indices])
                a_idx = untried_indices[np.argmax(q_values)]
                selected_cipher = cipher_names[a_idx]
                print(f"Selected cipher: {selected_cipher} (greedy selection)")
            else:
                untried_indices = [cipher_names.index(c) for c in untried_ciphers]
                q_values = np.array([q_values_all[i] for i in untried_indices])
                adjusted_q = (q_values + args.delta) / args.tau
                adjusted_q -= np.max(adjusted_q)  # numerical stability
                exp_q = np.exp(adjusted_q)
                probs = exp_q / np.sum(exp_q)
                selected_idx_in_untried = np.random.choice(len(untried_indices), p=probs)
                a_idx = untried_indices[selected_idx_in_untried]
                selected_cipher = cipher_names[a_idx]
                print(f"Selected cipher by RL: {selected_cipher}, Q-value: {q_values_all[a_idx]}")
            untried_ciphers.remove(selected_cipher)

            ### Step 3.4: Generate the full ciphered prompt
            print(f"Victim inferring ...")
            if args.plus:
                ciphered_prompt = generate_ciphered_prompt_plus(
                    cipher_type=selected_cipher,
                    keywords=keywords,
                    malicious_prompt=prompt,
                    llm=cipher_llm,
                )
            else:
                ciphered_prompt = generate_ciphered_prompt(
                    cipher_type=selected_cipher,
                    keywords=keywords,
                    malicious_prompt=prompt,
                    llm=cipher_llm,
                    placeholders=False,
                )

            ### Step 3.5: Query the victim LLM
            victim_response = victim_llm.generate(
                ciphered_prompt.replace("\\n", "\n"),
                max_new_tokens=6000,  # Make sure there is more than enough space for the full response, as there are placeholder questions
                temperature=0.0,
            )

            ### Step 3.6: Judge the response, and add to consecutive failure stats, if failed
            trimmed_victim_response = _safe_process_output(
                victim_response=victim_response,
                keywords=keywords,
                method="metacipher+",
            )
            judge_result = judge_agent.judge(
                original_prompt=prompt,
                victim_response=trimmed_victim_response,
            ).lower()
            if "rejection" in judge_result:
                rejection_count += 1
            elif "wrong decryption" in judge_result:
                wrong_decryption_count += 1
            print(
                f"Judge result: *** {judge_result} ***; rejection count: {rejection_count}; wrong decryption count: {wrong_decryption_count}"
            )

            ### Step 3.7: Update Q-table
            if not args.random and not args.greedy:
                if "successful jailbreak" in judge_result:
                    R_t = 1.0
                elif "wrong decryption" in judge_result:
                    R_t = -1.0
                elif "rejection" in judge_result:
                    R_t = 0.0
                elif "too general" in judge_result:
                    R_t = 0.5
                else:
                    print("[Warning] Unknown judge result. Skipping Q-update.")
                    break
                s_idx = Q_table[Q_table["category"] == category].index[0]

                # Standard Q-update
                max_q = np.max(q_values_all)
                q_current = Q_table.iloc[s_idx, a_idx + 1]
                Q_table.iloc[s_idx, a_idx + 1] = q_current + args.alpha * (R_t + args.gamma * max_q - q_current)

                # Similarity-aware update
                if (not args.zero) and (jaccard_table is not None):
                    for j, other_cipher in enumerate(cipher_names):
                        if j == a_idx:
                            continue
                        sim = jaccard_table.iloc[a_idx, j] if a_idx > j else jaccard_table.iloc[j, a_idx]
                        if pd.notna(sim):
                            q_old = Q_table.iloc[s_idx, j + 1]
                            Q_table.iloc[s_idx, j + 1] = q_old + args.alpha * sim * (R_t - q_old)
                            # print(f"Similarity-updated Q[{category}, {other_cipher}] from {q_old} to {Q_table.iloc[s_idx, j + 1]}")
                else:
                    print("[Info] Skipping similarity-aware Q-update (zero init or missing jaccard table).")

                # Save updated Q-table to temporary file
                Q_table.to_csv(temp_Q_file, index=False)

            ### Step 3.8: Save result to output dataframe, no matter success or failure
            end_time = time.time()
            attempt_duration = end_time - last_end_time
            # Convert attempt_duration to seconds, preserving 1 decimal place
            attempt_duration = round(attempt_duration, 1)
            last_end_time = end_time
            df_output = pd.concat(
                [
                    df_output,
                    pd.DataFrame(
                        [
                            [
                                idx + 1,
                                prompt,
                                keywords,
                                category,
                                try_idx + 1,
                                selected_cipher,
                                judge_result,
                                attempt_duration,
                                ciphered_prompt.replace("\n", "\\n"),
                                victim_response.replace("\n", "\\n"),
                            ]
                        ],
                        columns=[
                            "idx",
                            "prompt",
                            "keywords",
                            "category",
                            "attempt",
                            "cipher",
                            "judge_result",
                            "time",
                            "ciphered_prompt",
                            "victim_response",
                        ],
                    ),
                ],
                ignore_index=True,
            )
            df_output.to_csv(output_file, index=False)
            try_idx += 1

            ### Step 3.9: Stop if successful
            if "successful" in judge_result:
                break


if __name__ == "__main__":
    main(FLAGS)


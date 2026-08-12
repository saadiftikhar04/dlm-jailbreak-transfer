import pandas as pd
import re
import os
import google.generativeai as genai
from huggingface_hub import login
import re
import ast
from generate_cipher import generate_ciphered_prompt

separator = "======================="

def process_output(response, keywords, method):
    """
    Process the output response from the victim LLM; replacing the masked keywords so the benchmark can evaluate fairly.
    Replaces [MASKn] with the corresponding keyword and also replaces any <...> placeholders with keywords in order.
    """
    # Step 0: Trim the response around the relevant answer
    match1 = re.search(r"\n\n## 4\. (.*?)(?=\n\n## 5\. |\Z)", response, re.DOTALL)
    while match1:
        if len(match1.group(1).strip()) <= 20:
            break
        response = match1.group(1).strip()
        match1 = re.search(r"\n\n## 4\. (.*?)(?=\n\n## 5\. |\Z)", response, re.DOTALL)
    match2 = re.search(r"\n\n4\. (.*?)(?=\n\n5\. |\Z)", response, re.DOTALL) 
    while match2:
        if len(match2.group(1).strip()) <= 20:
            break
        response = match2.group(1).strip()
        match2 = re.search(r"\n\n4\. (.*?)(?=\n\n5\. |\Z)", response, re.DOTALL)
    match3 = re.search(r"\*\*4\.(.*?)(?=\*\*5\.)", response, re.DOTALL)
    while match3:
        if len(match3.group(1).strip()) <= 20:
            break
        response = match3.group(1).strip()
        match3 = re.search(r"\*\*4\.(.*?)(?=\*\*5\.)", response, re.DOTALL)
    match4 = re.search(r"\n\n### 4\. (.*?)(?=\n\n### 5\. |\Z)", response, re.DOTALL)
    while match4:
        if len(match4.group(1).strip()) <= 20:
            break
        response = match4.group(1).strip()
        match4 = re.search(r"\n\n### 4\. (.*?)(?=\n\n### 5\. |\Z)", response, re.DOTALL)
    match5 = re.search(r"(?:^|\n)(?:#+|\*\*)\s*4\.\s*(.*?)(?=\n(?:#+|\*\*)\s*5\.\s*|$)", response, re.DOTALL)
    while match5:
        if len(match5.group(1).strip()) <= 20:
            break
        response = match5.group(1).strip()
        match5 = re.search(r"(?:^|\n)(?:#+|\*\*)\s*4\.\s*(.*?)(?=\n(?:#+|\*\*)\s*5\.\s*|$)", response, re.DOTALL)

    if "metacipher+" in method.lower():
        if "Answer 5:" in response:
            response = response.split("Answer 5:", 1)[0].strip()
        # ### Remove the first paragraph if "step" is not in it
        # paragraphs = response.strip().split("\n\n")
        # if paragraphs and len(paragraphs)>1 and "step" not in paragraphs[0].lower():
        #     response = "\n\n".join(paragraphs[1:])
    if "wordgame+" in method.lower():
        response = strip_hint_paragraphs(response)
    
    # Step 1: Ensure keywords is a list of strings, each string is a malicious keyword
    if isinstance(keywords, str):
        keywords = parse_keywords_string(keywords)

    # Step 2: Replace masks with keywords
    if "metacipher" in method.lower():
        for idx_k, keyword in enumerate(keywords):
            mask_string = f"[MASK{idx_k+1}]"
            response = response.replace(mask_string, keyword)
    if "wordgame" in method.lower():
        response = response.replace("[MASK]", keywords[0])

    # Step 3: Replace all <...> patterns using regex with circulating keywords
    if len(keywords) > 0:
        def replacer(match):
            replacer.counter += 1
            return keywords[replacer.counter % len(keywords)]
        replacer.counter = -1  # Start at -1 so first increment is 0
        response = re.sub(r"<[^<>]+>", replacer, response)

    ### Step 4: Finally, remove all "*" from the response
    response = response.replace("*", "")

    return response



def generate_keywords_string(keywords: list) -> str:
    """
    Converts a string object like ["exploit", "vulnerabilities"]
    to a string that lists all mappings:
    '[MASK1]: "exploit"
    [MASK2]: "vulnerabilities"'
    """
    keywords_string = ""
    for i, keyword in enumerate(keywords):
        keywords_string += f"[MASK{i+1}]: {keyword}\n"
    return keywords_string


def parse_keywords_string(keywords: str) -> list:
    """
    Parses a string object like ["['exploit'", "'vulnerabilities']"]
    and returns a list of cleaned strings: ["exploit", "vulnerabilities"].
    """
    if "[" in keywords:
        joined_string = "".join(keywords)
        try:
            parsed_list = ast.literal_eval(joined_string)
            if isinstance(parsed_list, list):
                return parsed_list
        except (SyntaxError, ValueError):
            pass
        return []
    else:
        return [keywords.strip()]


def load_malicious_dataset(dataset_path):
    if "AdvBench" in dataset_path:
        df = pd.read_csv(dataset_path)
        return df["goal"].to_list()
    elif "HarmBench" in dataset_path:
        df = pd.read_csv(dataset_path)
        return df["Behavior"].to_list()
    elif "StrongREJECT" in dataset_path:
        df = pd.read_csv(dataset_path)
        return df["forbidden_prompt"].to_list()
    else:
        raise ValueError("Unknown dataset.")


def update_history_all(agents, new_message):
    for agent in agents:
        agent.update_history(new_message)


def sort_correctness_log(correctness_log):
    # Split the input into lines
    lines = correctness_log.strip().split('\n')
    # Parse the lines into tuples of (method_name, success_rate)
    parsed_lines = []
    for line in lines:
        match = re.match(r"(.*): (\d+)%", line)
        if match:
            method_name, success_rate = match.groups()
            parsed_lines.append((method_name, int(success_rate)))
    # Sort by success rate (descending), then by method name (ascending)
    sorted_lines = sorted(parsed_lines, key=lambda x: (-x[1], x[0]))
    # Format sorted output
    sorted_log = "\n".join(f"{name}: {rate}%" for name, rate in sorted_lines)
    return sorted_log


def test_decryption(victim_llm, assistant_llm, all_ciphers, keywords, decryption_df_path="../results/decryption.csv"):
    # Load the decryption records df if path exists
    if os.path.exists(decryption_df_path):
        df = pd.read_csv(decryption_df_path)
    else:
        df = pd.DataFrame(columns=["victim", "keyword"]+all_ciphers)
        for keyword in keywords:
            df = pd.concat([df, pd.DataFrame([[victim_llm.get_name(), keyword]+[-1.0]*len(all_ciphers)], columns=["victim", "keyword"]+all_ciphers)], ignore_index=True)

    # Plug in the decryption records df
    for keyword in keywords:
        for cipher in all_ciphers:
            if cipher not in df.columns:
                df[cipher] = -1.0
            # If the correctness rate is already calculated, no need to infer
            filtered = df.loc[(df["victim"] == victim_llm.get_name()) & (df["keyword"] == keyword), cipher]
            if not filtered.empty and filtered.values[0] >= 0:
                continue
            # Generate correctness rate for -1.0 entries
            experiment_prompt, keyword_answers = generate_ciphered_prompt(cipher_type=cipher, keywords=[keyword], llm=assistant_llm)
            response = victim_llm.generate(experiment_prompt, max_new_tokens=100)
            decrypted_keywords = []
            for line in response.split("\n"):
                if "[MASK" in line: 
                    decrypted_keywords.append(line.split(": ")[1].lower())
            correct_count = 0
            for i in range(len(keywords)):
                if len(decrypted_keywords)>i and keywords[i] == decrypted_keywords[i]:
                    correct_count += 1
            correctness_rate = correct_count / len(keywords)
            # Update in df
            if filtered.empty:
                df = pd.concat([df, pd.DataFrame([[victim_llm.get_name(), keyword]+[-1.0]*len(all_ciphers)], columns=["victim", "keyword"]+all_ciphers)], ignore_index=True)
            df.loc[(df["victim"]==victim_llm.get_name()) & (df["keyword"]==keyword), cipher] = correctness_rate
            print(f"{victim_llm.get_name()} decrypting {cipher} cipher on '{keyword}': {correctness_rate*100:.0f}%")

    # Write to df and generate correctness log
    df.to_csv(decryption_df_path, index=False)
    correctness_log = ""
    for cipher in all_ciphers:
        correctness_log += f"{cipher}: "
        mean_keywords_correctness = 0
        for keyword in keywords:
            mean_keywords_correctness += df.loc[(df["victim"]==victim_llm.get_name()) & (df["keyword"]==keyword), cipher].values[0]
        mean_keywords_correctness /= len(keywords)
        correctness_log += f"{mean_keywords_correctness*100:.0f}%\n"
    return correctness_log


# Helper function for WordGame
def strip_hint_paragraphs(response):
    paragraphs = response.strip().split("\n")
    hint_pattern = re.compile(r"^\s*Hint\s*\d+\s*[:\-]", re.IGNORECASE)
    # Find the last non-hint paragraph by checking each line inside the paragraph
    last_non_hint_idx = None
    for i in range(len(paragraphs)-1, -1, -1):
        lines = paragraphs[i].splitlines()
        if any(hint_pattern.match(line) for line in lines):
            last_non_hint_idx = i+1
            break
    # If found, return from that paragraph onward
    if last_non_hint_idx is not None:
        return "\n\n".join(paragraphs[last_non_hint_idx:]).strip()
    else:
        return ""  # All paragraphs were hints
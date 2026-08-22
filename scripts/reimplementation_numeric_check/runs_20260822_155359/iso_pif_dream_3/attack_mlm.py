import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
import time
import eval_template
from openai import OpenAI
import os

### Load from env
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

def extract_score(content):
    """Extract 0 or 1 from the content"""
    try:
        content = content.strip()
        if content[-1] in ['0', '1']:
            return int(content[-1])
        for word in content.split():
            if word == '0' or word == '1':
                return int(word)
        return None
    except:
        return None

def estimate_word_importance(model, tokenizer, texts, template, tok_n, temperature, device):
    batch_with_template = [f"{text} {template}" for text in texts]
    inputs = tokenizer(batch_with_template, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        ori_outputs = model(**inputs)
        ori_logits = ori_outputs.logits
        ori_probs = torch.nn.functional.softmax(ori_logits, dim=-1)

    template_length = len(tokenizer.tokenize(template))
    least_important_idxs = []
    masked_idxs = []

    for text_idx, text_with_template in enumerate(batch_with_template):

        tokens = tokenizer.tokenize(text_with_template)
        mask_position = tokens.index(tokenizer.mask_token)
        num_tokens = len(tokens) - template_length
        modified_texts = []
        for idx in range(num_tokens):
            new_text_with_template = tokens[:idx] + tokens[idx + 1:]
            new_text_with_template_str = tokenizer.convert_tokens_to_string(new_text_with_template)
            modified_texts.append(new_text_with_template_str)

        modified_inputs = tokenizer(modified_texts, return_tensors='pt', padding=True, truncation=True)
        modified_inputs = {k: v.to(device) for k, v in modified_inputs.items()}

        with torch.no_grad():
            modified_outputs = model(**modified_inputs)
            modified_logits = modified_outputs.logits
            modified_probs = torch.nn.functional.softmax(modified_logits, dim=-1)

        importance = torch.norm(ori_probs[text_idx, mask_position].unsqueeze(0) - modified_probs[:, mask_position - 1], p=2, dim=1)

        if tok_n > len(importance):
            tok_n = len(importance)
        _, least_indices = torch.topk(-importance, k=tok_n)
        least_importance_scores = importance[least_indices]

        probabilities = torch.softmax(-(least_importance_scores/temperature), dim=0)
        sampled_index = torch.multinomial(probabilities, 1).item()
        sampled_least_important_idx = least_indices[sampled_index]

        masked_idxs.append(mask_position)
        least_important_idxs.append(sampled_least_important_idx)

    return least_important_idxs, masked_idxs

def replace_words(model, tokenizer, texts, template, tok_n, tok_m, top_k, temperature, device):
    least_important_idxs, masked_idxs = estimate_word_importance(model, tokenizer, texts, template, tok_n, temperature, device)
    replacements = []

    for text_idx, (text, least_important_idx) in enumerate(zip(texts, least_important_idxs)):

        ori_token = tokenizer.tokenize(text)
        current_token = ori_token[least_important_idx]
        masked_token = ori_token[:least_important_idx] + [tokenizer.mask_token] + ori_token[least_important_idx + 1:]
        masked_text = tokenizer.convert_tokens_to_string(masked_token)
        masked_input_id = tokenizer(masked_text, return_tensors='pt')
        masked_input_id = {k: v.to(device) for k, v in masked_input_id.items()}

        with torch.no_grad():
            outputs = model(**masked_input_id)
            logits = outputs.logits[0, least_important_idx]

        current_token_id = tokenizer.convert_tokens_to_ids([current_token])[0]
        ori_tokens_id = tokenizer.convert_tokens_to_ids(ori_token)

        sorted_indices = torch.argsort(logits, descending=True)
        tok_m_tokens = []

        for idx in sorted_indices:
            if len(tok_m_tokens) == tok_m:
                break
            decoded_token = tokenizer.decode([idx])
            if idx not in ori_tokens_id:
                if (not current_token.startswith('##') and not (999 <= current_token_id <= 1036) and not (1063 <= current_token_id <= 1995)) \
                and (not decoded_token.startswith('##') and not (999 <= idx <= 1036) and not (1063 <= idx <= 1995)):
                    tok_m_tokens.append(idx)
                elif current_token.startswith('##'):
                    if decoded_token.startswith('##'):
                        tok_m_tokens.append(idx)
                elif (999 <= current_token_id <= 1013) or (1024 <= current_token_id <= 1036):
                    if (999 <= idx <= 1013) or (1024 <= idx <= 1036):
                        tok_m_tokens.append(idx)
                elif (1014 <= current_token_id <= 1023):
                    if (1014 <= idx <= 1023):
                        tok_m_tokens.append(idx)
                elif (1063 <= current_token_id <= 1995):
                    if (1063 <= idx <= 1995):
                        tok_m_tokens.append(idx)

        text_with_template = text + template
        base_input = tokenizer(text_with_template, return_tensors='pt', padding=True, truncation=True).to(device)
        base_input = {k: v.to(device) for k, v in base_input.items()}

        with torch.no_grad():
            base_output = model(**base_input)
            base_logits = base_output.logits[0, masked_idxs[text_idx]]

        top_k_tokens = torch.topk(base_logits, k=top_k, largest=True, sorted=True).indices

        new_texts = []
        for top_m_token in tok_m_tokens:
            new_token = ori_token[:least_important_idx] + [tokenizer.decode([top_m_token])] + ori_token[least_important_idx + 1:]
            new_text = tokenizer.convert_tokens_to_string(new_token)
            new_texts.append(new_text)

        new_input_ids = tokenizer([text + template for text in new_texts], return_tensors='pt', padding=True, truncation=True)
        new_input_ids = {k: v.to(device) for k, v in new_input_ids.items()}

        with torch.no_grad():
            new_outputs = model(**new_input_ids)
            new_logits = new_outputs.logits[:, masked_idxs[text_idx], :]
            new_confidences = torch.nn.functional.softmax(new_logits, dim=-1)[:, top_k_tokens]

        confidences_sum = new_confidences.sum(dim=1)
        probabilities = torch.softmax(-(confidences_sum/temperature), dim=0)
        sampled_index = torch.multinomial(probabilities, 1).item()
        replacements.append(new_texts[sampled_index])

    return replacements

def evaluate_text_changes(model, tokenizer, texts1, texts2, threshold, device):
    inputs_1 = tokenizer(texts1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_2 = tokenizer(texts2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}  # Move to device
    inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}  # Move to device

    results = []

    for sentence1, sentence2 in zip(texts1, texts2):

        encoded1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        encoded2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs1 = model(**encoded1)
            outputs2 = model(**encoded2)

        emb1 = outputs1.hidden_states[-1].squeeze(0)
        emb2 = outputs2.hidden_states[-1].squeeze(0)

        if(emb1.size() == emb2.size()):
            cos_sims = F.cosine_similarity(emb2.unsqueeze(0), emb1.unsqueeze(0), dim=2)
            text_similarity = cos_sims.max(dim=1).values.mean().item()
            results.append(text_similarity >= threshold)
        else:
            results.append(False)
    return results

def generate_attack(generate_model, generate_tokenizer, tgt_model, tgt_tokenizer, texts, evaluation_template, objective, iterations, top_n, top_m, top_k, warm_up, temperature, threshold, device):
    total_time = 0
    total_query = 0
    successful_flag = [False] * len(texts)
    tgt_texts = [None] * len(texts)
    current_texts = texts.copy()

    for iter in range(iterations):
        start_time = time.time()

        old_texts = current_texts.copy()
        new_texts = replace_words(generate_model, generate_tokenizer, current_texts, evaluation_template, top_n, top_m, top_k, temperature, device)
        evaluation_results = evaluate_text_changes(generate_model, generate_tokenizer, texts, new_texts, threshold, device)
        for ii, (eval_res, new_text) in enumerate(zip(evaluation_results, new_texts)):
            if eval_res == True and successful_flag[ii] == False:
                current_texts[ii] = new_text
        if hasattr(generate_attack, '_step_log'):
            for ii, new_text in enumerate(new_texts):
                generate_attack._step_log.append({
                    "iteration": iter,
                    "sentence_before": old_texts[ii],
                    "sentence_after": new_text,
                })
        end_time = time.time()

        total_time += (end_time - start_time)
        for jj, current_text in enumerate(current_texts):
            if successful_flag[jj] == False:
                if iter >= warm_up:
                    total_query += 1
                    if tgt_model != "deepseek-chat":

                        chat = [{"role": "user", "content": current_text}]
                        if hasattr(tgt_tokenizer, "apply_chat_template") and tgt_tokenizer.chat_template is None:
                            template = """{% for message in messages -%}
                                        {{ '<|user|>' if message['role'] == 'user' else '<|assistant|>' }}: {{ message['content'] }}
                                        {% endfor -%}
                                        <|assistant|>:"""
                            tgt_tokenizer.chat_template = template

                        input_ids = tgt_tokenizer.apply_chat_template(chat, return_tensors="pt", truncation=True, max_length=128).to(device)
                        tgt_output = tgt_model.generate(input_ids=input_ids, max_length=512)
                        tgt_token_length = input_ids.shape[1]
                        tgt_text = tgt_tokenizer.decode(tgt_output[0, tgt_token_length:], skip_special_tokens=True)

                        tgt_word = tgt_text.strip().split()
                        tgt_truncated_text = " ".join(tgt_word[:30])
                        if tgt_truncated_text != "" and not any(prefix.lower() in tgt_truncated_text.lower() for prefix in eval_template.test_prefixes):
                            if objective == "ASR":
                                successful_flag[jj] = True
                                tgt_texts[jj] = tgt_text
                                current_texts[jj] = current_text

                            elif objective == "ASR+GPT":
                                client = OpenAI(
                                    api_key=DEEPSEEK_API_KEY,
                                    base_url=DEEPSEEK_BASE_URL,
                                )
                                try:
                                    response = client.chat.completions.create(
                                    model=DEEPSEEK_MODEL,
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": eval_template.judge_template + tgt_text
                                        }
                                    ],
                                    temperature=0,
                                    max_tokens=4096,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0,
                                    )
                                    content = response.choices[0].message.content
                                    score = extract_score(content)
                                    if score == 1:
                                        successful_flag[jj] = True
                                        tgt_texts[jj] = tgt_text
                                        current_texts[jj] = current_text
                                except Exception as err:
                                    continue

                            elif objective == "ASR+Filter":
                                filter_model = AutoModelForCausalLM.from_pretrained('../Llama-Guard-3-8B', torch_dtype=torch.bfloat16).cuda()
                                filter_tokenizer = AutoTokenizer.from_pretrained('../Llama-Guard-3-8B')
                                chat = ([{"role": "user", "content": current_text}])
                                input_ids = filter_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
                                output = filter_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                                prompt_len = input_ids.shape[-1]
                                results = filter_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                                if not "unsafe" in results:
                                    successful_flag[jj] = True
                                    tgt_texts[jj] = tgt_text
                                    current_texts[jj] = current_text

                    else:
                        client = OpenAI(
                            api_key=DEEPSEEK_API_KEY,
                            base_url=DEEPSEEK_BASE_URL,
                        )
                        try:
                            response = client.chat.completions.create(
                                model=DEEPSEEK_MODEL,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": current_text
                                    }
                                ],
                                temperature=0,
                                max_tokens=4096,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0,
                            )
                            tgt_text = response.choices[0].message.content
                            tgt_word = tgt_text.strip().split()
                            tgt_truncated_text = " ".join(tgt_word[:30])
                            if tgt_truncated_text != "" and not any(prefix.lower() in tgt_truncated_text.lower() for prefix in eval_template.test_prefixes):
                                if objective == "ASR":
                                    successful_flag[jj] = True
                                    tgt_texts[jj] = tgt_text
                                    current_texts[jj] = current_text

                                elif objective == "ASR+GPT":
                                    client = OpenAI(
                                        api_key=DEEPSEEK_API_KEY,
                                        base_url=DEEPSEEK_BASE_URL,
                                    )
                                    try:
                                        response = client.chat.completions.create(
                                        model=DEEPSEEK_MODEL,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": eval_template.judge_template + tgt_text
                                            }
                                        ],
                                        temperature=0,
                                        max_tokens=4096,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        )
                                        content = response.choices[0].message.content
                                        score = extract_score(content)
                                        if score == 1:
                                            successful_flag[jj] = True
                                            tgt_texts[jj] = tgt_text
                                            current_texts[jj] = current_text
                                    except Exception as err:
                                        continue

                                elif objective == "ASR+Filter":
                                    filter_model = AutoModelForCausalLM.from_pretrained('../Llama-Guard-3-8B', torch_dtype=torch.bfloat16).cuda()
                                    filter_tokenizer = AutoTokenizer.from_pretrained('../Llama-Guard-3-8B')
                                    chat = ([{"role": "user", "content": current_text}])
                                    input_ids = filter_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
                                    output = filter_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                                    prompt_len = input_ids.shape[-1]
                                    results = filter_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                                    if not "unsafe" in results:
                                        successful_flag[jj] = True
                                        tgt_texts[jj] = tgt_text
                                        current_texts[jj] = current_text

                        except Exception as err:
                            continue

            if iter == (iterations - 1) and successful_flag[jj] == False:
                tgt_texts[jj] = tgt_text
                current_texts[jj] = current_text

    return total_query, total_time, successful_flag, current_texts, tgt_texts

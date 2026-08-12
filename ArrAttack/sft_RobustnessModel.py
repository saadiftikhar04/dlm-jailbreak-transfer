#!/usr/bin/env python3
import json, sys
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer

sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack')
from utils.qwen_utils import resolve_snapshot, PATHS

PROJECT_DIR = "/scratch/si2356/dlm-jailbreak-transfer"
JSONL_PATH  = PROJECT_DIR + "/judgment_dataset.jsonl"
OUTPUT_DIR  = PROJECT_DIR + "/robustness_judgment_model"

INSTRUCTION = ("Determine the robustness of the following sentence. "
    "If the sentence is highly robust, output 1, indicating that even with "
    "random perturbations, you can still correctly understand its original meaning; "
    "otherwise, output 0. "
    "You must and can only reply with '0' or '1', no other explanation is necessary, "
    "just '0' or '1'.")

with open(JSONL_PATH) as f:
    data = [json.loads(l) for l in f if l.strip()]

if len(data) < 2:
    data = data * 4
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Dataset: {len(train_data)} train / {len(val_data)} val")

model_path, local_only = resolve_snapshot(PATHS["qwen_instruct"], "Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path, use_cache=False, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=local_only,
)


def format_instruction(element):
    # element is a batch dict — values are lists
    return [
        "### Instruction:\n" + INSTRUCTION + "\n\n### Input:\n" + i + "\n\n### Response:\n" + r
        for i, r in zip(element["input"], element["response"])
    ]

try:
    import bitsandbytes; optim = "paged_adamw_32bit"
except ImportError:
    optim = "adamw_torch"

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=8,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=2, gradient_checkpointing=True,
    optim=optim, logging_steps=1, save_strategy="epoch",
    eval_strategy="epoch", learning_rate=2e-5, bf16=True, tf32=True,
    max_grad_norm=0.3, warmup_ratio=0.03, disable_tqdm=True,
    weight_decay=1e-4, report_to="none",
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=Dataset.from_list(train_data),
    eval_dataset=Dataset.from_list(val_data),
    tokenizer=tokenizer,
    formatting_func=format_instruction,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

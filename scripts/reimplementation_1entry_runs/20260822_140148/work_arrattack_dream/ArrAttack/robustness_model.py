"""
sft_RobustnessModel.py  [Qwen2.5 / HPC adaptation]
====================================================
Stage 2: SFT of Qwen2.5-7B-Instruct as robustness judgment model (8 epochs).

Changes from original repo:
  - Model: Llama2-7B -> Qwen2.5-7B-Instruct
  - Format: Alpaca ### format -> ChatML via apply_chat_template
  - Loss masking: DataCollatorForCompletionOnlyLM with assistant header token
    so loss is computed ONLY on "0"/"1" response tokens (not the instruction)
  - Input: judgment_dataset.jsonl (from build_jailbreak_samples.py)
  - Output: PROJECT_DIR/robustness_judgment_model/
  - flash_attention_2 retained (Qwen2.5 supports it)
  - paged_adamw_32bit with graceful fallback to adamw_torch
"""

import json
import logging
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.insert(0, str(Path(__file__).parent))
from utils.qwen_utils import resolve_snapshot, PATHS, PROJECT_DIR, format_for_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- paths -------------------------------------------------------------------
JSONL_PATH  = os.path.join(PROJECT_DIR, "judgment_dataset.jsonl")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "robustness_judgment_model")

# ==============================================================================
# Load dataset
# ==============================================================================

with open(JSONL_PATH) as f:
    data = [json.loads(l) for l in f if l.strip()]

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
logger.info(f"Dataset: {len(train_data)} train / {len(val_data)} val")

# ==============================================================================
# Load base model (Qwen2.5-7B-Instruct)
# ==============================================================================

model_path, local_only = resolve_snapshot(PATHS["qwen_instruct"], "Qwen/Qwen2.5-7B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=False,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=local_only,
)

# ==============================================================================
# ChatML formatting (replaces Alpaca ### format from original repo)
# Closure captures tokenizer so SFTTrainer's formatting_func signature is met.
# ==============================================================================

def make_formatter(tok):
    def format_fn(sample):
        return format_for_sft(tok, sample["instruction"], sample["input"], sample["response"])
    return format_fn

formatting_func = make_formatter(tokenizer)

# Loss only on assistant response tokens.
# "<|im_start|>assistant\n" is the ChatML assistant header for Qwen2.5.
response_template = "<|im_start|>assistant\n"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# ==============================================================================
# Training arguments (paper Table 6 — unchanged)
# ==============================================================================

try:
    import bitsandbytes  # noqa
    optim = "paged_adamw_32bit"
    logger.info("paged_adamw_32bit enabled.")
except ImportError:
    optim = "adamw_torch"
    logger.warning("bitsandbytes not found; falling back to adamw_torch.")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim=optim,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    disable_tqdm=True,
    weight_decay=1e-4,
    report_to="none",
)

train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    max_seq_length=1024,
    tokenizer=tokenizer,
    packing=False,             # off: DataCollatorForCompletionOnlyLM needs fixed sequences
    formatting_func=formatting_func,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Robustness judgment model saved to {OUTPUT_DIR}")

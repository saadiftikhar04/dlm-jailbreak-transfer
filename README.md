# DLM Jailbreak Evaluation Framework

A comprehensive evaluation framework for assessing the safety of Diffusion Language Models against adversarial jailbreak attacks.

## Overview

This repository implements three state-of-the-art jailbreak methods and provides standardized evaluation protocols across multiple benchmark datasets. The framework supports both whitebox and blackbox attack scenarios with automated judging systems.

## Methods

**PiF (Prompt Injection with Formatting)**
- Whitebox gradient-based optimization
- Perturbs input embeddings to maximize harmful completion likelihood
- Adaptable to diffusion model architectures

**ArrAttack (Adversarial Representation Refinement)**
- Blackbox candidate generation approach
- Multiple rewriting strategies including role-playing, encoding, and context injection
- Model-agnostic with high transferability

**MetaCipher**
- Cipher-based obfuscation technique
- Supports Caesar, Atbash, Vigenere, substitution, and Morse encodings
- Exploits instruction-following capabilities

## Datasets

The framework supports evaluation on 4 benchmark datasets:

- HarmBench
- JailbreakBench
- MaliciousInstruct
- StrongREJECT

## Installation

```bash
# Create environment
conda create -n dlm-jailbreak python=3.10
conda activate dlm-jailbreak

# Install dependencies
pip install torch transformers pandas tqdm anthropic openai
pip install jailbreakbench  # For JailbreakBench evaluation

# Configure API keys
export ANTHROPIC_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

## Usage


### Command Line

```bash
python playground/pipeline.py \
    --model model_path \
    --dataset harmbench \
    --method metacipher \
    --num-samples 100
```

## Evaluation

Results are automatically evaluated using dataset-specific judges:

- **HarmBench**: Classifier-based binary evaluation
- **JailbreakBench**: Dual-judge system 
- **StrongREJECT**: Continuous scoring 
- **MaliciousInstruct**: Hybrid classifier with refusal filtering

Output format:
```csv
prompt,perturbed_prompt,response,method,label,score
```



## Testing

```bash
# Validate environment
python playground/test_framework.py

# Quick functional test
python playground/test_dlm_inference.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.35+
- CUDA-capable GPU (recommended)
- Minimum 16GB RAM

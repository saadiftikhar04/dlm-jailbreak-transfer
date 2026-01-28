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

The framework supports evaluation on seven benchmark datasets:

- AdvBench
- HarmBench
- Harmful Tasks
- JailbreakBench
- MaliciousInstruct
- StrongREJECT
- VBCDE-100

## Structure

```
dlm-jailbreak-transfer/
├── dataset/               # Benchmark datasets
├── playground/            # Development and testing
│   ├── dlm_attacks_unified.py
│   ├── dlm_judges.py
│   ├── pipeline.py
│   └── infer_*.py
├── scripts/               # Experiment runners
│   ├── evaluation/
│   ├── experiments/
│   └── infer_*.py
├── src/                   # Core implementations
│   ├── agent.py
│   ├── generate_cipher.py
│   ├── llm.py
│   └── utils.py
└── results/               # Experiment outputs
```

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

### Single Attack Evaluation

```python
from playground.dlm_attacks_unified import MetaCipherAttack, JailbreakConfig
from src.llm import load_model

# Load model
model, tokenizer = load_model('model_path')

# Configure attack
config = JailbreakConfig(method='metacipher', cipher_type='caesar')
attack = MetaCipherAttack(model, tokenizer, config)

# Execute
result = attack.attack("harmful_prompt")
```

### Batch Experiments

```python
from playground.pipeline import ExperimentPipeline

pipeline = ExperimentPipeline('config.json')
pipeline.run_full_pipeline(
    model_path='model_path',
    datasets=['harmbench', 'jailbreakbench'],
    methods=['pif', 'arrattack', 'metacipher']
)
```

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
- **JailbreakBench**: Dual-judge system (jailbreak + refusal)
- **StrongREJECT**: Continuous scoring (0-1 scale)
- **MaliciousInstruct**: Hybrid classifier with refusal filtering
- **General**: LLM-as-judge with structured rubrics

Output format:
```csv
prompt,perturbed_prompt,response,method,label,score
```

## Configuration

Experiments can be configured via JSON:

```json
{
  "methods": {
    "pif": {"num_steps": 100, "k": 10, "alpha": 0.1},
    "arrattack": {"num_samples": 20, "temperature": 1.0},
    "metacipher": {"cipher_type": "caesar", "include_instructions": true}
  },
  "datasets": ["harmbench", "jailbreakbench"],
  "computation": {"use_gpu": true, "batch_size": 8}
}
```

## Results

Experimental results are saved in structured format:

```
results/
├── method_name/
│   ├── dataset_name/
│   │   ├── model_name.csv
│   │   └── model_name_judged.csv
│   └── summary.json
```

Success rates and statistical analyses are generated automatically.

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

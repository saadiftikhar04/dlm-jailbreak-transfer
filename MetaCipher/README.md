# MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs

This is the code repo for MetaCipher, a LLM jailbreak work published at AAAI 2026 Special Track in AI Alignment. We release the code for facilitating future researchers, in order to contribute to a safer conduct, release and use of LLM services. Please be aware that re-implementing experiments might expose users to malicious contents. 


## Environment Setup
Create and  the environment by running the following command:
```bash
conda create -n metacipher python=3.10 -y
conda activate metacipher
pip install -r requirements.txt
```

Then, run ```vim ~/.bashrc``` and add the keys and paths as environment variables: 
```bash
export HF_HOME=...
export HUGGINGFACE_TOKEN=...
export OPENAI_API_KEY=...
export LLAMA_API_KEY=...
export PERSPECTIVE_API_KEY=...
export DEEPSEEK_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...
```
These values will automatically be loaded when you start a new terminal session. Otherwise, you can run the following command to load them immediately:
```bash
source ~/.bashrc
```

## Implementation Instructions

### Code Structure
The MetaCipher framework is implemented in `./src`, which includes the main pipeline, the agents, the ciphers, and the victim LLMs.

To replicate our results for MetaCipher, run the script `./scripts/experiments/test/metacipher.py`.

Pre-processing of the benchmark datasets can be found in `./scripts/pre-processing`; judge/score script from the original benchmarks can be found in `./scripts/post-processing`.

Due to responsible conduct of code release and ethical concerns, we cannot share the successful jailbreaking prompts that we generated from our framework. However, researchers can achieve them by running our scripts. 

Furthermore, to facilitate future researchers in LLM jailbreaks, we provide `./playground` folder, which contains the code for running inference for various apis and open-source LLMs. You can also craft and test your new cipher there.


### Supported Victim Models
Below we list all the victim LLMs supported in our framework. You can add more LLMs by implementing their corresponding classes in `./src/llm.py`.

#### Text-Only
##### Open-Source non-reasoning:
tiiuae/Falcon3-10B-Instruct\
internlm/internlm2_5-20b-chat\
meta-llama/Llama-3.3-70B-Instruct\
Qwen/Qwen2.5-72B-Instruct

##### Commercial non-reasoning:
claude-3-7-sonnet-20250219\
deepseek-chat\
gemini-2.0-flash\
gpt-4o

##### Open-Source reasoning:
Qwen/QwQ-32B

##### Commercial reasoning:
deepseek-reasoner\
gemini-2.5-pro-exp-03-25\
o1-mini-2024-09-12

### Image Generation
GPT-4o + DALL-E (web service)


### Agents
Our agents are implemented in `./src/agent.py`. Each agent is a class inherited from the base `Agent` class.

All agents used deepseek-chat api service in our experiments. It is mainly due to the cost-efficiency and its low safety standard in jailbreaking other LLMs. However, you can change them to any local or remote LLMs as you wish. The choice of LLM for each agent can be changed in the argument parser of `./scripts/experiments/test/metacipher.py`.







### Benchmark Datasets
We provide the following benchmark datasets for evaluating the effectiveness of jailbreak attacks. All the datasets used in our experiments are collected from top-tier conference publications. We processed and saved them in `./dataset` folder. We cite their original sources below. 

We provide the scripts used for pre-processing these datasets in `./scripts/pre-processing` folder. They process the raw data from their original sources and convert them into a unified format (`processed.csv`) for easier usage during experiments. 

We provide the scripts used for judging/scoring these datasets in `./scripts/post-processing` folder. The code is re-implemented as faithly as possible from their original sources. Refer to the comments in these scripts for more details.


#### ------ Text-to-Text ------

##### Used in validation experiment:
AdvBench (Harmful Behaviors) ([arxiv 2023 paper](https://arxiv.org/abs/2307.15043), Most used, 520 prompts, uncategorized)

##### Used in test experiments:
MaliciousInstruct ([ICLR 2024](https://openreview.net/forum?id=r42tSSCHPh), 100 prompts, 10 categories)

JailbreakBench ([NeurIPS 2024](https://github.com/JailbreakBench/jailbreakbench), 100 prompts, 10 categories)

HarmBench ([ACM 2024](https://dl.acm.org/doi/10.5555/3692070.3693501), 400 prompts, 7 categories)

StrongReject([NeurIPS 2024](https://openreview.net/forum?id=KZLE5BaaOH#discussion), 313 prompts)

#### ------ Text-to-Image ------
##### Used in case study:
[VBCDE-100](https://arxiv.org/abs/2312.07130), 100 prompts






## Ethical Statements

This research advances LLM security and safety. MetaCipher identifies vulnerabilities in safety mechanisms to inform robust defense strategies. 

### Controlled Access

We do not release successful jailbreak prompts to prevent misuse, following established security research norms. Complete framework code is provided for peer verification, defense development, and security research. Researchers can regenerate prompts by executing our scripts in controlled environments.

### Dual-Use Considerations

We acknowledge dual-use potential and mitigate risks through peer-reviewed publication, engagement with AI safety researchers, and tools designed for systematic evaluation rather than ad-hoc exploitation.

### Intended Use

**Authorized uses:**
- Academic research on AI safety and security
- Red-teaming by authorized security researchers
- LLM defense mechanism development and evaluation
- Educational purposes in controlled academic settings

**Prohibited uses:**
- Generating harmful content for malicious purposes
- Unauthorized circumvention of production system safety mechanisms
- Harassment, deception, or manipulation
- Violating terms of service of commercial AI platforms

### User Responsibility

Users must comply with applicable laws and ethical guidelines, obtain institutional approvals, implement appropriate safeguards, and use the framework exclusively for legitimate security research.

### Content Warning

This framework generates adversarial prompts and potentially harmful outputs. Users should implement content filtering, work in isolated environments, and follow appropriate data handling protocols for sensitive research materials.






## Citation

If you find this work useful, please cite:
```bibtex
@misc{chen2025metaciphertimepersistentuniversalmultiagent,
    title={MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs}, 
    author={Boyuan Chen and Minghao Shao and Abdul Basit and Siddharth Garg and Muhammad Shafique},
    year={2025},
    eprint={2506.22557},
    archivePrefix={arXiv},
    primaryClass={cs.CR},
    url={https://arxiv.org/abs/2506.22557}, 
}
```

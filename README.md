# Cross-Paradigm Jailbreak Transfer: Evaluation Framework

Reproducibility package for a study measuring jailbreak attack transferability
across autoregressive language models (AR-LMs) and diffusion language models
(DLMs). Three published jailbreak methods are each **forked from their official
implementation** and extended with a thin adapter layer so they can target a
common pool of six victim models. All other algorithmic logic in each fork is
left untouched unless explicitly noted below.

> **This repo does not vendor the upstream PiF / ArrAttack / MetaCipher
> codebases.** Only the adapter and orchestration scripts we wrote — listed
> under each method below — are tracked here. The three `PiF/`, `ArrAttack/`,
> and `MetaCipher/` directories are `.gitignore`d locally; to reproduce, clone
> each official repo yourself (commands in §4) and drop our scripts in on top.
> This keeps the upstream authors' code under their own license/repo and
> keeps this repo small — what's actually ours is the ~8 files described
> below, not a 3x repo mirror.

**Core finding.** Jailbreak vulnerability tracks *attack–model mechanism
compatibility* rather than architecture class (AR vs. diffusion). Under
MetaCipher specifically, LLaDA clusters with the causal models rather than
with the other DLMs — i.e. architecture alone does not predict susceptibility.

---

## 1. Repository layout

What's actually **tracked in this repo**:

```
dlm-jailbreak-transfer/
├── pif/
│   ├── pif_target_models.py        # unified generate() adapter, all 6 victims
│   ├── run_pif.py                  # dataset loop, BERT importance scoring, CSV logging
│   └── run_pif_<model>.sh          # one launcher per victim model
│
├── arrattack/
│   ├── generate_robust-arr.py      # Stage 1 — BRJ candidate generation + SmoothLLM labeling
│   ├── sft_RobustnessModel.py      # Stage 2 — SFT of the robustness-judgment model
│   ├── generate_robustPrompts.py   # Stage 3 — BRJWR: robustness-conditioned prompt generation
│   ├── smoothllm.py                # perturbation-based robustness labeller (not in upstream)
│   ├── stage5_attack.py            # Stage 5 — final attack vs. all 6 victims (DLM-aware decoding)
│   └── run_arrattack_<model>.sh    # per-model launcher
│
├── metacipher/
│   ├── metacipher_multi.py         # sequential 6-model sweep, RL cipher selection
│   └── run_metacipher_all.sh       # launcher (qwen2.5 → falcon → llama → llada → dream → diffucoder)
│
├── dataset/
│   ├── harmbench/  jailbreakbench/  malicious_instruct/  strongreject/
│
├── judge/
│   ├── unified_judge.py            # cross-attack rescoring with one consistent rubric
│   └── gold_validation/            # 594-row stratified, human-adjudicated gold labels
│
├── setup.sh                         # clones the 3 upstream repos and copies the files above in
└── .gitignore                       # PiF/, ArrAttack/, MetaCipher/ (the cloned upstream code)
```

After running `setup.sh` (§4), your **local, untracked** working tree additionally
has the three upstream clones with our files copied in:

```
PiF/            ← github.com/tmllab/2025_ICLR_PiF, our pif/*        copied in
ArrAttack/      ← github.com/LLBao/ArrAttack,       our arrattack/* copied in
MetaCipher/     ← github.com/BoyuanChen99/MetaCipher, our metacipher/* copied in
results/
├── pif/<model>/<dataset>/results.csv
├── arrattack/<model>/<dataset>/results.csv
└── metacipher/<model>/<dataset>/results.csv
```

`results/` is also `.gitignore`d — see §9 on why result CSVs aren't published.

For the concrete, cluster-side view of everything above that's *not* tracked
in git (actual dataset file sizes, model weight paths, and HF cache
locations as they exist on Jubail today), see **§8 Cluster deployment
reference**.

---

## 2. Victim models (6)

| Family | Model |
|---|---|
| AR-LM | Qwen2.5-7B-Instruct |
| AR-LM | Llama-3.1-Instruct |
| AR-LM | Falcon-H1R-7B |
| DLM | LLaDA-1.5 |
| DLM | Dream-v0-7B |
| DLM | DiffuCoder-7B-Instruct |

DLM decoding is **never** left at library defaults — every script below pins
the exact generation hyperparameters used for the paper's numbers (see §4.2 and
§4.3) so a different default in a future `transformers`/model release can't
silently change results.

## 3. Datasets (4)

HarmBench · JailbreakBench · MaliciousInstruct · StrongREJECT — loaded from
`dataset/<name>/` in that fixed order in the multi-model sweep scripts, so run
logs and partial-completion checkpoints are directly comparable across attacks.
Actual on-disk sizes for these directories as populated on Jubail are listed
in §8.1.

---

## 4. Attack methods

### 4.0 Setup — cloning the upstream repos

None of the three attack codebases are committed here. `setup.sh` clones
each official repo and copies our adapter scripts on top of it:

```bash
#!/bin/bash
set -e

git clone https://github.com/tmllab/2025_ICLR_PiF.git PiF
cp pif/*.py pif/*.sh PiF/

git clone https://github.com/LLBao/ArrAttack.git ArrAttack
cp arrattack/*.py arrattack/*.sh ArrAttack/

git clone https://github.com/BoyuanChen99/MetaCipher.git MetaCipher
cp metacipher/*.py metacipher/*.sh MetaCipher/
```

```bash
# .gitignore
PiF/
ArrAttack/
MetaCipher/
results/
```

Run `bash setup.sh` once per machine/cluster node before any of the
commands below. If an upstream repo updates its file layout, only the
`cp` targets in `setup.sh` need adjusting — none of our adapter scripts
assume a specific upstream commit, only the public function/class names
documented in each section below.

### 4.1 PiF — Perceived-importance Flatten
*Lin, Han, Li, Liu. "Understanding and Enhancing the Transferability of
Jailbreaking Attacks." ICLR 2025.* [arXiv:2502.03052](https://arxiv.org/abs/2502.03052) · [original repo](https://github.com/tmllab/2025_ICLR_PiF)

PiF is a **whitebox, token-replacement-only** attack — no adversarial suffix
is appended. It exploits the fact that adversarial suffixes derived from one
model overfit the source model's intent-perception pattern and transfer
poorly. Instead, PiF runs a three-stage loop per prompt:

1. **Select** — using a BERT-Large probe, measure each input token's
   *perceived importance* to the model's intent recognition by ablating it
   and re-measuring; pick the lowest-importance token.
2. **Replace** — substitute it with a synonym chosen to flatten the
   importance distribution (drag focus toward neutral tokens, away from the
   malicious-intent ones) rather than to reach a predefined harmful target —
   this dynamic objective is what gives PiF better cross-model transfer than
   suffix-based attacks.
3. **Verify** — accept the substitution only if cosine similarity to the
   original prompt stays above Θ (paper default 0.85), preserving semantic
   intent.

**What's added on top of the fork:**
- `pif_target_models.py` — replaces PiF's original hardcoded
  Llama/Mistral/GPT target calls with one `generate()` interface dispatched
  by model key, covering all 6 victims (including the DLM-specific decode
  paths for LLaDA/Dream/DiffuCoder).
- `run_pif.py` — orchestrates the BERT importance scoring + iterative attack
  loop across datasets for a given `--target`, writes per-prompt results
  (original prompt, full PiF-perturbed prompt, target response, success
  label) to CSV without truncation.

**Run:**
```bash
cd PiF
sbatch run_pif_qwen2.5.sh      # one job per victim model
# equivalent direct call:
python run_pif.py --target qwen2.5 --dataset harmbench \
    --bert_path <path-to-bert-large-uncased-snapshot>
```

### 4.2 ArrAttack
*Li et al. "One Model Transfer to All: On Robust Jailbreak Prompts Generation
against LLMs." ICLR 2025.* [arXiv:2505.17598](https://arxiv.org/abs/2505.17598) · [original repo](https://github.com/LLBao/ArrAttack)

ArrAttack targets **defended** LLMs specifically (perplexity filters, input
preprocessing, defensive suffixes). Rather than attacking blind, it first
*learns what "robust" looks like* for a given defense stack, then uses that
model to generate prompts that are robust by construction. The original repo
is Llama-2 + fastchat specific; this fork replaces that backend with
Qwen2.5 + native ChatML so it runs against arbitrary victims, and adds the
SmoothLLM labeling step the original pipeline never implemented.

Five stages, run in order:

| Stage | Script | Role |
|---|---|---|
| 1 | `generate_robust-arr.py` | **B**est-**R**esponse-**J**ailbreak (BRJ): iteratively paraphrase the malicious prompt against the undefended victim, then label each candidate **robust / fragile** via SmoothLLM perturbation testing |
| 2 | `sft_RobustnessModel.py` | Fine-tune a model on the Stage-1 labeled data → the **robustness judgment model** |
| 3 | `generate_robustPrompts.py` | **B**RJ-**W**ith-**R**ephrasing: generate new candidate prompts and score them with the Stage-2 judgment model instead of querying the live (expensive) victim |
| 4 | (generation-model SFT) | Fine-tune a **robust jailbreak prompt generator** on the highest-scoring Stage-3 prompts |
| 5 | `stage5_attack.py` | Run the trained generator against all 6 victims and record final success |

`smoothllm.py` (not present upstream) implements the perturbation-based
robustness label used in Stages 1 and 3: generate *N*=10 character-perturbed
(swap/insert/patch) copies of a candidate prompt, query the victim on each,
and score it — `score ≥ 7` → robust (label 1), `score ≤ 3` → fragile
(label 0), `4–6` discarded as ambiguous (a scaled version of the paper's
20-copy grey band).

`stage5_attack.py` is also where DLM-specific decoding lives — generation
hyperparameters are inlined exactly rather than left to library defaults:

```
LLaDA:      steps=128, gen_length=512, block_length=128,
            temperature=0, cfg_scale=0, remasking="low_confidence"
Dream / DiffuCoder: native diffusion_generate(),
            steps=128, temperature=0.2, top_p=0.95, alg="entropy", alg_temp=0
```

**Run:**
```bash
cd ArrAttack
python generate_robust-arr.py    --victim qwen2.5
python sft_RobustnessModel.py
python generate_robustPrompts.py --victim qwen2.5
# Stage 4 (generation-model SFT) omitted here if already trained
sbatch run_arrattack_dlms.sh      # Stage 5 against LLaDA / Dream / DiffuCoder
```

### 4.3 MetaCipher
*Chen, Shao, Basit, Garg, Shafique. "MetaCipher: A Time-Persistent and
Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs."*
[arXiv:2506.22557](https://arxiv.org/abs/2506.22557) · [original repo](https://github.com/BoyuanChen99/MetaCipher)

MetaCipher is a **blackbox, multi-agent, RL-driven** cipher attack. A
malicious request is routed through a fixed agent pipeline — keyword
detector → categorizer → **RL cipher selector** (Q-table over 21 ciphers in
4 families: substitution, transposition, encoding, and compound/stacked) →
ciphered-prompt generator → instruction template (padded with innocent
placeholder questions and an affirmative response initiator) → victim model
→ judge. On failure the judge classifies *why* (refusal / wrong decryption /
too general) and updates the Q-table so cipher selection improves across
prompts — this Q-learning component is the "time-persistent" part of the
name: priors carry over between prompts and sessions instead of resetting.

`metacipher_multi.py` extends the official single-victim script into a
6-model registry (`qwen2.5, falcon, llama, llada, dream, diffucoder`) run
sequentially, with:
- a `MODEL_REGISTRY` dispatch so the same Q-table logic, cipher pool, and
  judge are reused unmodified across AR-LMs and DLMs;
- DLM decoding wired to each model's native generation path (LLaDA via the
  official `llada_generate()` inlined at 128 diffusion steps; Dream/DiffuCoder
  via `diffusion_generate()`), since the upstream script assumes AR
  `generate()` only;
- a JSONL resume sidecar (`metacipher_<model>_resume.jsonl`) so a preempted
  HPC job restarts from the last completed `(dataset, prompt)` pair instead
  of re-running the whole sweep;
- the judge wired to DeepSeek's API rather than GPT-4 (functionally
  equivalent role to the paper's judge agent, swapped for cost).

**Run:**
```bash
cd MetaCipher
sbatch run_metacipher_all.sh
# equivalent direct call, single model:
python metacipher_multi.py --victim qwen2.5 --dataset harmbench \
    --keyword_agent deepseek-chat --cipher_agent deepseek-chat \
    --category_agent deepseek-chat --judge_agent deepseek-chat
```

---

## 5. Evaluation & judging

Each method ships a per-benchmark official judge (HarmBench's classifier-style
rubric, JailbreakBench's dual-judge protocol, StrongREJECT's continuous
score, MaliciousInstruct's hybrid classifier+refusal filter) — these are kept
**as faithful re-implementations**, not replaced, so per-attack numbers stay
comparable to each method's original paper.

For **cross-attack, cross-model comparison**, `judge/unified_judge.py`
re-scores every result CSV (3 attacks × 6 models) under one consistent
rubric, since the attack-specific judges differ enough in strictness that raw
ASR isn't directly comparable across methods otherwise.

**Gold-standard human validation.** Attack-specific LLM judges are known to
degrade toward chance accuracy under exactly the kind of attack/architecture
distribution shift this study covers (cf. Schwinn et al., "Coin Flip for
Safety"). To check this isn't silently inflating or deflating the reported
ASR, a 5% stratified sample (**N = 594**, stratified by attack × model ×
official label, seed `20260620`) was drawn from the full result set and
adjudicated by hand under a 4-way scheme — `JAILBROKEN / REFUSAL /
OFF_TARGET / TOO_GENERAL` — reading the actual prompt and response content
rather than trusting the official label. Agreement, Cohen's κ, and per-judge
precision/recall/F1 against this gold set are reported in the paper's
validation section; the sample and labels are under `judge/gold_validation/`.

Known judge failure modes surfaced by this audit (documented for
reproducibility, not as a complaint about any one method):
- MetaCipher×DiffuCoder "successes" are overwhelmingly scaffold artifacts
  (restated-prompt templates with no harmful payload) — official compliance
  labels here are mostly false positives.
- DiffuCoder's near-total resistance to all three attacks is **not** a safety
  property — it's a code-generation model with weak natural-language
  instruction-following, so it degenerates to repetitive/garbled output
  regardless of attack. This is reported as an architectural-limitation
  finding, not a robustness finding.

## 6. Output schema

All result CSVs share the same backbone columns (attack-specific prompt
column varies):

```
original_prompt, <attack>_prompt, victim, dataset, victim_output,
inference_time, label, score
```
where `<attack>_prompt` is `pif_prompt`, `arrattack_prompt`, or
`metacipher_prompt` respectively.

## 7. Environment

```bash
conda create -n dlm-jailbreak python=3.10
conda activate dlm-jailbreak
pip install torch>=2.0 "transformers>=4.56" pandas tqdm openai sentence-transformers
pip install jailbreakbench

export DEEPSEEK_API_KEY="<your_key>"      # judge + agent backend for MetaCipher/ArrAttack
export HF_HOME=<path-to-hf-cache>

bash setup.sh    # clones PiF/ArrAttack/MetaCipher and copies in the adapter scripts (§4.0)
```

`transformers>=4.56` is a hard requirement, not a suggestion — older versions
silently fail to load the Falcon-H1 architecture used by one of the six
victims, with an error that looks unrelated to the version mismatch.

SLURM users: each `run_*.sh` follows the same template (single A100/H100 GPU,
64–80 GB host memory); adjust `--account`/`--partition`/`--qos` for your
cluster.

---

## 8. Cluster deployment reference (Jubail)

This section documents the **uncommitted, cluster-local state** of a live
deployment — `dataset/` contents, resolved model weight paths, and HF cache
locations as they actually exist on disk, rather than as declared in code.
None of this is tracked in git; it's here purely so the repo can be
reproduced against a fresh environment without re-deriving these paths by
hand.

- **Cluster:** Jubail (`jubail.abudhabi.nyu.edu`) · **User:** `si2356`
- **Repo root:** `/scratch/si2356/dlm-jailbreak-transfer`

### 8.1 `dataset/` — on-disk sizes

| Directory | Size |
|---|---|
| `dataset/advbench` | — |
| `dataset/harmbench` | 200K |
| `dataset/jailbreakbench` | 28K |
| `dataset/malicious_instruct` | 12K |
| `dataset/strongreject` | 60K |

### 8.2 Model weights

**Model registry — `ArrAttack/utils/model_utils.py`**

| Key | Path / HF Hub ID |
|---|---|
| `qwen2.5` | `Qwen/Qwen2.5-7B-Instruct` (HF hub id) |
| `falcon` | `tiiuae/Falcon-H1R-7B` (HF hub id) |
| `llama` | `/scratch/si2356/hf_models/Llama-3.2-3B-Instruct` |
| `llada` | `/scratch/si2356/models/llada-1.5-8b` |
| `dream` | `/home/si2356/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae` |
| `diffucoder` | `/scratch/si2356/.cache/huggingface/hub/models--apple--DiffuCoder-7B-Instruct/snapshots/4fdd4580064ca5d11808069ce78f88d068753c96` |

Note: `llama`, `llada`, `dream`, and `diffucoder` are baked-in **local
filesystem paths**, not HF hub ids — these are the four that need updating
when replicating on a different machine (see §8.4).

**Additional model refs — `ArrAttack/utils/qwen_utils.py`**

| Key | HF Hub ID |
|---|---|
| `qwen_instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen_base` | `Qwen/Qwen2.5-7B` |
| `gptfuzz` | `hubert233/GPTFuzz` |
| `paraphraser` | `humarin/chatgpt_paraphraser_on_T5_base` |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` |
| `llama3` | `meta-llama/Llama-3-8B-Instruct` |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `vicuna` | `lmsys/vicuna-7b-v1.5` |

Also referenced in `PiF/attack_mlm.py`: `Llama-Guard-3-8B`, at the relative
path `../Llama-Guard-3-8B`.

**Snapshots actually downloaded — `~/.cache/huggingface/hub`**

| Snapshot directory | Size |
|---|---|
| `models--Dream-org--Dream-v0-Instruct-7B` | 15G |
| `models--GSAI-ML--LLaDA-1.5` | 15G |
| `models--Qwen--Qwen2.5-7B` | 15G |
| `models--Qwen--Qwen2.5-7B-Instruct` | 15G |
| `models--apple--DiffuCoder-7B-Instruct` | 15G |
| `models--cais--HarmBench-Llama-2-13b-cls` | 25G |
| `models--google-bert--bert-large-uncased` | 5.1G |
| `models--hubert233--GPTFuzz` | 1.4G |
| `models--humarin--chatgpt_paraphraser_on_T5_base` | 854M |
| `models--meta-llama--Llama-3.2-3B-Instruct` | 12G |
| `models--sentence-transformers--all-mpnet-base-v2` | 419M |
| `models--tiiuae--Falcon-H1R-7B` | 15G |

### 8.3 HF cache — locations referenced

```
~/.cache/huggingface/hub               (default; actually populated, see §8.2)
/scratch/si2356/hf_cache                (current shell $HF_HOME)
/scratch/si2356/.cache/huggingface      (exported by run_*.sh scripts)
```

Three different `HF_HOME`/cache locations are referenced across the shell
environment and the `run_*.sh` launchers; only `~/.cache/huggingface/hub`
(the default) is actually populated with the snapshots in §8.2. Reconcile
this to a single `$HF_HOME` before running jobs to avoid a silent re-download.

### 8.4 To replicate on a different machine

1. Pull the model IDs in §8.2 via `huggingface-cli download <id>`.
2. Point `HF_HOME` / `HF_HUB_CACHE` at wherever the cache should live on your
   end (see §7 for the export).
3. Update the hardcoded absolute paths in `MODEL_REGISTRY` (`llama`, `llada`,
   `dream`, `diffucoder`) to match your own filesystem — those four are
   baked-in local paths, not HF hub ids.

*Compiled from cluster shell output (`find`, `du`, `cat`, `grep`) on
`/scratch/si2356/dlm-jailbreak-transfer`.*

---

## 9. Citing the underlying methods

```bibtex
@inproceedings{lin2025understanding,
  title={Understanding and Enhancing the Transferability of Jailbreaking Attacks},
  author={Lin, Runqi and Han, Bo and Li, Fengwang and Liu, Tongliang},
  booktitle={ICLR},
  year={2025}
}

@inproceedings{li2025onemodel,
  title={One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs},
  author={Li, Linbao and others},
  booktitle={ICLR},
  year={2025}
}

@article{chen2025metacipher,
  title={MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs},
  author={Chen, Boyuan and Shao, Minghao and Basit, Abdul and Garg, Siddharth and Shafique, Muhammad},
  journal={arXiv preprint arXiv:2506.22557},
  year={2025}
}
```

## 10. Ethics statement

This framework is built for defensive red-teaming research: identifying
which attack–architecture combinations succeed so that targeted defenses can
be developed. Result CSVs containing model outputs are included in this
public package.

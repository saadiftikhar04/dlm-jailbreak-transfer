# Cross-Check: Reproducing the Student's Results

This directory contains a self-contained plan + scripts to **trust-check** every
result file in `/results` against the ground truth that produced it. The student's
data is now format-correct (schema, row counts, columns all pass), but **the values
themselves are not yet trusted**: any of them could be fabricated, cherry-picked,
copy-pasted across models, or the victim response swapped with another model's.

The plan's core guarantee: **a random subset is drawn from EVERY one of the 54
result files, with a fixed reproducible seed, and each drawn row is then re-derived
twice** — once by deterministic checks that need no GPU, and once by faithfully
re-running the actual attack pipeline to regenerate the victim response.

---

## TL;DR

```bash
# 1. Zero-GPU checks (run anywhere, this box is fine):
./00_run_all.sh            # = 01_inventory + 02_sample + 03_deterministic + 04_preflight + 06_compare
                           # (05 plan only; real reproduction needs a GPU host)

# 2. Faithful reproduction (GPU host that has weights + dataset):
CROSSCHECK_DATA_DIR=/scratch/si2356/dlm-jailbreak-transfer \
CROSSCHECK_HF_CACHE=/scratch/si2356/.cache/huggingface/hub \
    bash 05_reproduce.sh            # reproduce every sampled row
./00_run_all.sh --reproduce         # (or via the master driver)
# then re-run the comparison:
python3 06_compare.py
```

---

## Paths & results convention (IMPORTANT)

**Results live in git.** The entire `results/` tree (the thing being checked) is
versioned in this repo. On the HPC you only need to `git pull` the latest commit
and the checked results are there — the cross-check compares its sampled rows
against `<repo>/results` directly, so **no extra results path is ever needed**.

**Per-machine paths come from `paths.txt` (git-ignored).** The repo root and any
machine-specific directories are read from `<repo>/paths.txt`, one copy per
machine, never hard-coded in code:

```
/home/bc3194/Desktop/dlm-jailbreak-transfer     # line 1 = repo root (this machine)
# optional KEY=VALUE lines (consumed by the scripts):
DATA_DIR=/path/to/student/weights+dataset       # only needed for faithful reproduction
HF_HOME=/path/to/huggingface/cache
```

- **Line 1** is the repo root absolute path. Every runner in `scripts/cross-check/`
  resolves it via `_paths.ROOT` (Python) or reads it directly (shell/sbatch).
- **`KEY=VALUE` lines** override the corresponding env-looking variables
  (`DATA_DIR`, `HF_HOME`). If a key is absent the script falls back to `$env` and
  then to a standard location; nothing is assumed.
- On the HPC, create your own `paths.txt` with the HPC root (e.g.
  `/scratch/bc3194/dlm-jailbreak-transfer`) — `.gitignore` already excludes it, so
  the two machines never clobber each other.

> The student's real weights + dataset source live at `/scratch/si2356/...`, which
> is mode `700` owned by `si2356` — **not readable by `bc3194`**. Faithful
> reproduction needs you (or the student/admin) to make them readable by `bc3194`
> and record the path as `DATA_DIR=` in the HPC `paths.txt`. Without that, the
> reproduction pass reports `MISSING` (honest, not a failure).

Read `/scripts/cross-check/outputs/{inventory.json, manifest.json,
deterministic_check.txt, preflight.txt, compare.txt}` for results.

---

## Why this plan exists

The user trusts nothing about the student's data beyond its format. Two failure
classes must both be killed:

1. **Self-contradiction** — the file internally disagrees with itself or with its
   sibling file (e.g. judged `asr_success=1` but the raw sibling never had a
   success; a `response_chars` column that doesn't match the response it describes;
   the same prompt repeated verbatim as a lazy fill). These are caught by the
   deterministic pass, **before any GPU is spent**.

2. **Fabrication / wrong attribution** — the recorded victim response is not what
   the model actually produced for that prompt. These are only caught by re-running
   the attack and diffing the new response against the recorded one. That is the
   reproduction pass.

The skill doc for this repo catalogs many past failure modes we are guarding
against: cross-labeled PIF llama/llada jailbreakbench, api-error text stuffed into
response columns, CJK/refusal text misjudged, `gpt_fuzz` vs `asr_success` identity
disagreements, candidate-fabricated successes in raw files, etc.

---

## Directory layout

```
scripts/cross-check/
  00_run_all.sh               master driver
  01_inventory.py             enumerate the 54 result files + row/col/sha256 + git snapshot
  02_sample.py                stratified, seeded random sample of every file
  03_deterministic_check.py   zero-GPU internal / cross-file consistency checks (C1..C4)
  04_preflight.py             which (attack,model) can be reproduced in THIS env
  05_reproduce.sh             faithful-reproduction driver (GPU host with weights+data)
  05b_select_samples.py       filter manifest -> TSV of rows to reproduce
  06_compare.py               diff recorded vs reproduced response; emit verdicts
  reproducers/
    repro_pif.py              re-run PiF for one (model,dataset) over sampled rows
    repro_metacipher.py       re-run MetaCipher for one model over sampled rows
    repro_arrattack.sh        re-run ArrAttack for one model over sampled rows
  outputs/                    generated reports (gitignored)
```

Every step is independently re-runnable and idempotent-safe (sampling uses a fixed
seed; reproduction writes only into `outputs/reproduced/`, never into `/results`).

---

## Step by step

### Step 01 — Inventory (what exists)

`python3 01_inventory.py`

Lists all 54 CSVs, classifies each as (role `raw|judged`, attack
`pif|metacipher|arrattack`, model, dataset), and records row counts, exact column
set, and SHA-256 of every file, plus a git snapshot of `results/`. This is your
version stamp: re-run it before/after any `git pull` so the sample always matches
the on-disk universe.

### Step 02 — Sample (draw the subset)

`python3 02_sample.py`   (defaults: `--n-per-file 8 --n-raw-pif 4 --n-raw-aa 6
--seed 20260809`)

* **PIF raw** (24 files, one per model×dataset): 4 rows per file.
* **PIF judged** (6 files @ 913): 8 rows per file, stratified so every dataset is
  represented (min 1/group).
* **MetaCipher** raw + judged (12 files @ 913): 8 per file, stratified by dataset.
* **ArrAttack raw** (6 files, multi-attempt): 6 **decision blocks** per file —
  distinct `(dataset, prompt_idx)` groups (a success/failure verdict must be judged
  at the decision level, not per attempt).
* **ArrAttack judged** (6 files @ 165): 8 per file, stratified.

Default total ≈ 324 sample rows spanning all 54 files. Override sample counts with
the CLI flags; the seed makes any choice reproducible.

### Step 03 — Deterministic cross-check (zero GPU)

`python3 03_deterministic_check.py`

Examines every drawn row against the file's OWN logic and its sibling file.
Checks:

* **C1 Response presence/plausibility** — empty, API-error-text (anchored patterns
  only, to avoid the "unauthorized" false-positive trap), CJK-only short text.
* **C2 Internal column consistency** — `original_prompt_chars`/`converted_prompt_chars`/
  `response_chars` vs actual `len()`; MetaCipher `asr_success` vs categorical
  `llm_judge`; `keyword_count` vs parsed `detected_keywords`; ArrAttack
  `asr_success==1` while `gpt_fuzz!='compliance'`.
* **C3 Raw/judged sibling consistency** — the sampled `(dataset, prompt_idx)` must
  exist in the sibling file with the same prompt text (ArrAttack raw is multi-attempt,
  so it falls back to prompt-text matching).
* **C4 Duplicate-prompts within a file** — a lazy fill often repeats one boilerplate
  prompt many times.

Outputs: `deterministic_check.{json,txt}`. On the current (format-clean) data this
is expected to be clean; its job is to catch contradictions regardless of whether
responses are true.

### Step 04 — Preflight (can we reproduce here?)

`python3 04_preflight.py`

The attack scripts hardcode the student's HPC paths as module constants
(`PROJECT="/scratch/si2356/dlm-jailbreak-transfer"`, `HF_CACHE`, `DATASETS`,
`DATASET_PLAN`, `PROJECT_DIR`), and the dataset files + most model weights are NOT
in this repo. So a faithful reproduction needs a host that has, for each target
model: the dataset source, the weights (incl. BERT-large for PiF), a GPU, and a
`DEEPSEEK_API_KEY` (MetaCipher judge).

This writes `preflight.txt` = a matrix that says, per (attack, model), whether the
current machine can faithfully reproduce it and, if not, what's missing. On this
box it will report mostly `repro-here=False` (weights + dataset live on student
HPC) — that's the honest ground truth you need before spending GPU. The one local
candidate is typically **falcon/PiF** (BERT + Falcon weights are present).

### Step 05 — Faithful reproduction (GPU host with weights)

`CROSSCHECK_DATA_DIR=... CROSSCHECK_HF_CACHE=... bash 05_reproduce.sh
[--attack pif|metacipher|arrattack] [--model <key>] [--limit N] [--plan]`

For each sampled row, the corresponding reproducer:

1. **Builds a slim dataset** — a copy of the target dataset file containing ONLY
   the sampled prompt (so we re-run a handful of rows, not all 913/165).
2. **Path-patches a scratch copy** of the attack script so its hardcoded student
   paths point at `CROSSCHECK_DATA_DIR` on this host.
3. **Runs the real attack** (genuine inference on the target model) for that row.
4. Writes the newly regenerated victim response into `outputs/reproduced/`.

Use `--plan` first to print every command without running it. `--limit N` does a
sanity run of the first N samples. Reproducers are separate per attack because the
three pipelines load models and write output very differently; see the files.

> `DEEPSEEK_API_KEY` is read from `$DEEPSEEK_API_KEY` or `~/.bashrc`.
> `CROSSCHECK_HF_CACHE` defaults to `$HF_HOME/hub` or `~/.cache/huggingface/hub`.

> **Known reproducibility blocker (verified 2026-08-09):** `scripts/run_pif.py`
> imports repo-local sibling modules `attack_mlm.py` and `deepseek_patch.py`, and
> neither exists anywhere in this checkout. The PiF pipeline therefore cannot be
> run as-is from this repo — the student must supply those two modules (and the
> `ArrAttack/` tree for `stage5_attack.py`) before PiF/ArrAttack responses can be
> faithfully reproduced. The reproducers detect this and report it as a
> `REPRO-BLOCKER` (exit 3) rather than silently producing a bogus response.

### Step 06 — Compare (verdict per sample)

`python3 06_compare.py`

Diffs the **recorded** response (from the student's CSV) against the **reproduced**
response. Verdicts:

| Verdict | Meaning | Action |
|---|---|---|
| `MATCH` | text ~identical (`norm≥0.99`, PiF only) | trust; expected for greedy PiF |
| `SIMILAR` | high overlap (`≥0.70`) or same behaviour | trust; expected for stochastic MC/AA |
| `DIFFERENT` | low overlap; re-run produced something else | investigate the sample |
| `EMPTY_RECORDED` | CSV claims empty but re-run produced text (or vice-versa) | investigate |
| `MISSING` | no reproduction possible in this env (weights/data absent) | not a failure; need the GPU host |

Rationale: MetaCipher and ArrAttack are stochastic (cipher sampling, GPTFuzz
attempt loop, LLM judge), so a single re-run should NOT byte-match; `SIMILAR`
means "the same weights+data reproduce the same behaviour", which is what we use
to trust the number. PiF runs greedy (`temperature≈0`), so a near-identical
response is expected there — a `DIFFERENT` on PiF is the strongest fabrication
signal.

---

## Missing dependencies: PiF & ArrAttack cannot run as-is (English context)

The two whitebox / code-coupled attacks import modules that are **absent from this
checkout**. This is a real reproducibility problem, not a cosmetic one: it means a
faithful re-run of PiF and ArrAttack responses is currently impossible with the
files in the repo alone. Here is the full picture, per attack.

### PiF (`scripts/run_pif.py`)

`run_pif.py` begins with:

```python
import deepseek_patch
import attack_mlm as _atk
import deepseek_patch as _p; _p.patch_repo_modules()
from attack_mlm import generate_attack
from pif_target_models import DLM_KEYS, load_target, MODEL_MAP
```

- `attack_mlm.py` implements `generate_attack(...)` — the core "perceived-importance
  flattening" gradient loop that rewrites the prompt and drives the target model.
  Without it there is no attack to run.
- `deepseek_patch.py` is a monkey-patch module (`patch_repo_modules()`), used to
  substitute the DeepSeek judge for whatever the original code called. It is the
  "copy the student's judge in" shim.
- `pif_target_models.py` **is** present (it is in the repo), so only the first two
  are missing.

Net effect: `python3 run_pif.py` fails immediately with
`ModuleNotFoundError: No module named 'deepseek_patch'`. The student must supply
`attack_mlm.py` and `deepseek_patch.py` (the exact versions that produced the
`results/pif/*` CSVs) for PiF to be reproducible.

### ArrAttack (`scripts/stage5_attack.py`)

`stage5_attack.py` hard-codes the student's HPC layout and imports from it:

```python
sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack')
from utils.qwen_utils import (chatml_instruction_prompt, resolve_snapshot,
    PATHS, PROJECT_DIR)
sys.path.insert(0, '/scratch/si2356/dlm-jailbreak-transfer/ArrAttack/utils')
```

- The whole `ArrAttack/` tree (GPTFuzz mutation engine, `utils/qwen_utils.py`,
  `PATHS`/`PROJECT_DIR` constants, the shared fine-tuned Qwen generation model under
  `generation_model/`) is referenced but **not in this repo**.
- The hard-coded `/scratch/si2356/...` prefix is another user's path; on any other
  host it must be remapped (the reproducer rewrites it to the local data dir).

Net effect: `stage5_attack.py` cannot import `utils.qwen_utils` on a host that does
not have that `ArrAttack/` directory. The student must provide the `ArrAttack/`
tree (and the generation-model checkpoint it points at) for ArrAttack responses to
be reproducible.

### MetaCipher — self-contained (no blocker)

`metacipher_multi.py` imports only third-party libraries (`transformers`, `openai`,
`numpy`, `pandas`) plus stdlib. It is self-contained: it needs the model weights in
the HF cache and a `DEEPSEEK_API_KEY`, but no repo-local source module. MetaCipher
responses are therefore the most straightforward to reproduce.

### How the reproducers behave

Each reproducer copies the attack script into a scratch dir, patches the hard-coded
student paths onto the local data dir, and — if a required repo-local module is
missing — stops with a **`REPRO-BLOCKER` (exit 3)** and marks that sample
`MISSING` in the comparison. It never fabricates a substitute, so a missing module
surfaces as a clear, actionable error instead of a plausible-looking but wrong
number.

---

## Running the full plan on the student HPC

The most faithful reproduction runs where the experiment actually ran. Because
results ship in git and per-machine paths live in `paths.txt`, the HPC run is just:

```bash
# 1) on the HPC, clone/pull this repo and create its paths.txt:
git clone <repo-url> /scratch/bc3194/dlm-jailbreak-transfer
cd /scratch/bc3194/dlm-jailbreak-transfer
printf '/scratch/bc3194/dlm-jailbreak-transfer\nDATA_DIR=/scratch/bc3194/dlm-jailbreak-data\nHF_HOME=/scratch/bc3194/huggingface_cache\n' > paths.txt

# 2) get the student's weights+dataset readable by bc3194 and set DATA_DIR to it.

# 3) submit the reproduction (ebrainccs condo, 3x A100-80G, 100G, 71:59:59):
cd scripts/cross-check && sbatch run_crosscheck.sbatch

# 4) when it finishes, read the verdict:
#    scripts/cross-check/outputs/compare.txt
```

The reproduction writes only under `outputs/reproduced/`, so it never overwrites
`/results`. After it completes, `compare.txt` tells you, row by row, which numbers
are trustworthy.

---

## Audit & decision summary

* **Which pass kills which fake:** deterministic pass (03) kills self-contradiction
  and lazy fills; reproduction pass (05→06) kills fabrication and wrong-attribution.
* **Trust signal:** a sample is trusted only when (a) it passes 03 AND (b) it gets
  `MATCH`/`SIMILAR` from 06.
* **Need a GPU:** only steps 05 (and 06 after it) require one. 01–04 and the 06
  comparison logic run on any machine, so the cheap safety net is always available.
* If **any** sample in a file comes back `DIFFERENT` or `EMPTY_RECORDED`, escalate
  that file for a full re-audit before trusting its per-model ASR numbers.
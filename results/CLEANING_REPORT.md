# Results Cleaning & Audit Resolution Report

This report documents what was fixed in the `results/` tree for the three jailbreak
attacks (PiF, ArrAttack, MetaCipher) across six victim models, and gives an explanation
for each audit finding that cannot be fixed by editing the data files themselves.

Every file in the accompanying `cleaned/results/` tree parses with a plain
`pandas.read_csv(path)` — no `encoding='utf-8-sig'`, no positional hacks, no
embedded-header handling. 54/54 files parse cleanly, and all judged-file ASRs
reproduce the audit's corrected numbers exactly.

---

## Part 1 — Issues fixed directly in the data

### 1. Embedded header rows in PiF raw CSVs (audit #1)

Four PiF raw files had a stray duplicate header row concatenated mid-data. These were
located by logical (CSV-aware) row scanning and removed:

| File | Embedded header at (1-indexed data) | Rows after fix |
|---|---|---|
| `pif/qwen/harmbench/results.csv` | 107 | 400 |
| `pif/qwen/strongreject/results.csv` | 10 | 313 |
| `pif/llama/harmbench/results.csv` | 244 | 400 |
| `pif/falcon/harmbench/results.csv` | 279 | 400 |

Removal was done by dropping any row whose first cell equals `prompt_idx` (prompt
indices are integers, so no genuine data row is affected). All 24 PiF raw files now land
at the expected `400 / 100 / 100 / 313` counts.

### 2. PiF schema A/B inconsistency (audit #2)

`qwen/harmbench` used **schema A** as its top header and switched to **schema B** at the
embedded row; every other file used schema B throughout. The two schemas are
**position-for-position identical** and differ only in four column *names*:

```
target  → victim           jailbreak_prompt → pif_prompt
target_response → victim_output   asr_gpt   → judge_gpt
```

I verified the data columns align across the A→B boundary in `qwen/harmbench`
(`prompt_idx` continues 105 → 106 → 107, `dataset`/`victim`/`success`/`judge` all
consistent), so this was a pure header rename — **no value re-alignment needed**. All 24
files were rewritten with the single canonical schema B header:

```
prompt_idx, dataset, victim, original_prompt, templated_prompt, pif_prompt,
victim_output, attack_success_internal, judge_gpt, total_query, pif_time_s, total_time_s
```

### 3. ArrAttack LLaDA missing columns (audit #4) — reconstructed, not dropped

LLaDA raw had 16 columns instead of 18 (missing `global_idx`, `dataset_idx`) and a
different column order. Investigating the canonical files revealed the exact semantics,
which let me **reconstruct** the missing columns rather than blank-fill them:

- In the five standard files, `dataset_idx == prompt_idx` (a within-dataset index,
  `1..75 / 1..20 / 1..20 / 1..50`), and `global_idx` is a running `1..165` prompt id.
- The `(global_idx) → (dataset, dataset_idx)` map is **identical across all five standard
  files**, so it is canonical.
- Critically, LLaDA's column *labelled* `prompt_idx` actually holds the **global index**
  (`1..165`, spanning datasets: HarmBench 1–75, StrongReject 76–125, JailbreakBench
  126–145, MaliciousInstruct 146–165). Mapping that against the canonical scheme gave
  **0 mismatches on all 165 prompts**.

Reconstruction applied to LLaDA:

```
global_idx  = old prompt_idx        (the 1..165 global id it already contained)
dataset_idx = canonical within-dataset index looked up by global_idx
prompt_idx  = dataset_idx           (standard within-dataset convention)
dataset     = lowercased
```

LLaDA now has the full 18-column canonical schema in canonical order, with
`unique(dataset, prompt_idx) = 165` and correct per-dataset index ranges.

### 4. Dataset name casing (audit #5)

- **ArrAttack LLaDA**: `HarmBench / JailbreakBench / MaliciousInstruct / StrongReject`
  were lowercased to match every other model (`harmbench / jailbreakbench /
  malicious_instruct / strongreject`).
- **PiF llama StrongREJECT directory**: `strong_reject/` was renamed to `strongreject/`
  to match the other five models (file content was already identical).

### 5. BOM in PIF_JUDGED headers (audit #6 — with a correction)

The audit stated 3 files carried a UTF-8 BOM (qwen, llama, diffucoder). Byte-level
inspection (`od -tx1` → `ef bb bf`) shows **four** files do: **qwen, llama, diffucoder,
and llada**. (`dream` and `falcon` are clean.) All four BOMs were stripped, so
`pandas.read_csv` finds `prompt_idx` — not `\ufeffprompt_idx` — as the first column
without needing `encoding='utf-8-sig'`.

### 6. Directory / filename normalization (audit #9)

Model directories and judged filenames are now consistent lowercase:

- ArrAttack raw: `Qwen2.5/` → `qwen/`; all raw files named `arrattack_results.csv`.
- ArrAttack judged: `arrattack_01_qwen2_5_judged.csv` → `arrattack_qwen_judged.csv`
  (and likewise for the other five).
- MetaCipher raw: `metacipher_qwen2.5_results_FINAL.csv` etc. → `metacipher_results.csv`
  under a lowercase `<model>/` directory.

### 7. Redundant LLaDA `.xlsx` duplicates

`pif/llada/malicious_instruct/results.xlsx` and `.../strongreject/results.xlsx` are
exact duplicates of the sibling `results.csv` (identical shape and columns). They were
**not** carried into the cleaned tree; the CSV is the single canonical copy.

---

## Part 2 — Issues that cannot be fixed by editing the data (explanations)

### A. The actual unified judge is not in the repo (audit #7)

`scripts/evaluation/judge.py` contains only placeholder `KeywordJudge`/`LLMJudge`
classes that do **not** produce the columns present in the `*_Judged/` files
(`asr_success`, `llm_judge`, `judge_reason`, `manual_review_note`, `prompt_type`, …). The
real judge that generated the judged CSVs is missing, and `judge.py` also hardcodes
another user's HPC scratch path (`/scratch/si2356/...`). This cannot be "fixed" by data
edits — the judged outputs are self-contained and correct, but the judge **code** should
be located and committed (or the placeholder removed and the hardcoded path deleted) for
reproducibility. Nothing in the cleaned data depends on it.

### B. ArrAttack judged "column misalignment" is a parser choice, not a bug (audit #8)

The `asr_success` field only looks corrupted when parsed with `awk -F','` because
`final_response` contains commas inside quoted fields. Parsed with any CSV-aware reader
(`pandas`, Python `csv`) the file is already correct: 22 clean columns, proper
`True/False`. There is nothing to fix in the file — the guidance is simply to **use a
CSV-aware parser**. Confirmed: all six ArrAttack judged files parse to `(165, 22)` with
clean booleans.

### C. Model-behavior findings (audit #10–#13) are results, not defects

Falcon CoT/`<think>` leakage, MetaCipher/PiF internal-judge false positives on
Dream/DiffuCoder/Falcon, and GPTFuzz returning 0% for Falcon are all **real modelling
findings** already correctly handled by the unified judge (the judged files are the
source of truth). They are not data errors and require no file edits — they are the
substance of the paper's "internal judge vs unified judge" analysis. The cleaned raw
files retain the internal-judge columns so this disagreement remains auditable.

### D. Data ↔ paper reconciliation (surfaced during cleaning)

Recomputing ASR from the cleaned judged files surfaces a few places where the **paper
text is stale relative to the data**. These are manuscript edits, not data fixes:

| Quantity | Value in cleaned data | Paper location(s) that disagree |
|---|---|---|
| ArrAttack LLaDA | **128/165 = 77.6%** | Table 1 says 127/165 (77.0%); Fig 2 already shows 77.6 |
| PiF LLaDA | **48/913 = 5.3%** | §5.2 / §5.7 / Fig 4 say 7.8% (Table 1 already shows 5.3%) |
| PiF Qwen2.5 | **106/913 = 11.6%** | §5.2 says 17.1% (Table 1 already shows 11.6%) |
| PiF Llama | **120/913 = 13.1%** | §5.2 says 23.1% (Table 1 already shows 13.1%) |
| MetaCipher LLaDA | **432/913 = 47.3%** | §5.2 / Fig 4 say 46.9% (Table 1 already shows 47.3%) |

The data-derived Table 1 column is the correct one; the discrepancies are confined to the
narrative sections (§5.2, §5.7) and Figure 4 captions, which appear to predate the final
judged run. Recommend regenerating all in-text numbers directly from the judged files.

---

## Cleaned tree layout

```
cleaned/results/
  pif/<model>/{harmbench,jailbreakbench,malicious_instruct,strongreject}/results.csv   (schema B, no BOM/embedded rows)
  pif/PIF_JUDGED/<model>_pif_final_judged.csv                                          (BOM stripped)
  arrattack/<model>/arrattack_results.csv                                             (18-col canonical; LLaDA reconstructed)
  arrattack/Arrattack_Judged/arrattack_<model>_judged.csv                             (normalized names)
  metacipher/<model>/metacipher_results.csv                                           (14-col clean)
  metacipher/Metacipher_Judged/<model>.csv                                            (27-col clean)
```

Row-count guarantees after cleaning: PiF raw/judged = 913 unique `(dataset, prompt_idx)`
per model; MetaCipher raw/judged = 913; ArrAttack raw = variable attempts but 165 unique
prompts, judged = 165.

#!/usr/bin/env python3
"""Build model-lineage crosswalk and same-initialization paired summary."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R4 = REPO / "revision_experiments" / "r4"
sys.path.insert(0, str(REPO / "tier0" / "00_shared"))
import common as C  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return 100 * (center - half) / denom, 100 * (center + half) / denom


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


MODEL_EVIDENCE = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "base_or_initialization": "Qwen2.5-7B base model; the exact checkpoint-specific post-training path is not separately itemized in the report.",
        "posttraining": "Instruction-tuned. The Qwen2.5 report describes over 1M SFT samples and multistage RL across the series.",
        "preference_optimization": "The series report includes offline DPO and online GRPO; exact use by this 7B checkpoint is not disaggregated.",
        "safety_tuning_documented": "Family-level helpful, honest, harmless alignment is described. No checkpoint-specific safety dataset or recipe is given in the cited materials.",
        "sources": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct ; https://arxiv.org/abs/2412.15115",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "base_or_initialization": "Llama 3.2 3B pretrained model.",
        "posttraining": "Instruction-tuned; the Meta model card describes SFT and RLHF for tuned versions.",
        "preference_optimization": "RLHF is documented at the family level; the 3B-specific preference recipe is not disclosed in the cited card.",
        "safety_tuning_documented": "The card states alignment for helpfulness and safety; detailed 3B-specific safety data and methods are not provided.",
        "sources": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "base_or_initialization": "Llama 3.1 8B pretrained model.",
        "posttraining": "Instruction-tuned; the Meta model card describes SFT and RLHF for tuned versions.",
        "preference_optimization": "RLHF is documented at the family level; the 8B-specific preference recipe is not disclosed in the cited card.",
        "safety_tuning_documented": "The card states alignment for helpfulness and safety; detailed 8B-specific safety data and methods are not provided.",
        "sources": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    },
    "tiiuae/Falcon-H1R-7B": {
        "base_or_initialization": "Falcon-H1-7B-Base.",
        "posttraining": "Cold-start supervised fine-tuning with long reasoning traces, followed by reinforcement learning.",
        "preference_optimization": "GRPO is documented for the H1R reasoning stage.",
        "safety_tuning_documented": "The cited model card describes reasoning training but does not specify a separate safety-tuning stage.",
        "sources": "https://huggingface.co/tiiuae/Falcon-H1R-7B ; https://huggingface.co/tiiuae/Falcon-H1-7B-Base",
    },
    "GSAI-ML/LLaDA-1.5": {
        "base_or_initialization": "LLaDA-8B-Instruct.",
        "posttraining": "Preference-optimized continuation of LLaDA-8B-Instruct.",
        "preference_optimization": "Variance-Reduced Preference Optimization (VRPO); the cited paper describes 350K preference pairs.",
        "safety_tuning_documented": "Preference alignment is documented; a separate safety-specific objective is not established by the cited card and paper.",
        "sources": "https://huggingface.co/GSAI-ML/LLaDA-1.5 ; https://arxiv.org/abs/2505.19223 ; https://github.com/ML-GSAI/LLaDA",
    },
    "Dream-org/Dream-v0-Instruct-7B": {
        "base_or_initialization": "Dream-7B is initialized from Qwen2.5-7B autoregressive weights, then adapted to diffusion. The instruction checkpoint is the SFT Dream variant, not the raw Qwen instruct checkpoint.",
        "posttraining": "Diffusion pretraining followed by supervised fine-tuning on 1.8M instruction-response pairs from Tulu 3 and SmolLM2.",
        "preference_optimization": "No preference optimization is reported for this Dream-Instruct checkpoint in the cited paper.",
        "safety_tuning_documented": "No separate safety-specific training objective is reported in the cited paper.",
        "sources": "https://huggingface.co/Dream-org/Dream-v0-Instruct-7B ; https://arxiv.org/abs/2508.15487",
    },
    "apple/DiffuCoder-7B-Instruct": {
        "base_or_initialization": "DiffuCoder-7B-Base, a diffusion code model trained on approximately 130B code tokens.",
        "posttraining": "Instruction-tuned from the base checkpoint on 436K OpenCoder instruction samples; the official card describes five SFT epochs.",
        "preference_optimization": "The separate DiffuCoder-7B-cpGRPO checkpoint adds coupled-GRPO, but that is not the Instruct checkpoint used here.",
        "safety_tuning_documented": "The cited training targets code instruction following; no separate safety-tuning stage is reported.",
        "sources": "https://huggingface.co/apple/DiffuCoder-7B-Instruct ; https://huggingface.co/apple/DiffuCoder-7B-Base ; https://arxiv.org/abs/2506.20639",
    },
}


def main() -> None:
    config_path = R4 / "A1_config_audit" / "config_matrix.csv"
    config = pd.read_csv(config_path)
    rows = []
    for checkpoint, facts in MODEL_EVIDENCE.items():
        subset = config.loc[config["victim_checkpoint_id"].astype(str) == checkpoint]
        if subset.empty:
            raise ValueError(f"A1 config matrix has no cells for checkpoint {checkpoint}")
        if subset["victim_revision_sha"].nunique() != 1:
            raise ValueError(f"A1 contains inconsistent pinned revisions for {checkpoint}")
        chat = sorted(set(subset["chat_template_and_system"].astype(str)))
        rows.append({
            "victim_checkpoint_id": checkpoint,
            "source_cells": ";".join(f"{r.attack}/{r.model_alias}" for r in subset.itertuples()),
            "runtime_revision_sha": str(subset["victim_revision_sha"].iloc[0]),
            **facts,
            "chat_template_and_system": " || ".join(chat),
            "run_code_revision": "unknown in retained historical runtime records",
            "qwen_dream_same_initialization_pair": checkpoint in {
                "Qwen/Qwen2.5-7B-Instruct", "Dream-org/Dream-v0-Instruct-7B"
            },
        })
    lineage_path = HERE / "a8_model_lineage.csv"
    pd.DataFrame(rows).to_csv(lineage_path, index=False)

    # Use A7's eligible intersection: all attacks and both checkpoints share
    # the same 90 row keys after copyright/context exclusions.
    pair_path = R4 / "A7_prompt_pool" / "a7_results_ledger.csv"
    pair_ledger = pd.read_csv(pair_path)
    heldout_keys = set(pair_ledger.loc[pair_ledger["attack"] == "arrattack", "canonical_key"])
    paired = pair_ledger.loc[
        pair_ledger["eligible_harmfulness_asr"] & pair_ledger["canonical_key"].isin(heldout_keys)
    ]
    pair_rows = []
    for attack in ("pif", "metacipher", "arrattack"):
        qwen = paired.loc[(paired["model"] == "qwen") & (paired["attack"] == attack), ["canonical_key", "official_success"]].copy()
        dream = paired.loc[(paired["model"] == "dream") & (paired["attack"] == attack), ["canonical_key", "official_success"]].copy()
        if len(qwen) != 90 or len(dream) != 90:
            raise ValueError(f"A8 eligible pair is not 90 canonical rows for {attack}")
        qwen = qwen.rename(columns={"official_success": "qwen_success"})
        dream = dream.rename(columns={"official_success": "dream_success"})
        merged = qwen.merge(dream, on="canonical_key", validate="one_to_one")
        if len(merged) != 90:
            raise ValueError(f"Qwen/Dream eligible pair key mismatch for {attack}")
        qs = int(merged["qwen_success"].sum())
        ds = int(merged["dream_success"].sum())
        q_only = int((merged["qwen_success"].astype(bool) & ~merged["dream_success"].astype(bool)).sum())
        d_only = int((~merged["qwen_success"].astype(bool) & merged["dream_success"].astype(bool)).sum())
        discordant = q_only + d_only
        p = 1.0 if discordant == 0 else float(binomtest(min(q_only, d_only), discordant, 0.5).pvalue)
        qlo, qhi = wilson(qs, len(merged))
        dlo, dhi = wilson(ds, len(merged))
        pair_rows.append({
            "attack": attack,
            "n_paired": len(merged),
            "qwen_successes": qs,
            "qwen_asr_pct": 100 * qs / len(merged),
            "qwen_wilson95_low_pct": qlo,
            "qwen_wilson95_high_pct": qhi,
            "dream_successes": ds,
            "dream_asr_pct": 100 * ds / len(merged),
            "dream_wilson95_low_pct": dlo,
            "dream_wilson95_high_pct": dhi,
            "qwen_only": q_only,
            "dream_only": d_only,
            "paired_difference_qwen_minus_dream_pp": 100 * (qs - ds) / len(merged),
            "mcnemar_exact_p": p,
            "judge_scope": "same attack-specific judge within pair; judges differ across attacks",
        })
    adjusted = holm([row["mcnemar_exact_p"] for row in pair_rows])
    for row, p_adj in zip(pair_rows, adjusted):
        row["holm_p_over_3_attack_tests"] = p_adj
    pair_csv = HERE / "a8_qwen_dream_paired_comparison.csv"
    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False, float_format="%.10g")

    lineage_md = [
        "# A8 model lineage and same-initialization comparison",
        "",
        "The lineage table uses A1's audited runtime checkpoint IDs, pinned revisions, chat-template descriptions, and system-prompt descriptions. Historical run-code revisions remain unknown where no immutable runtime record was retained. Public training summaries are reported only at the granularity documented by each source; absence of a described safety stage is recorded as not documented, not as proof that none existed.",
        "",
        "## Checkpoint identity correction",
        "",
        "The paper's earlier single Llama row combines Llama-3.2-3B-Instruct for PiF and MetaCipher with Llama-3.1-8B-Instruct for ArrAttack. These are separate rows in `a8_model_lineage.csv`; do not report a single Llama three-attack profile from these mixed identities.",
        "",
        "## Qwen and Dream pair",
        "",
        "The Dream 7B paper documents autoregressive initialization from Qwen2.5-7B weights. The compared endpoints are not post-training matched: the Qwen victim is Qwen2.5-7B-Instruct, while Dream-v0-Instruct-7B is a diffusion-adapted model followed by SFT. The pair controls for shared initialization only, not architecture, objective, post-training, safety training, inference decoding, or run-code revision.",
        "",
        "The paired table uses the 90 A7-eligible canonical held-out prompts. Each comparison uses the same attack-specific judge within the Qwen/Dream pair; the judge differs between attacks. Values are descriptive and are not the D3 common-judge comparison.",
        "",
        "| Attack | Qwen | Dream | Qwen minus Dream (pp) | Exact paired p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lineage_md.append(
            f"| {row['attack']} | {row['qwen_successes']}/{row['n_paired']} ({row['qwen_asr_pct']:.2f}%) | "
            f"{row['dream_successes']}/{row['n_paired']} ({row['dream_asr_pct']:.2f}%) | "
            f"{row['paired_difference_qwen_minus_dream_pp']:.2f} | {row['mcnemar_exact_p']:.6g} | "
            f"{row['holm_p_over_3_attack_tests']:.6g} |"
        )
    lineage_md.extend([
        "",
        "## Sources",
        "",
        "Official model cards and primary training reports are linked in `a8_model_lineage.csv`. The Dream initialization and training description is verified against [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487).",
        "",
    ])
    note_path = HERE / "a8_summary.md"
    note_path.write_text("\n".join(lineage_md), encoding="utf-8")

    input_paths = [config_path, pair_path]
    manifest = {
        "task_id": "A8",
        "lineage_rows": len(rows),
        "same_initialization_pair": "Qwen2.5-7B-Instruct and Dream-v0-Instruct-7B share Qwen2.5-7B autoregressive initialization; post-training and architecture differ",
        "paired_prompt_n": 90,
        "new_model_inference": False,
        "new_api_judging": False,
        "model_documentation_sources": sorted({url.strip() for row in rows for url in row["sources"].split(";")}),
        "input_sha256": {str(path.relative_to(REPO)): sha256(path) for path in input_paths},
        "outputs_sha256": {
            path.name: sha256(path) for path in (lineage_path, pair_csv, note_path)
        },
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (HERE / "a8_manifest.json").write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"A8 complete: {len(rows)} distinct runtime checkpoint identities; Qwen/Dream paired on 90 eligible prompts.")
    for row in pair_rows:
        print(f"{row['attack']}: Qwen {row['qwen_successes']}/{row['n_paired']}, Dream {row['dream_successes']}/{row['n_paired']}, Holm p={row['holm_p_over_3_attack_tests']:.6g}")


if __name__ == "__main__":
    main()

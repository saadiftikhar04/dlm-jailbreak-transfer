#!/usr/bin/env python3
"""Build reproducible, response-free G6 seed and paired-greedy summaries."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments" / "r4" / "G6_stochasticity"
JUDGE = OUT / "g6_d3_judge_ledger.csv"
SUMMARY = OUT / "g6_summary.csv"
PAIRED = OUT / "g6_paired_seed_greedy.csv"
SEED_SUMMARY = OUT / "g6_seed_summary.csv"
ANALYSIS_MANIFEST = OUT / "g6_posthoc_analysis_manifest.json"
MODELS = ("qwen", "llama")
SEEDS = (20261021, 20261022, 20261023)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def main() -> None:
    if not OUT.is_dir() or OUT.stat().st_mode & 0o077:
        raise RuntimeError("G6 output directory must exist and be restricted to its owner")
    judgments = read_csv(JUDGE)
    summary = read_csv(SUMMARY)
    expected_conditions = {f"seed_{seed}" for seed in SEEDS} | {"greedy"}
    if len(judgments) != 560:
        raise RuntimeError(f"Expected exactly 560 judge rows, found {len(judgments)}")
    key = lambda row: (row["model"], row["dataset"], int(row["prompt_idx"]), row["condition"])
    judge_map = {key(row): row for row in judgments}
    if len(judge_map) != 560:
        raise RuntimeError("Duplicate G6 judge keys detected")
    summary_map = {(row["model"], row["condition"]): row for row in summary}
    if len(summary_map) != 8:
        raise RuntimeError("Expected eight model-condition summary rows")

    paired_rows: list[dict] = []
    seed_summary_rows: list[dict] = []
    for model in MODELS:
        model_keys = {(dataset, int(idx)) for m, dataset, idx, condition in judge_map if m == model and condition == "greedy"}
        if len(model_keys) != 70:
            raise RuntimeError(f"Expected 70 paired greedy prompts for {model}; found {len(model_keys)}")
        seed_rates = []
        for seed in SEEDS:
            condition = f"seed_{seed}"
            seed_keys = {(dataset, int(idx)) for m, dataset, idx, cond in judge_map if m == model and cond == condition}
            if seed_keys != model_keys:
                raise RuntimeError(f"Seed/greedy prompt mismatch for {model} {condition}")
            seed_row = summary_map[(model, condition)]
            greedy_row = summary_map[(model, "greedy")]
            seed_rate = int(seed_row["successes_itt"]) / int(seed_row["n_itt"])
            seed_rates.append(seed_rate)

            wins = losses = ties = 0
            for dataset, prompt_idx in sorted(model_keys):
                seed_result = judge_map[(model, dataset, prompt_idx, condition)]
                greedy_result = judge_map[(model, dataset, prompt_idx, "greedy")]
                # Frozen intention-to-evaluate rule: unresolved judge/generation rows count as failure.
                seed_success = int(seed_result["status"] == "ok" and seed_result["success"] == "1")
                greedy_success = int(greedy_result["status"] == "ok" and greedy_result["success"] == "1")
                diff = seed_success - greedy_success
                wins += diff > 0
                losses += diff < 0
                ties += diff == 0
            n = len(model_keys)
            paired_rows.append({
                "model": model,
                "seed": seed,
                "seed_successes_itt": int(seed_row["successes_itt"]),
                "greedy_successes_itt": int(greedy_row["successes_itt"]),
                "n_paired": n,
                "seed_minus_greedy_pp": 100.0 * (wins - losses) / n,
                "seed_wins": wins,
                "seed_losses": losses,
                "ties": ties,
                "judge_errors_in_seed_arm": int(seed_row["judge_errors"]),
                "scope": "paired sample-conditional ITT; unresolved outcomes count as failure; unweighted",
            })
        greedy_row = summary_map[(model, "greedy")]
        seed_summary_rows.append({
            "model": model,
            "seed_conditions": ";".join(f"seed_{seed}" for seed in SEEDS),
            "seed_rates_itt": ";".join(f"{rate:.6f}" for rate in seed_rates),
            "three_seed_mean_itt": sum(seed_rates) / len(seed_rates),
            "three_seed_min_itt": min(seed_rates),
            "three_seed_max_itt": max(seed_rates),
            "three_seed_range_pp": 100.0 * (max(seed_rates) - min(seed_rates)),
            "greedy_successes_itt": greedy_row["successes_itt"],
            "greedy_n_itt": greedy_row["n_itt"],
            "greedy_rate_itt": greedy_row["rate_itt"],
            "mean_seed_minus_greedy_pp": sum(r["seed_minus_greedy_pp"] for r in paired_rows if r["model"] == model) / len(SEEDS),
            "scope": "sample-conditional; unweighted; three seeds summarized separately from greedy",
        })

    paired_fields = list(paired_rows[0])
    seed_summary_fields = list(seed_summary_rows[0])
    write_csv(PAIRED, paired_rows, paired_fields)
    write_csv(SEED_SUMMARY, seed_summary_rows, seed_summary_fields)
    manifest = {
        "experiment": "G6_decode_only_stochasticity_posthoc_analysis",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "judge_ledger_sha256": sha256(JUDGE),
        "base_summary_sha256": sha256(SUMMARY),
        "paired_seed_greedy_sha256": sha256(PAIRED),
        "seed_summary_sha256": sha256(SEED_SUMMARY),
        "judge_rows": len(judgments),
        "paired_rows": len(paired_rows),
        "seed_summary_rows": len(seed_summary_rows),
        "pairing_key": ["model", "dataset", "prompt_idx"],
        "unresolved_outcomes": "counted as failure under intention-to-evaluate",
        "sample_conditional": True,
        "pool_weighted": False,
        "raw_prompts_or_responses_in_outputs": False,
    }
    ANALYSIS_MANIFEST.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(ANALYSIS_MANIFEST, 0o600)
    for row in seed_summary_rows:
        print(
            f"{row['model']}: seed mean={row['three_seed_mean_itt']:.4f}; "
            f"range=[{row['three_seed_min_itt']:.4f}, {row['three_seed_max_itt']:.4f}] "
            f"({row['three_seed_range_pp']:.2f} pp); greedy={row['greedy_successes_itt']}/{row['greedy_n_itt']}; "
            f"mean paired seed-greedy difference={row['mean_seed_minus_greedy_pp']:.2f} pp"
        )


if __name__ == "__main__":
    main()

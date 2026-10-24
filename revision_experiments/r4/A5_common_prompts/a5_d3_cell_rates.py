"""Cell-level D3 rates and prompt-paired attack-versus-control lifts for J1.

Reads the merged label file so regenerated and retried rows are included, and
treats every non-`ok` status as a non-success under the frozen ITT rule.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

HERE = Path(__file__).resolve().parent
LABELS = HERE / "j1_final_labels.csv"
CLUSTERS = HERE.parent / "A7_prompt_pool" / "a7_duplicate_clusters.csv"
CELL_CSV = HERE / "j1_d3_cell_rates.csv"
CELL_MD = HERE / "j1_d3_cell_rates.md"
PAIR_CSV = HERE / "j1_d3_paired_lifts.csv"
PAIR_MD = HERE / "j1_d3_paired_lifts.md"

# Frozen pairing rule: the control must match the victim decoding configuration.
# LLaDA PiF has no matching control; the LLaDA ArrAttack comparison is the G4
# schedule arm and stays flagged rather than silently paired.
PAIRS = [
    ("pif", "qwen", "control:pif_arrattack_settings"),
    ("pif", "llama31_8b", "control:g1_pif_feedback"),
    ("pif", "dream", "control:pif_arrattack_settings"),
    ("pif", "diffucoder", "control:pif_arrattack_settings"),
    ("arrattack", "qwen", "control:pif_arrattack_settings"),
    ("arrattack", "llama31_8b", "control:pif_arrattack_settings"),
    ("arrattack", "dream", "control:pif_arrattack_settings"),
    ("arrattack", "diffucoder", "control:pif_arrattack_settings"),
    ("metacipher", "qwen", "control:metacipher_settings"),
    ("metacipher", "llama31_8b", "control:g1_metacipher"),
    ("metacipher", "dream", "control:metacipher_settings"),
    ("metacipher", "diffucoder", "control:metacipher_settings"),
]
UNPAIRED = [
    ("pif", "llada", "no decoding-matched plaintext control exists for this cell"),
    ("arrattack", "llada", "G4 schedule arm; pair only against the locked 128-step control by explicit decision"),
    ("metacipher", "llada", "no decoding-matched plaintext control exists for this cell"),
]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    keys = sorted(p_values, key=lambda key: p_values[key])
    total = len(keys)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(keys):
        value = min(1.0, (total - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def main() -> None:
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8", newline="")))
    by_cell: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(dict)
    for row in rows:
        outcome = 1 if (row["status"] == "ok" and row["success"] == "1") else 0
        by_cell[(row["scope"], row["model"])][row["canonical_key"]] = outcome

    cluster_of: dict[str, str] = {}
    if CLUSTERS.is_file():
        for row in csv.DictReader(CLUSTERS.open(encoding="utf-8", newline="")):
            for member in str(row["members"]).split(";"):
                member = member.strip()
                if member:
                    cluster_of[member] = row["cluster_id"]
    if not cluster_of:
        raise RuntimeError("duplicate-cluster map is required for the S2 cluster bootstrap")

    rng = np.random.default_rng(20261024)
    cells = []
    for (scope, model), outcomes in sorted(by_cell.items()):
        total = len(outcomes)
        successes = sum(outcomes.values())
        ok_rows = sum(
            1 for row in rows
            if row["scope"] == scope and row["model"] == model and row["status"] == "ok"
        )
        low, high = wilson(successes, total)
        by_cluster: dict[str, list[int]] = collections.defaultdict(list)
        for key, value in outcomes.items():
            by_cluster[cluster_of.get(key, key)].append(int(value))
        group_success = np.array([sum(values) for values in by_cluster.values()])
        group_size = np.array([len(values) for values in by_cluster.values()])
        draws = np.empty(10000)
        for index in range(10000):
            pick = rng.integers(0, len(group_success), len(group_success))
            draws[index] = group_success[pick].sum() / group_size[pick].sum()
        cluster_low, cluster_high = np.percentile(draws, [2.5, 97.5])
        cells.append({
            "scope": scope,
            "model": model,
            "rows": total,
            "d3_ok_rows": ok_rows,
            "non_scoreable_rows": total - ok_rows,
            "successes": successes,
            "asr_itt": round(successes / total, 6),
            "asr_itt_ci_low": round(low, 6),
            "asr_itt_ci_high": round(high, 6),
            "cluster_boot_low": round(float(cluster_low), 6),
            "cluster_boot_high": round(float(cluster_high), 6),
            "clusters": len(group_success),
            "asr_ok_only": round(successes / ok_rows, 6) if ok_rows else None,
        })

    with CELL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)

    lines = [
        "# J1 D3 cell rates",
        "",
        "Primary endpoint: D3 `actionable_compliance` on the first stored victim response.",
        "Every non-`ok` status counts as a non-success (D7 intention-to-treat).",
        "Intervals are Wilson 95% and duplicate-cluster bootstrap 95% (10,000 resamples,\nseed 20261024) on the ITT denominator of 913 prompts, using the A7 cluster map.",
        "",
        "| cell | rows | D3 ok | non-scoreable | successes | ASR ITT | Wilson 95% | cluster 95% | ASR (ok rows only) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        ok_only = "n/a" if cell["asr_ok_only"] is None else f"{cell['asr_ok_only']:.4f}"
        lines.append(
            f"| {cell['scope']} / {cell['model']} | {cell['rows']} | {cell['d3_ok_rows']} | "
            f"{cell['non_scoreable_rows']} | {cell['successes']} | {cell['asr_itt']:.4f} | "
            f"[{cell['asr_itt_ci_low']:.4f}, {cell['asr_itt_ci_high']:.4f}] | "
            f"[{cell['cluster_boot_low']:.4f}, {cell['cluster_boot_high']:.4f}] | {ok_only} |"
        )
    lines.append("")
    CELL_MD.write_text("\n".join(lines), encoding="utf-8")

    rng = np.random.default_rng(20261024)
    pairs = []
    for attack_scope, model, control_scope in PAIRS:
        attack = by_cell.get((attack_scope, model))
        control = by_cell.get((control_scope, model))
        if attack is None or control is None:
            continue
        keys = sorted(attack)
        if keys != sorted(control):
            raise RuntimeError(f"paired key sets differ for {attack_scope}/{model}")
        a = np.array([attack[key] for key in keys], dtype=int)
        c = np.array([control[key] for key in keys], dtype=int)
        both = int(((a == 1) & (c == 1)).sum())
        attack_only = int(((a == 1) & (c == 0)).sum())
        control_only = int(((a == 0) & (c == 1)).sum())
        neither = int(((a == 0) & (c == 0)).sum())
        discordant = attack_only + control_only
        p_value = 1.0 if discordant == 0 else float(
            binomtest(attack_only, discordant, 0.5, alternative="two-sided").pvalue
        )
        diff = a - c
        observed = float(diff.mean())
        draws = rng.choice(diff, size=(10000, diff.size), replace=True).mean(axis=1)
        pairs.append({
            "attack_cell": f"{attack_scope}|{model}",
            "control_cell": f"{control_scope}|{model}",
            "prompts": len(keys),
            "attack_successes": int(a.sum()),
            "control_successes": int(c.sum()),
            "lift_pp": round(100 * observed, 4),
            "lift_ci_low_pp": round(100 * float(np.percentile(draws, 2.5)), 4),
            "lift_ci_high_pp": round(100 * float(np.percentile(draws, 97.5)), 4),
            "attack_only": attack_only,
            "control_only": control_only,
            "both": both,
            "neither": neither,
            "mcnemar_exact_p": p_value,
        })

    adjusted = holm({row["attack_cell"]: row["mcnemar_exact_p"] for row in pairs})
    for row in pairs:
        row["holm_p"] = adjusted[row["attack_cell"]]

    with PAIR_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    lines = [
        "# J1 D3 paired attack-versus-control lifts",
        "",
        "Same canonical prompt key, same judge, decoding-matched control; ITT outcomes.",
        "95% intervals are prompt-level bootstrap (10,000 resamples); p-values are exact McNemar,",
        "Holm-adjusted within the attack-versus-control family.",
        "",
        "| attack cell | control cell | attack | control | lift (pp) | 95% CI (pp) | exact p | Holm p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['attack_cell']} | {row['control_cell']} | "
            f"{row['attack_successes']} | {row['control_successes']} | {row['lift_pp']:+.2f} | "
            f"[{row['lift_ci_low_pp']:+.2f}, {row['lift_ci_high_pp']:+.2f}] | "
            f"{row['mcnemar_exact_p']:.4g} | {row['holm_p']:.4g} |"
        )
    lines += ["", "## Not paired by this script", ""]
    for scope, model, reason in UNPAIRED:
        lines.append(f"- `{scope}|{model}`: {reason}")
    lines.append("")
    PAIR_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "cells": len(cells),
        "pairs": len(pairs),
        "cell_table": CELL_CSV.name,
        "pair_table": PAIR_CSV.name,
        "any_holm_significant": any(row["holm_p"] < 0.05 for row in pairs),
    }, indent=4))


if __name__ == "__main__":
    main()

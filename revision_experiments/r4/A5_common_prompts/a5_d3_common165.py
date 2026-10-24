"""A5/S2: three-attack comparison on the 165 common prompts under the J1 D3 judge.

Uses the merged J1 labels (so retried and gap-filled rows are included) and the
frozen ArrAttack held-out key mapping rebuilt from the stored source rows.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TIER0 = REPO / "tier0"
sys.path.insert(0, str(TIER0 / "00_shared"))
import common as C  # noqa: E402

POOL = TIER0 / "02_dedup_and_config_audit" / "pool_913.csv"
LABELS = HERE / "j1_final_labels.csv"
CLUSTERS = REPO / "revision_experiments/r4/A7_prompt_pool/a7_duplicate_clusters.csv"
OUT_CSV = HERE / "a5_d3_common165_cell_rates.csv"
OUT_Q = HERE / "a5_d3_common165_cochran_q.csv"
OUT_PAIR = HERE / "a5_d3_common165_mcnemar_holm.csv"
OUT_MD = HERE / "a5_d3_common165_summary.md"

VICTIMS = ["qwen", "llama31_8b", "llada", "dream", "diffucoder"]
SOURCE_MODEL = {"qwen": "qwen", "llama31_8b": "llama", "llada": "llada", "dream": "dream", "diffucoder": "diffucoder"}
ATTACKS = ("pif", "metacipher", "arrattack")


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for k paired binary treatments (rows are prompts)."""
    k = matrix.shape[1]
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    total = matrix.sum()
    denominator = k * total - int((row_totals ** 2).sum())
    if denominator <= 0:
        return 0.0, 1.0
    numerator = (k - 1) * (k * int((column_totals ** 2).sum()) - total ** 2)
    statistic = numerator / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    keys = sorted(p_values, key=lambda key: p_values[key])
    total = len(keys)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def heldout_keys() -> tuple[list[str], dict[str, str]]:
    pool = pd.read_csv(POOL)
    pool["_key"] = [f"{row.dataset}|{int(row.prompt_idx)}" for row in pool.itertuples()]
    reference: list[str] | None = None
    prompt_by_key: dict[str, str] = {}
    for model in ("qwen", "dream", "diffucoder"):
        frame = C.load("arrattack", model).copy()
        mapping: dict[str, str] = {}
        for dataset, selected in frame.groupby("dataset", sort=False):
            canonical_subset = pool.loc[pool["dataset"].astype(str) == str(dataset)].sort_values("prompt_idx")
            tail = canonical_subset.tail(len(selected)).reset_index(drop=True)
            ordered = selected.sort_values("prompt_idx")
            if ordered["prompt_idx"].astype(int).tolist() != list(range(1, len(ordered) + 1)):
                raise RuntimeError(f"{model}: non-contiguous ArrAttack local prompt indices")
            if ordered["original_prompt"].astype(str).tolist() != tail["original_prompt"].astype(str).tolist():
                raise RuntimeError(f"{model}: ArrAttack rows do not match the canonical tail")
            for source_row, pool_row in zip(ordered.itertuples(), tail.itertuples()):
                mapping[f"{source_row.dataset}|{int(source_row.prompt_idx)}"] = f"{pool_row.dataset}|{int(pool_row.prompt_idx)}"
        keys = sorted(mapping.values())
        if len(keys) != 165:
            raise RuntimeError(f"{model}: expected 165 held-out keys, got {len(keys)}")
        if reference is None:
            reference = keys
            prompt_by_key = {
                f"{row.dataset}|{int(row.prompt_idx)}": str(row.original_prompt) for row in pool.itertuples()
            }
        elif keys != reference:
            raise RuntimeError(f"{model}: held-out key set differs across victims")
    assert reference is not None
    return reference, prompt_by_key


def main() -> None:
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8", newline="")))
    outcome: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(dict)
    for row in rows:
        value = 1 if (row["status"] == "ok" and row["success"] == "1") else 0
        outcome[(row["scope"], row["model"])][row["canonical_key"]] = value

    keys, _ = heldout_keys()
    key_set = set(keys)

    cluster_of: dict[str, str] = {}
    if CLUSTERS.is_file():
        for row in csv.DictReader(CLUSTERS.open(encoding="utf-8", newline="")):
            canonical = row.get("canonical_key") or row.get("key")
            cluster = row.get("cluster_id") or row.get("cluster")
            if canonical and cluster:
                cluster_of[canonical] = cluster

    rng = np.random.default_rng(20261024)
    cell_rows = []
    q_rows = []
    matrices: dict[str, pd.DataFrame] = {}
    for victim in VICTIMS:
        matrix = pd.DataFrame(index=keys)
        for attack in ATTACKS:
            cell = outcome.get((attack, victim))
            if cell is None:
                raise RuntimeError(f"missing J1 cell {attack}/{victim}")
            missing = key_set - set(cell)
            if missing:
                raise RuntimeError(f"{attack}/{victim}: {len(missing)} held-out keys absent from J1")
            matrix[attack] = [cell[key] for key in keys]
        matrices[victim] = matrix
        for attack in ATTACKS:
            values = matrix[attack].to_numpy()
            successes = int(values.sum())
            low, high = wilson(successes, len(values))
            clusters = [cluster_of.get(key, key) for key in keys]
            by_cluster: dict[str, list[int]] = collections.defaultdict(list)
            for cluster, value in zip(clusters, values):
                by_cluster[cluster].append(int(value))
            groups = np.array([sum(v) for v in by_cluster.values()])
            sizes = np.array([len(v) for v in by_cluster.values()])
            draws = []
            for _ in range(10000):
                pick = rng.integers(0, len(groups), len(groups))
                draws.append(groups[pick].sum() / sizes[pick].sum())
            cluster_low, cluster_high = np.percentile(draws, [2.5, 97.5])
            cell_rows.append({
                "victim": victim,
                "attack": attack,
                "prompts": len(values),
                "successes": successes,
                "asr": round(successes / len(values), 6),
                "wilson_low": round(low, 6),
                "wilson_high": round(high, 6),
                "cluster_boot_low": round(float(cluster_low), 6),
                "cluster_boot_high": round(float(cluster_high), 6),
                "clusters": int(len(groups)),
            })
        statistic, p_value = cochran_q(matrix[list(ATTACKS)].to_numpy())
        q_rows.append({"victim": victim, "Q": round(statistic, 4), "df": len(ATTACKS) - 1,
                       "p_value": p_value, "prompts": len(keys)})

    pair_rows = []
    for victim in VICTIMS:
        matrix = matrices[victim]
        for first, second in (("pif", "metacipher"), ("pif", "arrattack"), ("metacipher", "arrattack")):
            a = matrix[first].to_numpy()
            b = matrix[second].to_numpy()
            first_only = int(((a == 1) & (b == 0)).sum())
            second_only = int(((a == 0) & (b == 1)).sum())
            discordant = first_only + second_only
            p_value = 1.0 if discordant == 0 else float(
                binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue
            )
            diff = a.astype(int) - b.astype(int)
            draws = rng.choice(diff, size=(10000, diff.size), replace=True).mean(axis=1)
            pair_rows.append({
                "victim": victim,
                "pair": f"{first}>{second}",
                "first_successes": int(a.sum()),
                "second_successes": int(b.sum()),
                "diff_pp": round(100 * float(diff.mean()), 4),
                "diff_ci_low_pp": round(100 * float(np.percentile(draws, 2.5)), 4),
                "diff_ci_high_pp": round(100 * float(np.percentile(draws, 97.5)), 4),
                "first_only": first_only,
                "second_only": second_only,
                "mcnemar_exact_p": p_value,
            })
    adjusted = holm({f"{row['victim']}|{row['pair']}": row["mcnemar_exact_p"] for row in pair_rows})
    for row in pair_rows:
        row["holm_p"] = adjusted[f"{row['victim']}|{row['pair']}"]

    for path, data in ((OUT_CSV, cell_rows), (OUT_Q, q_rows), (OUT_PAIR, pair_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    lines = [
        "# A5/S2 three-attack comparison on the 165 common prompts (J1 D3 judge)",
        "",
        "ITT outcomes from the merged J1 labels; the 165 ArrAttack held-out keys were rebuilt",
        "from stored source rows and verified identical across victims before analysis.",
        "",
        "## Cell rates",
        "",
        "| victim | attack | successes / 165 | ASR | Wilson 95% | duplicate-cluster 95% |",
        "|---|---|---|---|---|---|",
    ]
    for row in cell_rows:
        lines.append(
            f"| {row['victim']} | {row['attack']} | {row['successes']}/{row['prompts']} | "
            f"{row['asr']:.4f} | [{row['wilson_low']:.4f}, {row['wilson_high']:.4f}] | "
            f"[{row['cluster_boot_low']:.4f}, {row['cluster_boot_high']:.4f}] |"
        )
    lines += ["", "## Cochran's Q across the three attacks", "",
              "| victim | Q | df | p |", "|---|---|---|---|"]
    for row in q_rows:
        lines.append(f"| {row['victim']} | {row['Q']:.4f} | {row['df']} | {row['p_value']:.4g} |")
    lines += ["", "## Pairwise exact McNemar with Holm (attack-versus-attack family)", "",
              "| victim | pair | first | second | diff (pp) | 95% CI (pp) | exact p | Holm p |",
              "|---|---|---|---|---|---|---|---|"]
    for row in pair_rows:
        lines.append(
            f"| {row['victim']} | {row['pair']} | {row['first_successes']} | {row['second_successes']} | "
            f"{row['diff_pp']:+.2f} | [{row['diff_ci_low_pp']:+.2f}, {row['diff_ci_high_pp']:+.2f}] | "
            f"{row['mcnemar_exact_p']:.4g} | {row['holm_p']:.4g} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "prompts": len(keys),
        "cells": len(cell_rows),
        "cochran_significant": sum(1 for row in q_rows if row["p_value"] < 0.05),
        "pairwise_significant_holm": sum(1 for row in pair_rows if row["holm_p"] < 0.05),
        "outputs": [OUT_CSV.name, OUT_Q.name, OUT_PAIR.name, OUT_MD.name],
    }, indent=4))


if __name__ == "__main__":
    main()

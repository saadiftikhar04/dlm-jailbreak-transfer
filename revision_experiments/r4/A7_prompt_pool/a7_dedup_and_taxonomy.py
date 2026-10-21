#!/usr/bin/env python3
"""Build a response-free canonical prompt ledger and dedup sensitivity audit."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TIER0 = REPO / "tier0"
sys.path.insert(0, str(TIER0 / "00_shared"))
import common as C  # noqa: E402

OUT = HERE
KEY_COLS = ("dataset", "prompt_idx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(dataset: object, prompt_idx: object) -> str:
    return f"{dataset}|{int(prompt_idx)}"


def comparable_prompt(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).replace(chr(34) * 2, chr(34))).strip().casefold()


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return 100 * (center - half) / denom, 100 * (center + half) / denom


def cochran_q(matrix: list[list[bool]]) -> tuple[float, float]:
    n = len(matrix)
    k = len(matrix[0])
    col_totals = [sum(int(row[j]) for row in matrix) for j in range(k)]
    row_totals = [sum(int(value) for value in row) for row in matrix]
    total = sum(col_totals)
    numerator = (k - 1) * (k * sum(value * value for value in col_totals) - total * total)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        return 0.0, 1.0
    statistic = max(0.0, numerator / denominator)
    return statistic, float(chi2.sf(statistic, k - 1))


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def build_source_metadata(pool: pd.DataFrame) -> tuple[dict[str, dict], dict[str, str]]:
    source_meta: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}

    hb_path = REPO / "dataset/harmbench/text_all.csv"
    sr_path = REPO / "dataset/strongreject/strongreject.csv"
    jb_path = REPO / "dataset/jailbreakbench/jailbreakbench.csv"
    mi_path = REPO / "dataset/malicious_instruct/malicious_instruct.txt"
    hb = pd.read_csv(hb_path)
    sr = pd.read_csv(sr_path)
    jb = pd.read_csv(jb_path)
    mi = [line.strip() for line in mi_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_hashes.update({
        str(path.relative_to(REPO)): sha256(path) for path in (hb_path, sr_path, jb_path, mi_path)
    })

    native_by_dataset = {
        "harmbench": hb["Behavior"].astype(str).tolist(),
        "strongreject": sr["forbidden_prompt"].astype(str).tolist(),
        "jailbreakbench": jb["Goal"].astype(str).tolist(),
        "malicious_instruct": mi,
    }
    sizes = {"harmbench": 400, "strongreject": 313, "jailbreakbench": 100, "malicious_instruct": 100}
    for dataset, size in sizes.items():
        rows = pool.loc[pool["dataset"].astype(str) == dataset].sort_values("prompt_idx")
        prompts = native_by_dataset[dataset]
        if len(rows) != size or len(prompts) != size:
            raise ValueError(f"Unexpected source/pool size for {dataset}")
        for local_i, (pool_row, source_prompt) in enumerate(zip(rows.itertuples(), prompts), start=1):
            if str(pool_row.original_prompt) != source_prompt:
                raise ValueError(f"Source data mismatch at {dataset} local row {local_i}")
            key = canonical_key(pool_row.dataset, pool_row.prompt_idx)
            if dataset == "harmbench":
                functional = str(hb.iloc[local_i - 1]["FunctionalCategory"])
                semantic = str(hb.iloc[local_i - 1]["SemanticCategory"])
                tags = "" if pd.isna(hb.iloc[local_i - 1]["Tags"]) else str(hb.iloc[local_i - 1]["Tags"])
                context = "" if pd.isna(hb.iloc[local_i - 1]["ContextString"]) else str(hb.iloc[local_i - 1]["ContextString"])
                context_in_prompt = bool(context and context in source_prompt)
                if functional == "copyright":
                    shared = "copyright"
                    exclusion = "harmbench_copyright_behavior"
                elif functional == "contextual":
                    shared = "contextual_behavior"
                    exclusion = "harmbench_context_missing" if not context_in_prompt else ""
                else:
                    exclusion = ""
                    shared = {
                        "chemical_biological": "chemical_or_biological_harm",
                        "cybercrime_intrusion": "cybercrime_and_privacy",
                        "harassment_bullying": "harassment_and_discrimination",
                        "harmful": "other_harm",
                        "illegal": "crime_and_illicit_goods",
                        "misinformation_disinformation": "misinformation_and_deception",
                    }.get(semantic, "unmapped")
                meta = {
                    "source_category": functional,
                    "source_subcategory": semantic,
                    "source_tags": tags,
                    "shared_taxonomy": shared,
                    "context_string_present_in_source": bool(context),
                    "context_string_in_evaluated_prompt": context_in_prompt,
                    "eligible_harmfulness_asr": not bool(exclusion),
                    "exclusion_reason": exclusion,
                }
            elif dataset == "strongreject":
                native = str(sr.iloc[local_i - 1]["category"])
                shared = {
                    "Non-violent crimes": "crime_and_illicit_goods",
                    "Violence": "physical_harm",
                    "Hate, harassment and discrimination": "harassment_and_discrimination",
                    "Disinformation and deception": "misinformation_and_deception",
                    "Illegal goods and services": "crime_and_illicit_goods",
                    "Sexual content": "sexual_content",
                }.get(native, "unmapped")
                meta = {
                    "source_category": native,
                    "source_subcategory": "",
                    "source_tags": "",
                    "shared_taxonomy": shared,
                    "context_string_present_in_source": False,
                    "context_string_in_evaluated_prompt": False,
                    "eligible_harmfulness_asr": True,
                    "exclusion_reason": "",
                }
            elif dataset == "jailbreakbench":
                native = str(jb.iloc[local_i - 1]["Category"])
                shared = {
                    "Harassment/Discrimination": "harassment_and_discrimination",
                    "Malware/Hacking": "cybercrime_and_privacy",
                    "Physical harm": "physical_harm",
                    "Economic harm": "crime_and_illicit_goods",
                    "Fraud/Deception": "misinformation_and_deception",
                    "Disinformation": "misinformation_and_deception",
                    "Sexual/Adult content": "sexual_content",
                    "Privacy": "cybercrime_and_privacy",
                    "Expert advice": "high_impact_advice",
                    "Government decision-making": "government_and_civic_process",
                }.get(native, "unmapped")
                meta = {
                    "source_category": native,
                    "source_subcategory": "",
                    "source_tags": "",
                    "shared_taxonomy": shared,
                    "context_string_present_in_source": False,
                    "context_string_in_evaluated_prompt": False,
                    "eligible_harmfulness_asr": True,
                    "exclusion_reason": "",
                }
            else:
                meta = {
                    "source_category": "unavailable",
                    "source_subcategory": "",
                    "source_tags": "",
                    "shared_taxonomy": "unmapped",
                    "context_string_present_in_source": False,
                    "context_string_in_evaluated_prompt": False,
                    "eligible_harmfulness_asr": True,
                    "exclusion_reason": "",
                }
            source_meta[key] = meta
    return source_meta, source_hashes


def build_clusters(pool: pd.DataFrame) -> tuple[dict[str, str], list[dict], dict[str, str]]:
    pairs_path = TIER0 / "02_dedup_and_config_audit" / "near_duplicate_pairs_mpnet.csv"
    emb_path = TIER0 / "02_dedup_and_config_audit" / "mpnet_913.npy"
    idx_path = TIER0 / "02_dedup_and_config_audit" / "mpnet_index.csv"
    pairs = pd.read_csv(pairs_path)
    embedding = np.load(emb_path)
    mpnet_idx = pd.read_csv(idx_path)
    pool_keys = [canonical_key(r.dataset, r.prompt_idx) for r in pool.itertuples()]
    mpnet_keys = [canonical_key(r.dataset, r.prompt_idx) for r in mpnet_idx.itertuples()]
    if embedding.shape[0] != 913 or mpnet_keys != pool_keys:
        raise ValueError("MPNet embedding index does not align to the canonical pool")
    similarity = embedding @ embedding.T
    expected_edges = set()
    for i in range(len(pool_keys)):
        for j in range(i + 1, len(pool_keys)):
            if similarity[i, j] >= 0.90:
                expected_edges.add((pool_keys[i], pool_keys[j]))
    observed_edges = set()
    for row in pairs.itertuples():
        left = canonical_key(row.bench_a, row.idx_a)
        right = canonical_key(row.bench_b, row.idx_b)
        ordered = tuple(sorted((left, right), key=pool_keys.index))
        if row.cosine < 0.90:
            raise ValueError("MPNet pair file contains an edge below its frozen threshold")
        observed_edges.add(ordered)
    if observed_edges != expected_edges:
        raise ValueError(f"MPNet pair listing disagrees with the stored embeddings: expected={len(expected_edges)} observed={len(observed_edges)}")

    rank = {key: i for i, key in enumerate(pool_keys)}
    parent = {key: key for key in pool_keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            if rank[root_left] <= rank[root_right]:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    for left, right in observed_edges:
        union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for key in pool_keys:
        components[find(key)].append(key)
    sorted_components = sorted(components.values(), key=lambda members: min(rank[x] for x in members))
    cluster_by_key: dict[str, str] = {}
    cluster_rows = []
    for number, members in enumerate(sorted_components, start=1):
        cluster = f"C{number:04d}"
        members.sort(key=rank.__getitem__)
        for key in members:
            cluster_by_key[key] = cluster
        cluster_rows.append({
            "cluster_id": cluster,
            "cluster_size": len(members),
            "representative_key": members[0],
            "members": ";".join(members),
            "has_near_duplicate_edge": len(members) > 1,
        })
    return cluster_by_key, cluster_rows, {
        str(path.relative_to(REPO)): sha256(path) for path in (pairs_path, emb_path, idx_path)
    }


def map_cell_to_pool(attack: str, model: str, frame: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if attack in {"pif", "metacipher"}:
        frame["canonical_key"] = [canonical_key(r.dataset, r.prompt_idx) for r in frame.itertuples()]
        if frame["canonical_key"].duplicated().any():
            raise ValueError(f"Duplicate source row key: {attack}/{model}")
    else:
        key_map: dict[str, str] = {}
        for dataset, group in frame.groupby("dataset", sort=False):
            full = pool.loc[pool["dataset"].astype(str) == str(dataset)].sort_values("prompt_idx")
            tail = full.tail(len(group)).reset_index(drop=True)
            ordered = group.sort_values("prompt_idx")
            if ordered["prompt_idx"].astype(int).tolist() != list(range(1, len(ordered) + 1)):
                raise ValueError(f"ArrAttack local indices not contiguous for {model}/{dataset}")
            if ordered["original_prompt"].astype(str).tolist() != tail["original_prompt"].astype(str).tolist():
                raise ValueError(f"ArrAttack tail source mismatch for {model}/{dataset}")
            for source_row, pool_row in zip(ordered.itertuples(), tail.itertuples()):
                local = f"{source_row.dataset}|{int(source_row.prompt_idx)}"
                key_map[local] = canonical_key(pool_row.dataset, pool_row.prompt_idx)
        frame["canonical_key"] = [key_map[f"{r.dataset}|{int(r.prompt_idx)}"] for r in frame.itertuples()]
        if frame["canonical_key"].duplicated().any():
            raise ValueError(f"Duplicate mapped ArrAttack row key: {model}")
    return frame


def main() -> None:
    pool_path = TIER0 / "02_dedup_and_config_audit" / "pool_913.csv"
    pool = pd.read_csv(pool_path)
    if len(pool) != 913 or pool.duplicated(list(KEY_COLS)).any():
        raise ValueError("Canonical pool is not 913 unique keys")
    pool["canonical_key"] = [canonical_key(r.dataset, r.prompt_idx) for r in pool.itertuples()]
    source_meta, source_hashes = build_source_metadata(pool)
    cluster_by_key, cluster_rows, cluster_hashes = build_clusters(pool)
    source_hashes.update(cluster_hashes)
    source_hashes[str(pool_path.relative_to(REPO))] = sha256(pool_path)
    canonical_prompt_by_key = {
        canonical_key(row.dataset, row.prompt_idx): str(row.original_prompt)
        for row in pool.itertuples()
    }
    hb_source = pd.read_csv(REPO / "dataset/harmbench/text_all.csv")
    context_by_key = {}
    for i, row in hb_source.iterrows():
        key = canonical_key("harmbench", i + 1)
        context = "" if pd.isna(row["ContextString"]) else str(row["ContextString"])
        context_by_key[key] = context
    context_checks = 0

    prompt_rows = []
    rank_by_key = {key: i for i, key in enumerate(pool["canonical_key"])}
    for row in pool.itertuples():
        key = row.canonical_key
        meta = source_meta[key]
        prompt_rows.append({
            "dataset": str(row.dataset),
            "prompt_idx": int(row.prompt_idx),
            "canonical_key": key,
            "original_prompt_sha256": hashlib.sha256(str(row.original_prompt).encode("utf-8")).hexdigest(),
            "cluster_id": cluster_by_key[key],
            "cluster_size": next(x["cluster_size"] for x in cluster_rows if x["cluster_id"] == cluster_by_key[key]),
            "full_pool_cluster_representative": cluster_rows[int(cluster_by_key[key][1:]) - 1]["representative_key"] == key,
            **meta,
        })
    prompt_ledger = pd.DataFrame(prompt_rows)
    prompt_ledger.to_csv(OUT / "a7_prompt_ledger.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "a7_duplicate_clusters.csv", index=False)

    eval_rows = []
    sensitivity_rows = []
    for attack in C.ATTACKS:
        for model in C.MODELS:
            source_path = Path(C.path_for(attack, model))
            source_hashes[str(source_path.relative_to(REPO))] = sha256(source_path)
            frame = C.load(attack, model)
            frame = map_cell_to_pool(attack, model, frame, pool)
            for row in frame.itertuples():
                expected_prompt = canonical_prompt_by_key[row.canonical_key]
                if comparable_prompt(row.original_prompt) != comparable_prompt(expected_prompt):
                    raise ValueError(f"Evaluated original prompt mismatch for {attack}/{model}/{row.canonical_key}")
                if str(row.dataset) == "harmbench":
                    context = context_by_key[row.canonical_key]
                    if context:
                        context_checks += 1
                        if context.strip().casefold() in str(row.original_prompt).casefold():
                            raise ValueError(f"HarmBench context string was included unexpectedly for {attack}/{model}/{row.canonical_key}")
            frame["success"] = C.success_series(attack, frame).astype(bool).to_numpy()
            frame["_rank"] = frame["canonical_key"].map(rank_by_key)
            frame = frame.merge(
                prompt_ledger[["canonical_key", "cluster_id", "eligible_harmfulness_asr", "exclusion_reason", "shared_taxonomy"]],
                on="canonical_key", how="left", validate="one_to_one",
            )
            if frame["cluster_id"].isna().any():
                raise ValueError(f"Missing cluster assignment in {attack}/{model}")
            for row in frame.itertuples():
                eval_rows.append({
                    "attack": attack,
                    "model": model,
                    "dataset": str(row.dataset),
                    "prompt_idx": int(row.prompt_idx),
                    "canonical_key": row.canonical_key,
                    "response_sha256": hashlib.sha256(str(getattr(row, C.ATTACKS[attack]["response_col"])).encode("utf-8")).hexdigest(),
                    "cluster_id": row.cluster_id,
                    "shared_taxonomy": row.shared_taxonomy,
                    "eligible_harmfulness_asr": bool(row.eligible_harmfulness_asr),
                    "exclusion_reason": row.exclusion_reason,
                    "official_success": bool(row.success),
                })
            n_all = len(frame)
            s_all = int(frame["success"].sum())
            eligible = frame.loc[frame["eligible_harmfulness_asr"]].copy()
            eligible_dedup = eligible.sort_values("_rank").drop_duplicates("cluster_id", keep="first")
            n_eligible, s_eligible = len(eligible), int(eligible["success"].sum())
            n_dedup, s_dedup = len(eligible_dedup), int(eligible_dedup["success"].sum())
            lo_e, hi_e = wilson(s_eligible, n_eligible)
            lo_d, hi_d = wilson(s_dedup, n_dedup)
            sensitivity_rows.append({
                "attack": attack,
                "model": model,
                "as_stored_n": n_all,
                "as_stored_successes": s_all,
                "as_stored_asr_pct": 100 * s_all / n_all,
                "excluded_copyright_or_context_missing_n": n_all - n_eligible,
                "eligible_n": n_eligible,
                "eligible_successes": s_eligible,
                "eligible_asr_pct": 100 * s_eligible / n_eligible if n_eligible else "",
                "eligible_wilson95_low_pct": lo_e if n_eligible else "",
                "eligible_wilson95_high_pct": hi_e if n_eligible else "",
                "eligible_cluster_dedup_n": n_dedup,
                "eligible_cluster_dedup_successes": s_dedup,
                "eligible_cluster_dedup_asr_pct": 100 * s_dedup / n_dedup if n_dedup else "",
                "dedup_wilson95_low_pct": lo_d if n_dedup else "",
                "dedup_wilson95_high_pct": hi_d if n_dedup else "",
                "judge_scope": "attack_specific_as_stored",
            })
    results = pd.DataFrame(eval_rows)
    results.to_csv(OUT / "a7_results_ledger.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "a7_dedup_sensitivity.csv", index=False, float_format="%.5f")

    # Re-run A5's paired tests only on rows that remain eligible after A7's
    # predeclared copyright/context exclusions. ArrAttack remains held-out-only.
    heldout_keys = set(results.loc[results["attack"] == "arrattack", "canonical_key"])
    if len(heldout_keys) != 165:
        raise ValueError(f"Expected 165 canonical ArrAttack held-out keys, found {len(heldout_keys)}")
    pair_eligible = results.loc[
        results["eligible_harmfulness_asr"] & results["canonical_key"].isin(heldout_keys)
    ].copy()
    eligible_cell_rows = []
    for model in C.MODELS:
        for attack in ("pif", "metacipher", "arrattack"):
            cell = pair_eligible.loc[(pair_eligible["model"] == model) & (pair_eligible["attack"] == attack)]
            n = len(cell)
            successes = int(cell["official_success"].sum())
            lo, hi = wilson(successes, n)
            by_dataset = cell["dataset"].value_counts().to_dict()
            eligible_cell_rows.append({
                "model": model,
                "attack": attack,
                "successes": successes,
                "n": n,
                "asr_pct": 100 * successes / n if n else "",
                "wilson95_low_pct": lo if n else "",
                "wilson95_high_pct": hi if n else "",
                "dataset_counts": ";".join(f"{name}:{by_dataset[name]}" for name in sorted(by_dataset)),
                "judge_scope": "attack-specific as stored",
            })
    eligible_cell_rates = OUT / "a7_eligible_common_cell_rates.csv"
    pd.DataFrame(eligible_cell_rows).to_csv(eligible_cell_rates, index=False, float_format="%.6f")
    omnibus_rows = []
    pairwise_rows = []
    for model in C.MODELS:
        subset = pair_eligible.loc[pair_eligible["model"] == model]
        pivot = subset.pivot(index="canonical_key", columns="attack", values="official_success")
        if set(pivot.columns) != {"pif", "metacipher", "arrattack"} or pivot.isna().any().any():
            raise ValueError(f"A7 eligible paired attack matrix is incomplete for {model}")
        if len(pivot) != 90:
            raise ValueError(f"Expected 90 eligible held-out keys for {model}, found {len(pivot)}")
        q_stat, q_p = cochran_q(pivot[["pif", "metacipher", "arrattack"]].astype(bool).values.tolist())
        omnibus_rows.append({"model": model, "n_paired": len(pivot), "cochran_q": q_stat, "p_value": q_p})
        attacks = ("pif", "metacipher", "arrattack")
        for i, left in enumerate(attacks):
            for right in attacks[i + 1:]:
                a = pivot[left].astype(bool)
                b = pivot[right].astype(bool)
                a_only = int((a & ~b).sum())
                b_only = int((~a & b).sum())
                discordant = a_only + b_only
                p_value = 1.0 if discordant == 0 else float(
                    binomtest(min(a_only, b_only), discordant, 0.5).pvalue
                )
                pairwise_rows.append({
                    "model": model,
                    "attack_a": left,
                    "attack_b": right,
                    "n_paired": len(pivot),
                    "a_only": a_only,
                    "b_only": b_only,
                    "discordant_n": discordant,
                    "mcnemar_exact_p": p_value,
                })
    for row, adjusted in zip(pairwise_rows, holm([r["mcnemar_exact_p"] for r in pairwise_rows])):
        row["holm_p_over_18_tests"] = adjusted
    pd.DataFrame(omnibus_rows).to_csv(OUT / "a7_eligible_common_cochran_q.csv", index=False, float_format="%.8g")
    pd.DataFrame(pairwise_rows).to_csv(OUT / "a7_eligible_common_mcnemar_holm.csv", index=False, float_format="%.8g")

    context_rows = prompt_ledger[prompt_ledger["exclusion_reason"] == "harmbench_context_missing"]
    copyright_rows = prompt_ledger[prompt_ledger["exclusion_reason"] == "harmbench_copyright_behavior"]
    taxonomy_counts = prompt_ledger["shared_taxonomy"].value_counts().to_dict()
    n_near_edges = len(pd.read_csv(TIER0 / "02_dedup_and_config_audit" / "near_duplicate_pairs_mpnet.csv"))
    summary = [
        "# A7 prompt-pool and deduplication audit",
        "",
        f"The canonical pool contains {len(prompt_ledger)} row identities and {len(cluster_rows)} connected components at the frozen MPNet cosine threshold 0.90. The 45 listed edges were rechecked against `mpnet_913.npy`; all source prompts and each attack cell were mapped to canonical `(dataset, prompt_idx)` identities. The full row ledger contains no prompt or response text.",
        "",
        f"HarmBench metadata identifies {len(copyright_rows)} copyright behaviors and {len(context_rows)} contextual behaviors. The context string was checked against {context_checks} stored HarmBench prompt records across the three attacks and six victims; none contained its source context, so contextual rows are excluded from harmfulness ASR. The copyright set is also excluded rather than re-scored with an unavailable classifier. Counts and remaining per-cell denominators are in `a7_dedup_sensitivity.csv`.",
        "",
        "The shared taxonomy is a deterministic crosswalk from dataset-provided category metadata. MaliciousInstruct has no category field in the available source and is explicitly `unmapped`; no prompt-level human annotation or model-generated labels were added. Dataset-native categories are retained in `a7_prompt_ledger.csv` so the crosswalk can be audited.",
        "",
        "Deduplication sensitivity keeps the first eligible canonical pool row in each connected component. ArrAttack uses its observed held-out rows only and is not promoted to a full-pool estimate. Cell rates remain under attack-specific stored judgments, so they are not the D3 primary-judge results.",
        "",
        "After excluding HarmBench copyright and context-missing rows, the common held-out paired set contains 90 prompts: 50 StrongREJECT, 20 JailbreakBench, and 20 MaliciousInstruct. All 75 held-out HarmBench prompts are contextual and therefore excluded. The attack-specific-judge Cochran's Q and McNemar/Holm tests on these 90 keys are descriptive for this three-benchmark subset and are not reweighted estimates of the full eligible 713-prompt pool, which includes 200 standard HarmBench prompts absent from ArrAttack's held-out sample.",
        "",
        "## Shared taxonomy counts",
        "",
        "| Shared category | Rows |",
        "|---|---:|",
    ]
    for name, count in sorted(taxonomy_counts.items()):
        summary.append(f"| {name} | {count} |")
    summary.extend(["", "## Per-cell results", "", "`a7_eligible_common_cell_rates.csv` reports attack-specific stored-label rates for all six victims on the same 90 eligible held-out keys. It is an unweighted three-benchmark subset, not a reweighted estimate for the full eligible 713-prompt pool. `a7_dedup_sensitivity.csv` reports as-stored, eligible, and deduplicated full-cell rates. Paired eligible-subset tests are in `a7_eligible_common_cochran_q.csv` and `a7_eligible_common_mcnemar_holm.csv`.", ""])
    note_path = OUT / "a7_summary.md"
    note_path.write_text("\n".join(summary), encoding="utf-8")

    manifest = {
        "task_id": "A7",
        "pool_rows": len(pool),
        "mpnet_similarity_threshold": 0.90,
        "near_duplicate_edges_verified": n_near_edges,
        "connected_components": len(cluster_rows),
        "non_singleton_clusters": sum(int(row["cluster_size"] > 1) for row in cluster_rows),
        "harmbench_copyright_excluded_n": len(copyright_rows),
        "harmbench_context_missing_excluded_n": len(context_rows),
        "harmbench_context_inclusion_checks": context_checks,
        "arrattack_scope": "165 held-out prompts only",
        "shared_taxonomy": "crosswalk of available source category metadata; MaliciousInstruct is explicitly unmapped",
        "human_annotation": "none",
        "model_or_api_labels_added": False,
        "analysis_script_sha256": sha256(Path(__file__)),
        "source_sha256": source_hashes,
        "outputs_sha256": {
            path.name: sha256(path)
            for path in (
                OUT / "a7_prompt_ledger.csv",
                OUT / "a7_duplicate_clusters.csv",
                OUT / "a7_results_ledger.csv",
                OUT / "a7_dedup_sensitivity.csv",
                eligible_cell_rates,
                OUT / "a7_eligible_common_cochran_q.csv",
                OUT / "a7_eligible_common_mcnemar_holm.csv",
                note_path,
            )
        },
    }
    (OUT / "a7_manifest.json").write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "pool_rows": len(pool),
        "mpnet_edges": n_near_edges,
        "components": len(cluster_rows),
        "non_singleton_clusters": manifest["non_singleton_clusters"],
        "copyright_excluded": len(copyright_rows),
        "context_missing_excluded": len(context_rows),
    }, indent=4))


if __name__ == "__main__":
    main()

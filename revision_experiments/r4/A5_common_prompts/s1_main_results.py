"""S1: rebuild the main results table straight from the J1 label ledger and the A5/A7 artifacts.

One row per victim x attack. The primary block is the D3 ITT rate on 913 prompts;
the control block is the decoding-matched plaintext control under the same judge;
the lift block is the prompt-paired difference, and the last columns carry the
attack-specific-judge value with its own denominator and provenance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "j1_d3_cell_rates.csv"
PAIRS = HERE / "j1_d3_paired_lifts.csv"
COMMON165 = HERE / "a5_d3_common165_cell_rates.csv"
OFFICIAL = HERE / "a5_official_cell_rates.csv"
OUT_CSV = HERE / "s1_main_results.csv"
OUT_MD = HERE / "s1_main_results.md"

VICTIMS = ["qwen", "llama31_8b", "llada", "dream", "diffucoder"]
ATTACKS = ["pif", "metacipher", "arrattack"]
OFFICIAL_MODEL = {"qwen": "qwen", "llama31_8b": "llama", "llada": "llada",
                  "dream": "dream", "diffucoder": "diffucoder"}
PROVENANCE = {
    "llama31_8b": "legacy stored Llama cell (Llama-3.2), not the G1 Llama-3.1-8B replacement",
}


def load(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {tuple(row[field] for field in key_fields): row
                for row in csv.DictReader(stream)}


def main() -> None:
    cells = load(CELLS, ("scope", "model"))
    pairs = load(PAIRS, ("attack_cell",))
    common = load(COMMON165, ("victim", "attack"))
    official = load(OFFICIAL, ("model", "attack"))

    rows = []
    for victim in VICTIMS:
        for attack in ATTACKS:
            cell = cells[(attack, victim)]
            pair = pairs.get((f"{attack}|{victim}",))
            if pair is not None:
                control_scope = pair["control_cell"].split("|")[0]
                control_cell = cells[(control_scope, victim)]
                lift = {
                    "control_scope": control_scope,
                    "control_successes": int(pair["control_successes"]),
                    "control_asr": float(control_cell["asr_itt"]),
                    "control_wilson_low": float(control_cell["asr_itt_ci_low"]),
                    "control_wilson_high": float(control_cell["asr_itt_ci_high"]),
                    "lift_pp": float(pair["lift_pp"]),
                    "lift_ci_low_pp": float(pair["lift_ci_low_pp"]),
                    "lift_ci_high_pp": float(pair["lift_ci_high_pp"]),
                    "mcnemar_exact_p": float(pair["mcnemar_exact_p"]),
                    "holm_p": float(pair["holm_p"]),
                }
            else:
                lift = {name: "" for name in (
                    "control_scope", "control_successes", "control_asr", "control_wilson_low",
                    "control_wilson_high", "lift_pp", "lift_ci_low_pp", "lift_ci_high_pp",
                    "mcnemar_exact_p", "holm_p")}
                lift["control_scope"] = "none (no decoding-matched control)"

            secondary = official.get((OFFICIAL_MODEL[victim], attack))
            shared = common.get((victim, attack))
            rows.append({
                "victim": victim,
                "attack": attack,
                "judge": "D3 Nemotron-3-Ultra (DeepInfra)",
                "primary_successes": int(cell["successes"]),
                "primary_n": int(cell["rows"]),
                "primary_asr_itt": float(cell["asr_itt"]),
                "primary_wilson_low": float(cell["asr_itt_ci_low"]),
                "primary_wilson_high": float(cell["asr_itt_ci_high"]),
                "primary_cluster_low": float(cell["cluster_boot_low"]),
                "primary_cluster_high": float(cell["cluster_boot_high"]),
                "non_scoreable_rows": int(cell["non_scoreable_rows"]),
                "asr_ok_only": float(cell["asr_ok_only"]) if cell["asr_ok_only"] else "",
                **lift,
                "d3_common165_successes": shared["successes"] if shared else "",
                "d3_common165_asr": shared["asr"] if shared else "",
                "attack_specific_successes": secondary["successes"] if secondary else "",
                "attack_specific_n": secondary["n"] if secondary else "",
                "attack_specific_asr": (
                    round(float(secondary["asr_pct"]) / 100, 6) if secondary else ""
                ),
                "attack_specific_wilson_low": (
                    round(float(secondary["wilson95_low_pct"]) / 100, 6) if secondary else ""
                ),
                "attack_specific_wilson_high": (
                    round(float(secondary["wilson95_high_pct"]) / 100, 6) if secondary else ""
                ),
                "attack_specific_provenance": PROVENANCE.get(victim, "primary stored cell"),
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# S1 main results table (rebuilt from the row ledger)",
        "",
        "Primary judge: D3 `actionable_compliance` on the first stored victim response,",
        "intention-to-treat over 913 prompts per cell (non-`ok` statuses count as non-success).",
        "Controls are the decoding-matched plaintext runs under the same judge; the lift is the",
        "prompt-paired difference with a prompt bootstrap interval and an exact McNemar test",
        "Holm-adjusted inside the attack-versus-control family.",
        "",
        "| victim | attack | D3 successes / 913 | ASR | Wilson 95% | cluster 95% |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['victim']} | {row['attack']} | {row['primary_successes']} | "
            f"{row['primary_asr_itt']:.4f} | "
            f"[{row['primary_wilson_low']:.4f}, {row['primary_wilson_high']:.4f}] | "
            f"[{row['primary_cluster_low']:.4f}, {row['primary_cluster_high']:.4f}] |"
        )
    lines += ["", "## Paired plaintext control and lift", "",
              "| victim | attack | control cell | control successes | control ASR | lift (pp) | 95% CI (pp) | exact p | Holm p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["lift_pp"] == "":
            lines.append(
                f"| {row['victim']} | {row['attack']} | none | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {row['victim']} | {row['attack']} | {row['control_scope']} | "
            f"{row['control_successes']} | {float(row['control_asr']):.4f} | "
            f"{float(row['lift_pp']):+.2f} | "
            f"[{float(row['lift_ci_low_pp']):+.2f}, {float(row['lift_ci_high_pp']):+.2f}] | "
            f"{float(row['mcnemar_exact_p']):.4g} | {float(row['holm_p']):.4g} |"
        )
    lines += ["", "## Secondary columns", "",
              "| victim | attack | D3 on the 165 common prompts | attack-specific judge (as stored) | denominator | provenance |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['victim']} | {row['attack']} | "
            f"{row['d3_common165_successes']}/165 ({float(row['d3_common165_asr']):.4f}) | "
            f"{row['attack_specific_successes']}/{row['attack_specific_n']} "
            f"({float(row['attack_specific_asr']):.4f}) | {row['attack_specific_n']} | "
            f"{row['attack_specific_provenance']} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "rows": len(rows),
        "with_control": sum(1 for row in rows if row["lift_pp"] != ""),
        "holm_significant": sum(1 for row in rows if row["holm_p"] != "" and float(row["holm_p"]) < 0.05),
        "outputs": [OUT_CSV.name, OUT_MD.name],
    }, indent=4))


if __name__ == "__main__":
    main()

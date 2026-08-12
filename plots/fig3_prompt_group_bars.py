#!/usr/bin/env python3
"""
Figure 3 — Content-conditioned susceptibility: per-attack ASR across the six
high-support prompt groups.

Values are computed *at render time* from the unified judged CSVs, so the figure
can never drift from the data (this is the fix for the hardcoded
`fig-3-prompt_group_bars.py`).

Two category-assignment modes are supported:

  --category-mode native      (default, reproduces the paper)
      PiF and MetaCipher are grouped by MetaCipher's `prompt_type`
      (a per-prompt label covering all 913 prompts, joined to PiF by
      (dataset, prompt_idx)); ArrAttack is grouped by its OWN `prompt_type`
      over its 165-prompt held-out stage. This matches how the original
      figure was built.

  --category-mode canonical
      All three attacks share one per-prompt category (MetaCipher's
      `prompt_type`, joined by (dataset, prompt_idx)). Methodologically
      cleaner — identical category definition across attacks — but it changes
      ArrAttack's per-group numbers because its 165-prompt subset is
      re-partitioned by the shared labels.

ASR for each (attack, group) is pooled across all six victim models, exactly as
in the paper. Support counts (pooled cases per cell) are annotated; cells whose
per-model support falls below --min-support are hatched as low-confidence.

Usage:
    python fig3_prompt_group_bars.py --results-root results --outdir figures
    python fig3_prompt_group_bars.py --category-mode canonical
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODELS = ["qwen", "llama", "falcon", "llada", "dream", "diffucoder"]
ATTACKS = ["PiF", "MetaCipher", "ArrAttack"]

# six high-support display groups, in paper order
GROUPS = ["Cyber", "Fraud / financial", "Drugs",
          "Weapons / explosives", "Harassment / hate", "Other policy"]

# MetaCipher prompt_type -> display group (used for PiF, MetaCipher, and the
# canonical mode's ArrAttack labels)
META_MAP = {
    "cyber": "Cyber",
    "fraud_financial": "Fraud / financial",
    "drugs": "Drugs",
    "weapons_explosives": "Weapons / explosives",
    "harassment_hate_coercion": "Harassment / hate",
    "other_policy": "Other policy",
}
# ArrAttack's own prompt_type -> display group (native mode)
ARR_MAP = {
    "cyber": "Cyber",
    "fraud_scam": "Fraud / financial",
    "drugs": "Drugs",
    "violence_weapons": "Weapons / explosives",
    "hate_harassment": "Harassment / hate",
    "other_harmful": "Other policy",
}

# attack colours (colour-blind-safe)
COLORS = {"PiF": "#4C72B0", "MetaCipher": "#DD8452", "ArrAttack": "#55A868"}


# ------------------------------------------------------------------ data loading
def _succ(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0})
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _read(attack: str, model: str, root: Path) -> pd.DataFrame:
    if attack == "PiF":
        d = pd.read_csv(root / "pif" / "PIF_JUDGED" / f"{model}_pif_final_judged.csv")
        d = d.rename(columns={"llm_judge": "succ"})
        return d[["dataset", "prompt_idx", "succ"]]
    if attack == "MetaCipher":
        d = pd.read_csv(root / "metacipher" / "Metacipher_Judged" / f"{model}.csv")
        return d.rename(columns={"asr_success": "succ"})[
            ["dataset", "prompt_idx", "succ", "prompt_type"]]
    d = pd.read_csv(root / "arrattack" / "Arrattack_Judged" /
                    f"arrattack_{model}_judged.csv")
    return d.rename(columns={"asr_success": "succ"})[
        ["dataset", "prompt_idx", "succ", "prompt_type"]]


def category_map(root: Path) -> dict:
    """Canonical per-prompt category from MetaCipher (identical across models)."""
    d = pd.read_csv(root / "metacipher" / "Metacipher_Judged" / "qwen.csv")
    return d.set_index(["dataset", "prompt_idx"])["prompt_type"].to_dict()


def compute(root: Path, mode: str):
    canon = category_map(root)
    asr = pd.DataFrame(index=GROUPS, columns=ATTACKS, dtype=float)
    sup = pd.DataFrame(index=GROUPS, columns=ATTACKS, dtype=int)

    for attack in ATTACKS:
        parts = []
        for m in MODELS:
            d = _read(attack, m, root)
            d["succ"] = _succ(d["succ"])
            if attack == "ArrAttack" and mode == "native":
                d["group"] = d["prompt_type"].map(ARR_MAP)
            else:
                d["cat"] = list(zip(d.dataset, d.prompt_idx))
                d["group"] = d["cat"].map(canon).map(META_MAP)
            parts.append(d[["group", "succ"]])
        x = pd.concat(parts).dropna(subset=["group"])
        g = x.groupby("group")["succ"]
        for grp in GROUPS:
            if grp in g.groups:
                asr.loc[grp, attack] = 100 * g.get_group(grp).mean()
                sup.loc[grp, attack] = len(g.get_group(grp))
            else:
                asr.loc[grp, attack] = np.nan
                sup.loc[grp, attack] = 0
    return asr.astype(float), sup.astype(int)


# ----------------------------------------------------------------------- plotting
def make_figure(asr, sup, mode, min_support, outstem: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    })
    n_groups = len(GROUPS)
    x = np.arange(n_groups)
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.2))

    for i, attack in enumerate(ATTACKS):
        vals = asr[attack].values
        sups = sup[attack].values
        offs = (i - 1) * width
        bars = ax.bar(x + offs, vals, width, label=attack,
                      color=COLORS[attack], edgecolor="white", linewidth=0.7,
                      zorder=3)
        for b, v, s in zip(bars, vals, sups):
            # low per-model support (pooled over 6 models) -> hatch as caution
            if s / len(MODELS) < min_support:
                b.set_hatch("xxx")
                b.set_edgecolor(COLORS[attack])
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#222222", zorder=4)

    # support counts under each group label
    labels = []
    for grp in GROUPS:
        s = sup.loc[grp]
        labels.append(f"{grp}\n"
                      f"n: {s['PiF']}/{s['MetaCipher']}/{s['ArrAttack']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Observed ASR (%)", fontsize=11)
    ax.set_ylim(0, max(80, np.nanmax(asr.values) + 10))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    handles = [Patch(facecolor=COLORS[a], label=a) for a in ATTACKS]
    handles.append(Patch(facecolor="white", edgecolor="#777777", hatch="xxx",
                         label=f"< {min_support}/model support"))
    ax.legend(handles=handles, ncol=4, frameon=False, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, 1.10))

    sub = ("category per prompt: attack-native taxonomy"
           if mode == "native" else
           "category per prompt: shared canonical taxonomy")
    ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=9,
            style="italic", color="#666666")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results", type=Path)
    ap.add_argument("--outdir", default="figures", type=Path)
    ap.add_argument("--category-mode", choices=["native", "canonical"],
                    default="native")
    ap.add_argument("--min-support", type=int, default=10,
                    help="per-model support below which a cell is hatched")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    asr, sup = compute(args.results_root, args.category_mode)
    print(f"Content-group ASR (%)  [mode={args.category_mode}]:")
    print(asr.round(1).to_string())
    print("\nPooled support per cell (6 models):")
    print(sup.to_string())

    out = asr.round(1).copy()
    for a in ATTACKS:
        out[f"{a}_n"] = sup[a]
    out.to_csv(args.outdir / f"fig3_values_{args.category_mode}.csv")

    stem = args.outdir / f"fig3_prompt_group_bars_{args.category_mode}"
    make_figure(asr, sup, args.category_mode, args.min_support, stem)
    print(f"\nWrote {stem}.pdf/.png and fig3_values_{args.category_mode}.csv")


if __name__ == "__main__":
    main()

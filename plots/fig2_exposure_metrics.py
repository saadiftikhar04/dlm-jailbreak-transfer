#!/usr/bin/env python3
"""
Figure 2 — Mechanism Exposure Envelope (MEE) and Attack-Dependence Gap (MDG).

Values are computed *at render time* from the unified judged CSVs, so the figure
can never drift from the data (this is the fix for audit finding #3, where the
old script hardcoded stale MEE/MDG for Qwen and Llama).

Per-model ASR is taken over each attack's own final evaluation stage:
  - PiF / MetaCipher : full 913-prompt pool          (judged success column)
  - ArrAttack        : 165-prompt held-out final stage

  MEE(m) = max_a ASR(a, m)                (worst-case exposure)
  MDG(m) = max_a ASR(a, m) - min_a ASR(a, m)   (attack-dependence gap)

Usage:
    python fig2_exposure_metrics.py --results-root results --outdir figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------- config
# Display order (top -> bottom) and family grouping, matching the paper.
MODELS = ["Qwen2.5", "Llama-3.1", "Falcon-H1R", "DiffuCoder", "Dream", "LLaDA"]
FAMILY = {  # True = causal LLM, False = diffusion-family
    "Qwen2.5": True, "Llama-3.1": True, "Falcon-H1R": True,
    "DiffuCoder": False, "Dream": False, "LLaDA": False,
}
# internal model key used to locate each judged file
KEY = {
    "Qwen2.5": "qwen", "Llama-3.1": "llama", "Falcon-H1R": "falcon",
    "DiffuCoder": "diffucoder", "Dream": "dream", "LLaDA": "llada",
}

# colour palette (colour-blind-safe blue / orange)
C_CAUSAL = "#2166AC"
C_DIFF   = "#D6604D"


# ------------------------------------------------------------------- ASR computation
def _asr(series: pd.Series) -> float:
    """Mean of a boolean/0-1 success column, as a percentage."""
    s = series
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    return 100.0 * pd.to_numeric(s, errors="coerce").fillna(0).mean()


def compute_asr(root: Path) -> pd.DataFrame:
    """Return a tidy DataFrame: model, PiF, MetaCipher, ArrAttack (ASR %)."""
    rows = []
    for disp, key in KEY.items():
        pif = pd.read_csv(root / "pif" / "PIF_JUDGED" / f"{key}_pif_final_judged.csv")
        meta = pd.read_csv(root / "metacipher" / "Metacipher_Judged" / f"{key}.csv")
        arr = pd.read_csv(root / "arrattack" / "Arrattack_Judged" /
                          f"arrattack_{key}_judged.csv")
        rows.append({
            "model": disp,
            "PiF": _asr(pif["llm_judge"]),
            "MetaCipher": _asr(meta["asr_success"]),
            "ArrAttack": _asr(arr["asr_success"]),
        })
    return pd.DataFrame(rows).set_index("model").loc[MODELS]


def exposure_metrics(asr: pd.DataFrame) -> pd.DataFrame:
    attacks = ["PiF", "MetaCipher", "ArrAttack"]
    out = pd.DataFrame(index=asr.index)
    out["MEE"] = asr[attacks].max(axis=1)
    out["MDG"] = asr[attacks].max(axis=1) - asr[attacks].min(axis=1)
    return out


# --------------------------------------------------------------------------- plotting
def make_figure(metrics: pd.DataFrame, asr: pd.DataFrame, outstem: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
    })

    models = list(metrics.index)
    y = range(len(models))[::-1]  # first model at top
    colors = [C_CAUSAL if FAMILY[m] else C_DIFF for m in models]
    hatches = ["" if FAMILY[m] else "///" for m in models]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True,
        gridspec_kw={"wspace": 0.08},
        constrained_layout=True,
    )

    def draw(ax, values, xlabel, xmax):
        bars = ax.barh(list(y), values, height=0.66,
                       color=colors, edgecolor="white", linewidth=0.6)
        for b, h, c in zip(bars, hatches, colors):
            if h:
                b.set_hatch(h)
                b.set_edgecolor(c)          # hatch lines in bar colour
                b.set_linewidth(0.0)
        for yi, v in zip(y, values):
            ax.text(v + xmax * 0.012, yi, f"{v:.1f}",
                    va="center", ha="left", fontsize=10, fontweight="bold",
                    color="#222222")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    draw(axL, metrics["MEE"].values, "Maximum ASR (%)", 90)
    draw(axR, metrics["MDG"].values,
         "Max \u2212 min ASR (percentage points)", 82)

    axL.set_yticks(list(y))
    axL.set_yticklabels(models, fontsize=11)

    axL.set_title("(a)  Mechanism exposure envelope (MEE)",
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)
    axR.set_title("(b)  Attack-dependence gap (MDG)",
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)

    legend = [
        Patch(facecolor=C_CAUSAL, edgecolor="white", label="Causal LLMs"),
        Patch(facecolor=C_DIFF, edgecolor=C_DIFF, hatch="///",
              label="Diffusion-family"),
    ]
    fig.legend(handles=legend, loc="outside upper center", ncol=2,
               frameon=False, fontsize=10.5)

    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results", type=Path,
                    help="path to the results/ directory")
    ap.add_argument("--outdir", default="figures", type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    asr = compute_asr(args.results_root)
    metrics = exposure_metrics(asr)

    print("Per-model ASR (%):")
    print(asr.round(1).to_string())
    print("\nExposure metrics:")
    print(metrics.round(1).to_string())

    metrics.assign(**asr.round(1)).round(1).to_csv(args.outdir / "fig2_values.csv")
    make_figure(metrics, asr, args.outdir / "fig2_exposure_metrics")
    print(f"\nWrote {args.outdir}/fig2_exposure_metrics.pdf/.png and fig2_values.csv")


if __name__ == "__main__":
    main()

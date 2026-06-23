"""Figure 2: Worst-case exposure envelope (MEE) and attack-dependence gap (MDG)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight",
    }
)

# ============================================================================
# FIGURE 2: Worst-case exposure envelope and attack-dependence gap
# ============================================================================
models = ["Qwen2.5", "Llama-3.1", "Falcon-H1R", "DiffuCoder", "Dream", "LLaDA"]
mee = np.array([71.0, 70.5, 24.8, 31.5, 13.9, 77.0])
mdg = np.array([53.9, 47.4, 24.6, 31.4, 13.9, 69.2])

# First three models are causal LLMs; last three are diffusion-family models.
causal_color = "#416D9D"
diffusion_color = "#D17B00"
edge_color = "#222222"
bar_colors = [causal_color] * 3 + [diffusion_color] * 3
bar_hatches = [""] * 3 + ["//"] * 3

y = np.arange(len(models))

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.65, 3.62),
    sharey=True,
    gridspec_kw={"wspace": 0.08},
)

panels = [
    {
        "ax": axes[0],
        "values": mee,
        "title": "Worst-case exposure envelope (MEE)",
        "xlabel": "Maximum ASR (%)",
        "xlim": (0, 84),
        "xticks": np.arange(0, 81, 10),
        "panel": "(a)",
    },
    {
        "ax": axes[1],
        "values": mdg,
        "title": "Attack-dependence gap (MDG)",
        "xlabel": "Maximum - minimum ASR (percentage points)",
        "xlim": (0, 76),
        "xticks": np.arange(0, 71, 10),
        "panel": "(b)",
    },
]

for panel in panels:
    ax = panel["ax"]
    values = panel["values"]

    bars = ax.barh(
        y,
        values,
        height=0.62,
        color=bar_colors,
        edgecolor=edge_color,
        linewidth=0.7,
    )

    for index, (bar, hatch) in enumerate(zip(bars, bar_hatches)):
        bar.set_hatch(hatch)
        ax.text(
            values[index] + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{values[index]:.1f}",
            va="center",
            ha="left",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_title(panel["title"], fontsize=16, fontweight="bold", pad=8)
    ax.set_xlabel(panel["xlabel"], fontsize=14, labelpad=6)
    ax.set_xlim(*panel["xlim"])
    ax.set_xticks(panel["xticks"])
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)

    # Divider between the causal and diffusion-family model groups.
    ax.axhline(2.5, color="#8A8A8A", linewidth=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Standalone panel labels. Remove these two lines if LaTeX overlays them.
    ax.text(
        0.5,
        -0.29,
        panel["panel"],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
    )

axes[0].set_yticks(y)
axes[0].set_yticklabels(models, fontsize=14)
axes[0].invert_yaxis()

legend_handles = [
    Patch(facecolor=causal_color, edgecolor=edge_color, label="Causal LLMs"),
    Patch(
        facecolor=diffusion_color,
        edgecolor=edge_color,
        hatch="//",
        label="Diffusion-family",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False,
    fontsize=16,
    handlelength=1.3,
    columnspacing=2.0,
)

fig.subplots_adjust(left=0.14, right=0.985, top=0.76, bottom=0.25)
fig.savefig(OUTPUT_DIR / "exposure_metrics_final.pdf")
fig.savefig(OUTPUT_DIR / "exposure_metrics_final.png", dpi=300)
plt.close(fig)



print("Created exposure_metrics_final.pdf and exposure_metrics_final.png")

"""Figure 3: Content-group ASR breakdown for PiF, MetaCipher, and ArrAttack."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
# FIGURE 3: Content-group breakdown
# ============================================================================
categories = [
    "Cyber",
    "Fraud / financial",
    "Drugs",
    "Weapons / explosives",
    "Harassment / hate",
    "Other policy",
]

pif = np.array([10.0, 6.9, 6.5, 6.6, 2.4, 8.9])
metacipher = np.array([45.1, 40.9, 36.1, 33.8, 34.0, 34.3])
arrattack = np.array([63.5, 70.4, 42.3, 67.7, 7.7, 49.2])

x = np.arange(len(categories))
width = 0.24

fig, ax = plt.subplots(figsize=(10.906, 5.556))

bars_pif = ax.bar(
    x - width,
    pif,
    width,
    label="PiF",
    color="#3776B6",
    edgecolor="#222222",
    hatch="//",
    linewidth=1.2,
)
bars_metacipher = ax.bar(
    x,
    metacipher,
    width,
    label="MetaCipher",
    color="#D9822B",
    edgecolor="#222222",
    hatch="..",
    linewidth=1.2,
)
bars_arrattack = ax.bar(
    x + width,
    arrattack,
    width,
    label="ArrAttack",
    color="#49A37D",
    edgecolor="#222222",
    hatch="xx",
    linewidth=1.2,
)

# Deliberately no internal plot title: the LaTeX caption supplies the title.
ax.set_ylabel("Observed ASR (%)", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=12)
ax.set_ylim(0, 82)
ax.set_yticks(np.arange(0, 81, 10))
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", color="#BFC3C7", alpha=0.7)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.legend(
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    frameon=False,
    fontsize=18,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.6,
)


def add_bar_labels(bars, values):
    """Add value labels without percent signs or support-count labels."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,  # reduced so adjacent labels do not touch the bars
        )


add_bar_labels(bars_pif, pif)
add_bar_labels(bars_metacipher, metacipher)
add_bar_labels(bars_arrattack, arrattack)

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.82)
fig.savefig(OUTPUT_DIR / "prompt_group_bars_final.pdf")
fig.savefig(OUTPUT_DIR / "prompt_group_bars_final.png", dpi=300)
plt.close(fig)

print("Created prompt_group_bars_final.pdf and prompt_group_bars_final.png")

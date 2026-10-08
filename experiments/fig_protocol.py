"""Schematic of the walk-forward protocol (label maturity and calibration reserve) for one retraining origin.
Not to scale: block widths are chosen for legibility."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ROOT

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d3", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    blocks = [(0.0, 6.0, BLUE, "white", "scorer trains on\nperiods $\\leq \\tau - D - 1 - W_c$", None),
              (6.0, 8.4, AQUA, INK, "", "calibration reserve ($W_c$ periods):\nlabelled, out-of-sample for the scorer"),
              (8.4, 10.2, GREY, INK, "", "$D$ periods whose\nlabels have not matured"),
              (10.2, 13.2, ORANGE, "white", "decisions with this\nmodel ($R$ periods)", None)]
    for a, b, c, tc, inside, outside in blocks:
        ax.barh(0, b - a, left=a, height=0.5, color=c, edgecolor="#fcfcfb", linewidth=1.6)
        if inside:
            ax.text((a + b) / 2, 0, inside, ha="center", va="center", fontsize=7, color=tc)
    ax.annotate("", xy=(7.2, -0.27), xytext=(7.2, -0.62), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(7.2, -0.72, blocks[1][5], ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.annotate("", xy=(9.3, 0.27), xytext=(9.3, 0.62), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(9.3, 0.7, blocks[2][5], ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.annotate("", xy=(10.2, -0.27), xytext=(10.2, -0.62), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(11.4, -0.72, r"retraining origin $\tau$", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.text(6.6, 1.38, r"a decision in period $t$ may use event $i$ only if $s_i + D < t$", ha="center", fontsize=7.2, color=INK)
    ax.set_xlim(-0.2, 13.4); ax.set_ylim(-1.6, 1.7); ax.axis("off")
    fig.savefig(ROOT / "figures" / "fig_protocol.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_protocol.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

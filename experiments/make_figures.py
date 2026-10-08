"""Paper figures (vector PDF) from results/analysis_*_test.json and results/synth_mechanism.csv.
Palette: validated reference palette slots 1-3 (blue, orange, aqua) + neutrals; aqua is <3:1 on white, so every series is
also distinguished by line style/marker and direct-labelled (relief rule). One y-axis per panel; no dual axes.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.utils import ROOT

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "lines.linewidth": 1.6, "pdf.fonttype": 42, "legend.frameon": False})
STYLE = {  # policy -> (colour, linestyle, marker, label)
    "pi1": (MUTED, ":", "x", r"$\pi_1$ single threshold"), "pi2": (INK, "-", "o", r"$\pi_2$ Chow (control)"),
    "pi3": (BLUE, "-", "s", r"$\pi_3$ Chow + epistemic cap"), "pi4": (ORANGE, "-", "^", r"$\pi_4$ MAD"),
    "pi4_nodelta": (ORANGE, "--", "v", r"$\pi_4$ without cap"), "pi3_randu": (AQUA, "--", "D", r"$\pi_3$ random $u$"),
}


def load(dataset: str, era="test") -> dict:
    return json.loads((ROOT / "results" / f"analysis_{dataset}_{era}.json").read_text())


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_cost_curves(A, dataset, shock=None, name="fig_cost_curves"):
    """(a) cost per 1,000 of the single threshold and the Chow control; (b) difference from the control for the
    uncertainty-aware policies (positive = cheaper than the control)."""
    s = pd.DataFrame(A["series"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 4.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    for p in ("pi1", "pi2"):
        d = s[s.policy == p].sort_values("period")
        c, ls, mk, lab = STYLE[p]
        a1.plot(d.period, d.cost_per_1000, color=c, ls=ls, marker=mk, ms=3, label=lab)
    base = s[s.policy == "pi2"].set_index("period").cost_per_1000
    for p in ("pi3", "pi4", "pi3_randu"):
        d = s[s.policy == p].set_index("period").cost_per_1000
        c, ls, mk, lab = STYLE[p]
        a2.plot(d.index, (base - d).reindex(d.index), color=c, ls=ls, marker=mk, ms=3, label=lab)
    a2.axhline(0, color=MUTED, lw=0.7)
    for a in (a1, a2):
        if shock:
            a.axvline(shock - 0.5, color=MUTED, lw=0.8)
    if shock:
        a1.text(shock - 0.3, a1.get_ylim()[1] * 0.97, "regime break", fontsize=7, color=MUTED, va="top")
    a1.set_ylabel("cost per 1,000 events"); a1.legend(fontsize=6, loc="upper left")
    a2.set_ylabel("gain over $\\pi_2$ (cost per 1,000)\npositive = cheaper")
    a2.set_xlabel("decision period"); a2.legend(fontsize=6, loc="lower left")
    save(fig, name)


def fig_calibration(A, shock=None, name="fig_calibration_decay"):
    d = pd.DataFrame(A["diag_series"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 3.4), sharex=True)
    a1.plot(d.period, d.ece_raw, color=MUTED, marker="x", ms=3, label="raw score")
    a1.plot(d.period, d.ece_cal, color=BLUE, marker="s", ms=3, label="recalibrated (matured window)")
    a1.set_ylabel("ECE"); a1.legend(fontsize=6)
    a2.plot(d.period, d.ce_u_cal, color=ORANGE, marker="^", ms=3)
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_ylabel(r"$\mathrm{CE}_u$ (incremental)"); a2.set_xlabel("decision period")
    if shock:
        for a in (a1, a2):
            a.axvline(shock - 0.5, color=MUTED, lw=0.8)
    save(fig, name)


def fig_h2(A, shock=None, name="fig_h2_scatter"):
    pts = pd.DataFrame(A["h2_points"])
    h2 = A["confirmatory"]["H2"]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    post = pts.period >= (shock or 10**9)
    ax.scatter(pts[~post].ce_u_cal, pts[~post].gain, s=18, color=BLUE, marker="o", label="pre-break", zorder=3)
    ax.scatter(pts[post].ce_u_cal, pts[post].gain, s=22, color=ORANGE, marker="^", label="post-break", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$\mathrm{CE}_u$ (incremental conditional calibration error)")
    ax.set_ylabel(r"gain of $\pi_3$ over random-$u$ control" "\n(cost per 1,000)")
    ax.set_ylim(top=pts.gain.max() + 0.35 * (pts.gain.max() - pts.gain.min()))   # head-room so the annotation never covers a point
    ax.text(0.03, 0.95, rf"Spearman $\rho$={h2['rho']:.2f}  [{h2['lo']:.2f}, {h2['hi']:.2f}], n={int(h2['n'])}", transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(fontsize=6, loc="lower right")
    save(fig, name)


def fig_grid(A, name="fig_cost_grid"):
    g = pd.DataFrame(A["grid"])
    blocks = [b for b in ("T1", "T2") if b in set(g.block)] or sorted(set(g.block))
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ee", BLUE])
    fig, axes = plt.subplots(len(blocks), 2, figsize=(3.6, 2.0 * len(blocks)), squeeze=False, constrained_layout=True)
    for i, b in enumerate(blocks):
        for j, p in enumerate(("pi3", "pi4")):
            d = g[(g.block == b) & (g.policy == p)]
            piv = d.pivot(index="C_FN", columns="C_FP", values="gain")
            deg = d.pivot(index="C_FN", columns="C_FP", values="degenerate")
            v = np.nanmax(np.abs(piv.to_numpy())) or 1
            ax = axes[i, j]
            ax.imshow(np.where(deg.to_numpy(), np.nan, piv.to_numpy()), cmap=cmap, vmin=-v, vmax=v, aspect="auto")
            for (r, c), val in np.ndenumerate(piv.to_numpy()):
                ax.text(c, r, "n/a" if deg.to_numpy()[r, c] else (f"{val:+.1f}" if abs(val) < 10 else f"{val:+.0f}"), ha="center", va="center", fontsize=7, color=INK)
            ax.set_xticks(range(3), [f"{int(x)}" for x in piv.columns]); ax.set_yticks(range(3), [f"{int(x)}" for x in piv.index])
            ax.grid(False)
            ax.set_title(rf"{b}: $\pi_{p[-1]}$ vs $\pi_2$", fontsize=7)
            if i == len(blocks) - 1: ax.set_xlabel(r"$C_{FP}$")
            if j == 0: ax.set_ylabel(r"$C_{FN}$")
    save(fig, name)


def fig_scorers(A, block="T2", name="fig_scorers"):
    d = pd.DataFrame(A["scorers"])
    d = d[d.block == block].sort_values("cost_pi2")
    lab = [f"{r.scorer}/{r.fs.replace('F_', '')}" for r in d.itertuples()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.7), sharey=True)
    y = np.arange(len(d))
    a1.barh(y, d.cost_pi2, color=BLUE, height=0.62)
    a1.set_yticks(y, lab); a1.invert_yaxis(); a1.set_xlabel(r"$\pi_2$ cost per 1,000")
    a2.errorbar(d.gain_pi3, y, xerr=[d.gain_pi3 - d.gain_pi3_lo, d.gain_pi3_hi - d.gain_pi3], fmt="s", color=ORANGE, ms=3, capsize=2, lw=1)
    a2.axvline(0, color=MUTED, lw=0.7); a2.set_xlabel(r"gain of $\pi_3$ over $\pi_2$ (cost per 1,000)")
    save(fig, name)


def fig_dsweep(A, name="fig_dsweep"):
    d = pd.DataFrame(A["sweeps"])
    if d.empty:
        return
    d = d[d.R.astype(str) != "inf"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for sc, (c, mk) in {"rf": (INK, "o"), "xgb": (BLUE, "s"), "mlp": (ORANGE, "^")}.items():
        e = d[d.scorer == sc].sort_values("D")
        if len(e):
            ax.errorbar(e.D, e.gain_pi3, yerr=[e.gain_pi3 - e.gain_pi3_lo, e.gain_pi3_hi - e.gain_pi3], color=c, marker=mk, ms=3.5, capsize=2, label=sc)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("label delay D (periods)"); ax.set_ylabel(r"gain of $\pi_3$ over $\pi_2$" "\n(cost per 1,000)"); ax.legend(fontsize=6)
    save(fig, name)


def fig_synth(name="fig_synth_mechanism"):
    d = pd.read_csv(ROOT / "results" / "synth_mechanism.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharex=True)
    sty = {"local": (BLUE, "s", "informative $u$, local shift"), "local_randu": (AQUA, "D", "random $u$, same shift"),
           "uniform": (MUTED, "x", "uniform shift")}
    for ax, col, ttl in zip(axes, ("gain_pi3", "gain_pi4"), (r"$\pi_3$ (cap only)", r"$\pi_4$ (MAD)")):
        for fam, (c, mk, lab) in sty.items():
            e = d[d.family == fam]
            ax.errorbar(e.s, e[col], yerr=e[col + "_se"], color=c, marker=mk, ms=3.5, capsize=2, label=lab)
        ax.axhline(0, color=MUTED, lw=0.7); ax.set_xlabel("shift magnitude s"); ax.set_title(ttl, fontsize=8)
    axes[0].set_ylabel("gain over $\\pi_2$\n(cost per 1,000)"); axes[0].legend(fontsize=6)
    save(fig, name)


if __name__ == "__main__":
    A = load("elliptic")
    fig_cost_curves(A, "elliptic", shock=43)
    fig_calibration(A, shock=43)
    fig_h2(A, shock=43)
    fig_grid(A)
    fig_scorers(A)
    fig_dsweep(A)
    fig_synth()
    try:
        B = load("baf_Base")
        fig_cost_curves(B, "baf", shock=None, name="fig_cost_curves_baf")
    except FileNotFoundError:
        pass

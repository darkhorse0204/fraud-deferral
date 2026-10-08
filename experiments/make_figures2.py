"""Additional figures: pipeline diagram, forest plot, capacity frontier, recalibration sensitivity, reliability,
missed-vs-caught signals, MAD trajectory, analyst accuracy, seed variability, and the Study 2 figures.
Palette: validated slots 1-3 (blue, orange, aqua) + neutrals; every series also differs by marker / line style.
python -m experiments.make_figures2 [s1] [s2]
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments.make_figures import AQUA, BLUE, INK, MUTED, ORANGE, save
from src.utils import ROOT

R = ROOT / "results"
SHADE = "#dbe7f6"
NL = "\n"


def J(name):
    return json.loads((R / name).read_text())


# ------------------------------------------------------------------------------------------------ diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.06, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
    xs, ws = [0.5, 17.5, 38.5, 60.5, 81.0], [14.5, 18.5, 19.5, 18.0, 18.5]
    txt = [NL.join(["Event stream", "(features, period)"]), NL.join(["Scorer ensemble", "trained on matured", "periods only"]),
           NL.join(["Rolling recalibration", "on the matured,", "out-of-sample window"]), NL.join(["Decision policy", "(cost-based", "thresholds)"]),
           NL.join(["approve / review /", "block, then cost"])]
    fcs = ["white", SHADE, SHADE, SHADE, "white"]
    for x, w, t, fc in zip(xs, ws, txt, fcs):
        ax.add_patch(FancyBboxPatch((x, 13), w, 10, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=INK, lw=0.8))
        ax.text(x + w / 2, 18, t, ha="center", va="center", fontsize=6.3, color=INK)
    for i in range(4):
        ax.add_patch(FancyArrowPatch((xs[i] + ws[i] + 0.4, 18), (xs[i + 1] - 0.4, 18), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.8))
    ax.text((xs[1] + ws[1] + xs[2]) / 2, 25.2, "score, uncertainty", ha="center", fontsize=5.8, color=MUTED)
    ax.text((xs[2] + ws[2] + xs[3]) / 2, 25.2, "calibrated score", ha="center", fontsize=5.8, color=MUTED)
    ax.add_patch(FancyBboxPatch((17.5, 1.5), 61, 5, boxstyle="round,pad=0.2,rounding_size=0.8", fc="white", ec=MUTED, lw=0.7, ls="--"))
    ax.text(48, 4, "matured labels: an event of period $s$ is usable in period $t$ only if $s + D < t$", ha="center", va="center", fontsize=6.2, color=MUTED)
    for x, w in zip(xs[1:4], ws[1:4]):
        ax.add_patch(FancyArrowPatch((x + w / 2, 6.9), (x + w / 2, 12.7), arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7, ls="--"))
    ax.add_patch(FancyArrowPatch((xs[4] + ws[4] / 2, 12.7), (79.0, 4), arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7, ls="--", connectionstyle="arc3,rad=-0.3"))
    ax.text(50, 29.5, "Shaded boxes are evaluated in this paper; every policy consumes the same cached scores", ha="center", fontsize=6.6, color=INK)
    save(fig, "fig_pipeline")


# ------------------------------------------------------------------------------------------------ Study 1 extras
def fig_forest():
    a = J("analysis_elliptic_test.json")
    m = pd.DataFrame(a["main_table"])
    rows = [("pi3", r"$\pi_3$ epistemic cap"), ("pi3_randu", r"$\pi_3$ random $u$"), ("pi4", r"$\pi_4$ MAD"), ("pi4_nodelta", r"$\pi_4$ without cap"),
            ("pi4_randu", r"$\pi_4$ random $u$"), ("pi5", r"$\pi_5$ suppress-on-uncertainty")]
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.3), sharey=True, constrained_layout=True)
    for ax, b in zip(axes, ("T1", "T2")):
        for i, (p, lab) in enumerate(rows):
            r = m[(m.block == b) & (m.policy == p)].iloc[0]
            c = BLUE if p.startswith("pi3") else (ORANGE if p.startswith("pi4") else MUTED)
            ax.errorbar(r.gain_vs_pi2, i, xerr=[[r.gain_vs_pi2 - r.lo], [r.hi - r.gain_vs_pi2]], fmt="s" if "randu" not in p else "D", color=c, ms=3.5, capsize=2, lw=1)
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_title(f"block {b}", fontsize=8); ax.set_xlabel(r"gain over $\pi_2$ (cost per 1,000)", fontsize=7)
    axes[0].set_yticks(range(len(rows)), [l for _, l in rows], fontsize=7); axes[0].invert_yaxis()
    save(fig, "fig_forest")


def fig_capacity():
    c = pd.DataFrame(J("extra_elliptic.json")["capacity"])
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.2), constrained_layout=True)
    for ax, b in zip(axes, ("T2", "ALL")):
        d = c[(c.block == b) & (~c.unlimited)]
        for p, col, mk, ls, lab in (("pi1", MUTED, "x", ":", r"$\pi_1$"), ("pi2", INK, "o", "-", r"$\pi_2$"), ("pi3", BLUE, "s", "--", r"$\pi_3$")):
            e = d[d.policy == p].sort_values("capacity")
            ax.plot(100 * e.capacity, e.cost, color=col, marker=mk, ms=3, ls=ls, label=lab)
        un = c[(c.block == b) & c.unlimited & (c.policy == "pi2")].cost.iloc[0]
        ax.axhline(un, color=AQUA, lw=0.9, ls="-."); ax.text(0.27, un, r" $\pi_2$ unlimited", fontsize=6, color=MUTED, va="bottom")
        ax.set_xscale("log"); ax.set_xlabel("review capacity (% of events)", fontsize=7); ax.set_title("block T2" if b == "T2" else "whole test era", fontsize=8)
    axes[0].set_ylabel("cost per 1,000 events"); axes[0].legend(fontsize=6)
    save(fig, "fig_capacity")


def fig_recal():
    r = pd.DataFrame(J("extra_elliptic.json")["recalibration"])
    r = r[r.block == "ALL"]
    kinds, lab = ["none", "temp", "affine", "isotonic"], ["none", "temperature", "affine", "isotonic"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.2, 2.2), constrained_layout=True)
    x = np.arange(len(kinds)); wdt = 0.26
    for j, (W, col) in enumerate(zip((3, 5, 8), (BLUE, ORANGE, AQUA))):
        d = r[r.W == W].set_index("calib").loc[kinds]
        a1.bar(x + (j - 1) * wdt, d.cost_pi2, wdt * 0.92, color=col, label=f"window {W}")
        a2.bar(x + (j - 1) * wdt, d.gain_pi3, wdt * 0.92, color=col)
    a1.set_ylim(300, None); a1.set_ylabel(r"$\pi_2$ cost per 1,000"); a1.legend(fontsize=6)
    a2.axhline(0, color=INK, lw=0.7); a2.set_ylabel(r"gain of $\pi_3$ over $\pi_2$")
    for a in (a1, a2):
        a.set_xticks(x, lab, fontsize=6, rotation=20)
    save(fig, "fig_recal_sensitivity")


def fig_reliability():
    rel = J("extra_elliptic.json")["reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.3), sharey=True, constrained_layout=True)
    for ax, b in zip(axes, ("T1", "T2")):
        for col, c, mk, lab in (("p", MUTED, "x", "raw score"), ("pc", BLUE, "s", "recalibrated")):
            d = pd.DataFrame(rel[f"{b}_{col}"])
            ax.plot(d.mean_pred.clip(lower=1e-4), d.frac_pos.clip(lower=1e-4), color=c, marker=mk, ms=3, label=lab)
        ax.plot([1e-4, 1], [1e-4, 1], color=INK, lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title("block " + b, fontsize=8); ax.set_xlabel("mean predicted probability", fontsize=7)
    axes[0].set_ylabel("observed illicit fraction"); axes[0].legend(fontsize=6)
    save(fig, "fig_reliability")


def fig_missed():
    d = J("extra_elliptic.json")["missed_vs_caught"]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.3), constrained_layout=True)
    for ax, key, ttl, log in ((axes[0], "u", "ensemble variance", True), (axes[1], "nov", "novelty score", False)):
        pos = 0
        for b in ("T1", "T2"):
            for grp, col in (("missed", ORANGE), ("caught", BLUE)):
                q = d[b][f"{key}_{grp}"]
                ax.plot([pos, pos], [q[0], q[4]], color=col, lw=1)
                ax.add_patch(plt.Rectangle((pos - 0.3, q[1]), 0.6, q[3] - q[1], fc=col, ec=col, alpha=0.35))
                ax.plot([pos - 0.3, pos + 0.3], [q[2], q[2]], color=col, lw=1.6)
                pos += 1
            pos += 0.6
        ax.set_xticks([0.5, 3.1], ["T1", "T2"]); ax.set_title(ttl, fontsize=7.5)
        if log:
            ax.set_yscale("log")
    axes[0].plot([], [], color=ORANGE, lw=3, alpha=0.6, label="missed illicit"); axes[0].plot([], [], color=BLUE, lw=3, alpha=0.6, label="caught illicit")
    axes[0].legend(fontsize=6, loc="lower left")
    x = np.arange(2); w = 0.36
    axes[2].bar(x - w / 2, [d[b]["auroc_u"] for b in ("T1", "T2")], w * 0.92, color=BLUE, label="ensemble variance")
    axes[2].bar(x + w / 2, [d[b]["auroc_nov"] for b in ("T1", "T2")], w * 0.92, color=ORANGE, label="novelty")
    axes[2].axhline(0.5, color=INK, lw=0.7, ls="--"); axes[2].set_xticks(x, ["T1", "T2"]); axes[2].set_ylim(0, 1.25)
    axes[2].set_title(NL.join(["AUROC for missed", "(among illicit)"]), fontsize=7.5); axes[2].legend(fontsize=6, loc="upper right")
    save(fig, "fig_missed_signals")


def fig_mad():
    t = pd.DataFrame(J("error_analysis_elliptic.json")["trajectory"])
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.2), constrained_layout=True)
    axes[0].plot(t.period, t.ma, color=ORANGE, marker="^", ms=3); axes[0].axhline(1, color=INK, lw=0.7, ls="--")
    axes[0].set_title(NL.join(["MAD approve-threshold", "multiplier"]), fontsize=7.5)
    axes[1].plot(t.period, 100 * t.review_rate_pi2, color=INK, marker="o", ms=3, label=r"$\pi_2$")
    axes[1].plot(t.period, 100 * t.review_rate_pi4, color=ORANGE, marker="^", ms=3, ls="--", label=r"$\pi_4$ MAD")
    axes[1].set_title(NL.join(["events sent to", "review (%)"]), fontsize=7.5); axes[1].legend(fontsize=6)
    axes[2].plot(t.period, 100 * t.prev_win, color=AQUA, marker="D", ms=3, ls="--", label="matured window")
    axes[2].plot(t.period, 100 * t.prev_cur, color=BLUE, marker="s", ms=3, label="current period")
    axes[2].set_title(NL.join(["illicit prevalence", "(%)"]), fontsize=7.5); axes[2].legend(fontsize=6)
    for a in axes:
        a.axvline(42.5, color=MUTED, lw=0.8); a.set_xlabel("decision period", fontsize=7)
    save(fig, "fig_mad_trajectory")


def fig_rho_seeds():
    r = pd.DataFrame(J("extra_elliptic.json")["rho"]).query("block == 'T2'")
    sg = pd.DataFrame(J("analysis_elliptic_test.json")["seed_gains_T2"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.2, 2.2), constrained_layout=True)
    for p, col, mk, ls, lab in (("pi1", MUTED, "x", ":", r"$\pi_1$"), ("pi2", INK, "o", "-", r"$\pi_2$"), ("pi3", BLUE, "s", "--", r"$\pi_3$"), ("pi4", ORANGE, "^", "-", r"$\pi_4$")):
        a1.plot(r.rho, r[p], color=col, marker=mk, ms=3, ls=ls, label=lab)
    a1.invert_xaxis(); a1.set_xlabel(r"analyst accuracy $\rho$"); a1.set_ylabel("T2 cost per 1,000"); a1.legend(fontsize=6)
    rng = np.random.default_rng(0)
    for i, (col, c) in enumerate((("gain_pi3", BLUE), ("gain_pi4", ORANGE))):
        a2.scatter(i + rng.uniform(-0.12, 0.12, len(sg)), sg[col], s=14, color=c, zorder=3)
        a2.plot([i - 0.25, i + 0.25], [sg[col].mean()] * 2, color=INK, lw=1.2)
    a2.axhline(0, color=INK, lw=0.7, ls="--"); a2.set_xticks([0, 1], [r"$\pi_3$", r"$\pi_4$ MAD"]); a2.set_ylabel(NL.join([r"T2 gain over $\pi_2$", "(one dot per seed)"]))
    save(fig, "fig_rho_seeds")


# ------------------------------------------------------------------------------------------------ Study 2
def fig_s2_curves():
    a = J("analysis_study2.json")
    s = pd.DataFrame(a["series"]); pp = pd.DataFrame(a["per_period"])
    fig, axes = plt.subplots(3, 1, figsize=(4.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.75]}, constrained_layout=True)
    for p, col, mk, ls, lab in (("pi1", MUTED, "x", ":", r"$\pi_1$ single threshold"), ("pi2", INK, "o", "-", r"$\pi_2$ Chow band (control)")):
        d = s[s.policy == p].sort_values("period")
        axes[0].plot(d.period, d.cost_per_1000, color=col, marker=mk, ms=2.2, ls=ls, lw=1.1, label=lab)
    axes[0].set_ylabel(NL.join(["cost per 1,000", "events (BTC)"])); axes[0].set_yscale("log")
    axes[0].legend(fontsize=6, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    base = s[s.policy == "pi2"].set_index("period").cost_per_1000
    for p, col, mk, ls, lab in (("pi3", BLUE, "s", "-", r"$\pi_3$ ensemble variance"), ("pi4", ORANGE, "^", "-", r"$\pi_4$ MAD"), ("pi3_if", AQUA, "D", "--", r"$\pi_3$ isolation forest")):
        d = s[s.policy == p].set_index("period").cost_per_1000
        axes[1].plot(d.index, (base - d).reindex(d.index), color=col, marker=mk, ms=2.2, ls=ls, lw=1.1, label=lab)
    axes[1].axhline(0, color=INK, lw=0.7); axes[1].set_ylabel(NL.join([r"gain over $\pi_2$", "(positive = cheaper)"])); axes[1].legend(fontsize=6)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[2].bar(pp.period, pp.n_pos - pp.n_novel, color=BLUE, width=0.8, label="known family")
    axes[2].bar(pp.period, pp.n_novel, bottom=pp.n_pos - pp.n_novel, color=ORANGE, width=0.8, label="novel family")
    axes[2].set_ylabel(NL.join(["ransomware", "events"])); axes[2].set_xlabel("28-day period (52 = January 2015)"); axes[2].legend(fontsize=6)
    for a_ in axes:
        a_.axvline(64.5, color=MUTED, lw=0.8)
    save(fig, "fig_s2_curves")


def fig_s2_forest():
    a = J("analysis_study2.json")
    rows = [("pi1", r"$\pi_1$ single threshold"), ("pi3", r"$\pi_3$ ensemble variance"), ("pi3_randu", r"$\pi_3$ random $u$"), ("pi3_if", r"$\pi_3$ isolation forest"),
            ("pi3_maha", r"$\pi_3$ Mahalanobis"), ("pi3_knn", r"$\pi_3$ kNN distance"), ("pi4", r"$\pi_4$ MAD"), ("pi4_nodelta", r"$\pi_4$ without cap")]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.5), sharey=True, constrained_layout=True)
    for ax, b in zip(axes, ("T1", "T2", "ALL")):
        for i, (p, lab) in enumerate(rows):
            r = a["main"][b][p]
            e, lo, hi = (100 * r[k] / r["cost_ctrl"] for k in ("est", "lo", "hi"))
            c = MUTED if p == "pi1" else (ORANGE if p.startswith("pi4") else BLUE)
            ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="D" if "rand" in p else "s", color=c, ms=3.2, capsize=2, lw=1)
        ax.axvline(0, color=INK, lw=0.7); ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks([-2, 0, 2, 10, 30], ["−2", "0", "2", "10", "30"]); ax.minorticks_off()
        ax.set_title({"T1": "T1 (2015)", "T2": "T2 (2016-17)", "ALL": "whole test era"}[b], fontsize=8); ax.set_xlabel(r"gain over $\pi_2$ (% of its cost)", fontsize=6.5)
    axes[0].set_yticks(range(len(rows)), [l for _, l in rows], fontsize=7); axes[0].invert_yaxis()
    save(fig, "fig_s2_forest")


def fig_s2_novel():
    nv = J("analysis_study2.json")["novel_family"]
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.3), constrained_layout=True)
    keys = [("u", NL.join(["ensemble", "variance"])), ("u_if", NL.join(["isolation", "forest"])), ("u_maha", "Mahalanobis"), ("u_knn", NL.join(["kNN", "distance"]))]
    au = nv["auroc_for_novel_family"]
    axes[0].bar(range(4), [au[k] for k, _ in keys], color=[BLUE, ORANGE, ORANGE, ORANGE], width=0.62)
    axes[0].axhline(0.5, color=INK, lw=0.7, ls="--"); axes[0].set_ylim(0, 1); axes[0].set_xticks(range(4), [l for _, l in keys], fontsize=6)
    axes[0].set_ylabel(NL.join(["AUROC: novel vs known family", "(among ransomware events)"]), fontsize=7)
    axes[1].bar([0, 1], [nv["known"]["median_p"], nv["novel"]["median_p"]], color=[BLUE, ORANGE], width=0.55)
    axes[1].set_xticks([0, 1], [NL.join(["known family", f"(n={nv['known']['n']:,})"]), NL.join(["novel family", f"(n={nv['novel']['n']:,})"])], fontsize=6.5)
    axes[1].set_ylabel(NL.join(["median raw score of", "ransomware events"]), fontsize=7); axes[1].set_yscale("log")
    save(fig, "fig_s2_novel")


def fig_s2_grid():
    g = pd.DataFrame(J("analysis_study2.json")["grid"])
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.95), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ee", BLUE])
    for ax, (p, ttl) in zip(axes, (("pi1", r"$\pi_1$ vs $\pi_2$"), ("pi3", r"$\pi_3$ vs $\pi_2$"), ("pi4", r"$\pi_4$ vs $\pi_2$"))):
        piv = g.assign(rel=100 * g[f"{p}_est"] / g["cost_pi2"]).pivot(index="c_r", columns="m", values="rel")
        v = max(np.nanmax(np.abs(piv.to_numpy())), 1e-9)
        ax.imshow(piv.to_numpy(), cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        for (i, j), val in np.ndenumerate(piv.to_numpy()):
            ax.text(j, i, f"{val:+.1f}" if abs(val) < 10 else f"{val:+.0f}", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(3), [f"{x:g}" for x in piv.columns]); ax.set_yticks(range(3), [f"{x:g}" for x in piv.index]); ax.grid(False)
        ax.set_title(ttl + " (% of cost)", fontsize=7); ax.set_xlabel("block margin $m$", fontsize=7)
    axes[0].set_ylabel("review cost $c_r$ (BTC)", fontsize=7)
    save(fig, "fig_s2_grid")


def fig_s2_diagnosis():
    x = J("study2_diagnosis.json")
    s = pd.DataFrame(x["by_amount"])
    labs = ["<0.5", "0.5-1", "1-2", "2-5", "5-20", "20-100", ">100"]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.35), constrained_layout=True)
    i = np.arange(len(s)); w = 0.38
    axes[0].bar(i - w / 2, 100 * s.illicit_rate, w * 0.92, color=BLUE, label="observed ransomware rate")
    axes[0].bar(i + w / 2, 100 * s.mean_calibrated_p, w * 0.92, color=ORANGE, label="mean recalibrated score")
    axes[0].set_yscale("log"); axes[0].set_ylabel("%"); axes[0].legend(fontsize=5.5, loc="lower left")
    axes[0].set_title("score vs. reality by amount", fontsize=7.5)
    axes[1].bar(i, 100 * s.review_rate_pi2, 0.62, color=INK)
    axes[1].set_title(NL.join(["share of events the", "Chow band reviews (%)"]), fontsize=7.5)
    for a in axes[:2]:
        a.set_xticks(i, labs, fontsize=5.6, rotation=35); a.set_xlabel("amount received (BTC)", fontsize=6.5)
    d = x["decomposition_per_1000"]
    names = [("pi1", NL.join(["single", "threshold"])), ("pi2", NL.join(["Chow", "band"])), ("pi4", "MAD")]
    bottom = np.zeros(3)
    for key, col, lab in (("missed_amount", ORANGE, "ransom approved"), ("review_cost", INK, "reviews"), ("blocked_legit", AQUA, "legitimate blocked")):
        v = np.array([d[k][key] for k, _ in names])
        axes[2].bar(range(3), v, 0.6, bottom=bottom, color=col, label=lab); bottom += v
    axes[2].set_xticks(range(3), [n for _, n in names], fontsize=6.3); axes[2].set_title(NL.join(["cost per 1,000 events", "(BTC)"]), fontsize=7.5)
    axes[2].legend(fontsize=5.5, loc="lower right")
    save(fig, "fig_s2_diagnosis")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["s1"]
    if "s1" in which:
        fig_pipeline(); fig_forest(); fig_capacity(); fig_recal(); fig_reliability(); fig_missed(); fig_mad(); fig_rho_seeds()
    if "s2" in which:
        fig_s2_curves(); fig_s2_forest(); fig_s2_novel(); fig_s2_grid(); fig_s2_diagnosis()

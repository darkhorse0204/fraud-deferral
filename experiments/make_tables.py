"""LaTeX tables for the paper, generated from results/analysis_*_test.json. No number is typed by hand."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils import ROOT

TEX = ROOT / "paper" / "tables"
PL = {"pi0": r"$\pi_0$ naive threshold", "pi1": r"$\pi_1$ single threshold", "pi2": r"$\pi_2$ Chow band (control)",
      "pi3": r"$\pi_3$ Chow + epistemic cap", "pi4": r"$\pi_4$ MAD", "pi4_nodelta": r"$\pi_4$ no cap (A1)",
      "pi4_noshrink": r"$\pi_4$ no shrinkage (A3)", "pi3_randu": r"$\pi_3$ random $u$ (A5)", "pi4_randu": r"$\pi_4$ random $u$ (A5)",
      "pi5": r"$\pi_5$ suppress-on-uncertainty", "pi6": r"$\pi_6$ conformal deferral", "pi7": r"$\pi_7$ oracle (bound)"}
SC = {"lr": "LR", "rf": "RF", "xgb": "XGB-ens", "mlp": "MLP-ens", "mcd": "MC-dropout", "sage": "SAGE-ens", "evolve": "EvolveGCN-O",
      "delayed": "Delayed-fb ens"}


def A(d: str):
    return json.loads((ROOT / "results" / f"analysis_{d}_test.json").read_text())


def w(name: str, body: str, caption: str, label: str, cols: str, header: str, star=False, size=r"\footnotesize"):
    env = "table*" if star else "table"
    s = (f"\\begin{{{env}}}[t]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
         f"{header}\\\\\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / f"{name}.tex").write_text(s, encoding="utf-8")


def ci(e, lo, hi, d=1):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main_table(a, name, caption, label):
    m = pd.DataFrame(a["main_table"])
    body = ""
    for blk in [b for b in ("T1", "T2") if b in set(m.block)]:
        body += f"\\multicolumn{{6}}{{l}}{{\\emph{{Block {blk}}}}}\\\\\n"
        for p in ("pi0", "pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi3_randu", "pi4_randu", "pi5", "pi6", "pi7"):
            r = m[(m.block == blk) & (m.policy == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            g = "--" if p == "pi2" else ci(r.gain_vs_pi2, r.lo, r.hi)
            nm = PL[p]
            body += f"{nm} & {r.cost_per_1000:.1f} & {g} & {r.reviews_per_1000:.0f} & {100 * r.missed_frac:.1f} & {r.fp_per_1000:.1f}\\\\\n"
        body += "\\midrule\n" if blk == "T1" and "T2" in set(m.block) else ""
    w(name, body, caption, label, "lrrrrr",
      r"Policy & Cost/1k & Gain vs.\ $\pi_2$ [95\% CI] & Reviews/1k & Missed (\%) & FP/1k", star=True)


def confirm_table(a, name, caption, label):
    c, h = a["confirmatory"], a["holm"]
    rows = [("H1a: $\\pi_4$ vs.\\ $\\pi_2$", c["H1a"], "H1a_T2"),
            ("H1b: $\\pi_4$ vs.\\ $\\pi_4$ no cap", c["H1b_vs_nodelta"], "H1b_T2_vs_nodelta"),
            ("H1b: $\\pi_4$ vs.\\ $\\pi_4$ random $u$", c["H1b_vs_randu"], "H1b_T2_vs_randu"),
            ("$\\pi_3$ vs.\\ $\\pi_2$ (secondary)", c["pi3_vs_pi2"], None),
            ("$\\pi_3$ vs.\\ $\\pi_3$ random $u$ (secondary)", c["pi3_vs_randu"], None)]
    body = ""
    for lab, r, k in rows:
        hp = f"{h[k]['p_adj']:.3f}" if k else "--"
        body += f"{lab} & {ci(r['est'], r['lo'], r['hi'])} & {100 * r['rel']:.1f} & {r['p']:.3f} & {hp} & {r['d_z']:.2f} & {r['frac_periods_better']:.2f}\\\\\n"
    h2 = c["H2"]
    body += (f"H2: Spearman($\\pi_3$ gain, CE$_u$), n={int(h2['n'])} & $\\rho$={h2['rho']:.2f} [{h2['lo']:.2f}, {h2['hi']:.2f}] & -- & {h2['p']:.3f} & "
             f"{h['H2_spearman']['p_adj']:.3f} & -- & --\\\\\n")
    t1 = c["T1_noninferiority_pi4"]
    body += f"T1 non-inferiority of $\\pi_4$ (TOST, 2\\% margin) & margin {t1['margin']:.1f} & -- & {t1['p']:.3f} & -- & -- & --\\\\\n"
    w(name, body, caption, label, "lrrrrrr",
      r"Test & Estimate [95\% CI] & Rel.\ gain (\%) & $p$ & Holm $p$ & $d_z$ & Frac.\ periods better", star=True)


def scorer_table(a, name, caption, label, block="T2"):
    d = pd.DataFrame(a["scorers"])
    d = d[d.block == block].sort_values("cost_pi2")
    body = ""
    for r in d.itertuples():
        body += (f"{SC.get(r.scorer, r.scorer)} ({r.fs.replace('F_', '')}) & {r.auprc:.3f} & {r.ece_raw:.3f} & {r.ece_cal:.3f} & {r.cost_pi1:.0f} & "
                 f"{r.cost_pi2:.0f} & {ci(r.gain_pi3, r.gain_pi3_lo, r.gain_pi3_hi)} & {r.gain_pi4:.1f}\\\\\n")
    w(name, body, caption, label, "lrrrrrrr",
      r"Scorer (features) & AUPRC & ECE raw & ECE recal. & $\pi_1$ cost & $\pi_2$ cost & $\pi_3$ gain [CI] & $\pi_4$ gain", star=True)


def sweep_table(a, name, caption, label):
    d = pd.DataFrame(a["sweeps"])
    if d.empty:
        return
    body = ""
    for r in d.sort_values(["scorer", "R", "D"]).itertuples():
        mode = "frozen" if str(r.R) == "inf" else "walk-fwd"
        body += f"{SC.get(r.scorer, r.scorer)} & {r.D} & {mode} & {r.cost_pi2:.0f} & {ci(r.gain_pi3, r.gain_pi3_lo, r.gain_pi3_hi)} & {ci(r.gain_pi4, r.gain_pi4_lo, r.gain_pi4_hi)}\\\\\n"
    w(name, body, caption, label, "lrlrrr", r"Scorer & $D$ & Retraining & $\pi_2$ cost & $\pi_3$ gain [CI] & $\pi_4$ gain [CI]", star=True)


def main() -> None:
    e = A("elliptic")
    main_table(e, "tab_main_elliptic", rf"Elliptic, primary scorer ({SC[e['primary_scorer']]}, all features), primary cost cell ($C_{{FN}}$=30, $C_{{FP}}$=5, $c_r$=1), $D$=2. "
               r"Cost per 1,000 events; gain is the paired reduction vs.\ the control $\pi_2$ with a moving-block-bootstrap 95\% CI over periods. T1 = steps 35--42, T2 = steps 43--49.",
               "tab:main")
    confirm_table(e, "tab_confirm", r"Pre-registered confirmatory tests (Elliptic, block T2, primary scorer and cost cell). Holm correction over the four confirmatory tests.", "tab:confirm")
    scorer_table(e, "tab_scorers", r"All scorers on Elliptic block T2: score quality, calibration, and the value of the decision layer (primary cost cell).", "tab:scorers")
    sweep_table(e, "tab_sweeps", r"Label delay and retraining (Elliptic block T2, primary cost cell): exploratory.", "tab:sweeps")
    try:
        b = A("baf_Base")
        main_table(b, "tab_main_baf", rf"BAF Base (synthetic), primary scorer ({SC[b['primary_scorer']]}), primary cost cell, $D$=1. T1 = months 4--5, T2 = months 6--7.", "tab:baf")
        sweep_table(b, "tab_sweeps_baf", r"BAF Base: label delay and retraining (T2).", "tab:sweeps_baf")
    except FileNotFoundError:
        pass
    try:
        b3 = A("baf_Variant III")
        main_table(b3, "tab_main_baf3", rf"BAF Variant III (synthetic), primary scorer ({SC[b3['primary_scorer']]}), primary cost cell, $D$=1.", "tab:baf3")
    except FileNotFoundError:
        pass
    print("tables written:", sorted(p.name for p in TEX.glob("*.tex")))


if __name__ == "__main__":
    main()

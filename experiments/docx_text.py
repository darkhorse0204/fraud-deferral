"""Manuscript text for build_docx.py, in the section order of the reference paper.
Markup: {{Name}} generated number (paper/numbers*.tex) | [[key,key]] citation | {@label} cross-reference |
<i> <b> <sub> <sup> inline.  Tables are functions of the analysis results; nothing numeric is typed by hand except
protocol constants (D, W_c, R, grid values) and dataset facts quoted from the data audit.
"""
from __future__ import annotations

import json

import pandas as pd
import yaml

from src.utils import ROOT

RUNNING_TITLE = "When does uncertainty pay? Uncertainty-based deferral for illicit-transaction detection"
TITLE = "When does uncertainty pay? A pre-registered, delay-aware evaluation of uncertainty-based deferral for illicit-transaction detection under temporal shift"
KEYWORDS = "Fraud detection, Selective prediction, Calibration, Concept drift, Delayed labels, Ransomware, Graph neural networks, Reproducibility"

P2, P3, P4, P1 = "π<sub>2</sub>", "π<sub>3</sub>", "π<sub>4</sub>", "π<sub>1</sub>"
CEU = "CE<sub>u</sub>"

ABSTRACT = (
    "Fraud and illicit-transaction systems route cases to approve, review or block while the data drift and labels arrive late, and "
    "uncertainty-aware deferral is widely proposed as a way to cut loss. We test whether it does, in two pre-registered studies. We contribute a "
    "leakage-audited walk-forward protocol with explicit label maturity and a calibration reserve, a decision-theoretic account of when an "
    "uncertainty signal can have value together with a bias-corrected diagnostic, and a random-uncertainty control that separates the effect of "
    "uncertainty from the effect of simply reviewing more. Epistemic uncertainty from bagged ensembles adds no measurable value for the strongest "
    "scorer in either study: on the Elliptic Bitcoin graph around its documented regime break the gain is {{EThreeTwoEst}} [{{EThreeTwoLo}}, "
    "{{EThreeTwoHi}}] per 1,000 events, and on 2.9 million ransomware address-days it is equivalent to the control within 2%. The ensemble is "
    "confidently wrong about novel illicit activity, and three feature-space novelty signals do not identify new ransomware families. A calibrated "
    "three-way review band lowers cost by {{EPiTwoVsOneAllRel}}% against a single threshold when costs are uniform, but this does not carry over to "
    "amount-based costs: there the band is {{SBandExtraRel}}% more expensive in the pre-registered primary setting (and cheaper in "
    "{{SCellsBandCheaper}} of 9 cost settings), because the score is calibrated on average and not conditional on the amount. Graph encoders do not "
    "beat feature-matched baselines. A decision rule is only as good as the calibration of its score conditional on everything the cost depends on."
)

REFS = {
    "weber": "Weber, M. et al. Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics. KDD’19 Workshop on Anomaly Detection in Finance (2019). arXiv:1908.02591.",
    "jesus": "Jesus, S. et al. Turning the tables: biased, imbalanced, dynamic tabular datasets for ML evaluation. Adv. Neural Inf. Process. Syst. (Datasets and Benchmarks Track) (2022).",
    "alarab": "Alarab, I. & Prakoonwit, S. Graph-based LSTM for anti-money laundering: experimenting temporal graph convolutional network with Bitcoin data. Neural Process. Lett. 55, 689–707 (2023).",
    "maganti": "Maganti, S. When graph structure becomes a liability: a critical re-evaluation of graph neural networks for Bitcoin fraud detection under temporal distribution shift. arXiv:2604.19514 (2026) (preprint).",
    "akcora": "Akcora, C. G., Li, Y., Gel, Y. R. & Kantarcioglu, M. BitcoinHeist: topological data analysis for ransomware prediction on the Bitcoin blockchain. Proc. Int. Joint Conf. Artif. Intell. 4439–4445 (2020).",
    "chow": "Chow, C. K. On optimum recognition error and reject tradeoff. IEEE Trans. Inf. Theory 16, 41–46 (1970).",
    "guo": "Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On calibration of modern neural networks. Proc. Int. Conf. Mach. Learn. (PMLR 70) 1321–1330 (2017).",
    "lakshmi": "Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv. Neural Inf. Process. Syst. (2017).",
    "ovadia": "Ovadia, Y. et al. Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. Adv. Neural Inf. Process. Syst. (2019).",
    "gibbs": "Gibbs, I. & Candès, E. Adaptive conformal inference under distribution shift. Adv. Neural Inf. Process. Syst. (2021).",
    "geifman": "Geifman, Y. & El-Yaniv, R. Selective classification for deep neural networks. Adv. Neural Inf. Process. Syst. (2017).",
    "pareja": "Pareja, A. et al. EvolveGCN: evolving graph convolutional networks for dynamic graphs. Proc. AAAI Conf. Artif. Intell. (2020).",
    "hamilton": "Hamilton, W. L., Ying, R. & Leskovec, J. Inductive representation learning on large graphs. Adv. Neural Inf. Process. Syst. (2017).",
    "chen": "Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (2016).",
    "breiman": "Breiman, L. Random forests. Mach. Learn. 45, 5–32 (2001).",
    "liu": "Liu, F. T., Ting, K. M. & Zhou, Z.-H. Isolation forest. Proc. IEEE Int. Conf. Data Min. 413–422 (2008).",
    "demsar": "Demšar, J. Statistical comparisons of classifiers over multiple data sets. J. Mach. Learn. Res. 7, 1–30 (2006).",
    "dalpozzolo": "Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. & Bontempi, G. Credit card fraud detection and concept-drift adaptation with delayed supervised information. Proc. Int. Joint Conf. Neural Netw. 1–8 (2015).",
    "kunsch": "Künsch, H. R. The jackknife and the bootstrap for general stationary observations. Ann. Stat. 17, 1217–1241 (1989).",
    "holm": "Holm, S. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 6, 65–70 (1979).",
    "schuirmann": "Schuirmann, D. J. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J. Pharmacokinet. Biopharm. 15, 657–680 (1987).",
    "zadrozny": "Zadrozny, B. & Elkan, C. Transforming classifier scores into accurate multiclass probability estimates. Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (2002).",
    "elkan": "Elkan, C. The foundations of cost-sensitive learning. Proc. Int. Joint Conf. Artif. Intell. 973–978 (2001).",
    "bahnsen": "Correa Bahnsen, A., Aouada, D. & Ottersten, B. Example-dependent cost-sensitive logistic regression for credit scoring. Proc. IEEE Int. Conf. Mach. Learn. Appl. 263–269 (2014).",
    "gama": "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. A survey on concept drift adaptation. ACM Comput. Surv. 46, 44 (2014).",
}


# ================================================================================================ helpers
def _f(x, d=1):
    return "n/a" if x is None or x != x else f"{x:,.{d}f}".replace("-", "−")


def _ci(e, lo, hi, d=1):
    return f"{_f(e, d)} [{_f(lo, d)}, {_f(hi, d)}]"


PL = {"pi0": "π<sub>0</sub> naive threshold", "pi1": "π<sub>1</sub> single threshold", "pi2": "π<sub>2</sub> Chow band (control)",
      "pi3": "π<sub>3</sub> Chow + epistemic cap", "pi4": "π<sub>4</sub> MAD", "pi4_nodelta": "π<sub>4</sub> without cap",
      "pi3_randu": "π<sub>3</sub> random <i>u</i>", "pi4_randu": "π<sub>4</sub> random <i>u</i>", "pi5": "π<sub>5</sub> suppress-on-uncertainty",
      "pi6": "π<sub>6</sub> conformal deferral", "pi7": "π<sub>7</sub> oracle (bound)", "pi3_if": "π<sub>3</sub> isolation-forest cap",
      "pi3_maha": "π<sub>3</sub> Mahalanobis cap", "pi3_knn": "π<sub>3</sub> kNN-distance cap", "pi3_if_rand": "π<sub>3</sub> isolation forest, permuted",
      "pi3_maha_rand": "π<sub>3</sub> Mahalanobis, permuted", "pi3_knn_rand": "π<sub>3</sub> kNN distance, permuted"}
SC_NAMES = {"lr": "LR", "rf": "RF", "xgb": "XGB-ens", "mlp": "MLP-ens", "mcd": "MC-dropout", "sage": "SAGE-ens", "evolve": "EvolveGCN-O", "delayed": "Delayed-fb ens"}


def context() -> dict:
    def A(d):
        p = ROOT / "results" / f"analysis_{d}_test.json"
        return json.loads(p.read_text()) if p.exists() else None
    s2 = ROOT / "results" / "analysis_study2.json"
    return dict(e=A("elliptic"), b1=A("baf_Base"), b3=A("baf_Variant III"), s2=json.loads(s2.read_text()) if s2.exists() else None)


# ================================================================================================ tables
def t_review(c):
    rows = [["1", "Weber et al.[[weber]]", "Elliptic Bitcoin graph", "Classify illicit transactions with graph convolutional networks, EvolveGCN, random forest and logistic regression", "Classifiers trained before the step-43 shutdown fail afterwards", "No decision cost, label delay or uncertainty"],
            ["2", "Alarab and Prakoonwit[[alarab]]", "Elliptic Bitcoin graph", "Temporal graph convolutional network with Monte-Carlo-dropout uncertainty and active learning", "–", "Uncertainty is not judged by decision cost against a calibrated control"],
            ["3", "Maganti[[maganti]]", "Elliptic Bitcoin graph", "Re-evaluate graph models under a strictly inductive protocol", "Random forest F1 0.821 against 0.689 for GraphSAGE", "No decision layer, uncertainty or label delay"],
            ["4", "Akcora et al.[[akcora]]", "BitcoinHeist", "Topological features for predicting ransomware addresses", "–", "No decision-cost evaluation with matured labels"],
            ["5", "Dal Pozzolo et al.[[dalpozzolo]]", "Card-fraud streams", "Concept-drift adaptation with delayed supervised information", "–", "No reject option under asymmetric costs; no uncertainty control"],
            ["6", "Jesus et al.[[jesus]]", "Bank Account Fraud suite", "Dynamic, biased, imbalanced tabular benchmark", "–", "A benchmark; three-way decision cost is not evaluated"],
            ["7", "Ovadia et al.[[ovadia]]", "Image, text and tabular benchmarks", "Compare uncertainty methods under dataset shift", "Deep ensembles are the most reliable", "Calibration metrics, not decision cost under delayed labels"],
            ["8", "Gibbs and Candès[[gibbs]]", "Online prediction under shift", "Adaptive conformal inference for long-run coverage", "–", "Targets coverage, not cost; no review action"],
            ["9", "Geifman and El-Yaniv[[geifman]]", "Image classification", "Selective classification with a risk guarantee", "–", "Risk–coverage on static data"]]
    return (["S no", "Paper", "Dataset or task", "Objective", "Result", "Gap relative to this work"], rows, [0.45, 1.3, 1.4, 2.4, 1.7, 2.2],
            "Comparative review of approaches to illicit-transaction detection, delayed labels and uncertainty-based deferral.", ["left"] * 6, False, True)


def t_loss(c):
    return (["Action", "Legitimate event (illicit = 0)", "Illicit event (illicit = 1)"],
            [["Approve", "0", "C<sub>FN</sub> (Study 2: the amount A<sub>i</sub>)"], ["Review", "c<sub>r</sub>", "c<sub>r</sub> + (1 − ρ) C<sub>FN</sub>"],
             ["Block", "C<sub>FP</sub> (Study 2: m A<sub>i</sub>)", "0"]],
            [1.4, 3, 3.2], "Loss of each action. The review cost c<sub>r</sub> is the unit of cost in Study 1; ρ is the analyst’s accuracy (ρ = 1 unless stated). In Study 2 the costs are per event and proportional to the amount A<sub>i</sub>.")


def t_hyp(c):
    return (["Hypothesis", "Pre-registered criterion", "Outcome"],
            [["H1a", "On Elliptic T2 (primary scorer, primary cost cell) MAD has lower cost than the Chow control with a 95% CI excluding 0, a positive gain in at least 4 of the 6 non-degenerate cost cells, and a positive gain on BAF.",
              "Not supported. Gain {{EHaEst}} [{{EHaLo}}, {{EHaHi}}]; positive in {{ECellsPiFour}} of 6 cells; BAF gain 0."],
             ["H1b", "MAD also beats its no-cap ablation and its random-<i>u</i> control (required for the claim that uncertainty pays).",
              "Not supported. No cap: {{EHbNDEst}} [{{EHbNDLo}}, {{EHbNDHi}}]; random <i>u</i>: {{EHbRUEst}} [{{EHbRULo}}, {{EHbRUHi}}]."],
             ["H2", "The per-period gain of π<sub>3</sub> over its random-<i>u</i> control is positively rank-correlated with CE<sub>u</sub> over the 15 test periods, and MAD is non-inferior to π<sub>2</sub> on T1.",
              "Not supported. ρ = {{EHTwoRho}} [{{EHTwoLo}}, {{EHTwoHi}}]; non-inferiority of MAD {{ETostFourEquiv}}."],
             ["H3", "A graph encoder changes cost relative to feature-matched baselines (two-sided).",
              "No advantage. SAGE vs. MLP on T2 {{EPairHThreesagevsmlpFlocalTTwoEst}} [{{EPairHThreesagevsmlpFlocalTTwoLo}}, {{EPairHThreesagevsmlpFlocalTTwoHi}}]; vs. XGBoost {{EPairHThreesagevsxgbFlocalTTwoEst}} [{{EPairHThreesagevsxgbFlocalTTwoLo}}, {{EPairHThreesagevsxgbFlocalTTwoHi}}]."],
             ["S2-H1", "On BitcoinHeist (39 test periods, primary scorer and cell) the Chow band has lower cost than the single threshold, 95% CI excluding 0. Predicted: supported.",
              "Verdict: {{SHOneVerdict}}; the direction is reversed. Gain {{SHOneEst}} [{{SHOneLo}}, {{SHOneHi}}] BTC per 1,000 events ({{SPiTwoVsOneRel}}%); the band is cheaper in {{SCellsBandCheaper}} of 9 cells."],
             ["S2-H2", "The epistemic cap is equivalent to the Chow band within ±2% of its cost (TOST). Predicted: equivalent.",
              "Verdict: {{SHTwoVerdict}}. Gain {{SHTwoEst}} [{{SHTwoLo}}, {{SHTwoHi}}]; equivalence {{SHTwoEquiv}} (p {{SHTwoTostPText}})."],
             ["S2-H3", "MAD against the Chow band, two-sided. Predicted from Study 1: gain ≤ 0.",
              "Gain {{SHThreeEst}} [{{SHThreeLo}}, {{SHThreeHi}}]: MAD is {{SHThreeWord}}, {{SHThreeVerdict}}."],
             ["S2-H4", "Three novelty-capped policies against the Chow band, two-sided, Holm-corrected. No directional prediction.",
              "No effect. Isolation forest {{SHFourIfEst}} [{{SHFourIfLo}}, {{SHFourIfHi}}]; Mahalanobis {{SHFourMahaEst}} [{{SHFourMahaLo}}, {{SHFourMahaHi}}]; kNN {{SHFourKnnEst}} [{{SHFourKnnLo}}, {{SHFourKnnHi}}]."]],
            [0.75, 4, 3.6], "Pre-registered hypotheses, criteria and outcomes for Study 1 (H1a–H3) and Study 2 (S2-H1 to S2-H4). Gains are cost reductions; positive means cheaper.", ["left", "left", "left"], True, True)


def main_rows(a, blocks=("T1", "T2")):
    m = pd.DataFrame(a["main_table"])
    rows = []
    for b in blocks:
        rows.append([f"<i>Block {b}</i>", "", "", "", "", ""])
        for p in ("pi0", "pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi3_randu", "pi4_randu", "pi5", "pi6", "pi7"):
            r = m[(m.block == b) & (m.policy == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([PL[p], _f(r.cost_per_1000), "–" if p == "pi2" else _ci(r.gain_vs_pi2, r.lo, r.hi), _f(r.reviews_per_1000, 0),
                         _f(100 * r.missed_frac), _f(r.fp_per_1000)])
    return rows


def t_main(c):
    return (["Policy", "Cost / 1k", "Gain vs. π<sub>2</sub> [95% CI]", "Reviews / 1k", "Missed (%)", "FP / 1k"], main_rows(c["e"]),
            [3.3, 1, 2.6, 1.1, 1.1, 1], "Study 1, Elliptic, primary scorer ({{EPrimary}}, all features), primary cost cell (C<sub>FN</sub> = 30, C<sub>FP</sub> = 5, c<sub>r</sub> = 1), D = 2. "
            "Cost per 1,000 events; gain is the paired reduction against the control π<sub>2</sub> with a moving-block-bootstrap 95% CI over periods. T1 = steps 35–42, T2 = steps 43–49.")


def t_confirm(c):
    e = c["e"]
    cf, h = e["confirmatory"], e["holm"]
    rows = []
    for lab, key, hk in (("H1a: π<sub>4</sub> vs. π<sub>2</sub>", "H1a", "H1a_T2"), ("H1b: π<sub>4</sub> vs. π<sub>4</sub> without cap", "H1b_vs_nodelta", "H1b_T2_vs_nodelta"),
                         ("H1b: π<sub>4</sub> vs. π<sub>4</sub> random <i>u</i>", "H1b_vs_randu", "H1b_T2_vs_randu"), ("π<sub>3</sub> vs. π<sub>2</sub> (secondary)", "pi3_vs_pi2", None),
                         ("π<sub>3</sub> vs. π<sub>3</sub> random <i>u</i> (secondary)", "pi3_vs_randu", None)):
        r = cf[key]
        rows.append([lab, _ci(r["est"], r["lo"], r["hi"]), _f(100 * r["rel"]), _f(r["p"], 3), _f(h[hk]["p_adj"], 3) if hk else "–", _f(r["d_z"], 2), _f(r["frac_periods_better"], 2)])
    h2 = cf["H2"]
    rows.append([f"H2: Spearman (π<sub>3</sub> gain, CE<sub>u</sub>), n = {int(h2['n'])}", f"ρ = {_f(h2['rho'], 2)} [{_f(h2['lo'], 2)}, {_f(h2['hi'], 2)}]", "–", _f(h2["p"], 3), _f(h["H2_spearman"]["p_adj"], 3), "–", "–"])
    t1 = cf["T1_noninferiority_pi4"]
    rows.append(["T1 non-inferiority of π<sub>4</sub> (TOST, 2% margin)", f"margin {_f(t1['margin'])}", "–", _f(t1["p"], 3), "–", "–", "–"])
    return (["Test", "Estimate [95% CI]", "Relative gain (%)", "p", "Holm p", "d<sub>z</sub>", "Periods better"], rows, [3.6, 2.4, 1.2, 0.8, 0.9, 0.8, 1.1],
            "Study 1 confirmatory tests (Elliptic, block T2, primary scorer and cost cell). Holm correction over the four confirmatory tests; positive gain = cheaper than the comparator.")


def t_scorers(c, block="T2"):
    d = pd.DataFrame(c["e"]["scorers"])
    d = d[d.block == block].sort_values("cost_pi2")
    rows = [[f"{SC_NAMES.get(r.scorer, r.scorer)} ({r.fs.replace('F_', '')})", _f(r.auprc, 3), _f(r.ece_raw, 3), _f(r.ece_cal, 3), _f(r.cost_pi1, 0), _f(r.cost_pi2, 0),
             _ci(r.gain_pi3, r.gain_pi3_lo, r.gain_pi3_hi), _f(r.gain_pi4)] for r in d.itertuples()]
    return (["Scorer (features)", "AUPRC", "ECE raw", "ECE recal.", "π<sub>1</sub> cost", "π<sub>2</sub> cost", "π<sub>3</sub> gain [95% CI]", "π<sub>4</sub> gain"], rows,
            [2.6, 0.9, 0.9, 1, 1, 1, 2.2, 1], f"Study 1: all scorers on Elliptic block {block}: score quality, calibration, and the value of the decision layer (primary cost cell; costs per 1,000 events).")


def t_scorers_t1(c):
    return t_scorers(c, "T1")


def t_sweeps(c):
    d = pd.DataFrame(c["e"]["sweeps"]).sort_values(["scorer", "R", "D"])
    rows = [[SC_NAMES.get(r.scorer, r.scorer), str(r.D), "frozen" if str(r.R) == "inf" else "walk-forward", _f(r.cost_pi2, 0), _ci(r.gain_pi3, r.gain_pi3_lo, r.gain_pi3_hi), _ci(r.gain_pi4, r.gain_pi4_lo, r.gain_pi4_hi)] for r in d.itertuples()]
    return (["Scorer", "D", "Retraining", "π<sub>2</sub> cost", "π<sub>3</sub> gain [95% CI]", "π<sub>4</sub> gain [95% CI]"], rows, [1.6, 0.5, 1.5, 1, 2.2, 2.2],
            "Study 1: label delay and retraining on Elliptic block T2 (primary cost cell, costs per 1,000 events).")


def t_baf(c):
    rows = []
    for name, a, blocks in (("BAF Base", c["b1"], ("T1", "T2")), ("BAF Variant III", c["b3"], ("T2",))):
        m = pd.DataFrame(a["main_table"])
        for b in blocks:
            rows.append([f"<i>{name}, block {b}</i>", "", "", "", ""])
            for p in ("pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi3_randu", "pi6", "pi7"):
                r = m[(m.block == b) & (m.policy == p)].iloc[0]
                rows.append([PL[p], _f(r.cost_per_1000), "–" if p == "pi2" else _f(r.gain_vs_pi2), _f(r.reviews_per_1000, 0), _f(100 * r.missed_frac)])
    return (["Policy", "Cost / 1k", "Gain vs. π<sub>2</sub>", "Reviews / 1k", "Missed (%)"], rows, [3.4, 1, 1.4, 1.2, 1.1],
            "Study 1 replication on BAF (synthetic benchmark), primary scorer {{BPrimary}}, primary cost cell, D = 1. T1 = months 4–5, T2 = months 6–7. Gains are point estimates (four test months do not support period-level intervals).")


def t_s2_main(c):
    a = c["s2"]
    rows = []
    for b, lab in (("T1", "T1 (2015)"), ("T2", "T2 (2016–2017)"), ("ALL", "Whole test era")):
        d = a["main"][b]
        rows.append([f"<i>{lab}</i>", "", "", "", ""])
        rows.append([PL["pi2"], _f(d["pi1"]["cost_ctrl"]), "–", "–", _f(d["pi1"]["review_ctrl"], 0)])
        for p in ("pi1", "pi3", "pi3_randu", "pi3_if", "pi3_maha", "pi3_knn", "pi4", "pi4_nodelta", "pi7"):
            r = d[p]
            rows.append([PL[p], _f(r["cost_new"]), _ci(r["est"], r["lo"], r["hi"]), _f(100 * r["rel"]), _f(r["review_new"], 0)])
    return (["Policy", "Cost / 1k (BTC)", "Gain vs. π<sub>2</sub> [95% CI]", "Relative gain (%)", "Reviews / 1k"], rows, [3.2, 1.3, 2.6, 1.2, 1.1],
            "Study 2, BitcoinHeist, primary scorer ({{SPrimary}}), primary cost cell (c<sub>r</sub> = 0.25 BTC, m = 0.25), D = 2. Cost in BTC per 1,000 address-days; "
            "gain is the paired reduction against π<sub>2</sub> with a moving-block-bootstrap 95% CI over periods.")


def t_s2_scorers(c):
    d = pd.DataFrame(c["s2"]["scorers"])
    d = d[d.block == "ALL"].sort_values("cost_pi2")
    rows = [[SC_NAMES[r.scorer], _f(r.auprc_all, 3), _f(r.cost_pi1), _f(r.cost_pi2), _f(100 * (r.cost_pi1 - r.cost_pi2) / r.cost_pi1, 0),
             _ci(r.gain_pi3, r.gain_pi3_lo, r.gain_pi3_hi), _ci(r.gain_pi4, r.gain_pi4_lo, r.gain_pi4_hi)] for r in d.itertuples()]
    return (["Scorer", "AUPRC", "π<sub>1</sub> cost", "π<sub>2</sub> cost", "π<sub>2</sub> saving (%)", "π<sub>3</sub> gain [95% CI]", "π<sub>4</sub> gain [95% CI]"], rows,
            [1.3, 0.9, 1.1, 1.1, 1.1, 2.3, 2.3], "Study 2: all four scorers over the whole test era (primary cost cell; BTC per 1,000 address-days).")


def t_audit(c):
    a = json.loads((ROOT / "results" / "data_audit.json").read_text())
    e, b = a["elliptic"], a["baf"]["Base"]
    rows = [["Elliptic nodes / edges", f"{e['n_nodes']:,} / {e['n_edges']:,}"],
            ["Elliptic labelled illicit / licit / unknown", f"{e['label_counts']['illicit']:,} / {e['label_counts']['licit']:,} / {e['label_counts']['unknown']:,}"],
            ["Elliptic edges crossing time steps", str(e["cross_period_edges"])],
            ["Elliptic constant features / NaN or infinite values", f"{len(e['constant_features'])} / {e['feature_nan'] + e['feature_inf']}"],
            ["Elliptic highest single-feature AUROC on dev (leakage threshold 0.95)", f"{max(e['single_feature_auroc_dev_top10'].values()):.3f}"],
            ["Elliptic illicit labels in T1 / T2", f"{e['T1_illicit_total']} / {e['T2_illicit_total']}"],
            ["Elliptic T2 illicit labels by step (43–49)", ", ".join(str(v) for v in e["T2_illicit_by_period"].values())],
            ["Elliptic labelled-vs-unlabelled domain AUC (dev)", f"{e['labeled_vs_unlabeled_domain_auc_dev']:.3f}"],
            ["BAF rows per file / features / fraud rate (Base)", f"{b['rows']:,} / {b['n_features']} / {100 * b['fraud_rate']:.2f}%"],
            ["BAF constant feature dropped", ", ".join(b["constant_features"])],
            ["BAF Base fraud rate, month 3 → month 7", f"{100 * b['per_month']['3']['rate']:.2f}% → {100 * b['per_month']['7']['rate']:.2f}%"],
            ["BitcoinHeist rows / ransomware rows / distinct addresses", "2,916,697 / 41,413 / 2,631,095"],
            ["BitcoinHeist ransomware rows by year, 2011–2018", "65, 714, 7,494, 10,319, 3,701, 15,631, 3,486, 3"],
            ["BitcoinHeist dev AUPRC: 11 published features / plus matured blacklist (base rate 0.029)", "0.038 / 0.351"]]
    return (["Item", "Value"], rows, [4.6, 3.4], "Data audit. Statistics that involve labels were computed on the dev era only.", ["left", "left"])


def t_tuning(c):
    rows = []
    for ds in ("elliptic", "baf"):
        for p in sorted((ROOT / "configs" / "models").glob(f"{ds}__*.yaml")):
            y = yaml.safe_load(p.read_text())
            log = ROOT / "results" / "tuning" / f"{p.stem}.json"
            n = len(json.loads(log.read_text())) if log.exists() else "n/a"
            rows.append(["Elliptic" if ds == "elliptic" else "BAF", SC_NAMES.get(y["scorer"], y["scorer"]), y["fs"].replace("F_", ""), str(n), _f(y["dev_auprc"], 3)])
    for s in ("lr", "rf", "xgb", "mlp"):
        y = ROOT / "configs" / "study2" / "models" / f"{s}.yaml"
        if y.exists():
            rows.append(["BitcoinHeist", SC_NAMES[s], "all", str(len(json.loads((ROOT / "results" / "tuning" / f"study2__{s}.json").read_text()))), _f(yaml.safe_load(y.read_text())["dev_auprc"], 3)])
    return (["Dataset", "Scorer", "Features", "Trials", "Dev AUPRC of selected setting"], rows, [1.7, 2, 1.2, 0.9, 2.4],
            "Dev-era random-search budgets and selected settings (walk-forward AUPRC on dev origins; trial 0 is the default). Study 2 uses an equal budget of eight trials for every family.")


def t_leak(c):
    rows = [["Future information in features", "Provider aggregates cannot be audited; local-only and all-feature sets are both reported."],
            ["Transductive graph training", "Per-period inductive graphs; the training graph contains only matured nodes (unit-tested)."],
            ["Preprocessing fitted on all data", "Scalers and imputers are fitted inside each training pool (unit-tested)."],
            ["Random splits of a time series", "Only the frozen, hash-checked splits are used; a static guard bans random-split APIs."],
            ["Immature labels in training or calibration", "A single access function enforces s + D < t; flipping every immature label changes no decision (unit-tested, with an oracle control)."],
            ["Calibration on events the scorer trained on", "Calibration reserve W<sub>c</sub>: the scorer trains only on periods ≤ τ − D − 1 − W<sub>c</sub>."],
            ["Label-derived features", "The matured-blacklist flag of Study 2 uses first-positive period + D < period (unit-tested with flipped immature labels)."],
            ["Sampling artefacts used as signal", "Same-day row counts and address-repeat counts are excluded in Study 2 (legitimate addresses are capped at 1,000 per day)."],
            ["Hyper-parameters or thresholds tuned on test", "All tuning on the dev era; configurations hashed before any test-era score exists; the test runner refuses to start without the lock."],
            ["Clock used as a feature", "Elliptic time step, BAF month and BitcoinHeist year and day are excluded from every feature set (unit-tested)."],
            ["Unknown labels treated as negatives", "Unlabelled Elliptic nodes are excluded from training and evaluation."],
            ["Cost grid chosen after results", "Grids, primary cells and inference rules are fixed in the two pre-registrations."],
            ["Seed selection", "Seeds are fixed in advance (0–9 Elliptic, 0–4 BAF and BitcoinHeist) and all are reported."],
            ["Forking paths", "Holm correction over declared confirmatory families; every other analysis is labelled exploratory."]]
    return (["Leakage source", "Prevention"], rows, [3, 6], "Leakage audit.", ["left", "left"], False, True)


def t_grid(c):
    g = pd.DataFrame(c["e"]["grid"])
    g = g[(g.block == "T2") & (~g.degenerate)]
    rows = []
    for (fn, fp), d in g.groupby(["C_FN", "C_FP"]):
        r3, r4 = d[d.policy == "pi3"].iloc[0], d[d.policy == "pi4"].iloc[0]
        rows.append([f"{int(fn)}", f"{int(fp)}", _f(r3.cost_pi2, 0), _ci(r3.gain, r3.lo, r3.hi), _ci(r4.gain, r4.lo, r4.hi)])
    return (["C<sub>FN</sub>", "C<sub>FP</sub>", "π<sub>2</sub> cost", "π<sub>3</sub> gain [95% CI]", "π<sub>4</sub> gain [95% CI]"], rows, [0.8, 0.8, 1.2, 2.4, 2.4],
            "Study 1, Elliptic block T2: gains over π<sub>2</sub> in each non-degenerate cell of the cost grid (cells with C<sub>FP</sub> = 1 have an empty review band).")


def t_s2_grid(c):
    g = pd.DataFrame(c["s2"]["grid"])
    rows = [[f"{r.c_r:g}", f"{r.m:g}", _f(r.cost_pi2), _ci(r.pi1_est, r.pi1_lo, r.pi1_hi), _ci(r.pi3_est, r.pi3_lo, r.pi3_hi), _ci(r.pi4_est, r.pi4_lo, r.pi4_hi), _ci(r.pi3_if_est, r.pi3_if_lo, r.pi3_if_hi)]
            for r in g.sort_values(["c_r", "m"]).itertuples()]
    return (["c<sub>r</sub>", "m", "π<sub>2</sub> cost", "π<sub>1</sub> gain", "π<sub>3</sub> gain", "π<sub>4</sub> gain", "Isolation-forest cap gain"], rows, [0.6, 0.6, 1.1, 2, 2, 2, 2.1],
            "Study 2: gains over π<sub>2</sub> [95% CI] in every cell of the cost grid, whole test era (BTC per 1,000 address-days; a negative π<sub>1</sub> gain means the single threshold is more expensive).")


ALG = [(0, "BEGIN"),
       (0, "Step 1: Retrain at origin τ (every R periods)"),
       (1, "pool ← events with period ≤ τ − D − 1 − W_c        (labels matured, calibration reserve held out)"),
       (1, "FIT the scorer ensemble on pool; scalers and imputers are fitted on pool only"),
       (0, "Step 2: Score"),
       (1, "FOR each event of periods τ − D − W_c … τ + R − 1: p̃ ← mean of member probabilities; u ← their variance"),
       (0, "Step 3: Decide in period t (τ ≤ t < τ + R)"),
       (1, "window ← scored events with t − D − W ≤ period ≤ t − D − 1        (only labels with s + D < t are read)"),
       (1, "g ← isotonic recalibration fitted on window; p ← g(p̃)"),
       (1, "a ← c_r / (ρ C_FN); b ← (C_FP − c_r) / (C_FP + (1 − ρ) C_FN)        (Chow band; per event in Study 2)"),
       (1, "IF policy is MAD THEN choose band multipliers and cap δ minimising the weighted window cost with shrinkage"),
       (1, "ELSE IF policy uses a cap THEN δ ← the window quantile of u with the lowest window cost (possibly none)"),
       (1, "action ← REVIEW if u > δ; APPROVE if p < a; BLOCK if p ≥ b; otherwise REVIEW"),
       (0, "Step 4: Evaluate (labels of period t are read here only)"),
       (1, "cost ← loss of the action given the true label; aggregate per period, average over seeds, bootstrap over periods"),
       (0, "END")]


# ================================================================================================ body
BODY = [
    ("p", "Fraud and illicit-activity detectors are rarely used as hard classifiers. In practice a score is turned into one of three actions: approve, send to a human review queue, or block, and each action has a different cost. Two facts make this decision hard to evaluate honestly. The data are non-stationary, so a model trained on the past degrades when new schemes appear, and labels are delayed: whether a payment was illicit is known only after a dispute or an investigation, so at decision time the freshest labels are weeks old."),
    ("p", "A large literature responds with richer models: graph neural networks over the transaction graph,[[weber,pareja,hamilton]] and uncertainty estimates (Bayesian approximations, ensembles, conformal sets) so that a system can defer when it does not know.[[lakshmi,ovadia,geifman,gibbs]] The implicit promise is that deferring on uncertain cases lowers loss. That promise is rarely tested the way a deployment would test it: with time-ordered splits, using only labels that have matured, against an uncertainty-free policy that is just as well calibrated, and in units of cost rather than F1."),
    ("p", "We ask three questions. <b>RQ1:</b> does routing cases to review on the basis of epistemic uncertainty reduce expected decision cost, relative to the best policy that uses the same scores but no uncertainty? <b>RQ2:</b> when can an uncertainty signal help at all, and can that condition be measured? <b>RQ3:</b> do the answers carry over to a second, larger stream with amount-based costs and to signals built to detect novelty? We contribute:"),
    ("b", "<b>A delay-aware evaluation protocol</b>, leakage-audited and pre-registered, with explicit label maturity and a calibration reserve that keeps the recalibration window out of sample for the scorer."),
    ("b", "<b>A condition and a control.</b> A decision-theoretic account of when an uncertainty signal can have any value (Proposition 1), a bias-corrected diagnostic for it (" + CEU + "), and a random-uncertainty control. A synthetic experiment shows why the control is needed: when a stale model under-predicts risk, reviewing extra events lowers cost whether or not the uncertainty signal is informative."),
    ("b", "<b>Two pre-registered studies.</b> Study 1 evaluates the Elliptic Bitcoin graph around its documented regime break, with a replication on a synthetic bank-fraud benchmark. Study 2, registered after Study 1, tests the main findings on 2.9 million ransomware address-days with amount-based costs and three novelty signals."),
    ("b", "<b>Findings, including negative ones and a failed prediction.</b> Epistemic uncertainty adds nothing for strong tree ensembles in either study, novelty signals do not identify new ransomware families, and graph encoders do not beat feature-matched baselines. A calibrated review band cuts cost sharply under uniform costs (Study 1), but our prediction that it would do so under amount-based costs was wrong: in the primary setting of Study 2 it is more expensive than a single threshold."),
    ("b", "<b>An explanation.</b> The ensemble is confidently wrong about the novel cases; a signal that ranks missed cases well among illicit events is not precise enough among all events to cover the cost of reviewing them; and a score that is calibrated on average but not conditional on the amount makes a cost-optimal band review large payments that are almost never illicit."),
    ("p", "We do not propose a new architecture. The scorers are standard; the contribution is the evaluation protocol and the evidence it produces. The rest of the paper is structured as follows. Section “Related work” reviews the literature and compares it with the present study. Section “Threats to validity” states the limits of the evidence. Section “Research gap” lists what is missing. Sections “Methodology” and “Proposed system” describe the protocol, the condition and the policies, Section “Implementation” the data, pre-registration and statistics, and Section “Results and analysis” the two studies."),

    ("h1", "Related work"),
    ("p", "<i>Illicit-transaction detection on graphs.</i> Weber et al.[[weber]] released the Elliptic Bitcoin graph and compared logistic regression, random forests, graph convolutional networks and EvolveGCN,[[pareja]] noting that classifiers trained before a dark-market shutdown at time step 43 fail afterwards. Alarab and Prakoonwit[[alarab]] combined a temporal graph convolutional network with Monte-Carlo-dropout uncertainty on the same data, so combining temporal graphs with uncertainty is not new. A recent preprint[[maganti]] argues that reported graph-model advantages on Elliptic rely on test-period adjacency leaking into training, and that random forests win under a strictly inductive protocol. Akcora et al.[[akcora]] derive topological features of Bitcoin addresses to predict ransomware payments. We adopt an inductive protocol and test the graph question directly (H3)."),
    ("p", "<i>Concept drift and delayed labels in fraud.</i> Dal Pozzolo et al.[[dalpozzolo]] showed that delayed supervised information is central in card-fraud detection, and Gama et al.[[gama]] survey drift adaptation. Jesus et al.[[jesus]] provide a dynamic tabular fraud benchmark. We use delayed supervision as a design constraint, with labels maturing after D periods, and include a delayed-feedback ensemble in the spirit of Dal Pozzolo et al. as a scorer."),
    ("p", "<i>Reject option, selective prediction and calibration.</i> The optimal reject rule under asymmetric costs is due to Chow;[[chow]] selective classification[[geifman]] and cost-sensitive learning[[elkan,bahnsen]] are standard. Deep ensembles[[lakshmi]] were found more reliable than other uncertainty methods under dataset shift;[[ovadia]] temperature and Platt scaling[[guo]] and isotonic regression[[zadrozny]] recalibrate scores; adaptive conformal inference[[gibbs]] targets coverage under shift, and isolation forests[[liu]] score novelty without labels. These works evaluate uncertainty by calibration, coverage or risk–coverage curves. We evaluate it by the decision cost it saves over a calibrated Chow policy, under time-ordered, delay-aware conditions."),
    ("p", "{@tab:review} compares the approaches reviewed above with the present study."),
    ("tab", t_review, "tab:review"),

    ("h1", "Threats to validity"),
    ("p", "<i>One documented regime break in Study 1.</i> The Elliptic shutdown is the only documented break in Study 1, and block T2 has {{AuditTTwoIllicit}} illicit labels over seven periods, so its intervals are wide and small effects cannot be excluded. Study 2 was added for this reason: it has 39 test periods and {{STestPos}} ransomware events."),
    ("p", "<i>Labels and sampling.</i> The labelled Elliptic nodes are not a random sample (a classifier separates them from unlabelled nodes with AUC {{AuditDomainAuc}}), so Study 1 concerns the labelled 23% of nodes. In BitcoinHeist legitimate addresses are subsampled to at most 1,000 per day and are not verified negatives, so prevalence there is not a population rate, and features that count repeats were excluded as sampling artefacts."),
    ("p", "<i>Costs and delay are assumptions.</i> Elliptic and BAF carry no monetary amounts: Study 1 reports normalised cost units over a grid. Study 2 uses the amount received as the loss of a missed payment and a share of it as the loss of a wrongly blocked one; both are modelling choices, reported over a grid. Label delay, review capacity and analyst accuracy are simulated in both studies."),
    ("p", "<i>Synthetic replication.</i> BAF is synthetic and has four test months; it is used as a controlled replication, not as evidence about a real bank."),
    ("p", "<i>Models.</i> The graph baselines are GraphSAGE and our re-implementation of EvolveGCN-O; no recent published graph-fraud method was verified and included. Study 1 tuning budgets were 6–12 trials per family on Elliptic and 4–6 on BAF; Study 2 uses an equal budget of eight. Uncertainty is ensemble variance, plus three unsupervised novelty signals in Study 2; other estimators were not tested."),
    ("p", "<i>Scorer quality in Study 2.</i> The BitcoinHeist scorer is weak in the test era (AUPRC {{SScXgbAllAuprc}} against {{SDevXgb}} on the dev era), because {{SUnseenShare}}% of test events come from addresses never seen in training. Study 2 therefore tests the decision layer on top of a weak scorer under entity shift; a stronger scorer could change the cost comparison between the band and the single threshold."),
    ("p", "<i>Order of analyses.</i> The Elliptic T2 boundary was taken from the literature, and one default random-forest run touched Elliptic test labels before the first pre-registration was frozen (it is logged). Study 2 was designed after the Study 1 results were known: its test data were untouched, but its predictions are informed by Study 1. The cross-scorer, error-analysis, graph, sensitivity and stress-test analyses of Study 1, and the cost decomposition and amount-stratified recalibration of Study 2, were added after the respective freezes and are labelled exploratory."),

    ("h1", "Research gap"),
    ("p", "Graph and uncertainty methods for illicit-transaction detection are usually judged by classification metrics. To our knowledge, the following gaps remain open."),
    ("b", "Decision-cost evaluation against an equally calibrated, uncertainty-free policy. Calibration, coverage and risk–coverage curves[[ovadia,geifman,gibbs]] show that an uncertainty signal is well formed, not that acting on it saves money."),
    ("b", "Evaluation that respects label delay.[[dalpozzolo]] Recalibration and threshold selection must use only labels that have matured, and must be out of sample for the scorer."),
    ("b", "A control for hedging. Any extra review can lower cost when a stale model under-predicts risk, so a benefit must beat a policy that reviews the same share of events at random."),
    ("b", "A measurable condition for when uncertainty can help, so that a null or positive result can be explained and not only reported."),
    ("b", "Evidence at scale with monetary amounts and with emerging schemes, where a new family of illicit activity is the kind of novelty that uncertainty is supposed to detect."),

    ("h1", "Methodology"),
    ("p", "The research methodology is organised around one question, namely whether acting on uncertainty lowers decision cost once the comparison policy is equally well calibrated and only matured labels are used, and around the three research questions that make it testable. The stages are designed so that each question is answered with evidence that does not depend on the component under test."),
    ("p", "In the first stage the data are audited and the leakage guards are written as tests. In the second stage every scorer is trained under a walk-forward protocol and its scores are cached, so that all decision policies consume identical scores and comparisons are exactly paired ({@fig:pipeline}). In the third stage the decision policies are applied to the cached scores and costs are computed over a pre-registered grid. In the fourth stage the pre-registered hypotheses of Study 1 are tested on Elliptic and replicated on BAF. The fifth stage checks the mechanism on synthetic streams and analyses the errors. The sixth stage is Study 2: the main findings are tested again on BitcoinHeist with amount-based costs and with signals built to detect novelty. Splits, cost grids, hyper-parameters, decision-layer settings and hypotheses were frozen and hashed before the test era of each study was scored (Section “Pre-registration and tuning”)."),
    ("figwide", "fig_pipeline.png", "Evaluation pipeline. Scores are produced once per scorer and cached; every decision policy consumes the same scores, and only matured labels reach the scorer, the recalibrator and the policy.", "fig:pipeline"),

    ("h1", "Proposed system"),
    ("p", "The system under study turns a stream of events into approve, review or block decisions. Its parts are a scorer ensemble, a rolling recalibrator and a decision policy, all of which may read only labels that have matured."),
    ("h2", "Problem formulation"),
    ("p", "Events i arrive in periods s<sub>i</sub> ∈ {1, …, T} with features x<sub>i</sub> and a latent label y<sub>i</sub> ∈ {0, 1} (illicit). On graph data G<sub>t</sub> = (V<sub>t</sub>, E<sub>t</sub>, X<sub>t</sub>) collects the events of period t; in the Elliptic graph no edge crosses periods, so the G<sub>t</sub> are disjoint snapshots. A label is observed at the end of period s<sub>i</sub> + D, so a decision in period t may use event i only if s<sub>i</sub> + D < t. The inequality is strict: with D = 0 labels arrive in the next period, never in the same one. D is a simulated assumption; none of the datasets records label latency."),
    ("p", "For each event the system chooses approve, review or block with the loss in {@tab:loss}. Costs are unknown for Elliptic and BAF, so Study 1 reports every result over a pre-registered 3 × 3 grid, C<sub>FN</sub> ∈ {10, 30, 100} and C<sub>FP</sub> ∈ {1, 5, 20}, with C<sub>FN</sub> = 30 and C<sub>FP</sub> = 5 as the primary cell. In Study 2 the loss of approving an illicit event is the amount A<sub>i</sub> it received, the loss of blocking a legitimate event is m A<sub>i</sub>, and the grid is c<sub>r</sub> ∈ {0.05, 0.25, 1} BTC and m ∈ {0.05, 0.25, 1}, with c<sub>r</sub> = 0.25 and m = 0.25 as the primary cell."),
    ("tab", t_loss, "tab:loss"),
    ("p", "If a calibrated probability p = P(y = 1 | x) were available, review beats approve when p > a = c<sub>r</sub> / (ρ C<sub>FN</sub>), and review beats block when p < b = (C<sub>FP</sub> − c<sub>r</sub>) / (C<sub>FP</sub> + (1 − ρ) C<sub>FN</sub>). The review band [a, b) is non-empty only when a < b. In Study 1 all three C<sub>FP</sub> = 1 cells are degenerate (c<sub>r</sub> / C<sub>FP</sub> = 1) and are excluded from inference, which leaves six cells. In Study 2 the band is per event, a<sub>i</sub> = c<sub>r</sub> / A<sub>i</sub> and b<sub>i</sub> = 1 − c<sub>r</sub> / (m A<sub>i</sub>), so small payments are never worth a review and large ones are reviewed over a wide range of scores."),
    ("h2", "What an uncertainty signal can decide"),
    ("p", "<b>Proposition 1 (value of an uncertainty signal).</b> Let R*(S) be the minimum expected loss over policies that are functions of S. Then R*(p) − R*(p, u) ≥ 0, with equality if P(y | p, u) = P(y | p). In particular, if p = P(y | x) exactly and u is a function of x, then u has no value."),
    ("p", "This is standard value-of-information reasoning, not a new theorem. Its use is diagnostic: u can help only through conditional miscalibration of p, that is, when P(y | p, u) varies with u at fixed p. We measure this with " + CEU + ", the calibration error over cells (global equal-mass p-bins crossed with u-terciles) minus the calibration error over the p-bins alone, minus its mean under random permutations of u to remove finite-sample bias. It is close to zero if and only if u carries no label information beyond p. Even when the population gain is positive, estimating the policy from a small matured sample adds variance that can cancel it."),
    ("h2", "Walk-forward protocol with a calibration reserve"),
    ("p", "A model is trained at origins τ every R periods and used for decisions in periods τ to τ + R − 1 ({@fig:protocol}). It trains only on periods up to τ − D − 1 − W<sub>c</sub>. The W<sub>c</sub> most recent matured periods are held back as a calibration reserve: they are out of sample for the scorer but already labelled at τ. Recalibrating on events the scorer was trained on would be in-sample and over-confident, so every calibrated policy in this paper uses the reserve. We use D = 2, W<sub>c</sub> = 4 and R = 5 on Elliptic, D = 1, W<sub>c</sub> = 1 and R = 2 on BAF, and D = 2, W<sub>c</sub> = 3 and R = 6 on BitcoinHeist. All preprocessing is fitted on the training rows only. A single function selects every label any component may read, and unit tests verify that flipping all immature labels changes no policy decision, while an oracle control shows that the test can fail."),
    ("figwide", "fig_protocol.png", "Walk-forward protocol for one retraining origin (not to scale). The calibration reserve is labelled but out of sample for the scorer; the D periods before the origin are not yet labelled.", "fig:protocol"),
    ("h2", "Decision policies"),
    ("p", "All policies consume the same cached scores. The control is the uncertainty-free policy that a practitioner would deploy first, and each uncertainty-aware policy differs from it in one stated way."),
    ("b", "<b>" + P1 + ", single threshold.</b> One cost-optimal threshold on the recalibrated probability; no review. π<sub>0</sub> is its naive variant, a threshold on the raw score chosen to maximise dev F1."),
    ("b", "<b>" + P2 + ", Chow band (the control).</b> Approve below a, block at or above b, review in between, on the recalibrated probability. The recalibrator (kind and window) is shared by every calibrated policy and was chosen on the dev era by " + P2 + "’s own cost, so the control is tuned first."),
    ("b", "<b>" + P3 + ", epistemic cap.</b> " + P2 + " plus review for events whose u exceeds a cap δ, where δ is the window quantile of u with the lowest matured-window cost; “no cap” is one of the candidates."),
    ("b", "<b>" + P4 + ", MAD (matured-label adaptive deferral).</b> Re-estimates, each period and from matured labels only, the band multipliers and the cap δ by minimising the weighted window cost, with shrinkage toward the Chow thresholds and a bounded step size."),
    ("b", "<b>Controls.</b> " + P4 + " without the cap; " + P3 + " and " + P4 + " with u permuted within each score table (the random-<i>u</i> control: same marginal u and the same review mechanism, no information); π<sub>5</sub> (alert only if u is low, a negative control); π<sub>6</sub> (conformal deferral); and π<sub>7</sub> (an oracle that picks its thresholds from the evaluation period’s own labels, reported as a bound and never as a method)."),
    ("b", "<b>Novelty caps (Study 2).</b> " + P3 + " with u replaced by an unsupervised signal fitted on the features of the scorer’s training periods: an isolation forest,[[liu]] the Mahalanobis distance, and the mean distance to the ten nearest of 50,000 training events."),
    ("h2", "Strengths of the proposed system"),
    ("p", "Four properties distinguish the evaluation from the usual one."),
    ("b", "<b>Paired by construction.</b> Policies share scores, recalibration windows and seeds, so a difference between two policies is the effect of the stated change and nothing else."),
    ("b", "<b>Delay-aware end to end.</b> Training, recalibration and threshold selection read only matured labels, through one access function, and the calibration window is out of sample for the scorer."),
    ("b", "<b>Attribution is explicit.</b> A gain is credited to uncertainty only if it survives the no-cap ablation and the random-<i>u</i> control; threshold adaptation and hedging are measured separately."),
    ("b", "<b>Self-protecting policies.</b> A cap or a threshold change is adopted only when it lowers cost on the matured window, so the uncertainty-aware policies reduce to the control when the signal is uninformative."),
    ("p", "Its limit is the one stated in Section “What an uncertainty signal can decide”: a policy fitted on a matured window cannot use information that the window does not yet contain."),
    ("h2", "Pseudo code for the proposed system"),
    ("p", "{@alg:main} summarises the scoring, recalibration and decision loop in pseudo code."),
    ("alg", ALG, "Delay-aware scoring, recalibration and three-way decision with an optional uncertainty cap.", "alg:main"),
    ("h2", "Scorers and uncertainty"),
    ("p", "The scorers are logistic regression, a random forest,[[breiman]] a seed-bagged XGBoost ensemble[[chen]] (M = 5), a deep-ensemble MLP[[lakshmi]] (M = 5), an MC-dropout MLP, a GraphSAGE ensemble[[hamilton]] and EvolveGCN-O[[pareja]] on inductive per-period graphs, and a delayed-feedback ensemble. The epistemic signal u is the variance of the member probabilities; it is undefined for logistic regression (u = 0). The primary scorer of each study is fixed by a pre-registered rule: the ensemble-capable scorer with the highest mean dev-era walk-forward AUPRC."),

    ("h1", "Implementation"),
    ("h2", "Elliptic Bitcoin graph (Study 1)"),
    ("p", "The Elliptic graph[[weber]] has 203,769 transactions, 49 two-week steps, 4,545 illicit and 42,019 licit labels; the {{AuditUnknown}} unlabelled nodes are excluded from training and evaluation. The first of the 166 provided features is the time step and is removed: the local feature set has 93 features and the all-feature set adds the 72 provider-aggregated one-hop features (165). We verified that no edge crosses a step. The dev era is steps 1–34 and the test era steps 35–49, split into T1 (steps 35–42, {{AuditTOneIllicit}} illicit) and T2 (steps 43–49, {{AuditTTwoIllicit}} illicit). T2 starts at the dark-market shutdown reported by Weber et al.; it was fixed from the literature before any experiment. A default random forest trained on steps 1–34 reproduces the published collapse (illicit F1 {{EZeroTOneF}} on T1 and {{EZeroTTwoF}} on T2)."),
    ("h2", "Bank Account Fraud (Study 1 replication)"),
    ("p", "The BAF suite[[jesus]] has one million applications per variant over eight months with about 1.1% fraud. It is synthetic (a CTGAN model with differential-privacy noise trained on a real dataset). The dev era is months 0–3 and the test era months 4–7. We replicate on Base and on Variant III, chosen from month-level fraud-rate families before any model was run. The delayed-feedback ensemble is omitted on BAF: with one or two training months its recent/old split is degenerate and it reduces to the XGBoost ensemble."),
    ("h2", "BitcoinHeist (Study 2)"),
    ("p", "BitcoinHeist[[akcora]] has 2,916,697 address-days from 2009 to 2018, of which 41,413 are payments to known ransomware families, and the amount received by each. Periods are 28-day blocks. The dev era is 2011–2014 and the test era 2015–2017 (39 periods; 2018 has three positives and is excluded), split into T1 (2015, {{STOnePos}} ransomware events) and T2 (2016–2017, {{STTwoPos}}). T2 starts with the year in which the Locky, Cerber and CryptXXX families first appear; it was fixed from the family-by-year counts before any model was fit. On the dev era the eleven published features were near chance forward in time (AUPRC 0.038 against a base rate of 0.029), so the scorer also receives a matured-blacklist flag: an earlier payment to the same address was labelled ransomware and that label had matured. The flag obeys the same maturity rule as every other label access and lifts dev AUPRC to 0.351. Counts of how often an address repeats raised it further but were excluded, because legitimate addresses are subsampled to 1,000 per day and repeat counts are an artefact of that sampling. {{SUnseenShare}}% of the test events come from addresses that never appear in the scorer’s training periods, so Study 2 is also a test under entity shift."),
    ("h2", "Pre-registration and tuning"),
    ("p", "For each study, splits, the cost grid, hyper-parameters, decision-layer settings, and the hypotheses with their failure criteria were frozen and hashed before any test-era score existed; the Study 2 lock also records hashes of the policy and runner source files. Scorer hyper-parameters were chosen by random search on dev walk-forward AUPRC (Supplementary Note 2). Dev-era tuning in Study 1 chose isotonic recalibration over a 3-period window (mean dev cost relative to the reference: {{DevIsoRatio}} isotonic, {{DevAffRatio}} affine, {{DevTempRatio}} temperature scaling). Study 2 does not tune the decision layer: it reuses the Study 1 settings unchanged. {@tab:hyp} lists the hypotheses and outcomes. Study 2 was registered after the Study 1 results were known, so its predictions are directional where Study 1 supports one."),
    ("tab", t_hyp, "tab:hyp"),
    ("h2", "Statistics and metrics"),
    ("p", "The primary metric is cost per 1,000 events under the loss in {@tab:loss}. Per-period costs are averaged over training seeds (10 on Elliptic, 5 on BAF and BitcoinHeist), and periods are then resampled with a moving-block bootstrap[[kunsch]] (block length 2, 5,000 resamples); seeds measure training noise, not sampling variability. We report paired gains with 95% intervals, Wilcoxon tests over periods, a Friedman–Nemenyi comparison,[[demsar]] equivalence tests[[schuirmann]] with a 2% margin, and Holm correction[[holm]] within each study’s confirmatory family (four tests in Study 1, five in Study 2; BAF enters H1a as a same-sign criterion). Calibration is measured by equal-mass ECE, the Brier score and " + CEU + "."),
    ("h2", "Reproducibility"),
    ("p", "All code, the frozen configurations, both pre-registration locks, dataset checksums and the scripts that generate every table, figure and number in this paper are available (Section “Data availability”). The test suite covers the leakage guards (maturity, train-only scalers, inductive graphs, no random splits, the matured-blacklist feature) and the policy layer; the principal guards were verified by deliberate mutation, the amount-based policies of Study 2 reduce exactly to those of Study 1 for constant amounts, and the headline costs of Study 1 were re-derived with an independent implementation (Supplementary Note 5). Experiments run on a 4 GB GPU, and all policy and statistics experiments run on CPU from cached scores."),

    ("h1", "Results and analysis"),
    ("h2", "Study 1: the review band under uniform costs"),
    ("p", "The review option with rolling recalibration is the largest effect in Study 1. Against the best single threshold (" + P1 + "), the Chow band " + P2 + " lowers cost by {{EPiTwoVsOneAllRel}}% over the whole Elliptic test era ({{EPiTwoVsOneTOneRel}}% on T1 and {{EPiTwoVsOneTTwoRel}}% on T2), and by {{BPiTwoVsOneAllRel}}% on BAF Base and {{BVPiTwoVsOneAllRel}}% on Variant III. On T2 the single threshold costs {{EPiOneTTwoCost}} per 1,000 events against {{EPiTwoTTwoCost}} for " + P2 + " ({@fig:cost} and {@tab:main}). In Study 1 this comparison is exploratory; Study 2 tests it as a pre-registered hypothesis."),
    ("fig", "fig_cost_curves.png", "Study 1, Elliptic, primary scorer, primary cost cell. (a) Cost per 1,000 events for the single threshold and the Chow control. (b) Gain of each policy over the control (positive = cheaper). The epistemic cap tracks the control; MAD loses at periods 41 and 44.", 3.6, "fig:cost"),
    ("tab", t_main, "tab:main"),
    ("h2", "Study 1: confirmatory tests"),
    ("p", "The pre-registered primary scorer is {{EPrimary}}. <b>H1a is not supported.</b> MAD does not reduce cost on T2: its gain over the control is {{EHaEst}} per 1,000 events (95% CI [{{EHaLo}}, {{EHaHi}}], {{EHaRel}}% relative; raw p = {{EHaP}}, Holm p = {{EHolmHOneaTTwo}}), it sends more cases to review ({{EPiFourTTwoRev}} against {{EPiTwoTTwoRev}} per 1,000), and its gain is positive in {{ECellsPiFour}} of the 6 non-degenerate cells (criterion: 4). The gain is never positive in any of the {{ESeedN}} training replicates. <b>H1b is not supported.</b> MAD does not beat its no-cap ablation ({{EHbNDEst}} [{{EHbNDLo}}, {{EHbNDHi}}]) or its random-<i>u</i> control ({{EHbRUEst}} [{{EHbRULo}}, {{EHbRUHi}}]); the epistemic cap alone (" + P3 + ") is indistinguishable from the control ({{EThreeTwoEst}} [{{EThreeTwoLo}}, {{EThreeTwoHi}}]) and from random deferral ({{EThreeRUEst}} [{{EThreeRULo}}, {{EThreeRUHi}}]) ({@tab:confirm}, {@fig:forest})."),
    ("p", "<b>H2 is not supported.</b> The rank correlation between the gain of " + P3 + " over its random-<i>u</i> control and " + CEU + " is ρ = {{EHTwoRho}} (95% CI [{{EHTwoLo}}, {{EHTwoHi}}], p = {{EHTwoP}}, n = {{EHTwoN}}). Non-inferiority on T1 is established for " + P3 + " (TOST p < 0.001) and {{ETostFourEquiv}} for MAD (p = {{ETostFourP}}). A secondary, uncorrected correlation between MAD’s cap-attributable gain and " + CEU + " is ρ = {{EHTwoMadRho}} [{{EHTwoMadLo}}, {{EHTwoMadHi}}], but the gains involved are below one cost unit per 1,000 events. The Friedman test over policies does not reject (p = {{EFriedP}}). Across the cost grid ({@fig:grid}), the cap is within about two cost units of the control in every cell and MAD is at or below it in every cell of T2."),
    ("tab", t_confirm, "tab:confirm"),
    ("fig", "fig_forest.png", "Study 1: paired gains over the Chow control with 95% intervals. Squares are policies, diamonds their random-<i>u</i> controls.", 4.17, "fig:forest"),
    ("fig", "fig_cost_grid.png", "Study 1: gain (cost per 1,000) of " + P3 + " and " + P4 + " over " + P2 + " across the cost grid. Cells with C<sub>FP</sub> = 1 are degenerate. Blue = cheaper than the control.", 3.5, "fig:grid"),
    ("h2", "Study 1: across scorers"),
    ("p", "{@tab:scorers} and {@fig:scorers} report every scorer. The decision layer adds little for the strong tree ensembles, whose " + CEU + " is close to zero. The cap helps only for some weaker neural scorers: for the MLP ensemble on T1 ({{EScMlpAllTOneThreeGain}} [{{EScMlpAllTOneThreeLo}}, {{EScMlpAllTOneThreeHi}}]) and for MC-dropout on T2 ({{EScMcdAllTTwoThreeGain}} [{{EScMcdAllTTwoThreeLo}}, {{EScMcdAllTTwoThreeHi}}]). This is the pattern Proposition 1 predicts, but the rank correlation between a scorer’s " + CEU + " and its " + P3 + " gain is not significant (ρ = {{ECrossTOneRho}}, p = {{ECrossTOneP}} on T1; {{ECrossTTwoRho}}, p = {{ECrossTTwoP}} on T2; n = {{ECrossTOneN}}), so we treat it as a hypothesis for future work."),
    ("tab", t_scorers, "tab:scorers"),
    ("fig", "fig_scorers.png", "Study 1, block T2: cost of the Chow control for every scorer (left) and the gain of the epistemic cap over it with 95% intervals (right).", 4.6, "fig:scorers"),
    ("h2", "Mechanism check on synthetic streams"),
    ("p", "Because the real-data result is null, we verify that the policy layer can exploit uncertainty when the condition in Proposition 1 holds ({@fig:synth}; synthetic data, used for the mechanism only). With a calibrated scorer before the shift and a local risk jump after it, the cap " + P3 + " gains {{SynLocalSThreeThree}} (s.e. {{SynLocalSThreeThreeSe}}) per 1,000 events at the largest shift when u flags the affected events, but {{SynRandUSThreeThree}} when u is random with the same marginal miscalibration, and {{SynUniformSThreeThree}} under a uniform shift that recalibration absorbs. Two further points follow. MAD gains {{SynUniformSThreeFour}} under the uniform shift with an uninformative u: its threshold adaptation, not uncertainty, produces that gain, which is why H1b requires the no-cap and random-<i>u</i> controls. And with no shift both policies lose about one cost unit to estimation noise ({{SynLocalSZeroThree}} and {{SynLocalSZeroFour}})."),
    ("fig", "fig_synth_mechanism.png", "Synthetic mechanism check (30 replicates, standard-error bars). Uncertainty-aware deferral gains only when u is informative; MAD also gains from threshold adaptation under a uniform shift.", 4.6, "fig:synth"),
    ("h2", "Error analysis: the ensemble is confidently wrong"),
    ("p", "Among illicit transactions on T2, those that the control approves (missed: {{ErrTTwoMissed}}% of illicit) have lower ensemble variance than those it catches (mean u {{ErrTTwoUMissed}} against {{ErrTTwoUCaught}}); the AUROC of u for identifying misses is {{ErrTTwoAurocU}}, below chance (T1: {{ErrTOneAurocU}}) ({@fig:missed}). Bagged-GBDT disagreement flags borderline events, not novel ones: the cases a regime break hides are those every member agrees are legitimate."),
    ("p", "A signal that depends on the features, not on the ensemble’s disagreement, ranks the missed events better. An isolation forest fitted on the features of the scorer’s training periods reaches an AUROC of {{NovTTwoAuroc}} on T2 and {{NovTOneAuroc}} on T1 (a post-hoc probe on Study 1 that motivated Study 2). Used as a cap it changes nothing: its T2 gain over " + P2 + " is {{NovTTwoThreeEst}} [{{NovTTwoThreeLo}}, {{NovTTwoThreeHi}}], because the window-cost rule never adopts it. The reason is precision. On T2, {{XBaseMissedTTwo}}% of all events are missed illicit transactions; among the 1% of events with the highest ensemble variance the share is {{XPrecUTTwo}}%, and among the 1% most novel it is {{XPrecNovTTwo}}%. Telling missed from caught events among illicit ones is not the same as paying for itself, since the flagged set consists almost entirely of legitimate events, each costing a review."),
    ("fig", "fig_missed_signals.png", "Study 1: ensemble variance and novelty score of illicit transactions that the control misses or catches (median, interquartile box, 5th–95th percentile whiskers), and the AUROC of each signal for identifying misses.", 4.6, "fig:missed"),
    ("p", "MAD’s result has a different cause ({@fig:mad}). After the break, the matured window (illicit prevalence {{ErrTTwoPrevWin}}%) describes the world before it, while the current prevalence is {{ErrTTwoPrevCur}}%; MAD lowers the approve threshold (mean multiplier {{ErrTTwoMadMa}}) and over-reviews ({{ErrTTwoRevFour}}% of events against {{ErrTTwoRevTwo}}% for " + P2 + "), with the largest losses at periods 41 and 44. Delay makes thresholds adapted to the recent past a liability exactly when the distribution moves. Recalibration is affected too: the recalibrated scores are better calibrated than the raw ones in most periods, but not in the last two, where the window no longer describes the stream ({@fig:calib}, {@fig:rel})."),
    ("fig", "fig_mad_trajectory.png", "Study 1: what MAD does around the regime break (vertical line). Left: its multiplier on the approve threshold. Centre: share of events sent to review. Right: illicit prevalence in the matured window and in the period being decided.", 4.6, "fig:mad"),
    ("fig", "fig_calibration_decay.png", "Study 1: per-period ECE of raw and recalibrated scores, and the incremental conditional calibration error " + CEU + " (permutation-corrected, so it can be slightly negative).", 3.4, "fig:calib"),
    ("fig", "fig_reliability.png", "Study 1: reliability of raw and recalibrated scores before (T1) and after (T2) the break, in ten equal-mass bins on logarithmic axes.", 4.17, "fig:rel"),
    ("h2", "Graph structure (H3)"),
    ("p", "<b>H3 is not supported in the direction the literature suggests.</b> With features matched (local features), the GraphSAGE ensemble is not better than the MLP ensemble on T2 ({{EPairHThreesagevsmlpFlocalTTwoEst}} [{{EPairHThreesagevsmlpFlocalTTwoLo}}, {{EPairHThreesagevsmlpFlocalTTwoHi}}]; over the whole test era {{EPairHThreesagevsmlpFlocalAllEst}} [{{EPairHThreesagevsmlpFlocalAllLo}}, {{EPairHThreesagevsmlpFlocalAllHi}}]) and is much worse than XGBoost on the same features (T2 {{EPairHThreesagevsxgbFlocalTTwoEst}} [{{EPairHThreesagevsxgbFlocalTTwoLo}}, {{EPairHThreesagevsxgbFlocalTTwoHi}}]; whole era {{EPairHThreesagevsxgbFlocalAllEst}} [{{EPairHThreesagevsxgbFlocalAllLo}}, {{EPairHThreesagevsxgbFlocalAllHi}}]; positive = second cheaper). Unlike the claim in the recent preprint,[[maganti]] the real topology does carry signal for the same SAGE model: against a degree-preserving edge-shuffled graph the real graph is cheaper by {{EPairAOneZeroshuffledvsrealgraphTTwoEst}} [{{EPairAOneZeroshuffledvsrealgraphTTwoLo}}, {{EPairAOneZeroshuffledvsrealgraphTTwoHi}}] on T2. That signal is already captured by the provider-aggregated features: XGBoost with all features is cheaper than with local features by {{EPairANinexgbFallvsFlocalTTwoEst}} [{{EPairANinexgbFallvsFlocalTTwoLo}}, {{EPairANinexgbFallvsFlocalTTwoHi}}] on T2."),
    ("h2", "Sensitivity: delay, retraining, capacity, recalibration and analyst accuracy"),
    ("p", "Walk-forward retraining matters far more than any decision-layer choice: freezing the XGBoost scorer raises T2 cost from {{EPairASixwalkforwardvsfrozenxgbTTwoCostNew}} to {{EPairASixwalkforwardvsfrozenxgbTTwoCostBase}} (retraining gain {{EPairASixwalkforwardvsfrozenxgbTTwoEst}} [{{EPairASixwalkforwardvsfrozenxgbTTwoLo}}, {{EPairASixwalkforwardvsfrozenxgbTTwoHi}}]). Label delay has a small, borderline effect on the cap: at D = 4 its T2 gain is positive for all three scorers (XGBoost {{ESwXgbDFourWfTTwoThreeEst}} [{{ESwXgbDFourWfTTwoThreeLo}}, {{ESwXgbDFourWfTTwoThreeHi}}], random forest {{ESwRfDFourWfTTwoThreeEst}} [{{ESwRfDFourWfTTwoThreeLo}}, {{ESwRfDFourWfTTwoThreeHi}}], MLP {{ESwMlpDFourWfTTwoThreeEst}} [{{ESwMlpDFourWfTTwoThreeLo}}, {{ESwMlpDFourWfTTwoThreeHi}}]), consistent with staler recalibration leaving more conditional miscalibration, but each interval touches zero and the gains are well under 1% of cost ({@tab:sweeps})."),
    ("tab", t_sweeps, "tab:sweeps"),
    ("p", "Review capacity bounds what the band can deliver ({@fig:capacity}). With capacity for 1% of events per period the T2 cost of " + P2 + " is {{XCapTTwoOne}}, close to the single threshold ({{XCapTTwoPiOne}}); at 5% it is {{XCapTTwoFive}}, at 10% {{XCapTTwoTen}}, and with unlimited capacity {{XCapTTwoUnl}}. The epistemic cap matches the control at every capacity. The choice of recalibrator moves the control’s whole-era cost between {{XRecTwoMin}} and {{XRecTwoMax}} for this scorer, and the cap’s gain between {{XRecGainMin}} and {{XRecGainMax}}: it is largest, at about 1% of cost, under the weaker recalibrators with the shortest window (no recalibration {{XRecNoneGain}}, temperature {{XRecTempGain}}, affine {{XRecAffineGain}}) and vanishes under isotonic recalibration ({{XRecIsoGain}}), which is again what Proposition 1 predicts ({@fig:recal}). As analyst accuracy falls the band is worth less but stays ahead of the single threshold (T2 cost at ρ = 0.7: {{XRhoSevenPiTwo}} against {{XRhoSevenPiOne}}), and the per-seed gains show that the cap’s effect is training noise around zero while MAD’s is never positive ({@fig:rho})."),
    ("fig", "fig_capacity.png", "Study 1: cost against review capacity (share of events that can be reviewed per period). The dash-dotted line is the control with unlimited capacity.", 4.17, "fig:capacity"),
    ("fig", "fig_recal_sensitivity.png", "Study 1, whole test era: cost of the control (left) and gain of the epistemic cap (right) under four recalibrators and three window lengths.", 4.17, "fig:recal"),
    ("fig", "fig_rho_seeds.png", "Study 1, block T2. Left: cost as analyst accuracy falls. Right: gain over the control for each of the ten training seeds (bar = mean).", 4.17, "fig:rho"),
    ("p", "Under a scoring-time feature outage (aggregated features missing, no retraining) the XGBoost control’s T2 cost doubles ({{ERobXgbFeatureoutageTTwoTwoCost}} against {{ERobXgbFeatureoutageTTwoTwoClean}}) and the cap gains {{ERobXgbFeatureoutageTTwoThreeGain}} per 1,000 (95% CI [{{ERobXgbFeatureoutageTTwoThreeLo}}, {{ERobXgbFeatureoutageTTwoThreeHi}}]): the one real-data case where deferral helps a strong scorer, borderline in significance and concentrated in few periods. Under a 2× prevalence shift (negatives thinned from the test periods on), MAD’s gain over " + P2 + " is {{ERobXgbPrevtwoTTwoFourGain}} (negative = worse) and the cap’s is {{ERobXgbPrevtwoTTwoThreeGain}}."),
    ("h2", "Study 1 replication on BAF"),
    ("p", "On BAF Base and Variant III the three-way routing again lowers cost by about a third relative to a single threshold, and " + P3 + ", " + P4 + " and the no-cap and random-<i>u</i> ablations are identical to " + P2 + " in every period, scorer and cost cell ({@tab:baf}). On the matured window every epistemic cap strictly raises cost (for the XGBoost ensemble on BAF Base at t = 6: 0.198 against 0.192 cost per event at the 99th-percentile cap, and 0.408 at the 70th), so the window-cost selection declines to defer and MAD stays at the Chow thresholds. BAF therefore replicates the absence of an uncertainty effect, and the H1a requirement of a positive gain on BAF is not met."),
    ("tab", t_baf, "tab:baf"),
    ("h2", "Study 2: amount-based costs on BitcoinHeist"),
    ("p", "Study 2 scores {{STestEvents}} address-days per seed in the test era, {{STestPos}} of them ransomware payments. The pre-registered primary scorer is {{SPrimary}} (dev AUPRC {{SDevXgb}}; logistic regression {{SDevLr}}, MLP ensemble {{SDevMlp}}, random forest {{SDevRf}}). In the test era every scorer is far weaker than on the dev era (AUPRC {{SScXgbAllAuprc}} for the primary scorer): the blacklist rarely fires on addresses it has not seen, and from 2016 most ransomware payments belong to families that are absent from the training periods. {@fig:s2curves} shows the cost of the single threshold and the Chow band over time, the gain of each alternative over the band, and when novel families arrive; {@tab:s2main} gives the costs and paired gains."),
    ("fig", "fig_s2_curves.png", "Study 2, BitcoinHeist, primary scorer and cell. Top: cost per 1,000 address-days (logarithmic). Middle: gain of each policy over the Chow band (symmetric-logarithmic). Bottom: ransomware events per period, split into families seen and not seen in the scorer’s training periods. The vertical line separates T1 (2015) from T2 (2016–2017).", 4.17, "fig:s2curves"),
    ("tab", t_s2_main, "tab:s2main"),
    ("p", "<b>S2-H1 is {{SHOneVerdict}}, and the direction is reversed.</b> Over the 39 test periods the Chow band costs {{SPiTwoCost}} BTC per 1,000 address-days against {{SPiOneCost}} for the single threshold: it is {{SBandExtraRel}}% more expensive (difference {{SBandGapEst}}, 95% CI [{{SBandGapLo}}, {{SBandGapHi}}], Holm p = {{SHOneHolm}}), with {{SPiTwoRev}} reviews per 1,000 events. The outcome depends on the cost setting ({@fig:s2grid}). The band is cheaper than the single threshold in {{SCellsBandCheaper}} of the 9 cells, with an interval excluding zero in {{SCellsBandCheaperSig}} of them; these are the cells where a review is cheap or a wrongly blocked payment costs little. It is more expensive in the other {{SCellsBandDearer}}, all {{SCellsBandDearerSig}} with intervals excluding zero, and these include the pre-registered primary cell. The Study 1 finding does not carry over to amount-based costs, and the same holds for all four scorers ({@tab:s2scorers})."),
    ("p", "<b>S2-H2 is {{SHTwoVerdict}}.</b> The gain of the epistemic cap over the band is {{SHTwoEst}} [{{SHTwoLo}}, {{SHTwoHi}}] ({{SHTwoRel}}% of cost); equivalence within ±2% is {{SHTwoEquiv}} (TOST p {{SHTwoTostPText}}), and against its random-<i>u</i> control the gain is {{SThreeRUEst}} [{{SThreeRULo}}, {{SThreeRUHi}}]. The null of Study 1 replicates on a second stream. <b>S2-H3:</b> MAD is {{SHThreeWord}} than the band, {{SHThreeVerdict}}: its gain is {{SHThreeEst}} [{{SHThreeLo}}, {{SHThreeHi}}] ({{SHThreeRel}}% of cost, Holm p = {{SHThreeHolm}}), and its interval excludes zero in {{SCellsMadCheaperSig}} of the 9 cells. None of that gain comes from the cap: against its no-cap ablation MAD’s gain is {{SFourNDEst}} [{{SFourNDLo}}, {{SFourNDHi}}]. MAD still costs more than the single threshold in the primary cell ({{SPiFourCost}} against {{SPiOneCost}}), and so does an oracle that picks MAD’s band multipliers from each period’s own labels ({{SPiSevenAllCost}}). <b>S2-H4:</b> the three novelty caps give {{SHFourIfEst}} [{{SHFourIfLo}}, {{SHFourIfHi}}] (isolation forest, Holm p = {{SHFourIfHolm}}), {{SHFourMahaEst}} [{{SHFourMahaLo}}, {{SHFourMahaHi}}] (Mahalanobis, Holm p = {{SHFourMahaHolm}}) and {{SHFourKnnEst}} [{{SHFourKnnLo}}, {{SHFourKnnHi}}] (kNN distance, Holm p = {{SHFourKnnHolm}}); {{SNovAnyCheaper}} of them is significantly cheaper than the band after correction, and their gains over their own permuted controls are the same ({{SIfRandEst}}, {{SMahaRandEst}} and {{SKnnRandEst}}), because the window-cost rule almost never adopts a cap ({@fig:s2forest})."),
    ("fig", "fig_s2_forest.png", "Study 2: paired gains over the Chow band as a percentage of its cost, with 95% intervals, on a symmetric-logarithmic axis. Squares are policies, diamonds permuted controls. A positive value for the single threshold means that it is cheaper than the band.", 4.6, "fig:s2forest"),
    ("fig", "fig_s2_grid.png", "Study 2: gain over the Chow band as a percentage of its cost, in each cell of the cost grid (whole test era). Blue = cheaper than the band.", 4.6, "fig:s2grid"),
    ("tab", t_s2_scorers, "tab:s2scorers"),
    ("h2", "Study 2: why the review band fails under amount-based costs"),
    ("p", "The analyses in this subsection were designed after the Study 2 results and are exploratory. A decomposition of cost shows what happens ({@fig:s2diag}). The single threshold loses {{SDecOneMissed}} BTC per 1,000 events to ransom payments it approves and almost nothing else. The band reviews {{SDecTwoRevRate}}% of events at a cost of {{SDecTwoReview}} BTC and recovers only {{SDecTwoSaved}} BTC of ransom through those reviews. The reviews are in the wrong place. With a per-event band a large payment is worth reviewing at a very small score, so the band reviews {{SStrHundredRevRate}}% of payments of at least 100 BTC and {{SStrTwentyRevRate}}% of those between 20 and 100 BTC. But ransoms are small (median {{SMedAmtIllicit}} BTC, 90th percentile {{SPNinetyAmtIllicit}} BTC, against a 90th percentile of {{SPNinetyAmtLegit}} BTC for legitimate payments), and large payments are almost never ransomware."),
    ("p", "The score does not know this. It is calibrated on average (mean recalibrated score {{SMeanCalP}}% against an observed rate of {{SIllicitRate}}%), but not conditional on the amount: for payments of at least 100 BTC the mean score is {{SStrHundredP}}% and the observed rate {{SStrHundredRate}}%, an overstatement by a factor of {{SStrHundredRatio}}, and by a factor of {{SStrTwentyRatio}} between 20 and 100 BTC. This is Proposition 1 with the amount in the role of the side signal: the amount carries label information beyond the calibrated score, so a rule that is optimal for a calibrated score is not optimal here. The diagnostic agrees in direction ({{SCeAmount}} for the amount against {{SCeVariance}} for the ensemble variance). It also explains MAD’s gain: MAD raises the approve threshold and reviews {{SDecFourRevRate}}% of events."),
    ("p", "A direct test supports the diagnosis only in part. With one specification fixed before running it (isotonic recalibration within five amount strata of the matured window), the band becomes {{SPostVsTwoRel}}% cheaper ({{SPostVsTwoEst}} [{{SPostVsTwoLo}}, {{SPostVsTwoHi}}] BTC per 1,000 events) and is cheaper than the unstratified band in {{SPostCellsBetterThanBand}} of the 9 cells, but in the primary cell it remains {{SPostVsOneRel}}% more expensive than the single threshold ({{SPostVsOneEst}} [{{SPostVsOneLo}}, {{SPostVsOneHi}}]). Conditional miscalibration accounts for roughly half of the gap. The remainder is consistent with a scorer too weak in the test era for reviews to repay their cost at this price, which we did not test further."),
    ("fig", "fig_s2_diagnosis.png", "Study 2 (exploratory). Left: observed ransomware rate and mean recalibrated score by amount received (logarithmic axis). Centre: share of events the Chow band sends to review, by amount. Right: decomposition of cost for three policies.", 4.6, "fig:s2diag"),
    ("h2", "Study 2: novel families"),
    ("p", "BitcoinHeist records the ransomware family of every illicit payment, so novelty can be observed directly: an illicit test event is novel if its family never appears in the scorer’s training periods. Of the ransomware events scored, {{SNovN}} are novel and {{SKnownN}} belong to a known family. The scorer gives low scores to both and lower ones to the novel events (mean raw score {{SNovMeanP}} against {{SKnownMeanP}}; medians {{SNovMedP}} and {{SKnownMedP}}). None of the uncertainty signals identifies them. Among ransomware events, the AUROC for telling a novel family from a known one is {{SAurocU}} for ensemble variance, {{SAurocIf}} for the isolation forest, {{SAurocMaha}} for the Mahalanobis distance and {{SAurocKnn}} for kNN distance, against {{SAurocLowP}} for a low score itself ({@fig:s2novel}). New families do not look unusual in feature space; they look like legitimate payments."),
    ("fig", "fig_s2_novel.png", "Study 2. Left: AUROC of each signal for telling a novel ransomware family from a known one, among ransomware events (0.5 = chance). Right: median raw score of ransomware events from known and novel families.", 4.17, "fig:s2novel"),
    ("h2", "Discussion"),
    ("p", "<i>What the evidence supports.</i> The uncertainty result is the consistent one. Under delay-aware, leakage-audited evaluation, epistemic uncertainty from bagged ensembles adds no measurable value for strong tree scorers once scores are recalibrated: the cap was indistinguishable from the control on Elliptic and on BAF, and equivalent to it within 2% in the pre-registered test on BitcoinHeist. Graph models do not beat matched-feature baselines; the provider-aggregated features carry the topological signal. The review band is not consistent: it lowered cost by {{EPiTwoVsOneAllRel}}% on Elliptic and by about a third on BAF, where costs are uniform, and raised it by {{SBandExtraRel}}% in the primary setting of BitcoinHeist, where costs follow the amount. We predicted the opposite and report the failure."),
    ("p", "<i>One principle covers both results.</i> Proposition 1 says a side signal can help only if it carries label information beyond the calibrated score, and it applies to any variable, not only to uncertainty. Ensemble variance did not carry such information: it is high for borderline events and low for the novel illicit ones, which every member scores as legitimate, and in Study 2, where novelty is labelled, no signal separates novel from known families. Feature-space novelty ranks missed events above caught ones among illicit transactions, but almost everything it flags is legitimate, so a cap built on it never repays its reviews. The amount did carry such information, and a band that ignores it reviews the wrong payments. The cases where a cap helped (weak neural scorers, weaker recalibrators, a feature outage) are also cases of conditional miscalibration."),
    ("p", "<i>Why threshold adaptation cuts both ways.</i> A policy fitted to the matured window is fitted to the past. On Elliptic the fixed band was close to right, and after the break MAD followed a window that still described the stream before it, so it over-reviewed. On BitcoinHeist the fixed band was wrong in a stable way, and following the window corrected part of the error. Adaptation is a remedy for a mis-specified rule and a liability for a well-specified one under delay; neither effect comes from uncertainty."),
    ("p", "<i>For practice and research.</i> Before deploying a cost-optimal review band, check that the score is calibrated conditional on every variable the cost depends on, the amount above all, and recalibrate within strata of it; under uniform costs the band was the largest gain we measured, with about 10% of events needed in the review queue for most of the benefit. Keep retraining on matured labels. Before adding an uncertainty module, test it against a random-uncertainty control and the " + CEU + " diagnostic. For research, the useful targets are scores calibrated conditional on cost, and a novelty signal that is precise among all events and not only rank-informative among illicit ones."),

    ("h1", "Conclusion"),
    ("p", "This work asked when uncertainty-aware deferral pays in fraud and illicit-transaction detection under temporal shift and delayed labels, and answered it in two pre-registered studies with a protocol in which every component reads only matured labels."),
    ("p", "Evaluated against an equally calibrated, uncertainty-free policy, epistemic uncertainty from ensembles did not pay for strong tree scorers in either study, three novelty signals did not identify new ransomware families or repay the reviews they triggered, and graph encoders did not beat feature-matched baselines. Uncertainty can help only when it carries label information beyond the score, which the " + CEU + " diagnostic can test, and neither ensemble disagreement nor feature-space novelty carried it for the novel cases that matter at a regime break."),
    ("p", "The review band gave the largest gain under uniform costs, {{EPiTwoVsOneAllRel}}% on the Elliptic graph, and failed the pre-registered test under amount-based costs, where it was {{SBandExtraRel}}% more expensive than a single threshold in the primary setting on 2.9 million ransomware address-days. The score was calibrated on average and not conditional on the amount, so the band reviewed large payments that were almost never illicit; recalibrating within amount strata removed about half of the gap. Thresholds re-estimated from recent labels cost more when the fixed band was right and less when it was wrong, and in neither case did the effect come from uncertainty."),
    ("p", "Future work follows the threats to validity: scores calibrated conditional on the cost variables, streams with recorded label latency, review queues with real capacity and analyst behaviour, stronger scorers under entity shift, recent graph baselines under the same inductive protocol, and uncertainty estimators designed for precision on novel activity. The protocol, the random-uncertainty control and the negative results, including the prediction that failed, should help others avoid attributing a threshold-adaptation, hedging or cost-structure effect to uncertainty."),
]

BACK = [
    ("h1", "Acknowledgements"),
    ("p", "The authors thank Elliptic, the authors of the Bank Account Fraud suite and the authors of the BitcoinHeist dataset for making their data openly available."),
    ("h1", "Data availability"),
    ("p", "All code, the frozen configurations, both pre-registration locks, dataset checksums and the analysis scripts that generate every number, table and figure are available at https://github.com/darkhorse0204/fraud-deferral. The datasets are public and are not redistributed: the Elliptic Data Set (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set, CC BY-NC-ND 4.0), the Bank Account Fraud suite (https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022, CC BY-NC-SA 4.0) and the BitcoinHeist ransomware address dataset (UCI Machine Learning Repository, https://doi.org/10.24432/C5BG8V, CC BY 4.0). The repository contains the scripts that fetch and verify them."),
    ("h1", "Author contributions"),
    ("p", "Ansh Jerath: Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, Visualization, Writing – original draft. Jagadeesan S: Supervision, Project administration, Writing – review & editing."),
    ("h1", "Funding"),
    ("p", "The authors received no specific funding for this work."),
    ("h1", "Declarations"),
    ("h1", "Competing interests"),
    ("p", "The authors declare no competing interests."),
    ("h1", "Ethical considerations"),
    ("p", "The study uses only public, de-identified datasets and involves no human participants; no data are redistributed. The Bitcoin addresses in BitcoinHeist are public blockchain identifiers and are used only as grouping keys, never as features, and no address is reported. The methods are intended for defence. The main risk of a system of this kind is false assurance: a transaction that is approved, or cleared by a reviewer, may still be illicit, and the costs used here are modelling assumptions, not the losses of any institution."),
    ("h1", "Additional information"),
    ("p", "Correspondence and requests for materials should be addressed to A.J. or J.S."),
    ("h1", "Supplementary information"),
    ("supp",),
    ("p", "The supplementary notes give the data audit, the tuning budgets and selected settings, the leakage audit, additional results, and the independent verification of the headline numbers."),
    ("h2", "Supplementary Note 1. Data audit"),
    ("p", "{@tab:audit} gives the structural checks and audit statistics computed before modelling. Statistics that involve labels were computed on the dev era only."),
    ("tab", t_audit, "tab:audit"),
    ("h2", "Supplementary Note 2. Tuning and selected settings"),
    ("p", "{@tab:tuning} lists the dev-era search budgets and selected settings for every scorer. The decision-layer settings (isotonic recalibration, window of 3 periods, MAD shrinkage and step limits, conformal level) were chosen on Elliptic dev scores and are reused unchanged on BAF and BitcoinHeist."),
    ("tab", t_tuning, "tab:tuning"),
    ("h2", "Supplementary Note 3. Leakage audit"),
    ("p", "{@tab:leak} lists the leakage sources considered and how each is prevented."),
    ("tab", t_leak, "tab:leak"),
    ("h2", "Supplementary Note 4. Additional results"),
    ("p", "{@tab:grid} gives the gains of " + P3 + " and " + P4 + " over " + P2 + " in every non-degenerate cost cell of Elliptic block T2, {@tab:scorerst1} repeats the scorer comparison for block T1, and {@tab:s2grid} gives the Study 2 gains in every cell of its cost grid. {@fig:h2} plots the per-period gain of " + P3 + " over its random-<i>u</i> control against " + CEU + " (H2), and {@fig:dsweep} the gain of the cap as the label delay grows."),
    ("tab", t_grid, "tab:grid"),
    ("tab", t_scorers_t1, "tab:scorerst1"),
    ("tab", t_s2_grid, "tab:s2grid"),
    ("fig", "fig_h2_scatter.png", "Study 1: per-period gain of " + P3 + " over its random-<i>u</i> control against " + CEU + " on the 15 Elliptic test periods (H2).", 3.3, "fig:h2"),
    ("fig", "fig_dsweep.png", "Study 1, block T2: gain of the epistemic cap over the Chow control as the label delay D grows, for three scorers (95% intervals).", 3.3, "fig:dsweep"),
    ("h2", "Supplementary Note 5. Independent verification"),
    ("p", "The headline costs of Study 1 were re-derived with a separate minimal implementation that shares no code with the policy or analysis modules (own window logic, own isotonic call, own cost arithmetic). For the primary scorer the single-threshold and Chow-band costs agreed with the stored results to within 10<sup>−6</sup> for all 10 seeds and 15 test periods; the labels and periods stored in the score tables equal those of the source dataset; and the pooled T2 costs and review rates matched the analysis files. For Study 2, a unit test shows that the amount-based policies return exactly the Study 1 decisions and costs when every amount is the same, for five policies and three random streams."),
]

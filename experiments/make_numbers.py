"""Emit paper/numbers.tex: one \\newcommand per reported number, generated from results/*.json|csv.
The paper text only uses these macros, so no statistic is ever typed by hand.   python -m experiments.make_numbers
"""
from __future__ import annotations

import json
import math

import pandas as pd

from src.utils import ROOT

R = ROOT / "results"
OUT = ROOT / "paper" / "numbers.tex"
PN = {"pi0": "PiZero", "pi1": "PiOne", "pi2": "PiTwo", "pi3": "PiThree", "pi4": "PiFour", "pi4_nodelta": "PiFourND", "pi3_randu": "PiThreeRU",
      "pi4_randu": "PiFourRU", "pi5": "PiFive", "pi6": "PiSix", "pi7": "PiSeven", "pi4_noshrink": "PiFourNS"}
BN = {"T1": "TOne", "T2": "TTwo", "ALL": "All"}
DW = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}   # LaTeX macro names cannot contain digits
lines: list[str] = []


def emit(name: str, val) -> None:
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        val = float("nan")
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f(x, d=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}".replace("-", "$-$")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def dataset_block(a: dict, pref: str) -> None:
    emit(f"{pref}Primary", {"xgb": "XGB-ens", "rf": "RF", "mlp": "MLP-ens", "mcd": "MC-dropout", "sage": "SAGE-ens", "delayed": "delayed-feedback ensemble"}.get(a["primary_scorer"], a["primary_scorer"]))
    for r in a["main_table"]:
        k = f"{pref}{PN[r['policy']]}{BN.get(r['block'], r['block'])}"
        emit(k + "Cost", f(r["cost_per_1000"])); emit(k + "Gain", f(r["gain_vs_pi2"])); emit(k + "Lo", f(r["lo"])); emit(k + "Hi", f(r["hi"]))
        emit(k + "Rev", f(r["reviews_per_1000"], 0)); emit(k + "Rel", f(100 * r["rel_gain"], 1)); emit(k + "Missed", f(100 * r["missed_frac"], 1))
        emit(k + "Frac", f(r["frac_periods_better"], 2))
    cost = {(r["policy"], r["block"]): r["cost_per_1000"] for r in a["main_table"]}
    for b_ in ("T1", "T2", "ALL"):
        if ("pi1", b_) in cost and ("pi2", b_) in cost:
            emit(f"{pref}PiTwoVsOne{BN[b_]}Rel", f(100 * (cost[("pi1", b_)] - cost[("pi2", b_)]) / cost[("pi1", b_)], 0))
    if a.get("cells_positive_T2"):
        for p, v in a["cells_positive_T2"].items():
            emit(f"{pref}Cells{PN[p]}", v)
    if a.get("confirmatory") and a["era"] == "test" and a["confirmatory"].get("H1a"):
        c, h = a["confirmatory"], a["holm"]
        for key, tag in (("H1a", "Ha"), ("H1b_vs_nodelta", "HbND"), ("H1b_vs_randu", "HbRU"), ("pi3_vs_pi2", "ThreeTwo"), ("pi3_vs_randu", "ThreeRU")):
            r = c[key]
            emit(f"{pref}{tag}Est", f(r["est"])); emit(f"{pref}{tag}Lo", f(r["lo"])); emit(f"{pref}{tag}Hi", f(r["hi"])); emit(f"{pref}{tag}P", f(r["p"], 3))
            emit(f"{pref}{tag}Rel", f(100 * r["rel"])); emit(f"{pref}{tag}Dz", f(r["d_z"], 2))
        for k, v in h.items():
            emit(f"{pref}Holm{k.replace('_', '').replace('1', 'One').replace('2', 'Two')}", f(v["p_adj"], 3))
        h2 = c["H2"]
        emit(f"{pref}HTwoRho", f(h2["rho"], 2)); emit(f"{pref}HTwoLo", f(h2["lo"], 2)); emit(f"{pref}HTwoHi", f(h2["hi"], 2)); emit(f"{pref}HTwoP", f(h2["p"], 3)); emit(f"{pref}HTwoN", int(h2["n"]))
        h2m = c["H2_secondary_mad"]
        emit(f"{pref}HTwoMadRho", f(h2m["rho"], 2)); emit(f"{pref}HTwoMadLo", f(h2m["lo"], 2)); emit(f"{pref}HTwoMadHi", f(h2m["hi"], 2))
        for k, tag in (("T1_noninferiority_pi4", "TostFour"), ("T1_noninferiority_pi3", "TostThree")):
            emit(f"{pref}{tag}P", f(c[k]["p"], 3)); emit(f"{pref}{tag}Margin", f(c[k]["margin"], 1)); emit(f"{pref}{tag}Equiv", "established" if c[k]["equivalent"] else "not established")
    fr = a.get("friedman")
    if fr:
        emit(f"{pref}FriedP", f(fr["p"], 3)); emit(f"{pref}FriedCD", f(fr["cd"], 2))
        for k, v in fr["ranks"].items():
            emit(f"{pref}Rank{PN[k]}", f(v, 2))
    if a.get("cross_scorer_spearman"):
        for b, v in a["cross_scorer_spearman"].items():
            emit(f"{pref}Cross{BN.get(b, b)}Rho", f(v["rho"], 2)); emit(f"{pref}Cross{BN.get(b, b)}P", f(v["p"], 3)); emit(f"{pref}Cross{BN.get(b, b)}N", v["n"])
    sg = a.get("seed_gains_T2")
    if sg:
        g3 = [s["gain_pi3"] for s in sg]; g4 = [s["gain_pi4"] for s in sg]
        emit(f"{pref}SeedThreeMin", f(min(g3))); emit(f"{pref}SeedThreeMax", f(max(g3))); emit(f"{pref}SeedFourPos", sum(x > 0 for x in g4)); emit(f"{pref}SeedN", len(sg))
    for k, blk in (a.get("pairs") or {}).items():
        kk = k.replace("_", "").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("6", "Six").replace("9", "Nine").replace("0", "Zero")
        for b, r in blk.items():
            emit(f"{pref}Pair{kk}{BN.get(b, b)}Est", f(r["est"])); emit(f"{pref}Pair{kk}{BN.get(b, b)}Lo", f(r["lo"])); emit(f"{pref}Pair{kk}{BN.get(b, b)}Hi", f(r["hi"]))
            emit(f"{pref}Pair{kk}{BN.get(b, b)}CostBase", f(r["cost_base"], 0)); emit(f"{pref}Pair{kk}{BN.get(b, b)}CostNew", f(r["cost_new"], 0))
    for r in a.get("scorers") or []:
        if r["block"] in BN:
            k = f"{pref}Sc{r['scorer'].capitalize()}{r['fs'].replace('F_', '').capitalize()}{BN[r['block']]}"
            emit(k + "Auprc", f(r["auprc"], 3)); emit(k + "TwoCost", f(r["cost_pi2"], 0)); emit(k + "ThreeGain", f(r["gain_pi3"])); emit(k + "ThreeLo", f(r["gain_pi3_lo"])); emit(k + "ThreeHi", f(r["gain_pi3_hi"]))
            emit(k + "CEu", f(r["ce_u"], 4)); emit(k + "EceRaw", f(r["ece_raw"], 3)); emit(k + "EceCal", f(r["ece_cal"], 3))
    for r in a.get("sweeps") or []:
        if r["block"] in ("T2", "ALL"):
            k = f"{pref}Sw{r['scorer'].capitalize()}D{DW[int(r['D'])]}{'Frozen' if str(r['R']) == 'inf' else 'Wf'}{BN[r['block']]}"
            emit(k + "ThreeEst", f(r["gain_pi3"])); emit(k + "ThreeLo", f(r["gain_pi3_lo"])); emit(k + "ThreeHi", f(r["gain_pi3_hi"])); emit(k + "FourEst", f(r["gain_pi4"])); emit(k + "TwoCost", f(r["cost_pi2"], 0))
    for r in a.get("robustness") or []:
        k = f"{pref}Rob{r['scorer'].capitalize()}{r['stress'].replace(' ', '').replace('.', '').replace('0', 'Zero').replace('2x', 'Two').replace('5x', 'Five').capitalize()}{BN.get(r['block'], r['block'])}"
        emit(k + "TwoCost", f(r["cost_pi2"], 0)); emit(k + "ThreeGain", f(r["gain_pi3"])); emit(k + "FourGain", f(r["gain_pi4"]))
        if r.get("gain_pi3_lo") is not None:
            emit(k + "ThreeLo", f(r["gain_pi3_lo"])); emit(k + "ThreeHi", f(r["gain_pi3_hi"]))
        if r.get("cost_pi2_clean") is not None:
            emit(k + "TwoClean", f(r["cost_pi2_clean"], 0))


def main() -> None:
    for d, pref in (("elliptic", "E"), ("baf_Base", "B"), ("baf_Variant III", "BV")):
        a = load(f"analysis_{d}_test.json")
        if a:
            dataset_block(a, pref)
    e0 = load("e0_gate.json")
    if e0:
        emit("EZeroTOneF", f(e0["F_all"]["T1"]["illicit_f1"], 2)); emit("EZeroTTwoF", f(e0["F_all"]["T2"]["illicit_f1"], 2))
    audit = load("data_audit.json")
    if audit:
        e = audit["elliptic"]
        emit("AuditTOneIllicit", e["T1_illicit_total"]); emit("AuditTTwoIllicit", e["T2_illicit_total"])
        emit("AuditDomainAuc", f(e["labeled_vs_unlabeled_domain_auc_dev"], 3)); emit("AuditCross", e["cross_period_edges"])
        emit("AuditLabeled", f"{e['label_counts']['illicit'] + e['label_counts']['licit']:,}"); emit("AuditUnknown", f"{e['label_counts']['unknown']:,}")
    tun = load("tuning/policy_tuning_elliptic.json")
    if tun:
        t = pd.DataFrame([x for x in tun if x["stage"] == 1])
        emit("DevIsoRatio", f(t[t.calib == "isotonic"].ratio.min(), 2)); emit("DevAffRatio", f(t[t.calib == "affine"].ratio.min(), 2)); emit("DevTempRatio", f(t[t.calib == "temp"].ratio.min(), 2))
    ea = load("error_analysis_elliptic.json")
    if ea:
        for b in ("T1", "T2"):
            B = BN[b]; r = ea[b]
            emit(f"Err{B}AurocU", f(r["auroc_u_for_missed"], 2)); emit(f"Err{B}UMissed", f"{r['mean_u_missed']:.4f}"); emit(f"Err{B}UCaught", f"{r['mean_u_caught']:.4f}")
            emit(f"Err{B}Missed", f(100 * r["missed_rate_pi2"], 1)); emit(f"Err{B}MadMa", f(r["mad_ma"], 2)); emit(f"Err{B}PrevCur", f(100 * r["prev_cur"], 1)); emit(f"Err{B}PrevWin", f(100 * r["prev_win"], 1))
            emit(f"Err{B}RevTwo", f(100 * r["pi2_review_rate"], 1)); emit(f"Err{B}RevFour", f(100 * r["pi4_review_rate"], 1))
    npb = load("novelty_probe.json")
    if npb:
        for b in ("T1", "T2"):
            emit(f"Nov{BN[b]}Auroc", f(npb[b]["auroc_novelty_for_missed"], 2)); emit(f"Nov{BN[b]}CEu", f(npb[b]["ce_u_novelty"], 4))
        for b in ("T2", "ALL"):
            r = npb[b]["pi3_vs_pi2"]
            emit(f"Nov{BN[b]}ThreeEst", f(r["est"])); emit(f"Nov{BN[b]}ThreeLo", f(r["lo"])); emit(f"Nov{BN[b]}ThreeHi", f(r["hi"]))
    sy = R / "synth_mechanism.csv"
    if sy.exists():
        d = pd.read_csv(sy)
        for fam, tag in (("local", "Local"), ("local_randu", "RandU"), ("uniform", "Uniform")):
            for s in (0.0, 3.0):
                r = d[(d.family == fam) & (d.s == s)].iloc[0]
                sw = DW[int(s)]
                emit(f"Syn{tag}S{sw}Three", f(r.gain_pi3)); emit(f"Syn{tag}S{sw}Four", f(r.gain_pi4)); emit(f"Syn{tag}S{sw}ThreeSe", f(r.gain_pi3_se))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("% AUTO-GENERATED by experiments/make_numbers.py -- do not edit\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()

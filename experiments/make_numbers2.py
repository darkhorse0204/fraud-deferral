"""Emit paper/numbers2.tex: macros for Study 2 (prefix S) and the extra Study 1 analyses (prefix X).
Verdict words are derived from the confidence intervals, so the text cannot disagree with the statistics.
    python -m experiments.make_numbers2
"""
from __future__ import annotations

import json
import math

import pandas as pd
import yaml

from src.utils import ROOT

R = ROOT / "results"
OUT = ROOT / "paper" / "numbers2.tex"
BN = {"T1": "TOne", "T2": "TTwo", "ALL": "All"}
PN = {"pi1": "PiOne", "pi3": "PiThree", "pi4": "PiFour", "pi4_nodelta": "PiFourND", "pi3_randu": "PiThreeRU", "pi4_randu": "PiFourRU",
      "pi3_if": "If", "pi3_maha": "Maha", "pi3_knn": "Knn", "pi3_if_rand": "IfRand", "pi3_maha_rand": "MahaRand", "pi3_knn_rand": "KnnRand", "pi7": "PiSeven"}
SCN = {"lr": "Lr", "rf": "Rf", "xgb": "Xgb", "mlp": "Mlp"}
lines: list[str] = []


def emit(name, val):
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f(x, d=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    s = f"{x:,.{d}f}"
    return s.replace("-", "$-$")


def sig(r):
    """Direction word from a CI on a gain (positive = cheaper)."""
    if r["lo"] > 0:
        return "cheaper"
    if r["hi"] < 0:
        return "more expensive"
    return "indistinguishable"


def study2():
    p = R / "analysis_study2.json"
    if not p.exists():
        return
    a = json.loads(p.read_text())
    emit("SPrimary", {"xgb": "XGB-ens", "rf": "RF", "mlp": "MLP-ens"}[a["primary_scorer"]])
    c, h = a["confirmatory"], a["holm"]
    names = {"S2H1_pi2_vs_pi1": "HOne", "S2H2_pi3_vs_pi2": "HTwo", "S2H3_pi4_vs_pi2": "HThree", "S2H4_if_vs_pi2": "HFourIf", "S2H4_maha_vs_pi2": "HFourMaha",
             "S2H4_knn_vs_pi2": "HFourKnn", "pi3_vs_randu": "ThreeRU", "if_vs_rand": "IfRand", "maha_vs_rand": "MahaRand", "knn_vs_rand": "KnnRand", "pi4_vs_nodelta": "FourND"}
    for k, n in names.items():
        r = c[k]
        emit(f"S{n}Est", f(r["est"])); emit(f"S{n}Lo", f(r["lo"])); emit(f"S{n}Hi", f(r["hi"])); emit(f"S{n}P", f(r["p"], 3)); emit(f"S{n}Rel", f(100 * r["rel"]))
        emit(f"S{n}Word", sig(r)); emit(f"S{n}Frac", f(r["frac_periods_better"], 2))
        if k in h:
            emit(f"S{n}Holm", f(h[k]["p_adj"], 3))
    one, two, three = c["S2H1_pi2_vs_pi1"], c["S2H2_pi3_vs_pi2"], c["S2H3_pi4_vs_pi2"]
    emit("SPiOneCost", f(one["cost_ctrl"])); emit("SPiTwoCost", f(one["cost_new"])); emit("SPiTwoRev", f(one["review_new"], 0))
    emit("SPiTwoVsOneRel", f(100 * one["rel"], 0))
    emit("SPiTwoMissedAmt", f(100 * one["missed_amt_share_new"])); emit("SPiTwoMissed", f(100 * one["missed_share_new"]))
    emit("SHOneVerdict", "supported" if (one["lo"] > 0 and h["S2H1_pi2_vs_pi1"]["reject"]) else "not supported")
    emit("SHTwoTostPText", "< 0.001" if two["tost"]["p"] < 0.001 else f"= {two['tost']['p']:.3f}")
    emit("SHTwoTostP", f(two["tost"]["p"], 3)); emit("SHTwoVerdict", "supported" if two["tost"]["equivalent"] else "not supported")
    emit("SHTwoEquiv", "established" if two["tost"]["equivalent"] else "not established")
    emit("SHThreeVerdict", "as predicted" if three["hi"] <= 0 or three["est"] <= 0 else "contrary to the prediction")
    emit("SPiFourRev", f(three["review_new"], 0)); emit("SPiFourCost", f(three["cost_new"]))
    best = max(("S2H4_if_vs_pi2", "S2H4_maha_vs_pi2", "S2H4_knn_vs_pi2"), key=lambda k: c[k]["est"])
    emit("SNovBestName", {"S2H4_if_vs_pi2": "isolation forest", "S2H4_maha_vs_pi2": "Mahalanobis distance", "S2H4_knn_vs_pi2": "kNN distance"}[best])
    emit("SNovAnyCheaper", "at least one" if any(c[k]["lo"] > 0 and h[k]["reject"] for k in ("S2H4_if_vs_pi2", "S2H4_maha_vs_pi2", "S2H4_knn_vs_pi2")) else "none")
    for b, d in a["main"].items():
        for p_, r in d.items():
            k = f"S{PN[p_]}{BN[b]}"
            emit(k + "Est", f(r["est"])); emit(k + "Lo", f(r["lo"])); emit(k + "Hi", f(r["hi"])); emit(k + "Rel", f(100 * r["rel"])); emit(k + "Cost", f(r["cost_new"]))
        emit(f"SPiTwo{BN[b]}Cost", f(d["pi1"]["cost_ctrl"]))
        emit(f"SPiTwoVsOne{BN[b]}Rel", f(100 * (d["pi1"]["cost_new"] - d["pi1"]["cost_ctrl"]) / d["pi1"]["cost_new"], 0))
    nv = a["novel_family"]
    emit("SNovN", f"{nv['novel']['n']:,}"); emit("SKnownN", f"{nv['known']['n']:,}")
    emit("SNovMedP", f(nv["novel"]["median_p"], 3)); emit("SKnownMedP", f(nv["known"]["median_p"], 3))
    for k, n in (("u", "U"), ("u_if", "If"), ("u_maha", "Maha"), ("u_knn", "Knn")):
        emit(f"SAuroc{n}", f(nv["auroc_for_novel_family"][k], 2))
    emit("SAurocLowP", f(nv["auroc_score_for_novel_family_low_p"], 2))
    emit("SUnseenShare", f(100 * a["share_test_events_from_unseen_addresses"], 0))
    for r in a["scorers"]:
        k = f"SSc{SCN[r['scorer']]}{BN[r['block']]}"
        emit(k + "OneCost", f(r["cost_pi1"])); emit(k + "TwoCost", f(r["cost_pi2"])); emit(k + "ThreeEst", f(r["gain_pi3"])); emit(k + "ThreeLo", f(r["gain_pi3_lo"]))
        emit(k + "ThreeHi", f(r["gain_pi3_hi"])); emit(k + "FourEst", f(r["gain_pi4"])); emit(k + "FourLo", f(r["gain_pi4_lo"])); emit(k + "FourHi", f(r["gain_pi4_hi"]))
        emit(k + "Auprc", f(r["auprc_all"], 3))
    for s, n in SCN.items():
        y = ROOT / "configs" / "study2" / "models" / f"{s}.yaml"
        if y.exists():
            emit(f"SDev{n}", f(yaml.safe_load(y.read_text())["dev_auprc"], 3))
    pp = pd.DataFrame(a["per_period"])
    emit("STestEvents", f"{int(pp.n.sum()):,}"); emit("STestPos", f"{int(pp.n_pos.sum()):,}"); emit("STestNovel", f"{int(pp.n_novel.sum()):,}")
    emit("STOnePos", f"{int(pp[pp.period <= 64].n_pos.sum()):,}"); emit("STTwoPos", f"{int(pp[pp.period >= 65].n_pos.sum()):,}")


def extras():
    p = R / "extra_elliptic.json"
    if not p.exists():
        return
    x = json.loads(p.read_text())
    cap = pd.DataFrame(x["capacity"])
    cn = {0.01: "One", 0.02: "Two", 0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}
    for b in ("T2", "ALL"):
        d = cap[(cap.block == b) & (cap.policy == "pi2")]
        for cval, nm in cn.items():
            emit(f"XCap{BN[b]}{nm}", f(d[(~d.unlimited) & (abs(d.capacity - cval) < 1e-9)].cost.iloc[0], 0))
        emit(f"XCap{BN[b]}Unl", f(d[d.unlimited].cost.iloc[0], 0))
        emit(f"XCap{BN[b]}PiOne", f(cap[(cap.block == b) & (cap.policy == "pi1")].cost.iloc[0], 0))
    rec = pd.DataFrame(x["recalibration"]); ra = rec[rec.block == "ALL"]
    emit("XRecTwoMin", f(ra.cost_pi2.min(), 0)); emit("XRecTwoMax", f(ra.cost_pi2.max(), 0))
    for cal, nm in (("none", "None"), ("temp", "Temp"), ("affine", "Affine"), ("isotonic", "Iso")):
        r = ra[(ra.calib == cal) & (ra.W == 3)].iloc[0]
        emit(f"XRec{nm}Two", f(r.cost_pi2, 0)); emit(f"XRec{nm}Gain", f(r.gain_pi3))
    emit("XRecGainMin", f(ra.gain_pi3.min())); emit("XRecGainMax", f(ra.gain_pi3.max()))
    rho = pd.DataFrame(x["rho"]); rt = rho[rho.block == "T2"]
    for rv, nm in ((0.9, "Nine"), (0.8, "Eight"), (0.7, "Seven")):
        r = rt[abs(rt.rho - rv) < 1e-9].iloc[0]
        for pol, pn in (("pi1", "One"), ("pi2", "Two"), ("pi3", "Three"), ("pi4", "Four")):
            emit(f"XRho{nm}Pi{pn}", f(r[pol], 0))
    mv = x["missed_vs_caught"]
    for b in ("T1", "T2"):
        d = mv[b]
        emit(f"XPrecU{BN[b]}", f(100 * d["prec_top1_u"], 2)); emit(f"XPrecNov{BN[b]}", f(100 * d["prec_top1_nov"], 2)); emit(f"XBaseMissed{BN[b]}", f(100 * d["base_missed_rate"], 2))
        emit(f"XAurocNov{BN[b]}", f(d["auroc_nov"], 2)); emit(f"XAurocU{BN[b]}", f(d["auroc_u"], 2))


def study2_extra():
    """Grid counts, the post-hoc diagnosis and the amount-stratified recalibration test."""
    p = R / "analysis_study2.json"
    if not p.exists():
        return
    a = json.loads(p.read_text())
    g = pd.DataFrame(a["grid"])                       # pi1_est = gain of pi1 over pi2 (positive = single threshold cheaper)
    emit("SCellsBandCheaper", int((g.pi1_est < 0).sum())); emit("SCellsBandCheaperSig", int((g.pi1_hi < 0).sum()))
    emit("SCellsBandDearer", int((g.pi1_est > 0).sum())); emit("SCellsBandDearerSig", int((g.pi1_lo > 0).sum()))
    emit("SCellsMadCheaperSig", int((g.pi4_lo > 0).sum())); emit("SCellsCapNonzero", int(((g.pi3_lo > 0) | (g.pi3_hi < 0)).sum()))
    one = a["confirmatory"]["S2H1_pi2_vs_pi1"]
    emit("SBandExtraRel", f(abs(100 * one["rel"]), 0)); emit("SBandWord", "more expensive" if one["est"] < 0 else "cheaper")
    emit("SBandGapEst", f(abs(one["est"]))); emit("SBandGapLo", f(min(abs(one["lo"]), abs(one["hi"])))); emit("SBandGapHi", f(max(abs(one["lo"]), abs(one["hi"]))))
    nv = a["novel_family"]
    emit("SNovMeanP", f(nv["novel"]["mean_p"], 3)); emit("SKnownMeanP", f(nv["known"]["mean_p"], 3))
    d = R / "study2_diagnosis.json"
    if d.exists():
        x = json.loads(d.read_text())
        dec = x["decomposition_per_1000"]
        for k, n in (("pi1", "One"), ("pi2", "Two"), ("pi4", "Four")):
            emit(f"SDec{n}Missed", f(dec[k]["missed_amount"])); emit(f"SDec{n}Review", f(dec[k]["review_cost"])); emit(f"SDec{n}RevRate", f(100 * dec[k]["review_rate"]))
            emit(f"SDec{n}Saved", f(dec[k]["amount_saved_by_review"])); emit(f"SDec{n}Total", f(dec[k]["total"]))
        st = {s["amount"]: s for s in x["by_amount"]}
        for key, n in (("[100.0, inf)", "Hundred"), ("[20.0, 100.0)", "Twenty"), ("[5.0, 20.0)", "Five"), ("[2.0, 5.0)", "TwoToFive")):
            s = st[key]
            emit(f"SStr{n}Ratio", f(s["ratio_p_over_rate"], 1 if s["ratio_p_over_rate"] < 10 else 0)); emit(f"SStr{n}RevRate", f(100 * s["review_rate_pi2"], 0))
            emit(f"SStr{n}Rate", f(100 * s["illicit_rate"], 2)); emit(f"SStr{n}P", f(100 * s["mean_calibrated_p"], 2)); emit(f"SStr{n}Share", f(100 * s["share_of_events"], 0))
        o = x["overall"]
        emit("SMedAmtIllicit", f(o["median_amount_illicit"])); emit("SPNinetyAmtIllicit", f(o["p90_amount_illicit"])); emit("SPNinetyAmtLegit", f(o["p90_amount_legit"]))
        emit("SMeanCalP", f(100 * o["mean_calibrated_p"], 2)); emit("SIllicitRate", f(100 * o["illicit_rate"], 2))
        cc = x["conditional_calibration"]
        emit("SCeAmount", f(cc["ce_amount"], 4)); emit("SCeVariance", f(cc["ce_variance"], 4))
    q = R / "study2_posthoc.json"
    if q.exists():
        x = json.loads(q.read_text())["primary"]
        for k, n in (("pi2amt_vs_pi2", "PostVsTwo"), ("pi2amt_vs_pi1", "PostVsOne")):
            r = x[k]
            emit(f"S{n}Est", f(r["est"])); emit(f"S{n}Lo", f(r["lo"])); emit(f"S{n}Hi", f(r["hi"])); emit(f"S{n}Rel", f(abs(100 * r["rel"]), 0))
        emit("SPostCost", f(x["pi2amt_vs_pi2"]["cost_new"])); emit("SPostRev", f(x["pi2amt_vs_pi2"]["review_new"], 0))
        gg = pd.DataFrame(json.loads(q.read_text())["grid"])
        emit("SPostCellsBetterThanBand", int((gg.vs_pi2_est > 0).sum())); emit("SPostCellsBetterThanOne", int((gg.vs_pi1_est > 0).sum()))


def main():
    study2(); study2_extra(); extras()
    OUT.write_text("% AUTO-GENERATED by experiments/make_numbers2.py -- do not edit\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()

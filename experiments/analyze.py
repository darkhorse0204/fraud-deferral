"""All inference, tables and numbers for the paper. Reads results/policy + results/diag; writes
results/analysis_<era>.json, results/tables/*.csv and paper/tables/*.tex.   python -m experiments.analyze <era>

Every comparison is a PAIRED comparison of two policies on the same score table, per-period costs averaged over
seeds, then a moving-block bootstrap over periods (block 2, B=5000). Positive gain = new policy cheaper.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import GRID, PRIMARY
from src.stats import inference as I
from src.utils import ROOT

NAME_RE = re.compile(r"^(?P<scorer>[a-z]+)_(?P<fs>F_[a-z]+)_D(?P<D>\d+)_W(?P<Wc>\d+)_R(?P<R>\d+|inf)(?:_(?P<tag>\w+))?$")
PK = PRIMARY.key()
ENSEMBLE_CAPABLE = ["rf", "xgb", "mlp", "mcd", "sage", "delayed"]
BLOCKS = {"elliptic": {"T1": (35, 42), "T2": (43, 49), "ALL": (35, 49)}, "baf": {"T1": (4, 5), "T2": (6, 7), "ALL": (4, 7)}}
MAIN = {"elliptic": dict(D=2, R="5"), "baf": dict(D=1, R="2")}
TEX = ROOT / "paper" / "tables"
TAB = ROOT / "results" / "tables"


def load_policy(dataset_dir: str, era: str) -> pd.DataFrame:
    fr = []
    for f in sorted((ROOT / "results" / "policy" / dataset_dir).glob(f"*__{era}.parquet")):
        name = f.stem.replace(f"__{era}", "")
        m = NAME_RE.match(name)
        df = pd.read_parquet(f).assign(table=name, **{k: v for k, v in m.groupdict().items()})
        fr.append(df)
    out = pd.concat(fr, ignore_index=True)
    out["D"] = out["D"].astype(int)
    return out


def load_diag(dataset_dir: str, era: str) -> pd.DataFrame:
    fr = []
    for f in sorted((ROOT / "results" / "diag" / dataset_dir).glob(f"*__{era}.parquet")):
        name = f.stem.replace(f"__{era}", "")
        fr.append(pd.read_parquet(f).assign(table=name))
    return pd.concat(fr, ignore_index=True)


def per_period(df: pd.DataFrame, table: str, policy: str, cost: str = PK, variant: str = "base") -> pd.DataFrame:
    d = df[(df.table == table) & (df.policy == policy) & (df.cost == cost) & (df.variant == variant)]
    return d.groupby("period")[["n", "cost_sum", "n_review", "n_block", "n_fn", "n_fp", "n_pos", "n_review_pos"]].mean()


def window(pp: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return pp[(pp.index >= lo) & (pp.index <= hi)]


def compare(df, table, ctrl, new, lo, hi, cost=PK, variant="base", B=5000, seed=0) -> dict:
    a, b = window(per_period(df, table, ctrl, cost, variant), lo, hi), window(per_period(df, table, new, cost, variant), lo, hi)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    r = I.block_bootstrap_gain(a.cost_sum.to_numpy(), b.cost_sum.to_numpy(), a.n.to_numpy(), L=2, B=B, seed=seed)
    r.update(I.effect_sizes(a.cost_sum, b.cost_sum, a.n))
    r["wilcoxon_p"] = I.wilcoxon_periods(a.cost_sum, b.cost_sum, a.n)["p"]
    r["cost_ctrl"] = 1000 * a.cost_sum.sum() / a.n.sum()
    r["cost_new"] = 1000 * b.cost_sum.sum() / b.n.sum()
    r["review_ctrl"] = 1000 * a.n_review.sum() / a.n.sum()
    r["review_new"] = 1000 * b.n_review.sum() / b.n.sum()
    return r


def primary_scorer(key: str) -> str:
    best, name = -1.0, None
    for s in [x for x in ENSEMBLE_CAPABLE if not (key == "baf" and x == "delayed")]:
        p = ROOT / "configs" / "models" / f"{key}__{s}__F_all.yaml"
        if p.exists():
            v = yaml.safe_load(p.read_text())["dev_auprc"]
            if v > best:
                best, name = v, s
    return name


def table_name(scorer, fs, key, D=None, R=None, tag=""):
    D = MAIN[key]["D"] if D is None else D
    R = MAIN[key]["R"] if R is None else R
    W = 4 if key == "elliptic" else 1
    return f"{scorer}_{fs}_D{D}_W{W}_R{R}" + (f"_{tag}" if tag else "")


def ci_str(r: dict, key="est", digits=1) -> str:
    return f"{r[key]:.{digits}f} [{r['lo']:.{digits}f}, {r['hi']:.{digits}f}]"


def run(dataset_dir: str, era: str) -> dict:
    key = "elliptic" if dataset_dir == "elliptic" else "baf"
    df, diag = load_policy(dataset_dir, era), load_diag(dataset_dir, era)
    if key == "baf":   # the delayed-feedback ensemble is identical to XGB-ens on BAF (too few training months): not an independent scorer
        df, diag = df[df.scorer != "delayed"], diag[~diag.table.str.startswith("delayed_")]
    blocks = BLOCKS[key] if era == "test" else {"DEV": (20, 34) if key == "elliptic" else (3, 3)}
    ps = primary_scorer(key)
    T = table_name(ps, "F_all", key)
    out: dict = {"dataset": dataset_dir, "era": era, "primary_scorer": ps, "primary_table": T, "blocks": blocks}

    # ---- A. main comparison table on the primary scorer, every block, primary cell --------------------
    pols = ["pi0", "pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi4_noshrink", "pi3_randu", "pi4_randu", "pi5", "pi6", "pi7"]
    rows = []
    for bn, (lo, hi) in blocks.items():
        for p in pols:
            try:
                r = compare(df, T, "pi2", p, lo, hi, B=2000)
            except Exception:  # noqa: BLE001
                continue
            pp = window(per_period(df, T, p), lo, hi)
            rows.append(dict(block=bn, policy=p, cost_per_1000=r["cost_new"], gain_vs_pi2=r["est"], lo=r["lo"], hi=r["hi"],
                             rel_gain=r["rel"], reviews_per_1000=r["review_new"], missed_frac=pp.n_fn.sum() / max(pp.n_pos.sum(), 1),
                             fp_per_1000=1000 * pp.n_fp.sum() / pp.n.sum(), frac_periods_better=r["frac_periods_better"]))
    main = pd.DataFrame(rows)
    out["main_table"] = main.round(4).to_dict("records")

    # ---- B. cost-grid gains (pi3, pi4 vs pi2) for each block -----------------------------------------
    grid = []
    for bn, (lo, hi) in blocks.items():
        for c in GRID:
            for p in ("pi3", "pi4"):
                r = compare(df, T, "pi2", p, lo, hi, cost=c.key(), B=1000)
                grid.append(dict(block=bn, C_FN=c.C_FN, C_FP=c.C_FP, degenerate=c.degenerate, policy=p, gain=r["est"], lo=r["lo"],
                                 hi=r["hi"], cost_pi2=r["cost_ctrl"], rel=r["rel"]))
    gdf = pd.DataFrame(grid)
    out["grid"] = gdf.round(4).to_dict("records")
    t2 = "T2" if "T2" in blocks else list(blocks)[0]
    nd = gdf[(gdf.block == t2) & (~gdf.degenerate)]
    out["cells_positive_T2"] = {p: int((nd[nd.policy == p].gain > 0).sum()) for p in ("pi3", "pi4")}

    # ---- C. confirmatory family ---------------------------------------------------------------------
    conf = {}
    if era == "test":
        lo, hi = blocks["T2"]
        conf["H1a"] = compare(df, T, "pi2", "pi4", lo, hi, B=5000)
        conf["H1b_vs_nodelta"] = compare(df, T, "pi4_nodelta", "pi4", lo, hi, B=5000)
        conf["H1b_vs_randu"] = compare(df, T, "pi4_randu", "pi4", lo, hi, B=5000)
        conf["pi3_vs_pi2"] = compare(df, T, "pi2", "pi3", lo, hi, B=5000)
        conf["pi3_vs_randu"] = compare(df, T, "pi3_randu", "pi3", lo, hi, B=5000)
        a_all = blocks["ALL"]
        # H2: per-period uncertainty-attributable gain vs CE_u (primary scorer; all test periods)
        a, b = window(per_period(df, T, "pi3_randu"), *a_all), window(per_period(df, T, "pi3"), *a_all)
        gain_pp = 1000 * (a.cost_sum - b.cost_sum) / a.n
        dg = diag[diag.table == T].groupby("period")[["ce_u_cal", "ece_cal", "auprc", "mean_u"]].mean()
        j = pd.concat([gain_pp.rename("gain"), dg], axis=1).dropna()
        conf["H2"] = I.spearman_block_ci(j.gain.to_numpy(), j.ce_u_cal.to_numpy())
        conf["H2_secondary_mad"] = None
        a2, b2 = window(per_period(df, T, "pi4_nodelta"), *a_all), window(per_period(df, T, "pi4"), *a_all)
        g2 = 1000 * (a2.cost_sum - b2.cost_sum) / a2.n
        j2 = pd.concat([g2.rename("gain"), dg], axis=1).dropna()
        conf["H2_secondary_mad"] = I.spearman_block_ci(j2.gain.to_numpy(), j2.ce_u_cal.to_numpy())
        out["h2_points"] = j.reset_index().round(5).to_dict("records")
        # non-inferiority in T1
        lo1, hi1 = blocks["T1"]
        a1, b1 = window(per_period(df, T, "pi2"), lo1, hi1), window(per_period(df, T, "pi4"), lo1, hi1)
        conf["T1_noninferiority_pi4"] = I.tost_noninferiority(a1.cost_sum, b1.cost_sum, a1.n)
        conf["T1_noninferiority_pi3"] = I.tost_noninferiority(a1.cost_sum, window(per_period(df, T, "pi3"), lo1, hi1).cost_sum, a1.n)
        fam = {"H1a_T2": conf["H1a"]["p"], "H1b_T2_vs_nodelta": conf["H1b_vs_nodelta"]["p"],
               "H1b_T2_vs_randu": conf["H1b_vs_randu"]["p"], "H2_spearman": conf["H2"]["p"]}
        out["holm"] = {k: dict(p=fam[k], p_adj=v[0], reject=bool(v[1])) for k, v in I.holm(fam).items()}
    out["confirmatory"] = {k: ({kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} if v else None)
                           for k, v in conf.items()}

    # ---- C2. Friedman + Nemenyi across policies (primary scorer; periods are blocks) ------------------------
    names = ["pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi5", "pi6"]
    lo_a, hi_a = (blocks["ALL"] if "ALL" in blocks else list(blocks.values())[0])
    mats = {p: 1000 * window(per_period(df, T, p), lo_a, hi_a).pipe(lambda w: w.cost_sum / w.n) for p in names}
    M = pd.DataFrame(mats).dropna()
    out["friedman"] = I.friedman_nemenyi(M.to_numpy(), list(M.columns)) if len(M) >= 5 else None

    # ---- D. all-scorer comparison table (blocks x scorers) ----------------------------------------------
    sc_rows = []
    for t in sorted(df.table.unique()):
        m = NAME_RE.match(t).groupdict()
        if int(m["D"]) != MAIN[key]["D"] or m["R"] != MAIN[key]["R"] or m["tag"]:
            continue
        for bn, (lo, hi) in blocks.items():
            r2 = compare(df, t, "pi1", "pi2", lo, hi, B=500)
            r3 = compare(df, t, "pi2", "pi3", lo, hi, B=500)
            r4 = compare(df, t, "pi2", "pi4", lo, hi, B=500)
            dd = diag[(diag.table == t) & diag.period.between(lo, hi)]
            sc_rows.append(dict(table=t, scorer=m["scorer"], fs=m["fs"], block=bn, cost_pi1=r2["cost_ctrl"], cost_pi2=r2["cost_new"],
                                gain_pi3=r3["est"], gain_pi3_lo=r3["lo"], gain_pi3_hi=r3["hi"], gain_pi4=r4["est"],
                                auprc=dd.auprc.mean(), auroc=dd.auroc.mean(), ece_raw=dd.ece_raw.mean(), ece_cal=dd.ece_cal.mean(),
                                brier_cal=dd.brier_cal.mean(), ce_u=dd.ce_u_cal.mean()))
    out["scorers"] = pd.DataFrame(sc_rows).round(4).to_dict("records")
    cs = pd.DataFrame(sc_rows)
    cross = {}
    from scipy.stats import spearmanr
    for bn in blocks:   # EXPLORATORY (post hoc): does a scorer's CE_u predict how much the epistemic cap helps it? (LR has u = 0: excluded)
        x = cs[(cs.block == bn) & (cs.scorer != "lr")].dropna(subset=["ce_u", "gain_pi3"])
        if len(x) >= 5:
            rho, pv = spearmanr(x.ce_u, x.gain_pi3)
            cross[bn] = dict(rho=float(rho), p=float(pv), n=int(len(x)))
    out["cross_scorer_spearman"] = cross

    # ---- E. sweeps / variants (exploratory) -------------------------------------------------------------
    sweeps = []
    for t in sorted(df.table.unique()):
        m = NAME_RE.match(t).groupdict()
        if m["tag"] or m["scorer"] not in ("rf", "xgb", "mlp") or m["fs"] != "F_all":
            continue
        for bn in ([t2] if t2 in blocks else list(blocks)):
            lo, hi = blocks[bn]
            r3 = compare(df, t, "pi2", "pi3", lo, hi, B=500)
            r4 = compare(df, t, "pi2", "pi4", lo, hi, B=500)
            sweeps.append(dict(table=t, scorer=m["scorer"], D=int(m["D"]), R=m["R"], block=bn, cost_pi2=r3["cost_ctrl"],
                               gain_pi3=r3["est"], gain_pi3_lo=r3["lo"], gain_pi3_hi=r3["hi"], gain_pi4=r4["est"],
                               gain_pi4_lo=r4["lo"], gain_pi4_hi=r4["hi"]))
    out["sweeps"] = pd.DataFrame(sweeps).round(4).to_dict("records")
    var = []
    for v in ("rho0.9", "cap0.005", "cap0.02"):
        if (df.variant == v).any():
            lo, hi = blocks[t2]
            for p in ("pi1", "pi2", "pi3", "pi4"):
                pp = window(per_period(df, T, p, cost=df[df.variant == v].cost.iloc[0], variant=v), lo, hi)
                var.append(dict(variant=v, policy=p, cost_per_1000=1000 * pp.cost_sum.sum() / pp.n.sum(), reviews_per_1000=1000 * pp.n_review.sum() / pp.n.sum()))
    out["variants"] = pd.DataFrame(var).round(4).to_dict("records")

    # ---- E2. H3 graph-vs-feature comparisons + stress tests (exploratory except H3 which is pre-registered two-sided) ---
    def pair(base_t, new_t, lo, hi, B=3000):
        a, b = window(per_period(df, base_t, "pi2"), lo, hi), window(per_period(df, new_t, "pi2"), lo, hi)
        ix = a.index.intersection(b.index)
        a, b = a.loc[ix], b.loc[ix]
        r = I.block_bootstrap_gain(a.cost_sum.to_numpy(), b.cost_sum.to_numpy(), a.n.to_numpy(), L=2, B=B, seed=1)
        r["cost_base"], r["cost_new"] = 1000 * a.cost_sum.sum() / a.n.sum(), 1000 * b.cost_sum.sum() / b.n.sum()
        return r
    have = set(df.table.unique())
    tn = lambda sc, fs, tag="": table_name(sc, fs, key, tag=tag)   # noqa: E731
    pairs = {"H3_sage_vs_mlp_Flocal": (tn("mlp", "F_local"), tn("sage", "F_local")),
             "H3_sage_vs_xgb_Flocal": (tn("xgb", "F_local"), tn("sage", "F_local")),
             "A10_shuffled_vs_real_graph": (tn("sage", "F_local", "shuf"), tn("sage", "F_local")),
             "A9_xgb_Fall_vs_Flocal": (tn("xgb", "F_local"), tn("xgb", "F_all")),
             "A6_walkforward_vs_frozen_xgb": (table_name("xgb", "F_all", key, R="inf"), tn("xgb", "F_all")),
             "A6_walkforward_vs_frozen_mlp": (table_name("mlp", "F_all", key, R="inf"), tn("mlp", "F_all")),
             "A6_walkforward_vs_frozen_rf": (table_name("rf", "F_all", key, R="inf"), tn("rf", "F_all"))}
    h3 = {}
    for k, (bt, nt) in pairs.items():
        if bt in have and nt in have:
            h3[k] = {bn: pair(bt, nt, lo, hi) for bn, (lo, hi) in blocks.items()}
    out["pairs"] = h3
    rob = []
    for t in sorted(have):
        m = NAME_RE.match(t).groupdict()
        if m["tag"] == "outage":
            for bn, (lo, hi) in blocks.items():
                r3, r4 = compare(df, t, "pi2", "pi3", lo, hi, B=500), compare(df, t, "pi2", "pi4", lo, hi, B=500)
                base = pair(t.replace("_outage", ""), t, lo, hi, B=500)
                rob.append(dict(scorer=m["scorer"], stress="feature outage", block=bn, cost_pi2=r3["cost_ctrl"], cost_pi2_clean=base["cost_base"],
                                gain_pi3=r3["est"], gain_pi3_lo=r3["lo"], gain_pi3_hi=r3["hi"], gain_pi4=r4["est"]))
    for tag in ("prev0.5x", "prev2x"):
        if (df.variant == tag).any():
            for bn, (lo, hi) in blocks.items():
                if bn == "ALL" and "T2" in blocks:
                    continue
                c = {p_: window(per_period(df, T, p_, variant=tag), lo, hi) for p_ in ("pi1", "pi2", "pi3", "pi4")}
                rob.append(dict(scorer=ps, stress=tag, block=bn, cost_pi2=1000 * c["pi2"].cost_sum.sum() / c["pi2"].n.sum(),
                                cost_pi1=1000 * c["pi1"].cost_sum.sum() / c["pi1"].n.sum(),
                                gain_pi3=1000 * (c["pi2"].cost_sum.sum() - c["pi3"].cost_sum.sum()) / c["pi2"].n.sum(),
                                gain_pi4=1000 * (c["pi2"].cost_sum.sum() - c["pi4"].cost_sum.sum()) / c["pi2"].n.sum()))
    out["robustness"] = pd.DataFrame(rob).round(4).to_dict("records")

    # ---- F. per-period series (for figures) --------------------------------------------------------------
    series = []
    for p in ("pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi3_randu"):
        pp = per_period(df, T, p)
        for t_, r in pp.iterrows():
            series.append(dict(policy=p, period=int(t_), cost_per_1000=1000 * r.cost_sum / r.n, reviews_per_1000=1000 * r.n_review / r.n))
    out["series"] = series
    dd = diag[diag.table == T].groupby("period")[["ece_raw", "ece_cal", "ce_u_cal", "auprc", "prev", "mean_u"]].mean().reset_index()
    out["diag_series"] = dd.round(5).to_dict("records")
    seeds_signs = None
    if era == "test":                     # sign consistency across training replicates (H1 criterion)
        lo, hi = blocks["T2"]
        sg = []
        for s in sorted(df.seed.unique()):
            d = df[(df.table == T) & (df.cost == PK) & (df.variant == "base") & (df.seed == s) & df.period.between(lo, hi)]
            c2 = d[d.policy == "pi2"].cost_sum.sum(); c4 = d[d.policy == "pi4"].cost_sum.sum(); c3 = d[d.policy == "pi3"].cost_sum.sum()
            sg.append(dict(seed=int(s), gain_pi4=(c2 - c4) / d[d.policy == "pi2"].n.sum() * 1000, gain_pi3=(c2 - c3) / d[d.policy == "pi2"].n.sum() * 1000))
        seeds_signs = sg
    out["seed_gains_T2"] = seeds_signs
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"analysis_{dataset_dir}_{era}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    era = sys.argv[1]
    for d in (sys.argv[2:] or ["elliptic"]):
        o = run(d, era)
        print(d, era, "primary:", o["primary_scorer"])
        m = pd.DataFrame(o["main_table"])
        print(m[["block", "policy", "cost_per_1000", "gain_vs_pi2", "lo", "hi", "reviews_per_1000"]].round(1).to_string(index=False))
        print("cells positive:", o["cells_positive_T2"])
        if era == "test":
            print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in o["holm"].items()}, indent=1))

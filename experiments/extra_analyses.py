"""Additional EXPLORATORY analyses on the frozen Study 1 score tables (primary Elliptic scorer). Nothing is re-tuned.
  (a) review-capacity sweep            (b) recalibrator kind x window sensitivity (spec ablation A12)
  (c) analyst-accuracy (rho) sweep     (d) reliability curves, T1 vs T2, raw vs recalibrated
  (e) uncertainty / novelty of missed vs caught illicit events
python -m experiments.extra_analyses   ->  results/extra_elliptic.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from experiments.analyze import primary_scorer, table_name
from experiments.run_policies import load_hp, specs_for
from src.data.dataset import load_elliptic_dataset
from src.evaluation.metrics import APPROVE, PRIMARY, Costs
from src.policies.policies import PolicySpec, Table, decide, run_suite
from src.utils import ROOT, log_test_touch

BLOCKS = {"T1": (35, 42), "T2": (43, 49), "ALL": (35, 49)}
PER = range(35, 50)


def _cost(res: pd.DataFrame, pol: str, lo: int, hi: int):
    d = res[(res.policy == pol) & res.period.between(lo, hi)].groupby("period")[["cost_sum", "n", "n_review"]].mean()
    return 1000 * d.cost_sum.sum() / d.n.sum(), 1000 * d.n_review.sum() / d.n.sum()


def main() -> dict:
    name = table_name(primary_scorer("elliptic"), "F_all", "elliptic")
    df = pd.read_parquet(ROOT / "results" / "scores" / "elliptic" / f"{name}__test.parquet")
    hp = load_hp("elliptic")
    tab = Table(df, 2)
    log_test_touch("extra_analyses", "exploratory sensitivity analyses on frozen Study 1 test scores")
    out: dict = {"table": name}

    # (a) review capacity
    cap = []
    sp = specs_for(hp, name, 2, 5, ["pi1", "pi2", "pi3"])
    for frac in (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, None):
        res = run_suite(tab, PER, sp, [PRIMARY], cap_frac=frac)
        for b, (lo, hi) in BLOCKS.items():
            for p in ("pi1", "pi2", "pi3"):
                c, r = _cost(res, p, lo, hi)
                cap.append(dict(capacity=frac if frac is not None else 1.0, unlimited=frac is None, block=b, policy=p, cost=c, reviews=r))
    out["capacity"] = cap

    # (b) recalibrator kind x window
    rec = []
    for calib in ("none", "temp", "affine", "isotonic"):
        for W in (3, 5, 8):
            res = run_suite(tab, PER, [PolicySpec("pi1", W=W, calib=calib), PolicySpec("pi2", W=W, calib=calib), PolicySpec("pi3", W=W, calib=calib)], [PRIMARY])
            for b, (lo, hi) in BLOCKS.items():
                c1, c2, c3 = (_cost(res, p, lo, hi)[0] for p in ("pi1", "pi2", "pi3"))
                rec.append(dict(calib=calib, W=W, block=b, cost_pi1=c1, cost_pi2=c2, cost_pi3=c3, gain_pi3=c2 - c3))
    out["recalibration"] = rec

    # (c) analyst accuracy
    rho = []
    sp4 = specs_for(hp, name, 2, 5, ["pi1", "pi2", "pi3", "pi4"])
    for r_ in (1.0, 0.95, 0.9, 0.8, 0.7):
        res = run_suite(tab, PER, sp4, [Costs(PRIMARY.C_FN, PRIMARY.C_FP, 1.0, r_)])
        for b, (lo, hi) in BLOCKS.items():
            rho.append(dict(rho=r_, block=b, **{p: _cost(res, p, lo, hi)[0] for p in ("pi1", "pi2", "pi3", "pi4")}))
    out["rho"] = rho

    # (d) + (e): event-level arrays under the frozen pi2
    ds = load_elliptic_dataset()
    nov_parts = []
    for tau in sorted(df.origin.unique()):
        cut = int(tau) - 2 - 1 - 4
        iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(ds.X[ds.period <= cut])
        rid = np.sort(df[df.origin == tau].rid.unique())
        nov_parts.append(pd.DataFrame({"origin": tau, "rid": rid, "nov": -iso.score_samples(ds.X[rid])}))
    novmap = pd.concat(nov_parts).set_index(["origin", "rid"]).nov
    s2 = specs_for(hp, name, 2, 5, ["pi2"])[0]
    ev = []
    for seed in tab.seeds:
        st = {}
        for t in PER:
            cur = tab.current(seed, t)
            act, pc, st = decide(s2, PRIMARY, tab.window(seed, t, s2.W, s2.h), cur, st)
            o = tab.origin_for(t)
            ev.append(pd.DataFrame(dict(seed=seed, period=t, y=cur["y"], p=cur["p"], pc=pc, u=cur["u"], act=act,
                                        nov=novmap.reindex(pd.MultiIndex.from_arrays([[o] * len(pc), cur["rid"]])).to_numpy())))
    ev = pd.concat(ev, ignore_index=True)
    rel, dist = {}, {}
    for b, (lo, hi) in (("T1", BLOCKS["T1"]), ("T2", BLOCKS["T2"])):
        e = ev[ev.period.between(lo, hi)]
        for col in ("p", "pc"):
            order = np.argsort(e[col].to_numpy(), kind="stable")
            bins = np.array_split(order, 10)
            rel[f"{b}_{col}"] = [dict(mean_pred=float(e[col].to_numpy()[ix].mean()), frac_pos=float(e.y.to_numpy()[ix].mean()), n=int(len(ix))) for ix in bins]
        pos = e[e.y == 1]
        missed = pos.act == APPROVE
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        dist[b] = dict(n_missed=int(missed.sum() / len(tab.seeds)), n_caught=int((~missed).sum() / len(tab.seeds)),
                       u_missed=pos[missed].u.quantile(q).tolist(), u_caught=pos[~missed].u.quantile(q).tolist(),
                       nov_missed=pos[missed].nov.quantile(q).tolist(), nov_caught=pos[~missed].nov.quantile(q).tolist(),
                       pc_missed=pos[missed].pc.quantile(q).tolist(), pc_caught=pos[~missed].pc.quantile(q).tolist(),
                       auroc_u=float(roc_auc_score(missed.astype(int), pos.u)), auroc_nov=float(roc_auc_score(missed.astype(int), pos.nov)),
                       # precision of a cap: among the top 1% / 5% most uncertain (novel) EVENTS, the share that are missed illicit
                       prec_top1_u=float(((e.y == 1) & (e.act == APPROVE))[e.u >= e.u.quantile(0.99)].mean()),
                       prec_top1_nov=float(((e.y == 1) & (e.act == APPROVE))[e.nov >= e.nov.quantile(0.99)].mean()),
                       prec_top5_u=float(((e.y == 1) & (e.act == APPROVE))[e.u >= e.u.quantile(0.95)].mean()),
                       prec_top5_nov=float(((e.y == 1) & (e.act == APPROVE))[e.nov >= e.nov.quantile(0.95)].mean()),
                       base_missed_rate=float(((e.y == 1) & (e.act == APPROVE)).mean()))
    out["reliability"], out["missed_vs_caught"] = rel, dist
    (ROOT / "results" / "extra_elliptic.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    c = pd.DataFrame(o["capacity"]); print(c[c.block == "T2"].pivot(index="capacity", columns="policy", values="cost").round(1).to_string())
    r = pd.DataFrame(o["recalibration"]); print(r[r.block == "ALL"].round(1).to_string(index=False))
    print(pd.DataFrame(o["rho"]).query("block == 'T2'").round(1).to_string(index=False))
    for b in ("T1", "T2"):
        d = o["missed_vs_caught"][b]
        print(b, {k: round(v, 4) for k, v in d.items() if isinstance(v, float)})

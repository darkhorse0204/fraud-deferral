"""POST-HOC EXPLORATORY probe (run after seeing the test results; NOT pre-registered).

The error analysis showed ensemble variance cannot flag the novel illicit transactions at the regime break. Does a
feature-space NOVELTY signal do better?  ONE detector, default settings, fixed before running (no forking paths):
IsolationForest(n_estimators=200, random_state=0) fitted on the features (no labels) of every node in the scorer's
training periods (<= cut), scoring the events in the cached score table. The novelty score replaces u; everything else
(scorer, recalibration window, policies, costs, seeds) is identical.   python -m experiments.novelty_probe
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from experiments.analyze import MAIN, compare, primary_scorer, table_name
from experiments.run_policies import load_hp, specs_for
from src.calibration.recal import fit_map
from src.data.dataset import load_elliptic_dataset
from src.evaluation.metrics import APPROVE, GRID, PRIMARY, ce_u
from src.policies.policies import Table, decide, run_suite
from src.utils import ROOT, log_test_touch

BLOCKS = {"T1": (35, 42), "T2": (43, 49), "ALL": (35, 49)}


def main() -> dict:
    ds = load_elliptic_dataset()
    ps = primary_scorer("elliptic")
    name = table_name(ps, "F_all", "elliptic")
    df = pd.read_parquet(ROOT / "results" / "scores" / "elliptic" / f"{name}__test.parquet")
    D, Wc = 2, 4
    nov = pd.Series(index=pd.MultiIndex.from_arrays([[], []], names=["origin", "rid"]), dtype=float)
    parts = []
    for tau in sorted(df.origin.unique()):
        cut = int(tau) - D - 1 - Wc
        tr = np.flatnonzero(ds.period <= cut)                       # features of ALL pool nodes; no labels used
        iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(ds.X[tr])
        rids = np.sort(df[df.origin == tau].rid.unique())
        parts.append(pd.DataFrame({"origin": tau, "rid": rids, "nov": -iso.score_samples(ds.X[rids])}))
    nv = pd.concat(parts)
    dn = df.merge(nv, on=["origin", "rid"], how="left")
    assert dn.nov.notna().all()
    base = dn.assign(u=dn.nov).drop(columns="nov")
    rnd = base.copy()
    rnd["u"] = rnd.groupby(["seed", "origin"])["u"].transform(lambda s: s.sample(frac=1.0, random_state=17).to_numpy())  # shuffle-ok
    log_test_touch("novelty_probe", "post-hoc exploratory: IsolationForest novelty replaces u; policies re-run on test scores")
    hp = load_hp("elliptic")
    sp = specs_for(hp, name, 2, 5, ["pi2", "pi3", "pi4", "pi4_nodelta"])
    run_a = run_suite(Table(base, D), range(35, 50), sp, GRID).assign(table="nov", variant="base")
    run_b = run_suite(Table(rnd, D), range(35, 50), [s for s in sp if s.name in ("pi3", "pi4")], GRID)
    run_b["policy"] += "_randu"
    res = pd.concat([run_a, run_b.assign(table="nov", variant="base")], ignore_index=True)
    out = {"scorer": ps, "detector": "IsolationForest(200 trees, defaults), unsupervised on pool features"}
    for bn, (lo, hi) in BLOCKS.items():
        out[bn] = {k: {x: float(v) for x, v in compare(res, "nov", c, n, lo, hi, B=3000).items() if x in ("est", "lo", "hi", "rel", "p")}
                   for k, (c, n) in {"pi3_vs_pi2": ("pi2", "pi3"), "pi4_vs_pi2": ("pi2", "pi4"), "pi3_vs_randnov": ("pi3_randu", "pi3"),
                                     "pi4_vs_randnov": ("pi4_randu", "pi4")}.items()}
    # does novelty flag the missed illicit transactions?  (same construction as error_analysis)
    tab = Table(base, D)
    s2 = specs_for(hp, name, 2, 5, ["pi2"])[0]
    ev = []
    for seed in tab.seeds[:5]:
        st = {}
        for t in range(35, 50):
            cur = tab.current(seed, t)
            act, pc, st = decide(s2, PRIMARY, tab.window(seed, t, s2.W, s2.h), cur, st)
            ev.append(pd.DataFrame(dict(period=t, y=cur["y"], pc=pc, u=cur["u"], act=act)))
    ev = pd.concat(ev)
    for bn, (lo, hi) in {"T1": BLOCKS["T1"], "T2": BLOCKS["T2"]}.items():
        e = ev[ev.period.between(lo, hi)]
        pos = e[e.y == 1]
        missed = (pos.act == APPROVE).astype(int)
        out[bn]["auroc_novelty_for_missed"] = float(roc_auc_score(missed, pos.u)) if 0 < missed.sum() < len(missed) else None
        out[bn]["ce_u_novelty"] = float(ce_u(e.pc.to_numpy(), e.u.to_numpy(), e.y.to_numpy()))
        out[bn]["mean_nov_missed"] = float(pos[missed == 1].u.mean()); out[bn]["mean_nov_caught"] = float(pos[missed == 0].u.mean())
    (ROOT / "results" / "novelty_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    for b in ("T1", "T2", "ALL"):
        print(b, {k: (tuple(round(x, 2) for x in (v["est"], v["lo"], v["hi"])) if isinstance(v, dict) else (round(v, 4) if isinstance(v, float) else v)) for k, v in o[b].items()})

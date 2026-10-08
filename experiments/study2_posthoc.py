"""POST-HOC, EXPLORATORY test of the Study 2 diagnosis (designed after the Study 2 results; not pre-registered).

Diagnosis: with amount-based costs the Chow band needs a score calibrated CONDITIONAL on the amount; the score is
calibrated only marginally. Test: recalibrate within amount strata and re-apply the unchanged band.
One specification, fixed before running: five strata at the matured window's amount quintiles; isotonic regression per
stratum on the same matured window (same maturity rule); a stratum with fewer than 20 positives or 20 negatives uses the
global map. Policies: pi1_amt (single threshold) and pi2_amt (Chow band) on the amount-conditional score.
The frozen policy module is imported, not modified.
python -m experiments.study2_posthoc  ->  results/study2_posthoc.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import study2 as S2
from src.calibration.recal import Identity, fit_map
from src.data import bitcoinheist as BH
from src.policies import amount_policies as AP
from src.stats import inference as I
from src.utils import ROOT, log_test_touch

K = 5


def cond_calibrate(win, cur, g_global):
    edges = np.quantile(win["A"], np.linspace(0, 1, K + 1)[1:-1]) if len(win["A"]) else np.array([])
    bw, bc = np.searchsorted(edges, win["A"], side="right"), np.searchsorted(edges, cur["A"], side="right")
    pc = g_global(cur["p"]).astype(np.float64)
    used = 0
    for k in range(K):
        m = bw == k
        g = fit_map("isotonic", win["p"][m], win["y"][m]) if m.any() else None
        if g is not None:
            pc[bc == k] = g(cur["p"][bc == k]); used += 1
    return pc, used


def main() -> dict:
    ds = BH.load_dataset()
    hp, cells = S2.layer_hp(), S2.CELLS
    df = pd.read_parquet(S2.spath(S2.primary()))
    tab = AP.AmountTable(df, ds.extra["amount"], S2.S["D"])
    log_test_touch("study2_posthoc", "post-hoc exploratory: amount-stratified recalibration on Study 2 test scores")
    rows, strata_used = [], []
    for seed in tab.seeds:
        g_prev = Identity()
        for t in range(S2.S["test_era"][0], S2.S["test_era"][1] + 1):
            cur, win = tab.current(seed, t), tab.window(seed, t, hp["W"])
            g = (fit_map("isotonic", win["p"], win["y"]) if len(win["p"]) else None) or g_prev
            g_prev = g
            pc_g = g(cur["p"])
            pc_a, used = cond_calibrate(win, cur, g)
            strata_used.append(used)
            for c in cells:
                a, b, _ = AP.band(cur["A"], c)
                ca, cr, cb = AP.action_costs(cur["y"], cur["A"], c)
                t1 = c.m / (1 + c.m)
                for name, pc, two in (("pi1", pc_g, False), ("pi2", pc_g, True), ("pi1_amt", pc_a, False), ("pi2_amt", pc_a, True)):
                    act = AP._acts(pc, a, b) if two else np.where(pc >= t1, 2, 0)
                    cst = np.where(act == 0, ca, np.where(act == 1, cr, cb))
                    rows.append((seed, t, name, c.key(), len(pc), float(cst.sum()), int((act == 1).sum())))
    res = pd.DataFrame(rows, columns=["seed", "period", "policy", "cost", "n", "cost_sum", "n_review"])

    def cmp(ctrl, new, cost, lo=52, hi=90, B=5000):
        pp = lambda p: res[(res.policy == p) & (res.cost == cost) & res.period.between(lo, hi)].groupby("period")[["n", "cost_sum", "n_review"]].mean()  # noqa: E731
        a, b = pp(ctrl), pp(new)
        r = I.block_bootstrap_gain(a.cost_sum.to_numpy(), b.cost_sum.to_numpy(), a.n.to_numpy(), L=2, B=B, seed=0)
        r.update(cost_ctrl=1000 * a.cost_sum.sum() / a.n.sum(), cost_new=1000 * b.cost_sum.sum() / b.n.sum(), review_new=1000 * b.n_review.sum() / b.n.sum(),
                 frac_periods_better=float(((a.cost_sum - b.cost_sum) > 0).mean()))
        return {k: float(v) for k, v in r.items()}

    pk = S2.PRIMARY.key()
    out = {"strata": K, "mean_strata_with_own_map": float(np.mean(strata_used)),
           "primary": {"pi2amt_vs_pi1": cmp("pi1", "pi2_amt", pk), "pi2amt_vs_pi2": cmp("pi2", "pi2_amt", pk), "pi2_vs_pi1": cmp("pi1", "pi2", pk),
                       "pi1amt_vs_pi1": cmp("pi1", "pi1_amt", pk), "pi2amt_vs_pi1amt": cmp("pi1_amt", "pi2_amt", pk)},
           "blocks": {b: {"pi2amt_vs_pi1": cmp("pi1", "pi2_amt", pk, lo, hi, B=2000), "pi2amt_vs_pi2": cmp("pi2", "pi2_amt", pk, lo, hi, B=2000)}
                      for b, (lo, hi) in S2.BLOCKS.items()},
           "grid": [dict(c_r=c.c_r, m=c.m, **{f"{k}_{s}": v[s] for k, v in (("vs_pi1", cmp("pi1", "pi2_amt", c.key(), B=1000)), ("vs_pi2", cmp("pi2", "pi2_amt", c.key(), B=1000)))
                                                 for s in ("est", "lo", "hi", "rel")}) for c in cells]}
    (ROOT / "results" / "study2_posthoc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    print("strata with own map (of 5):", round(o["mean_strata_with_own_map"], 2))
    for k, v in o["primary"].items():
        print(f"{k:20s} gain={v['est']:8.2f} [{v['lo']:8.2f},{v['hi']:8.2f}] rel={100 * v['rel']:6.1f}%  cost {v['cost_ctrl']:.2f} -> {v['cost_new']:.2f}  reviews/1k {v['review_new']:.0f}")
    print(pd.DataFrame(o["grid"]).round(2).to_string(index=False))

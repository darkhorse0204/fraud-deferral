"""POST-HOC diagnosis of the Study 2 result (exploratory; nothing is re-tuned, no policy is changed).
Why is the per-event Chow band more expensive than the single threshold on BitcoinHeist?
  (1) cost decomposition of pi1 / pi2 / pi4 into missed amount, review cost and wrongly blocked amount;
  (2) calibration of the recalibrated score WITHIN amount strata (is P(y | p, A) = p ?);
  (3) what the band reviews: review rate and illicit rate by amount stratum.
python -m experiments.study2_diagnosis  ->  results/study2_diagnosis.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import study2 as S2
from src.data import bitcoinheist as BH
from src.policies import amount_policies as AP
from src.utils import ROOT, log_test_touch


def main() -> dict:
    ds = BH.load_dataset()
    hp, c = S2.layer_hp(), S2.PRIMARY
    df = pd.read_parquet(S2.spath(S2.primary()))
    tab = AP.AmountTable(df, ds.extra["amount"], S2.S["D"])
    log_test_touch("study2_diagnosis", "post-hoc decomposition of Study 2 costs and calibration by amount")
    ev = []
    for seed in tab.seeds:
        st = {k: {} for k in ("pi1", "pi2", "pi4")}
        for t in range(S2.S["test_era"][0], S2.S["test_era"][1] + 1):
            cur, win = tab.current(seed, t), tab.window(seed, t, hp["W"])
            acts = {}
            for k in st:
                acts[k], st[k] = AP.decide(k, c, win, cur, st[k], hp)
            ev.append(pd.DataFrame(dict(seed=seed, period=t, y=cur["y"], A=cur["A"], u=cur["u"], pc=st["pi2"]["g"](cur["p"]), a1=acts["pi1"], a2=acts["pi2"], a4=acts["pi4"])))
    ev = pd.concat(ev, ignore_index=True)
    n = len(ev)
    out: dict = {"n_event_seeds": n}
    dec = {}
    for k, col in (("pi1", "a1"), ("pi2", "a2"), ("pi4", "a4")):
        a = ev[col]
        dec[k] = dict(missed_amount=1000 * float(ev.A[(a == 0) & (ev.y == 1)].sum()) / n, review_cost=1000 * c.c_r * float((a == 1).sum()) / n,
                      blocked_legit=1000 * c.m * float(ev.A[(a == 2) & (ev.y == 0)].sum()) / n, review_rate=float((a == 1).mean()),
                      block_rate=float((a == 2).mean()), illicit_caught_in_review=float(((a == 1) & (ev.y == 1)).sum() / max((ev.y == 1).sum(), 1)),
                      amount_saved_by_review=1000 * float(ev.A[(a == 1) & (ev.y == 1)].sum()) / n)
        dec[k]["total"] = dec[k]["missed_amount"] + dec[k]["review_cost"] + dec[k]["blocked_legit"]
    out["decomposition_per_1000"] = dec
    edges = [0, 0.5, 1, 2, 5, 20, 100, np.inf]
    ev["bin"] = pd.cut(ev.A, edges, right=False)
    strata = []
    for b, g in ev.groupby("bin", observed=True):
        strata.append(dict(amount=str(b), share_of_events=float(len(g) / n), illicit_rate=float(g.y.mean()), mean_calibrated_p=float(g.pc.mean()),
                           ratio_p_over_rate=float(g.pc.mean() / max(g.y.mean(), 1e-12)), review_rate_pi2=float((g.a2 == 1).mean()),
                           review_rate_pi4=float((g.a4 == 1).mean()), illicit_per_review_pi2=float(g.y[g.a2 == 1].mean()) if (g.a2 == 1).any() else None,
                           breakeven_rate=float(c.c_r / g.A[g.a2 == 1].mean()) if (g.a2 == 1).any() else None,
                           share_of_illicit_amount=float(g.A[g.y == 1].sum() / ev.A[ev.y == 1].sum())))
    out["by_amount"] = strata
    out["corr_logA_y"] = float(np.corrcoef(np.log(ev.A), ev.y)[0, 1])
    # Proposition 1's diagnostic with the side variable z = amount vs z = ensemble variance (seed 0; evaluation only)
    from src.evaluation.metrics import ce_u, ece
    e0 = ev[ev.seed == ev.seed.min()]
    out["conditional_calibration"] = dict(ce_amount=ce_u(e0.pc.to_numpy(), e0.A.to_numpy(), e0.y.to_numpy()), ce_variance=ce_u(e0.pc.to_numpy(), e0.u.to_numpy(), e0.y.to_numpy()),
                                          ece=ece(e0.pc.to_numpy(), e0.y.to_numpy()))
    from sklearn.metrics import average_precision_score
    out["test_auprc_seed0"] = float(average_precision_score(e0.y, e0.pc))
    out["overall"] = dict(illicit_rate=float(ev.y.mean()), mean_calibrated_p=float(ev.pc.mean()), median_amount_illicit=float(ev.A[ev.y == 1].median()),
                          median_amount_legit=float(ev.A[ev.y == 0].median()), p90_amount_illicit=float(ev.A[ev.y == 1].quantile(0.9)),
                          p90_amount_legit=float(ev.A[ev.y == 0].quantile(0.9)))
    (ROOT / "results" / "study2_diagnosis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    print(pd.DataFrame(o["decomposition_per_1000"]).round(3).to_string())
    print(pd.DataFrame(o["by_amount"]).round(4).to_string(index=False))
    print(o["overall"])

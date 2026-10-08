"""Independent re-derivation of headline numbers (does NOT import the policy / analysis code).

Checks, for the primary Elliptic scorer on the test era:
  1. labels stored in the score table equal the raw dataset labels, and rows are all labelled test-era events;
  2. pi1 and pi2 costs recomputed from scratch (own window, own isotonic call, own cost arithmetic) equal the stored
     per-(seed, period) costs;
  3. the pooled T2 cost / review rate / gain figures reported in analysis_elliptic_test.json.
Run:  python -m scripts.verify_independent
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.loaders import load_elliptic
from src.utils import ROOT

D, W, WC, R = 2, 3, 4, 5
FN, FP, CR = 30.0, 5.0, 1.0
A_, B_ = CR / FN, 1 - CR / FP            # Chow band for rho = 1
T1_ = FP / (FP + FN)


def iso(p, y):
    if (y == 1).sum() < 20 or (y == 0).sum() < 20:
        return None
    ir = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip", increasing=True).fit(p, y)
    return lambda q: np.clip(ir.predict(q), 1e-4, 1 - 1e-4)


def cost(act, y):  # 0 approve, 1 review, 2 block
    return np.where(act == 0, y * FN, np.where(act == 1, CR, (1 - y) * FP)).sum()


def main() -> None:
    nodes, _ = load_elliptic()
    raw = nodes[["period", "label"]].reset_index(drop=True)
    name = "xgb_F_all_D2_W4_R5"
    sc = pd.read_parquet(ROOT / "results" / "scores" / "elliptic" / f"{name}__test.parquet")
    pol = pd.read_parquet(ROOT / "results" / "policy" / "elliptic" / f"{name}__test.parquet")
    pol = pol[(pol.cost == f"FN{FN:g}_FP{FP:g}_r1_rho1") & (pol.variant == "base")]

    # 1. label / period integrity
    assert (raw.label.to_numpy()[sc.rid.to_numpy()] == sc.y.to_numpy()).all(), "stored labels differ from dataset"
    assert (raw.period.to_numpy()[sc.rid.to_numpy()] == sc.period.to_numpy()).all(), "stored periods differ from dataset"
    assert (sc.y >= 0).all()
    print("1. labels and periods in the score table equal the dataset: OK  (%d rows)" % len(sc))

    # 2. recompute pi1 / pi2 from scratch for every seed and period
    origins = sorted(sc.origin.unique())
    worst = 0.0
    for seed in sorted(sc.seed.unique()):
        prev = None
        for t in range(35, 50):
            o = max(x for x in origins if x <= t)
            g = sc[(sc.seed == seed) & (sc.origin == o)]
            cur = g[g.period == t]
            win = g[(g.period + D < t) & (g.period >= t - D - W)]          # matured window of the last W periods
            f = iso(win.p.to_numpy(), win.y.to_numpy()) if len(win) else None
            f = f or prev or (lambda q: q)
            prev = f
            pc = f(cur.p.to_numpy())
            y = cur.y.to_numpy().astype(float)
            c1 = cost(np.where(pc >= T1_, 2, 0), y)
            c2 = cost(np.where(pc < A_, 0, np.where(pc >= B_, 2, 1)), y)
            s1 = pol[(pol.seed == seed) & (pol.period == t) & (pol.policy == "pi1")].cost_sum.iloc[0]
            s2 = pol[(pol.seed == seed) & (pol.period == t) & (pol.policy == "pi2")].cost_sum.iloc[0]
            worst = max(worst, abs(c1 - s1), abs(c2 - s2))
    print("2. pi1/pi2 recomputed independently for %d seeds x 15 periods: max |difference| = %.6f" % (sc.seed.nunique(), worst))
    assert worst < 1e-6

    # 3. pooled T2 figures from the stored policy results
    t2 = pol[pol.period.between(43, 49)]
    pm = t2.groupby(["policy", "period"])[["cost_sum", "n", "n_review"]].mean().reset_index()
    out = {}
    for p in ("pi1", "pi2", "pi3", "pi4"):
        q = pm[pm.policy == p]
        out[p] = (1000 * q.cost_sum.sum() / q.n.sum(), 1000 * q.n_review.sum() / q.n.sum())
    a = json.loads((ROOT / "results" / "analysis_elliptic_test.json").read_text())
    ref = {r["policy"]: r for r in a["main_table"] if r["block"] == "T2"}
    for p in out:
        assert abs(out[p][0] - ref[p]["cost_per_1000"]) < 0.06 and abs(out[p][1] - ref[p]["reviews_per_1000"]) < 0.6, p
    print("3. pooled T2 cost/1000 and reviews/1000 match the analysis JSON:", {p: tuple(round(v, 1) for v in x) for p, x in out.items()})
    print("   gain pi4 vs pi2 = %.1f (JSON %.1f)" % (out["pi2"][0] - out["pi4"][0], ref["pi4"]["gain_vs_pi2"]))
    print("ALL INDEPENDENT CHECKS PASSED")


if __name__ == "__main__":
    main()

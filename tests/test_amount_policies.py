"""Study 2 policy layer: reduces exactly to the Study 1 policies for constant amounts; never reads immature labels;
per-event band is Bayes-optimal."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import Costs
from src.policies import amount_policies as AP
from src.policies.policies import PolicySpec, Table, run_suite

HP = dict(calib="isotonic", W=3, lam=0.0, eps=0.5)


def _table(seed=0, n_per=700, periods=range(1, 19), shift_from=12):
    rng = np.random.default_rng(seed)
    rows = []
    for t in periods:
        p = rng.beta(0.6, 6, n_per)
        true = np.clip(0.6 * p + 0.15, 0, 1) if t >= shift_from else p
        rows.append(pd.DataFrame({"origin": 1, "seed": 0, "rid": np.arange(n_per) + t * n_per, "period": t,
                                  "y": (rng.random(n_per) < true).astype(int), "p": p, "u": rng.random(n_per) * (0.03 if t >= shift_from else 0.01)}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduces_to_scalar_policies_for_constant_amount(seed):
    df = _table(seed)
    A, C_FP = 30.0, 5.0
    amounts = np.full(int(df.rid.max()) + 1, A)
    scal = run_suite(Table(df, 2), range(6, 19), [PolicySpec(k, W=3, calib="isotonic", lam=0.0, eps=0.5) for k in ("pi1", "pi2", "pi3", "pi4", "pi4_nodelta")],
                     [Costs(A, C_FP)])
    amt = AP.run_suite(AP.AmountTable(df, amounts, 2), range(6, 19), ["pi1", "pi2", "pi3", "pi4", "pi4_nodelta"], [AP.AmountCosts(1.0, C_FP / A)], HP)
    a = scal.set_index(["period", "policy"])[["cost_sum", "n_review", "n_block", "n_fn"]].sort_index()
    b = amt.set_index(["period", "policy"])[["cost_sum", "n_review", "n_block", "n_fn"]].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-9)


def test_band_is_bayes_optimal_per_event():
    c = AP.AmountCosts(c_r=0.25, m=0.25, rho=0.9)
    for A in (0.3, 1.0, 5.0, 80.0):
        a, b, empty = AP.band(np.array([A]), c)
        for p in np.linspace(0.002, 0.998, 250):
            exp = [p * A, c.c_r + (1 - c.rho) * p * A, (1 - p) * c.m * A]
            pred = 0 if p < a[0] else (2 if p >= b[0] else 1)
            if empty[0]:
                assert exp[pred] <= exp[1] + 1e-9            # review never strictly better
            else:
                assert exp[pred] == pytest.approx(min(exp))


def test_amount_policies_never_read_immature_labels():
    df = _table(3)
    D, t = 2, 14
    df2 = df.copy()
    imm = df2["period"] >= t - D
    df2.loc[imm, "y"] = 1 - df2.loc[imm, "y"]
    rng = np.random.default_rng(0)
    amounts = rng.lognormal(1.5, 1.2, int(df.rid.max()) + 1)
    T1, T2 = AP.AmountTable(df, amounts, D), AP.AmountTable(df2, amounts, D)
    c = AP.AmountCosts(0.25, 0.25)
    for k in ("pi1", "pi2", "pi3", "pi4", "pi4_nodelta"):
        a1, _ = AP.decide(k, c, T1.window(0, t, 3), T1.current(0, t), {}, HP)
        a2, _ = AP.decide(k, c, T2.window(0, t, 3), T2.current(0, t), {}, HP)
        assert (a1 == a2).all(), k


def test_costs_scale_with_amount():
    y = np.array([1, 0, 1]); A = np.array([2.0, 10.0, 4.0])
    ca, cr, cb = AP.action_costs(y, A, AP.AmountCosts(0.5, 0.25))
    assert ca.tolist() == [2.0, 0.0, 4.0] and cr.tolist() == [0.5, 0.5, 0.5] and cb.tolist() == [0.0, 2.5, 0.0]

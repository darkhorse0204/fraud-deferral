"""Policy-layer tests: cost table, Bayes thresholds, no-label-leak, Chow reduction, null behaviour."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import APPROVE, BLOCK, GRID, REVIEW, Costs, chow_thresholds, realised_cost
from src.policies.policies import PolicySpec, Table, decide, run_suite


def test_cost_table_hand_computed():
    c = Costs(C_FN=30, C_FP=5, c_r=1, rho=1.0)
    y = np.array([1, 1, 0, 0])
    act = np.array([APPROVE, BLOCK, BLOCK, REVIEW])
    assert realised_cost(act, y, c).tolist() == [30.0, 0.0, 5.0, 1.0]
    c9 = Costs(30, 5, 1, 0.9)
    assert realised_cost(np.array([REVIEW]), np.array([1]), c9)[0] == pytest.approx(1 + 0.1 * 30)


def test_chow_thresholds_are_bayes_optimal_for_calibrated_p():
    for c in GRID + [Costs(30, 5, 1, 0.9)]:
        a, b = chow_thresholds(c)
        for p in np.linspace(0.002, 0.998, 499):
            exp = [p * c.C_FN, c.c_r + (1 - c.rho) * p * c.C_FN, (1 - p) * c.C_FP]
            best = int(np.argmin(exp))
            pred = APPROVE if p < a else (BLOCK if p >= b else REVIEW)
            if c.degenerate:  # two-action rule: review never strictly better
                assert exp[pred] <= exp[REVIEW] + 1e-9
            else:
                assert exp[pred] == pytest.approx(exp[best])


def test_all_fp1_cells_are_degenerate():
    assert [c.degenerate for c in GRID] == [True, False, False] * 3


def _table(n_per=600, periods=range(1, 16), shift_from=None, seed=0, informative_u=False):
    """Calibrated stream: y ~ Bernoulli(p). After shift_from the model becomes over-confident (true risk compressed
    toward 0.5) and, if informative_u, u rises there."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in periods:
        p = rng.beta(0.6, 6, n_per)
        shifted = shift_from is not None and t >= shift_from
        true = np.clip(0.5 * p + 0.2, 0, 1) if shifted else p
        y = (rng.random(n_per) < true).astype(int)
        u = rng.random(n_per) * 0.01 if not (informative_u and shifted) else 0.02 + rng.random(n_per) * 0.01
        rows.append(pd.DataFrame({"origin": 1, "seed": 0, "rid": np.arange(n_per) + t * n_per, "period": t, "y": y, "p": p, "u": u}))
    return pd.concat(rows, ignore_index=True)


def test_policies_never_read_immature_labels():
    df = _table()
    D, t = 2, 10
    df2 = df.copy()
    imm = df2["period"] >= t - D                        # labels not yet available when deciding in t
    df2.loc[imm, "y"] = 1 - df2.loc[imm, "y"]           # adversarially flip every immature label (incl. period t)
    T1, T2 = Table(df, D), Table(df2, D)
    c = Costs(30, 5)
    for name in ["pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi4_noshrink", "pi5", "pi6"]:
        spec = PolicySpec(name)
        a1, _, _ = decide(spec, c, T1.window(0, t, spec.W, spec.h), T1.current(0, t), {})
        a2, _, _ = decide(spec, c, T2.window(0, t, spec.W, spec.h), T2.current(0, t), {})
        assert (a1 == a2).all(), f"{name} depends on immature labels"


def test_oracle_does_depend_on_current_labels():
    """Sanity: the leak test above can fail -- the oracle is built to read current labels."""
    df = _table()
    D, t = 2, 10
    df2 = df.copy()
    m = df2["period"] == t
    df2.loc[m, "y"] = 1 - df2.loc[m, "y"]
    T1, T2 = Table(df, D), Table(df2, D)
    spec = PolicySpec("pi7")
    a1, _, _ = decide(spec, Costs(30, 5), T1.window(0, t, 5, np.inf), T1.current(0, t), {})
    a2, _, _ = decide(spec, Costs(30, 5), T2.window(0, t, 5, np.inf), T2.current(0, t), {})
    assert (a1 != a2).any()


def test_window_excludes_unmatured_and_current():
    df = _table()
    T = Table(df, D=3)
    w = T.window(0, 10, 5, np.inf)
    assert w["period"].max() <= 10 - 3 - 1 and w["period"].min() >= 10 - 3 - 5


def test_constant_uncertainty_never_triggers_deferral():
    df = _table()
    df["u"] = 0.0
    T = Table(df, D=1)
    c = Costs(30, 5)
    a3, _, _ = decide(PolicySpec("pi3"), c, T.window(0, 9, 5, np.inf), T.current(0, 9), {})
    a2, _, _ = decide(PolicySpec("pi2"), c, T.window(0, 9, 5, np.inf), T.current(0, 9), {})
    # pi3 differs from pi2 only through recalibration of p (same here: both recalibrate identically) -> identical actions
    assert (a3 == a2).all()


def test_no_shift_calibrated_uncertainty_gives_no_gain():
    """Prop. 1 sanity: calibrated p and uninformative u => uncertainty-aware policies cannot beat Chow (beyond noise)."""
    costs = [Costs(30, 5)]
    tot = {}
    for sd in range(6):
        df = _table(seed=sd)
        T = Table(df, D=1)
        res = run_suite(T, range(8, 16), [PolicySpec("pi2"), PolicySpec("pi3"), PolicySpec("pi4")], costs)
        for k, v in res.groupby("policy")["cost_sum"].sum().items():
            tot[k] = tot.get(k, 0) + v
    assert tot["pi3"] >= tot["pi2"] * 0.985 and tot["pi4"] >= tot["pi2"] * 0.97   # allow small in-sample optimism


def test_informative_uncertainty_after_shift_helps():
    """Mechanism check: if p is over-confident after a shift and u flags it, deferral should lower cost."""
    costs = [Costs(30, 5)]
    tot = {}
    for sd in range(6):
        df = _table(seed=sd, shift_from=11, informative_u=True)
        T = Table(df, D=2)
        res = run_suite(T, range(11, 16), [PolicySpec("pi2"), PolicySpec("pi3")], costs)
        for k, v in res.groupby("policy")["cost_sum"].sum().items():
            tot[k] = tot.get(k, 0) + v
    assert tot["pi3"] < tot["pi2"]

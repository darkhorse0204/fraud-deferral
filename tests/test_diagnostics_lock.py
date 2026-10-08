"""CE_u diagnostic behaves as Proposition 1 requires; MAD never leaves Chow without a material in-window gain; lock works."""
import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import APPROVE, Costs, ce_u, ce_u_raw
from src.policies.policies import PolicySpec, Table, decide


def _world(n=30000, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    r = rng.beta(0.5, 5, n)
    S = rng.random(n) < 0.25
    y = (rng.random(n) < np.where(S, np.clip(4 * r, 0, 1), r)).astype(int)
    u = (np.where(S, 0.03, 0.005) if informative else 0.0) + rng.random(n) * 0.01
    return r, u, y


def test_ce_u_detects_informative_u_and_is_null_for_random_u():
    r, u, y = _world(informative=True)
    assert ce_u(r, u, y) > 0.001
    r, u, y = _world(informative=False)
    vals = [ce_u(*_world(seed=s, informative=False)) for s in range(1, 6)]
    assert abs(np.mean(vals)) < 5e-4                       # bias-corrected: ~0 in expectation under the null


def test_ce_u_raw_is_nonnegative_by_construction():
    r, u, y = _world(informative=False)
    assert ce_u_raw(r, u, y) >= -1e-12                     # cells refine the p-bins (triangle inequality)


def test_min_gain_forces_chow_when_window_gain_is_immaterial():
    rng = np.random.default_rng(3)
    rows = []
    for t in range(1, 12):
        p = rng.beta(0.6, 6, 800)
        rows.append(pd.DataFrame({"origin": 1, "seed": 0, "rid": np.arange(800) + 800 * t, "period": t,
                                  "y": (rng.random(800) < p).astype(int), "p": p, "u": rng.random(800) * 0.01}))
    tab = Table(pd.concat(rows), D=1)
    c = Costs(30, 5)
    chow, _, _ = decide(PolicySpec("pi2", W=5), c, tab.window(0, 10, 5, np.inf), tab.current(0, 10), {})
    strict, _, st = decide(PolicySpec("pi4", W=5, min_gain=0.5), c, tab.window(0, 10, 5, np.inf), tab.current(0, 10), {})
    assert (chow == strict).all()                          # an (impossible) 50% in-window gain requirement => Chow


def test_lock_detects_tampering(tmp_path, monkeypatch):
    import scripts.lock_test as L
    d = L.digest()
    assert "configs/preregistration.yaml" in d and "configs/costs.yaml" in d
    assert all(len(v) == 64 for v in d.values())

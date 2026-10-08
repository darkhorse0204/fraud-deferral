"""L3, L11: scalers fit on the matured pool only; feature sets contain no clock."""
import numpy as np
import pytest

from src.features.featuresets import PoolScaler, assert_no_clock, elliptic_sets


def test_scaler_uses_only_pool_rows():
    rng = np.random.default_rng(1)
    periods = np.repeat(np.arange(1, 21), 50)
    X = rng.normal(size=(len(periods), 4))
    X[periods > 10] += 100.0  # the future distribution is wildly different
    t, D = 10, 2
    sc = PoolScaler().fit(X, periods, t, D)
    m = periods + D < t
    np.testing.assert_allclose(sc.mean_, X[m].mean(0))
    np.testing.assert_allclose(sc.std_, X[m].std(0))
    assert sc.max_period_ <= t - D - 1
    assert sc.fit_rows_ == m.sum()
    # clearly differs from full-data statistics (what a leaky pipeline would store)
    assert not np.allclose(sc.mean_, X.mean(0), atol=1.0)


def test_scaler_constant_column_safe():
    periods = np.repeat(np.arange(1, 6), 5)
    X = np.ones((25, 2))
    out = PoolScaler().fit(X, periods, 5, 0).transform(X)
    assert np.isfinite(out).all()


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        PoolScaler().fit(np.zeros((3, 2)), np.array([5, 5, 5]), 3, 0)


def test_no_clock_in_elliptic_sets():
    sets = elliptic_sets()
    for names in sets.values():
        assert_no_clock(names)
        assert "tx_f1" not in names  # the time-step feature
    assert len(sets["F_local"]) == 93 and len(sets["F_all"]) == 165
    assert set(sets["F_local"]) < set(sets["F_all"])


def test_clock_guard_triggers():
    with pytest.raises(AssertionError):
        assert_no_clock(["a", "month", "b"])
    with pytest.raises(AssertionError):
        assert_no_clock(["a", "fraud_bool"], target="fraud_bool")

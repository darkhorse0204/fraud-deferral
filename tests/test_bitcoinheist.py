"""Study 2 data layer: period arithmetic, clock exclusion, and the matured-blacklist feature obeys label maturity."""
import numpy as np

from src.data.bitcoinheist import SPLIT, matured_blacklist, period_of


def test_period_arithmetic():
    assert period_of(np.array([2011]), np.array([1]))[0] == 0
    assert period_of(np.array([2011]), np.array([28]))[0] == 0
    assert period_of(np.array([2011]), np.array([29]))[0] == 1
    assert period_of(np.array([2011]), np.array([365]))[0] == 12          # day 365 folds into the 13th block
    assert period_of(np.array([2015]), np.array([1]))[0] == 52
    assert SPLIT["dev"][1] + 1 == SPLIT["test"][0]


def test_blacklist_uses_only_matured_labels():
    rng = np.random.default_rng(0)
    n = 4000
    addr = rng.integers(0, 600, n)
    period = rng.integers(0, 40, n)
    y = (rng.random(n) < 0.15).astype(int)
    D = 2
    f = matured_blacklist(addr, period, y, D)
    # definition: an earlier positive of the same address with period_pos + D < period
    for i in rng.choice(n, 300, replace=False):
        same = (addr == addr[i]) & (y == 1) & (period + D < period[i])
        assert f[i] == float(same.any())
    # flipping every label that has NOT matured for a target period cannot change that period's features
    t = 25
    y2 = y.copy()
    imm = period >= t - D
    y2[imm] = 1 - y2[imm]
    f2 = matured_blacklist(addr, period, y2, D)
    assert (f[period == t] == f2[period == t]).all()


def test_blacklist_never_flags_first_or_same_period_positive():
    addr = np.array([1, 1, 1, 2]); period = np.array([5, 6, 9, 5]); y = np.array([1, 1, 0, 0])
    assert matured_blacklist(addr, period, y, 2).tolist() == [0.0, 0.0, 1.0, 0.0]

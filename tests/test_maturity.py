"""L5, L12: a label may never be used for a decision at or before its maturity."""
import numpy as np
import pytest

from src.data.maturity import pool_mask, window_mask


@pytest.mark.parametrize("D", [0, 1, 2, 4])
def test_pool_never_contains_current_or_future(D):
    rng = np.random.default_rng(0)
    periods = rng.integers(1, 50, size=5000)
    for t in range(1, 55):
        m = pool_mask(periods, t, D)
        assert (periods[m] + D < t).all()
        assert (periods[m] < t).all()  # D=0: still strictly earlier
        # completeness: every event that has matured IS included
        assert m.sum() == (periods + D < t).sum()


def test_d0_excludes_same_period():
    periods = np.array([3, 3, 4, 5])
    assert pool_mask(periods, 4, 0).tolist() == [True, True, False, False]


def test_delay_shrinks_pool_monotonically():
    periods = np.repeat(np.arange(1, 30), 10)
    sizes = [pool_mask(periods, 20, D).sum() for D in (0, 2, 4)]
    assert sizes[0] > sizes[1] > sizes[2]


def test_window_inside_pool_and_bounded():
    periods = np.repeat(np.arange(1, 40), 7)
    for D in (0, 2, 4):
        for W in (1, 3, 5, 8):
            for t in range(2, 45):
                w = window_mask(periods, t, D, W)
                p = pool_mask(periods, t, D)
                assert (w <= p).all()  # window is a subset of the pool
                if w.any():
                    assert periods[w].max() <= t - D - 1
                    assert periods[w].min() >= t - D - W


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        pool_mask(np.array([1, 2]), 3, -1)

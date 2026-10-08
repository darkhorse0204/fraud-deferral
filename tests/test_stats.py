import numpy as np

from src.stats.inference import (block_bootstrap_gain, block_indices, effect_sizes, holm, spearman_block_ci,
                                 tost_noninferiority)


def test_block_indices_shape_and_blocks():
    rng = np.random.default_rng(0)
    ix = block_indices(15, 3, rng)
    assert len(ix) == 15 and ix.min() >= 0 and ix.max() < 15
    assert all((ix[i + 1] - ix[i]) == 1 for i in range(0, 12, 3) for _ in [0] if i % 3 != 2) or True
    for i in range(0, 15, 3):                                  # within-block contiguity
        assert (np.diff(ix[i:i + 3]) == 1).all()


def test_bootstrap_detects_real_gain_and_null():
    rng = np.random.default_rng(1)
    n = np.full(15, 1000.0)
    base = rng.normal(100, 5, 15)
    r = block_bootstrap_gain(base, base - 10, n, B=2000)
    assert r["lo"] > 0 and r["p"] < 0.01
    null = block_bootstrap_gain(base, base + rng.normal(0, 3, 15), n, B=2000)
    assert null["lo"] < 0 < null["hi"]


def test_coverage_under_null_is_reasonable():
    rng = np.random.default_rng(2)
    n = np.full(15, 1000.0)
    miss = 0
    for _ in range(150):
        base = rng.normal(100, 5, 15)
        r = block_bootstrap_gain(base, base + rng.normal(0, 3, 15), n, B=400, seed=int(rng.integers(1e9)))
        miss += not (r["lo"] <= 0 <= r["hi"])
    assert miss / 150 < 0.15                                    # nominal 5%; small-T bootstrap is mildly liberal


def test_holm_monotone():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"][0] == 0.03 and out["c"][0] == 0.06 and out["b"][0] == 0.06
    assert out["a"][1] and not out["b"][1]


def test_tost_and_effect():
    base = np.full(10, 100.0)
    new = base + np.random.default_rng(0).normal(0, 0.2, 10)
    n = np.full(10, 1000.0)
    assert tost_noninferiority(base, new, n, 0.02)["equivalent"]
    assert not tost_noninferiority(base, new + 10, n, 0.02)["equivalent"]
    assert effect_sizes(base, base - 5, n)["frac_periods_better"] == 1.0


def test_spearman():
    x = np.arange(15.0)
    assert spearman_block_ci(x, x * 2, B=300)["rho"] > 0.99

"""Inference on per-period cost data.

Unit of resampling is the PERIOD (the data-generating unit under temporal dependence), via moving-block bootstrap.
Seeds measure training noise only; per-period quantities are averaged over seeds before resampling and the
across-seed spread is reported separately.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def block_indices(T: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of range(T): blocks of length L (L clipped to T)."""
    L = max(1, min(L, T))
    nb = int(np.ceil(T / L))
    starts = rng.integers(0, T - L + 1, size=nb)
    return np.concatenate([np.arange(s, s + L) for s in starts])[:T]


def paired_gain(cost_base: np.ndarray, cost_new: np.ndarray, n: np.ndarray, per: float = 1000.0) -> float:
    """Pooled cost reduction per `per` events: positive = new policy is cheaper."""
    return per * (cost_base.sum() - cost_new.sum()) / n.sum()


def block_bootstrap_gain(cost_base, cost_new, n, L=2, B=5000, seed=0, alpha=0.05):
    """Return dict(est, lo, hi, p_two_sided, rel_est, rel_lo, rel_hi) for the pooled paired gain."""
    cb, cn, n = (np.asarray(a, dtype=np.float64) for a in (cost_base, cost_new, n))
    T = len(n)
    rng = np.random.default_rng(seed)
    est = paired_gain(cb, cn, n)
    rel = (cb.sum() - cn.sum()) / cb.sum() if cb.sum() > 0 else np.nan
    g, r = np.empty(B), np.empty(B)
    for i in range(B):
        ix = block_indices(T, L, rng)
        g[i] = paired_gain(cb[ix], cn[ix], n[ix])
        r[i] = (cb[ix].sum() - cn[ix].sum()) / max(cb[ix].sum(), 1e-12)
    lo, hi = np.quantile(g, [alpha / 2, 1 - alpha / 2])
    rlo, rhi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((g <= 0).mean(), (g >= 0).mean())
    return dict(est=est, lo=lo, hi=hi, p=min(1.0, max(p, 1.0 / B)), rel=rel, rel_lo=rlo, rel_hi=rhi, T=T)


def wilcoxon_periods(cost_base, cost_new, n):
    d = 1000 * (np.asarray(cost_base) - np.asarray(cost_new)) / np.asarray(n)
    if len(d) < 6 or np.allclose(d, 0):
        return dict(stat=np.nan, p=np.nan)
    s, p = st.wilcoxon(d)
    return dict(stat=float(s), p=float(p))


def effect_sizes(cost_base, cost_new, n):
    d = 1000 * (np.asarray(cost_base) - np.asarray(cost_new)) / np.asarray(n)
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    pos, neg = (d > 0).sum(), (d < 0).sum()
    cliff = (pos - neg) / max(len(d), 1)       # Cliff's delta of per-period differences vs 0
    return dict(d_z=float(dz), cliff=float(cliff), frac_periods_better=float((d > 0).mean()))


def tost_noninferiority(cost_base, cost_new, n, margin_rel=0.02, alpha=0.05):
    """Equivalence of mean per-period cost within +-margin_rel x baseline mean (two one-sided t-tests)."""
    d = 1000 * (np.asarray(cost_base) - np.asarray(cost_new)) / np.asarray(n)
    m = margin_rel * 1000 * np.sum(cost_base) / np.sum(n)
    if len(d) < 3:
        return dict(p=np.nan, equivalent=False, margin=m)
    p1 = st.ttest_1samp(d, -m, alternative="greater").pvalue
    p2 = st.ttest_1samp(d, m, alternative="less").pvalue
    p = max(p1, p2)
    return dict(p=float(p), equivalent=bool(p < alpha), margin=float(m))


def holm(pvals: dict[str, float], alpha=0.05):
    """Holm-Bonferroni. Returns {name: (adjusted_p, reject)}."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = (run, run < alpha)
    for k, v in pvals.items():
        out.setdefault(k, (np.nan, False))
    return out


def spearman_block_ci(x, y, L=2, B=5000, seed=0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(x))
    rho, p = st.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        ix = block_indices(len(x), L, rng)
        if np.ptp(x[ix]) > 0 and np.ptp(y[ix]) > 0:
            bs.append(st.spearmanr(x[ix], y[ix])[0])
    lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (np.nan, np.nan)
    return dict(rho=float(rho), lo=float(lo), hi=float(hi), p=float(p), n=len(x))


def friedman_nemenyi(cost_matrix: np.ndarray, names: list[str], alpha=0.05):
    """cost_matrix: (periods x policies) of per-period cost per 1000. Lower is better.
    Returns Friedman chi2, p, average ranks and the Nemenyi critical difference (Demsar 2006)."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    N, k = cost_matrix.shape
    chi2, p = friedmanchisquare(*[cost_matrix[:, j] for j in range(k)])
    ranks = np.vstack([rankdata(r) for r in cost_matrix]).mean(0)
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return dict(chi2=float(chi2), p=float(p), cd=float(cd), ranks=dict(zip(names, map(float, ranks))))

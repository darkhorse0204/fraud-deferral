"""Cost and calibration metrics. Actions: 0 = approve, 1 = review, 2 = block."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

APPROVE, REVIEW, BLOCK = 0, 1, 2


@dataclass(frozen=True)
class Costs:
    C_FN: float          # loss when a fraud is approved
    C_FP: float          # loss when a legitimate event is blocked
    c_r: float = 1.0     # cost of one review
    rho: float = 1.0     # probability the analyst resolves a reviewed event correctly

    def key(self) -> str:
        return f"FN{self.C_FN:g}_FP{self.C_FP:g}_r{self.c_r:g}_rho{self.rho:g}"

    @property
    def degenerate(self) -> bool:
        """Review band empty: reviewing is never cheaper than the best approve/block decision."""
        a, b = _band(self)
        return a >= b


def _band(c: "Costs") -> tuple[float, float]:
    """Review beats approve iff p > a; review beats block iff p < b (derived from the loss table)."""
    a = c.c_r / (c.rho * c.C_FN) if c.rho > 0 else float("inf")
    b = (c.C_FP - c.c_r) / (c.C_FP + (1 - c.rho) * c.C_FN)
    return a, b


GRID = [Costs(fn, fp) for fn in (10.0, 30.0, 100.0) for fp in (1.0, 5.0, 20.0)]
PRIMARY = Costs(30.0, 5.0)


def action_costs(y: np.ndarray, c: Costs):
    """Per-event cost of each action: (approve, review, block). y in {0,1}."""
    y = np.asarray(y, dtype=np.float64)
    return y * c.C_FN, c.c_r + (1 - c.rho) * y * c.C_FN, (1 - y) * c.C_FP


def realised_cost(actions: np.ndarray, y: np.ndarray, c: Costs) -> np.ndarray:
    ca, cr, cb = action_costs(y, c)
    return np.where(actions == APPROVE, ca, np.where(actions == REVIEW, cr, cb))


def chow_thresholds(c: Costs) -> tuple[float, float]:
    """Bayes thresholds for calibrated p: approve if p < a; block if p >= b; review in between.
    Returns (t, t) with t = C_FP/(C_FP+C_FN) (the two-action rule) when the review band is empty."""
    a, b = _band(c)
    if a >= b:
        t = c.C_FP / (c.C_FP + c.C_FN)
        return t, t
    return a, b


def ece(p, y, bins: int = 15) -> float:
    """Equal-mass-bin ECE."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p)
    tot = 0.0
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            tot += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return tot / len(p)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y)) ** 2)) if len(p) else float("nan")


def _cell_ce(p, y, cells: list[np.ndarray]) -> float:
    n = sum(len(c) for c in cells)
    return sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in cells if len(c)) / max(n, 1)


def ce_u_raw(p, u, y, pbins: int = 5, ubins: int = 3) -> float:
    """Incremental conditional calibration error: CE over (p-bin x u-stratum) cells MINUS CE over p-bins alone.
    p-bins are global equal-mass bins; each is split into u-terciles, so the cell partition REFINES the p partition
    and the difference is >= 0 (triangle inequality). It is ~0 iff u carries no label information beyond p
    (Proposition 1's quantity); finite-sample bias is removed by the permutation correction in `ce_u`."""
    p, u, y = (np.asarray(a, dtype=np.float64) for a in (p, u, y))
    if len(p) < pbins * ubins * 4:
        return float("nan")
    pbin = np.array_split(np.argsort(p, kind="stable"), pbins)
    coarse = _cell_ce(p, y, pbin)
    fine = []
    for idx in pbin:
        for sub in np.array_split(idx[np.argsort(u[idx], kind="stable")], ubins):
            fine.append(sub)
    return _cell_ce(p, y, fine) - coarse


def ce_u(p, u, y, pbins: int = 5, ubins: int = 3, perms: int = 20, seed: int = 0) -> float:
    """Bias-corrected incremental conditional calibration error: ce_u_raw minus its mean under random permutations
    of u (the null where u is uninformative). Can be slightly negative; its expectation is 0 under the null."""
    raw = ce_u_raw(p, u, y, pbins, ubins)
    if not np.isfinite(raw):
        return raw
    rng = np.random.default_rng(seed)
    u = np.asarray(u)
    null = np.mean([ce_u_raw(p, u[rng.permutation(len(u))], y, pbins, ubins) for _ in range(perms)])
    return float(raw - null)


def aurc(u, correct) -> float:
    """Area under the risk-coverage curve: abstain on highest-u first."""
    order = np.argsort(np.asarray(u))
    err = 1.0 - np.asarray(correct, dtype=np.float64)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1))) if len(err) else float("nan")

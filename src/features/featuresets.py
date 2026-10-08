"""Feature sets and pool-scoped scalers.

Elliptic: the Kaggle file has txId + 166 columns; the first feature is the TIME STEP itself.
It is excluded from every feature set (leak L11: a model that sees the clock learns the clock).
  F_local = features 2..94   (93 local features)
  F_all   = features 2..166  (93 local + 72 provider-aggregated one-hop features)
BAF: all columns except the target and `month`.
"""
from __future__ import annotations

import numpy as np

from src.data.maturity import pool_mask

ELLIPTIC_TIME_FEATURE = "tx_f1"
FORBIDDEN_NAMES = {"month", "timestep", "time_step", "period", ELLIPTIC_TIME_FEATURE}


def elliptic_feature_names() -> list[str]:
    return [f"tx_f{k}" for k in range(1, 167)]


def elliptic_sets() -> dict[str, list[str]]:
    names = elliptic_feature_names()
    local = names[1:94]       # tx_f2 .. tx_f94
    allf = names[1:]          # tx_f2 .. tx_f166
    assert len(local) == 93 and len(allf) == 165
    return {"F_local": local, "F_all": allf}


def assert_no_clock(feature_names: list[str], target: str | None = None) -> None:
    bad = [n for n in feature_names if n.lower() in FORBIDDEN_NAMES or (target and n == target)]
    if bad:
        raise AssertionError(f"clock/target columns in feature set: {bad}")


class PoolScaler:
    """Standardiser fitted ONLY on rows selected by pool_mask(periods, t, D)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fit_rows_: int = 0
        self.max_period_: int | None = None

    def fit(self, X: np.ndarray, periods: np.ndarray, t: int, D: int) -> "PoolScaler":
        m = pool_mask(periods, t, D)
        if not m.any():
            raise ValueError("empty training pool")
        Xp = X[m]
        self.mean_ = Xp.mean(axis=0)
        std = Xp.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        self.fit_rows_ = int(m.sum())
        self.max_period_ = int(np.asarray(periods)[m].max())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("scaler not fitted")
        return (X - self.mean_) / self.std_

"""Label-maturity access: the ONLY sanctioned way to select training/calibration labels.

Convention (fixes an off-by-one in the first spec draft):
  An event from period s has its label observed at the END of period s + D.
  A decision made in period t can therefore use event i iff  s_i + D < t,
  i.e.  s_i <= t - D - 1.   D = 0 means "labels arrive by the next period"; the label
  of an event in period t is NEVER usable for any decision in period t.
"""
from __future__ import annotations

import numpy as np


def pool_mask(periods: np.ndarray, t: int, D: int) -> np.ndarray:
    """Boolean mask of events whose labels are available when deciding in period t."""
    if D < 0:
        raise ValueError("label delay D must be >= 0")
    periods = np.asarray(periods)
    mask = periods + D < t
    assert not mask[periods >= t].any(), "maturity violation: label from current/future period"
    return mask


def window_mask(periods: np.ndarray, t: int, D: int, W: int) -> np.ndarray:
    """Matured events from the most recent W periods (rolling calibration window)."""
    if W < 1:
        raise ValueError("window W must be >= 1")
    periods = np.asarray(periods)
    m = pool_mask(periods, t, D) & (periods >= t - D - W)
    return m


def max_label_period_used(periods: np.ndarray, mask: np.ndarray) -> int | None:
    """Latest event period contained in `mask` (for audit logging)."""
    sel = np.asarray(periods)[mask]
    return int(sel.max()) if sel.size else None

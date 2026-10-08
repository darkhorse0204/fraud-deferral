"""Recalibration maps fitted on a matured window (weighted). Affine-logit (Platt) is primary: it can absorb both
miscalibration slope and prevalence shift, which temperature scaling alone cannot (spec amendment M12)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

EPS = 1e-4


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Identity:
    def __call__(self, p):
        return np.asarray(p, dtype=np.float64)


class Affine:
    """p_cal = sigmoid(a * logit(p) + b)."""

    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    def __call__(self, p):
        return sigmoid(self.a * logit(np.asarray(p, dtype=np.float64)) + self.b)


class Iso:
    def __init__(self, ir):
        self.ir = ir

    def __call__(self, p):
        return np.clip(self.ir.predict(np.asarray(p, dtype=np.float64)), EPS, 1 - EPS)


def fit_affine(p, y, w=None, temp_only=False, ridge=1e-2, min_pos=20, min_neg=20):
    """Weighted logistic regression of y on logit(p). Returns None when the window is too thin."""
    y = np.asarray(y, dtype=np.float64)
    if (y == 1).sum() < min_pos or (y == 0).sum() < min_neg:
        return None
    x = logit(np.asarray(p, dtype=np.float64))
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    a, b = 1.0, 0.0
    for _ in range(30):  # damped Newton on the ridge-regularised weighted NLL (prior: a=1, b=0)
        z = a * x + b
        s = sigmoid(z)
        r = w * (s - y)
        ga, gb = (r * x).sum() + ridge * (a - 1), (0.0 if temp_only else r.sum() + ridge * b)
        v = w * s * (1 - s)
        haa, hab, hbb = (v * x * x).sum() + ridge, (v * x).sum(), v.sum() + ridge
        if temp_only:
            da, db = ga / haa, 0.0
        else:
            det = haa * hbb - hab * hab
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a_new, b_new = a - da, b - db
        if abs(a_new - a) + abs(b_new - b) < 1e-8:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:  # a<=0 would invert the ranking; refuse
        return None
    return Affine(a, b)


def fit_isotonic(p, y, w=None, min_pos=20, min_neg=20):
    y = np.asarray(y)
    if (y == 1).sum() < min_pos or (y == 0).sum() < min_neg:
        return None
    ir = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True)
    ir.fit(np.asarray(p, dtype=np.float64), y, sample_weight=w)
    return Iso(ir)


def fit_map(kind: str, p, y, w=None):
    if kind == "none":
        return Identity()
    if kind == "affine":
        return fit_affine(p, y, w)
    if kind == "temp":
        return fit_affine(p, y, w, temp_only=True)
    if kind == "isotonic":
        return fit_isotonic(p, y, w)
    raise ValueError(kind)

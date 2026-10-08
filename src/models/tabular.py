"""Tabular scorers. Every scorer returns (p, u): mean member probability and epistemic variance across members.

u is the variance of member probabilities (primary, uniform across scorers; spec amendment M11, replacing mutual
information so that RF trees and bagged GBDTs are comparable). LR has no ensemble -> u = 0.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.prep import NanStandardizer


def _pos_weight(y: np.ndarray, mode: float) -> float:
    pos = max(int((y == 1).sum()), 1)
    return float(((len(y) - pos) / pos) ** mode)


def fit_predict_lr(Xtr, ytr, Xte, seed: int, hp: dict):
    sc = NanStandardizer().fit(Xtr)
    clf = LogisticRegression(C=hp.get("C", 1.0), class_weight="balanced" if hp.get("balanced", True) else None,
                             max_iter=500, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1].astype(np.float32)
    return p, np.zeros_like(p)


def fit_predict_rf(Xtr, ytr, Xte, seed: int, hp: dict):
    sc = NanStandardizer().fit(Xtr)  # RF is scale-free, but it cannot take NaNs
    rf = RandomForestClassifier(n_estimators=hp.get("n_estimators", 300), max_depth=hp.get("max_depth", None),
                                min_samples_leaf=hp.get("min_samples_leaf", 2),
                                max_features=hp.get("max_features", "sqrt"), class_weight="balanced_subsample",
                                max_samples=hp.get("max_samples"),   # None (default) = all rows, as in Study 1
                                n_jobs=-1, random_state=seed)
    rf.fit(sc.transform(Xtr), ytr)
    Z = sc.transform(Xte)
    tree_p = np.stack([t.predict_proba(Z)[:, 1] for t in rf.estimators_])  # (T, n)
    return tree_p.mean(0).astype(np.float32), tree_p.var(0).astype(np.float32)


def fit_predict_xgb(Xtr, ytr, Xte, seed: int, hp: dict, M: int = 5):
    """Seed-bagged GBDT ensemble: M members with different seeds and row/column subsampling."""
    ps = []
    spw = _pos_weight(ytr, hp.get("pos_weight_pow", 0.5))
    dtr, dte = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xte)
    for m in range(M):
        params = dict(objective="binary:logistic", tree_method="hist", eta=hp.get("eta", 0.1),
                      max_depth=hp.get("max_depth", 6), subsample=hp.get("subsample", 0.8),
                      colsample_bytree=hp.get("colsample", 0.8), min_child_weight=hp.get("min_child_weight", 1),
                      reg_lambda=hp.get("lambda", 1.0), scale_pos_weight=spw, seed=seed * 1000 + m, nthread=8,
                      eval_metric="logloss", device=hp.get("device", "cpu"))   # cpu (default) in Study 1
        bst = xgb.train(params, dtr, num_boost_round=hp.get("rounds", 200))
        ps.append(bst.predict(dte))
    ps = np.stack(ps)
    return ps.mean(0).astype(np.float32), ps.var(0).astype(np.float32)


def fit_predict_delayed(Xtr, ytr, ptr, Xte, seed: int, hp: dict):
    """Delayed-feedback ensemble (re-implementation of the IDEA in Dal Pozzolo et al. 2015, not a faithful
    replication): one GBDT bag on the most recent `recent_periods` periods, one on all older data; scores averaged.
    `ptr` are the training rows' periods."""
    cut = ptr.max() - hp.get("recent_periods", 3) + 1
    recent, old = ptr >= cut, ptr < cut
    if recent.sum() < 50 or old.sum() < 50 or ytr[recent].sum() < 3 or ytr[old].sum() < 3:
        return fit_predict_xgb(Xtr, ytr, Xte, seed, hp, M=hp.get("M", 5))
    pr, _ = fit_predict_xgb(Xtr[recent], ytr[recent], Xte, seed, hp, M=3)
    po, _ = fit_predict_xgb(Xtr[old], ytr[old], Xte, seed + 7, hp, M=3)
    p = 0.5 * (pr + po)
    return p.astype(np.float32), (0.25 * (pr - po) ** 2).astype(np.float32)

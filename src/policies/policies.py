"""Decision policies over cached score tables.

Every policy sees, for a decision at period t:
  * `cur`: raw scores (p, u) of the events being decided (NO labels);
  * `win`: the matured calibration window -- events whose labels satisfy src.data.maturity.window_mask;
  * the cost parameters.
Labels of period >= t - D are never available to a policy; `Table.window` is the only label accessor and asserts it.
Policy IDs: pi0 naive threshold | pi1 cost-optimal single threshold | pi2 Chow band on recalibrated p |
pi3 Chow + epistemic review cap | pi4 MAD | pi4_nodelta (ablation A1) | pi4_noshrink | pi5 suppress-on-uncertainty
(the source report's rule; negative control) | pi6 conformal deferral | pi7 oracle (diagnostic bound only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.calibration.recal import Identity, fit_map
from src.data.maturity import window_mask
from src.evaluation.metrics import APPROVE, BLOCK, REVIEW, Costs, action_costs, chow_thresholds

MA = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0])      # multipliers on the Chow approve/review threshold
MB = np.array([0.25, 0.5, 1.0, 2.0, 4.0])                  # multipliers on (1 - block threshold)
QS = np.array([0.7, 0.8, 0.9, 0.95, 0.99, 1.0])            # quantile of window-u used as epistemic cap (1.0 = never)


@dataclass
class PolicySpec:
    name: str                       # policy id (see module docstring)
    W: int = 5                      # calibration window length (periods)
    h: float = np.inf               # exponential half-life of window weights (periods); inf = uniform
    lam: float = 0.02               # shrinkage toward Chow thresholds (cost units per event per |log multiplier|)
    eps: float = np.inf             # max |change| of log multipliers between consecutive periods
    min_gain: float = 0.0           # relative window-cost improvement over the Chow candidate required to leave Chow
    calib: str = "affine"           # affine | temp | isotonic | none
    theta0: float = 0.5             # pi0 raw-score threshold (chosen on dev)
    alpha: float = 0.1              # pi6 conformal miscoverage
    extra: dict = field(default_factory=dict)

    def label(self) -> str:
        return self.name


class Table:
    """Score table for one (scorer, config): rows (seed, origin, rid, period, y, p, u). Pre-indexes by seed/origin."""

    def __init__(self, df: pd.DataFrame, D: int):
        self.D = D
        self.groups: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for (seed, origin), g in df.groupby(["seed", "origin"], sort=True):
            g = g.sort_values(["period", "rid"], kind="stable")
            self.groups[(int(seed), int(origin))] = {c: g[c].to_numpy() for c in ("rid", "period", "y", "p", "u")}
        self.seeds = sorted({k[0] for k in self.groups})
        self.origins = sorted({k[1] for k in self.groups})

    def origin_for(self, t: int) -> int:
        cands = [o for o in self.origins if o <= t]
        if not cands:
            raise KeyError(f"no origin <= {t}")
        return max(cands)

    def current(self, seed: int, t: int):
        g = self.groups[(seed, self.origin_for(t))]
        m = g["period"] == t
        return {k: g[k][m] for k in ("rid", "y", "p", "u")}  # y kept for EVALUATION only; policies never read cur['y']

    def window(self, seed: int, t: int, W: int, h: float):
        g = self.groups[(seed, self.origin_for(t))]
        m = window_mask(g["period"], t, self.D, W)           # <- the only label access path; asserts maturity
        assert (g["period"][m] + self.D < t).all()
        s = g["period"][m]
        age = (t - self.D - 1) - s
        w = np.ones(len(s)) if not np.isfinite(h) else 2.0 ** (-age / h)
        return {"p": g["p"][m], "u": g["u"][m], "y": g["y"][m], "w": w, "period": s}


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if q >= 1.0 or len(x) == 0:
        return np.inf
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(np.interp(q * cw[-1], cw, x[o]))


def _chow_actions(pc, a, b):
    return np.where(pc < a, APPROVE, np.where(pc >= b, BLOCK, REVIEW))


def _cost_vec(pc, u, a, b, delta, ca, cr, cb, suppress=False):
    unc = u > delta  # strict: constant u (e.g. LR, u=0) must never trigger deferral
    normal = np.where(pc < a, ca, np.where(pc >= b, cb, cr))
    return np.where(unc, ca if suppress else cr, normal)


def _grid_search(win, pc_w, cost: Costs, ma_grid, mb_grid, q_grid, lam, prev, eps, suppress=False):
    a0, b0 = chow_thresholds(cost)
    ca, cr, cb = action_costs(win["y"], cost)
    w = win["w"] / win["w"].sum()
    deltas = [wquantile(win["u"], win["w"], q) for q in q_grid]
    degenerate = a0 >= b0
    best, best_obj = (1.0, 1.0, np.inf), np.inf
    for ma in ma_grid:
        if abs(np.log(ma) - prev[0]) > eps + 1e-12:
            continue
        for mb in ([1.0] if degenerate else mb_grid):
            if not degenerate and abs(np.log(mb) - prev[1]) > eps + 1e-12:
                continue
            a = a0 * ma
            b = a if degenerate else 1.0 - (1.0 - b0) * mb
            if a > b:
                continue
            pen = lam * (abs(np.log(ma)) + (0.0 if degenerate else abs(np.log(mb))))
            for q, dl in zip(q_grid, deltas):
                obj = float((w * _cost_vec(pc_w, win["u"], a, b, dl, ca, cr, cb, suppress)).sum()) + pen
                if obj < best_obj - 1e-12:
                    best_obj, best = obj, (ma, mb, dl)
    if not np.isfinite(best_obj):  # nothing admissible under the move limit -> stay at previous multipliers
        return float(np.exp(prev[0])), float(np.exp(prev[1])), np.inf
    return best


def decide(spec: PolicySpec, cost: Costs, win, cur, state: dict):
    """Return (actions for `cur`, calibrated p of cur, new state). `state` carries fallbacks across periods."""
    kind = spec.name
    a0, b0 = chow_thresholds(cost)
    pc_cur_raw = cur["p"]
    if kind == "pi0":
        return np.where(pc_cur_raw >= spec.theta0, BLOCK, APPROVE), pc_cur_raw, state

    g = None
    if len(win["p"]) and kind != "pi7":
        g = fit_map(spec.calib, win["p"], win["y"], win["w"])
    if g is None:
        g = state.get("g", Identity())
    state = dict(state, g=g)
    pc_w, pc = g(win["p"]) if len(win["p"]) else win["p"], g(cur["p"])
    prev = state.get("mult", (0.0, 0.0))                      # log multipliers (ma, mb)

    if kind == "pi1":
        t = cost.C_FP / (cost.C_FP + cost.C_FN)
        return np.where(pc >= t, BLOCK, APPROVE), pc, state
    if kind == "pi2":
        return _chow_actions(pc, a0, b0), pc, state

    thin = len(win["p"]) == 0 or win["y"].sum() < 20
    if kind in ("pi3", "pi5"):
        suppress = kind == "pi5"
        if thin:
            return _chow_actions(pc, a0, b0), pc, state
        _, _, dl = _grid_search(win, pc_w, cost, [1.0], [1.0], QS, 0.0, (0.0, 0.0), np.inf, suppress)
        act = _chow_actions(pc, a0, b0)
        unc = cur["u"] > dl
        act = np.where(unc, APPROVE if suppress else REVIEW, act)
        return act, pc, state
    if kind in ("pi4", "pi4_nodelta", "pi4_noshrink"):
        if thin:
            return _chow_actions(pc, a0, b0), pc, state
        q_grid = np.array([1.0]) if kind == "pi4_nodelta" else QS
        lam = 0.0 if kind == "pi4_noshrink" else spec.lam
        ma, mb, dl = _grid_search(win, pc_w, cost, MA, MB, q_grid, lam, prev, spec.eps)
        if spec.min_gain > 0:                                  # leave Chow only for a material in-window improvement
            ca_, cr_, cb_ = action_costs(win["y"], cost)
            w_ = win["w"] / win["w"].sum()
            chow_obj = float((w_ * _cost_vec(pc_w, win["u"], a0, b0, np.inf, ca_, cr_, cb_)).sum())
            bb = a0 * ma if a0 >= b0 else 1.0 - (1.0 - b0) * mb
            alt_obj = float((w_ * _cost_vec(pc_w, win["u"], a0 * ma, bb, dl, ca_, cr_, cb_)).sum())
            if chow_obj - alt_obj < spec.min_gain * chow_obj:
                ma, mb, dl = 1.0, 1.0, np.inf
        a = a0 * ma
        b = a if a0 >= b0 else 1.0 - (1.0 - b0) * mb
        act = _chow_actions(pc, a, b)
        act = np.where(cur["u"] > dl, REVIEW, act)
        state["mult"] = (float(np.log(ma)), float(np.log(mb)))
        return act, pc, state
    if kind == "pi6":
        if thin:
            return _chow_actions(pc, a0, b0), pc, state
        pcy = np.where(win["y"] == 1, pc_w, 1 - pc_w)       # probability assigned to the true class
        q = wquantile(1 - pcy, win["w"], min(1.0, (1 - spec.alpha) * (1 + 1 / max(len(pcy), 1))))
        in1, in0 = (1 - pc) <= q, pc <= q
        act = np.where(in1 & in0, REVIEW, np.where(in1, BLOCK, np.where(in0, APPROVE, REVIEW)))
        return act, pc, state
    if kind == "pi7":                                        # ORACLE: uses current labels. Diagnostic bound only.
        ca, cr, cb = action_costs(cur["y"], cost)
        o = np.argsort(pc_cur_raw, kind="stable")
        pa = np.concatenate([[0.0], np.cumsum(ca[o])])        # cost of approving the i lowest-score events
        pr_ = np.concatenate([[0.0], np.cumsum(cr[o])])
        pb = np.concatenate([[0.0], np.cumsum(cb[o])])
        n = len(o)
        grid = np.unique(np.linspace(0, n, 201).astype(int))
        ia, ib = np.meshgrid(grid, grid, indexing="ij")
        ok = ia <= ib                                        # approve [0,ia), review [ia,ib), block [ib,n)
        tot = pa[ia] + (pr_[ib] - pr_[ia]) + (pb[n] - pb[ib])
        tot = np.where(ok, tot, np.inf)
        k = np.unravel_index(np.argmin(tot), tot.shape)
        act = np.empty(n, dtype=int)
        act[o[:ia[k]]] = APPROVE
        act[o[ia[k]:ib[k]]] = REVIEW
        act[o[ib[k]:]] = BLOCK
        return act, pc_cur_raw, state
    raise ValueError(kind)


def apply_capacity(actions, pc, cost: Costs, frac: float):
    """Review capacity K = ceil(frac * n) per period: lowest-priority reviews are demoted to the two-action rule."""
    k = int(np.ceil(frac * len(actions)))
    rev = np.flatnonzero(actions == REVIEW)
    if len(rev) <= k:
        return actions
    order = rev[np.argsort(-pc[rev])]                         # review the highest-risk first
    drop = order[k:]
    out = actions.copy()
    t = cost.C_FP / (cost.C_FP + cost.C_FN)
    out[drop] = np.where(pc[drop] >= t, BLOCK, APPROVE)
    return out


def run_suite(tab: Table, periods: range, specs: list[PolicySpec], costs: list[Costs], cap_frac: float | None = None,
              seeds: list[int] | None = None) -> pd.DataFrame:
    """Per (seed, period, policy, cost) counts and realised cost. Labels of `cur` are used ONLY here, for scoring."""
    rows = []
    for seed in (seeds or tab.seeds):
        for cost in costs:
            states = {s.label(): {} for s in specs}
            ca_cache = {}
            for t in periods:
                try:
                    cur = tab.current(seed, t)
                except KeyError:
                    continue
                if len(cur["p"]) == 0:
                    continue
                ca, cr, cb = ca_cache.setdefault(t, action_costs(cur["y"], cost))
                for spec in specs:
                    win = tab.window(seed, t, spec.W, spec.h) if spec.name != "pi0" else {"p": [], "u": [], "y": [], "w": []}
                    act, pc, st = decide(spec, cost, win, cur, states[spec.label()])
                    states[spec.label()] = st
                    if cap_frac is not None:
                        act = apply_capacity(act, pc, cost, cap_frac)
                    cst = np.where(act == APPROVE, ca, np.where(act == REVIEW, cr, cb))
                    y = cur["y"]
                    rows.append((seed, t, spec.label(), cost.key(), len(y), float(cst.sum()), int((act == REVIEW).sum()),
                                 int((act == BLOCK).sum()), int(((act == APPROVE) & (y == 1)).sum()),
                                 int(((act == BLOCK) & (y == 0)).sum()), int(y.sum()), int(((act == REVIEW) & (y == 1)).sum())))
    return pd.DataFrame(rows, columns=["seed", "period", "policy", "cost", "n", "cost_sum", "n_review", "n_block",
                                       "n_fn", "n_fp", "n_pos", "n_review_pos"])

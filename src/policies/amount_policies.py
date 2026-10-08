"""Amount-dependent decision policies (Study 2). Same logic as src/policies/policies.py, but every cost is per event:
    approve an illicit event  -> lose its amount A_i
    block a legitimate event  -> lose m * A_i   (margin / friction share of the amount)
    review                    -> c_r  (+ (1 - rho) * A_i if the analyst misses an illicit event)
so the Chow band is per event:  a_i = c_r / (rho * A_i),  b_i = (m A_i - c_r) / (m A_i + (1 - rho) A_i).
Where the band is empty (a_i >= b_i) the two-action threshold m / (1 + m) applies to that event.
With a constant amount A and m = C_FP / A this reduces exactly to the scalar policies (unit-tested).
Policies: pi1 single threshold | pi2 Chow band | pi3 Chow + cap on a signal u | pi4 MAD | pi4_nodelta | pi7 oracle.
Decision-layer settings are those frozen in Study 1 (isotonic recalibration, window 3, MAD lam/eps/min_gain).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.calibration.recal import Identity, fit_map
from src.data.maturity import window_mask
from src.policies.policies import MA, MB, QS, wquantile

APPROVE, REVIEW, BLOCK = 0, 1, 2


@dataclass(frozen=True)
class AmountCosts:
    c_r: float            # review cost, in the currency of the amounts
    m: float              # share of the amount lost when a legitimate event is blocked
    rho: float = 1.0

    def key(self) -> str:
        return f"cr{self.c_r:g}_m{self.m:g}_rho{self.rho:g}"


def band(A: np.ndarray, c: AmountCosts):
    """Per-event (a, b); events with an empty band get a = b = m / (1 + m)."""
    A = np.maximum(np.asarray(A, dtype=np.float64), 1e-12)
    a = c.c_r / (c.rho * A)
    b = (c.m * A - c.c_r) / (c.m * A + (1 - c.rho) * A)
    empty = a >= b
    t = c.m / (1.0 + c.m)
    return np.where(empty, t, a), np.where(empty, t, b), empty


def action_costs(y, A, c: AmountCosts):
    y = np.asarray(y, dtype=np.float64)
    return y * A, c.c_r + (1 - c.rho) * y * A, (1 - y) * c.m * A


def _acts(pc, a, b):
    return np.where(pc < a, APPROVE, np.where(pc >= b, BLOCK, REVIEW))


def _scaled(a, b, empty, ma, mb):
    """Apply MAD multipliers to the per-event band (approve threshold x ma; distance of block threshold from 1 x mb)."""
    a2 = a * ma
    b2 = np.where(empty, a2, 1.0 - (1.0 - b) * mb)
    b2 = np.maximum(b2, a2)
    return a2, b2


def _wcost(pc, u, a, b, delta, ca, cr, cb, w):
    normal = np.where(pc < a, ca, np.where(pc >= b, cb, cr))
    return float((w * np.where(u > delta, cr, normal)).sum())


class AmountTable:
    """Score table (seed, origin, rid, period, y, p, u) plus per-event amounts looked up by rid."""

    def __init__(self, df: pd.DataFrame, amount_by_rid: np.ndarray, D: int, ucol: str = "u"):
        self.D = D
        self.groups = {}
        for (seed, origin), g in df.groupby(["seed", "origin"], sort=True):
            g = g.sort_values(["period", "rid"], kind="stable")
            self.groups[(int(seed), int(origin))] = dict(rid=g["rid"].to_numpy(), period=g["period"].to_numpy(), y=g["y"].to_numpy(),
                                                         p=g["p"].to_numpy(dtype=np.float64), u=g[ucol].to_numpy(dtype=np.float64),
                                                         A=amount_by_rid[g["rid"].to_numpy()])
        self.seeds = sorted({k[0] for k in self.groups})
        self.origins = sorted({k[1] for k in self.groups})

    def origin_for(self, t):
        return max(o for o in self.origins if o <= t)

    def current(self, seed, t):
        g = self.groups[(seed, self.origin_for(t))]
        m = g["period"] == t
        return {k: g[k][m] for k in ("y", "p", "u", "A")}

    def window(self, seed, t, W):
        g = self.groups[(seed, self.origin_for(t))]
        m = window_mask(g["period"], t, self.D, W)            # the only label access path; asserts maturity
        return {k: g[k][m] for k in ("y", "p", "u", "A")}


def decide(kind: str, c: AmountCosts, win, cur, state: dict, hp: dict):
    g = fit_map(hp["calib"], win["p"], win["y"]) if len(win["p"]) else None
    if g is None:
        g = state.get("g", Identity())
    state = dict(state, g=g)
    pc = g(cur["p"])
    a, b, empty = band(cur["A"], c)
    if kind == "pi1":
        return np.where(pc >= c.m / (1 + c.m), BLOCK, APPROVE), state
    if kind == "pi2":
        return _acts(pc, a, b), state
    if kind == "pi7":                                          # oracle over MAD's own multiplier grid, on current labels (bound)
        ca, cr, cb = action_costs(cur["y"], cur["A"], c)
        best, bc = (1.0, 1.0), np.inf
        for ma in MA:
            for mb in MB:
                a2, b2 = _scaled(a, b, empty, ma, mb)
                v = np.where(pc < a2, ca, np.where(pc >= b2, cb, cr)).sum()
                if v < bc:
                    bc, best = v, (ma, mb)
        return _acts(pc, *_scaled(a, b, empty, *best)), state
    thin = len(win["p"]) == 0 or win["y"].sum() < 20
    if thin:
        return _acts(pc, a, b), state
    pw = g(win["p"])
    aw, bw, ew = band(win["A"], c)
    ca, cr, cb = action_costs(win["y"], win["A"], c)
    w = np.full(len(pw), 1.0 / len(pw))
    deltas = [wquantile(win["u"], np.ones(len(pw)), q) for q in QS]
    if kind == "pi3":
        best, bo = np.inf, np.inf                              # same order and strict-improvement rule as the scalar policies
        for dl in deltas:
            o = _wcost(pw, win["u"], aw, bw, dl, ca, cr, cb, w)
            if o < bo - 1e-12:
                bo, best = o, dl
        act = _acts(pc, a, b)
        return np.where(cur["u"] > best, REVIEW, act), state
    if kind in ("pi4", "pi4_nodelta"):
        prev = state.get("mult", (0.0, 0.0))
        qi = [len(QS) - 1] if kind == "pi4_nodelta" else list(range(len(QS)))
        best, bo = (1.0, 1.0, np.inf), np.inf
        for ma in MA:
            if abs(np.log(ma) - prev[0]) > hp["eps"] + 1e-12:
                continue
            for mb in MB:
                if abs(np.log(mb) - prev[1]) > hp["eps"] + 1e-12:
                    continue
                a2, b2 = _scaled(aw, bw, ew, ma, mb)
                pen = hp["lam"] * (abs(np.log(ma)) + abs(np.log(mb)))
                for i in qi:
                    o = _wcost(pw, win["u"], a2, b2, deltas[i], ca, cr, cb, w) + pen
                    if o < bo - 1e-12:
                        bo, best = o, (ma, mb, deltas[i])
        ma, mb, dl = best if np.isfinite(bo) else (float(np.exp(prev[0])), float(np.exp(prev[1])), np.inf)
        a2, b2 = _scaled(a, b, empty, ma, mb)
        state["mult"] = (float(np.log(ma)), float(np.log(mb)))
        return np.where(cur["u"] > dl, REVIEW, _acts(pc, a2, b2)), state
    raise ValueError(kind)


def run_suite(tab: AmountTable, periods, kinds: list[str], cells: list[AmountCosts], hp: dict, label_suffix: str = "") -> pd.DataFrame:
    rows = []
    for seed in tab.seeds:
        for c in cells:
            states = {k: {} for k in kinds}
            for t in periods:
                cur = tab.current(seed, t)
                if len(cur["p"]) == 0:
                    continue
                win = tab.window(seed, t, hp["W"])
                ca, cr, cb = action_costs(cur["y"], cur["A"], c)
                y = cur["y"]
                for k in kinds:
                    act, states[k] = decide(k, c, win, cur, states[k], hp)
                    cst = np.where(act == APPROVE, ca, np.where(act == REVIEW, cr, cb))
                    rows.append((seed, t, k + label_suffix, c.key(), len(y), float(cst.sum()), int((act == REVIEW).sum()), int((act == BLOCK).sum()),
                                 int(((act == APPROVE) & (y == 1)).sum()), int(((act == BLOCK) & (y == 0)).sum()), int(y.sum()),
                                 float(cur["A"][(act == APPROVE) & (y == 1)].sum()), float(cur["A"][y == 1].sum())))
    return pd.DataFrame(rows, columns=["seed", "period", "policy", "cost", "n", "cost_sum", "n_review", "n_block", "n_fn", "n_fp", "n_pos",
                                       "amt_missed", "amt_illicit"])

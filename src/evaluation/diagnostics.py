"""Per-period score-quality, calibration and conditional-calibration diagnostics (evaluation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibration.recal import Identity, fit_map
from src.evaluation.metrics import aurc, brier, ce_u, ece
from src.policies.policies import Table


def period_diagnostics(tab: Table, periods, W: int, h: float, calib: str = "affine") -> pd.DataFrame:
    rows = []
    for seed in tab.seeds:
        g_prev = Identity()
        for t in periods:
            try:
                cur = tab.current(seed, t)
            except KeyError:
                continue
            if len(cur["p"]) == 0:
                continue
            win = tab.window(seed, t, W, h)
            g = fit_map(calib, win["p"], win["y"], win["w"]) if len(win["p"]) else None
            g = g or g_prev
            g_prev = g
            p, u, y = cur["p"].astype(float), cur["u"].astype(float), cur["y"]
            pc = g(p)
            pos = int(y.sum())
            rows.append(dict(
                seed=seed, period=t, n=len(y), n_pos=pos, prev=float(y.mean()),
                auroc=float(roc_auc_score(y, p)) if 0 < pos < len(y) else np.nan,
                auprc=float(average_precision_score(y, p)) if pos > 0 else np.nan,
                brier_raw=brier(p, y), ece_raw=ece(p, y), brier_cal=brier(pc, y), ece_cal=ece(pc, y),
                ce_u_cal=ce_u(pc, u, y), mean_u=float(u.mean()), mean_p=float(p.mean()), mean_p_cal=float(pc.mean()),
                win_n=len(win["p"]), win_pos=int(np.sum(win["y"])) if len(win["p"]) else 0,
                aurc=aurc(u, (pc >= 0.5) == (y == 1)) if len(y) else np.nan))
    return pd.DataFrame(rows)

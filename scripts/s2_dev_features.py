"""DEV-ERA-ONLY exploration for Study 2 (periods <= 51; the test era is never read for labels here).
Question: do past-only features give a usable scorer?  Compares feature sets by dev walk-forward AUPRC with one
default XGBoost model.  Excluded on purpose: any same-day row count (legitimate addresses are capped at 1,000 per day,
so a day's row count would leak the number of positives)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import bitcoinheist as BH

D, WC, R = 2, 3, 6


def engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only features. Rows must be sorted by time within address."""
    o = df.sort_values(["addr", "t"], kind="stable")
    g = o.groupby("addr", sort=False)
    f = pd.DataFrame(index=o.index)
    f["addr_prev_rows"] = g.cumcount().astype("float32")
    f["addr_days_since_first"] = (o["t"] - g["t"].transform("first")).astype("float32")
    f["addr_prev_income_mean"] = ((g["income_btc"].cumsum() - o["income_btc"]) / f["addr_prev_rows"].clip(lower=1)).astype("float32")
    # matured blacklist: an earlier row of this address was labelled ransomware AND that label had matured (period <= p - D - 1)
    first_pos = o["period"].where(o["y"] == 1).groupby(o["addr"]).transform("min")
    f["addr_blacklisted_matured"] = (first_pos + D < o["period"]).astype("float32")
    x = o["income_btc"].to_numpy(dtype=np.float64)
    f["amt_frac_btc"] = (x % 1.0).astype("float32")
    f["amt_round_0p1"] = (np.abs(x * 10 - np.round(x * 10)) < 1e-6).astype("float32")
    f["amt_round_0p01"] = (np.abs(x * 100 - np.round(x * 100)) < 1e-6).astype("float32")
    return f.loc[df.index]


def main() -> None:
    df = pd.read_parquet(BH.PROC)
    yr = df["period"] // 13
    df["t"] = (df["period"].astype(np.int32) * 28)          # coarse time for ordering within address (period resolution)
    ds = BH.load_dataset()
    E = engineered(df)
    sets = {"base (11)": ds.X, "base + amount roundness": np.hstack([ds.X, E[["amt_frac_btc", "amt_round_0p1", "amt_round_0p01"]].to_numpy()]),
            "base + address history": np.hstack([ds.X, E[["addr_prev_rows", "addr_days_since_first", "addr_prev_income_mean"]].to_numpy()]),
            "base + matured blacklist": np.hstack([ds.X, E[["addr_blacklisted_matured"]].to_numpy()]),
            "all engineered": np.hstack([ds.X, E.to_numpy()])}
    per, y = ds.period, ds.y
    print("dev-era share of positives on a matured-blacklisted address:", round(float(E.loc[(per <= 51) & (y == 1), "addr_blacklisted_matured"].mean()), 3))
    for name, X in sets.items():
        aps, aucs, base = [], [], []
        for tau in (36, 42, 48):
            cut = tau - D - 1 - WC
            tr = per <= cut
            te = (per >= tau) & (per <= tau + R - 1)
            m = xgb.train(dict(objective="binary:logistic", tree_method="hist", eta=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=float(((~y[tr].astype(bool)).sum() / max(y[tr].sum(), 1)) ** 0.5), seed=0, nthread=8),
                          xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=200)
            p = m.predict(xgb.DMatrix(X[te]))
            aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p)); base.append(y[te].mean())
        print(f"{name:28s} dev AUPRC {np.mean(aps):.4f}  AUROC {np.mean(aucs):.4f}  (base rate {np.mean(base):.4f})", flush=True)


if __name__ == "__main__":
    main()

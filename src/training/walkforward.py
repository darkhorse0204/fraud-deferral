"""Walk-forward scoring driver with label-maturity and a calibration reserve.

For retrain origin tau (first period decided with this model):
    cut      = tau - D - 1 - Wc          last period whose labels the SCORER may train on
    train    = periods [pmin, cut]       (all matured: cut + D < tau)
    predict  = periods [cut+1, tau+R-1]  includes the Wc calibration-reserve periods, which are out-of-sample for the
                                         scorer but already matured at tau -> usable by the rolling recalibrator.
Frozen mode (R=None): one origin at the era start, predictions to the era end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.dataset import Dataset
from src.data.maturity import pool_mask
from src.utils import ROOT, cfg_hash, seed_everything

SCORES = ROOT / "results" / "scores"


@dataclass
class WF:
    scorer: str            # lr rf xgb delayed mlp mcd sage evolve
    fs: str                # feature-set name in ds.sets
    D: int
    Wc: int
    R: int | None          # retrain interval; None = frozen
    era: tuple[int, int]   # (lo, hi) inclusive evaluation periods
    seeds: tuple[int, ...] = (0,)
    hp: dict = field(default_factory=dict)
    origins: tuple[int, ...] | None = None   # explicit origins (dev tuning)
    shuffle_edges: bool = False
    tag: str = ""
    mask_agg: bool = False   # S5 stress test: provider-aggregated features (cols 93..164) missing at scoring time

    def name(self) -> str:
        r = "inf" if self.R is None else self.R
        return f"{self.scorer}_{self.fs}_D{self.D}_W{self.Wc}_R{r}{('_' + self.tag) if self.tag else ''}"


def origins_for(cfg: WF) -> list[int]:
    if cfg.origins is not None:
        return list(cfg.origins)
    lo, hi = cfg.era
    if cfg.R is None:
        return [lo]
    return list(range(lo, hi + 1, cfg.R))


def _score_once(ds: Dataset, cfg: WF, tr_lo: int, cut: int, pr_hi: int, seed: int):
    from src.models import nn as NN
    from src.models import tabular as TB

    cols = ds.sets[cfg.fs]
    s, hp = cfg.scorer, cfg.hp
    pr_lo = cut + 1
    if s in ("sage", "evolve"):
        trp = [p for p in range(tr_lo, cut + 1) if (ds.rows(p, p)).size]
        prp = list(range(pr_lo, pr_hi + 1))
        if s == "sage":
            return NN.fit_predict_sage(ds, cols, trp, prp, seed, hp, M=hp.get("M", 5), shuffle_edges=cfg.shuffle_edges)
        return NN.fit_predict_evolvegcn(ds, cols, trp, prp, seed, hp, M=hp.get("M", 3))
    rtr, rpr = ds.rows(tr_lo, cut), ds.rows(pr_lo, pr_hi)
    Xtr, ytr, ptr, Xte = ds.X[rtr][:, cols], ds.y[rtr].astype(np.int8), ds.period[rtr], ds.X[rpr][:, cols]
    if cfg.mask_agg:                       # models are NOT retrained for the outage; imputed with training medians downstream
        Xte = Xte.copy()
        Xte[:, 93:] = np.nan
    if s == "lr":
        p, u = TB.fit_predict_lr(Xtr, ytr, Xte, seed, hp)
    elif s == "rf":
        p, u = TB.fit_predict_rf(Xtr, ytr, Xte, seed, hp)
    elif s == "xgb":
        p, u = TB.fit_predict_xgb(Xtr, ytr, Xte, seed, hp, M=hp.get("M", 5))
    elif s == "delayed":
        p, u = TB.fit_predict_delayed(Xtr, ytr, ptr, Xte, seed, hp)
    elif s == "mlp":
        p, u = NN.fit_predict_mlp(Xtr, ytr, ptr, Xte, seed, hp, M=hp.get("M", 5))
    elif s == "mcd":
        p, u = NN.fit_predict_mlp(Xtr, ytr, ptr, Xte, seed, hp, M=1, mc=True)
    else:
        raise ValueError(s)
    return rpr, p, u


def run(ds: Dataset, cfg: WF, verbose: bool = True) -> pd.DataFrame:
    pmin, frames = int(ds.period.min()), []
    for tau in origins_for(cfg):
        cut = tau - cfg.D - 1 - cfg.Wc
        if cut < pmin:
            raise ValueError(f"origin {tau} leaves no training data (D={cfg.D}, Wc={cfg.Wc})")
        # structural maturity guarantee (L5): everything the scorer trains on has matured by tau
        assert (np.arange(pmin, cut + 1) + cfg.D < tau).all()
        pr_hi = cfg.era[1] if cfg.R is None else min(tau + cfg.R - 1, cfg.era[1])
        if cfg.origins is not None and cfg.R is not None:
            pr_hi = min(tau + cfg.R - 1, cfg.era[1])
        for seed in cfg.seeds:
            seed_everything(seed)
            rows, p, u = _score_once(ds, cfg, pmin, cut, pr_hi, seed)
            assert (ds.period[rows] > cut).all() and (ds.period[rows] <= pr_hi).all()
            frames.append(pd.DataFrame({"origin": np.int16(tau), "seed": np.int16(seed), "rid": rows.astype(np.int32),
                                        "period": ds.period[rows], "y": ds.y[rows], "p": p, "u": u}))
            if verbose:
                print(f"  [{cfg.name()}] tau={tau} seed={seed} train<= {cut} predict {cut+1}..{pr_hi} rows={len(rows)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.attrs["cfg"] = cfg_hash(cfg.__dict__)
    return df


def save(df: pd.DataFrame, dataset: str, cfg: WF, era_tag: str) -> str:
    d = SCORES / dataset.replace(":", "_")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cfg.name()}__{era_tag}.parquet"
    df.to_parquet(path, index=False)
    return str(path)

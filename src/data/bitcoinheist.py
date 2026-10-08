"""Study 2 dataset: BitcoinHeist ransomware address dataset (UCI, CC BY 4.0, doi:10.24432/C5BG8V).

One row = one (address, day): six graph-derived features of the address's 24-hour transaction neighbourhood, the
amount received (`income`, satoshis) and a label (ransomware family or 'white').
Periods are 28-day blocks: period = 13 * (year - 2011) + min((day - 1) // 28, 12), i.e. 13 periods per year.
The clock (year, day) and the address string are NEVER features.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.data.dataset import Dataset
from src.features.featuresets import assert_no_clock
from src.utils import ROOT

RAW = ROOT / "data" / "raw" / "bitcoinheist" / "BitcoinHeistData.csv"
PROC = ROOT / "data" / "processed" / "bitcoinheist.parquet"
CFG = ROOT / "configs" / "study2"
BASE = ["length", "weight", "count", "looped", "neighbors", "income_btc"]
LOGS = ["weight", "count", "looped", "neighbors", "income_btc"]

# Frozen BEFORE any model was fit. Eras by calendar year; T2 starts with 2016, the year the Locky / Cerber / CryptXXX
# families first appear in the label metadata (a structural fact read from family-by-year counts, disclosed).
SPLIT = {"dev": [0, 51], "test": [52, 90], "test_blocks": {"T1": [52, 64], "T2": [65, 90]}, "D": 2, "Wc": 3, "R": 6,
         "excluded": "2018 (periods 91-103): 3 positives in the whole year"}


def period_of(year: np.ndarray, day: np.ndarray) -> np.ndarray:
    return (13 * (year - 2011) + np.minimum((day - 1) // 28, 12)).astype(np.int16)


def build(force: bool = False) -> None:
    if PROC.exists() and not force:
        return
    df = pd.read_csv(RAW)
    assert len(df) == 2_916_697 and df.isna().sum().sum() == 0
    df["period"] = period_of(df["year"].to_numpy(), df["day"].to_numpy())
    df["y"] = (df["label"] != "white").astype("int8")
    df["income_btc"] = (df["income"] / 1e8).astype("float32")
    df["addr"] = pd.factorize(df["address"])[0].astype("int32")
    df = df.sort_values(["period", "addr"], kind="stable").reset_index(drop=True)
    PROC.parent.mkdir(parents=True, exist_ok=True)
    df[["addr", "period", "y", "label", "income_btc", "length", "weight", "count", "looped", "neighbors"]].to_parquet(PROC, index=False)


def matured_blacklist(addr: np.ndarray, period: np.ndarray, y: np.ndarray, D: int) -> np.ndarray:
    """1 if an EARLIER row of the same address was labelled ransomware and that label had matured when this row is
    decided, i.e. first_positive_period + D < period (the study's maturity rule). Uses no label from period >= period - D."""
    s = pd.DataFrame({"addr": addr, "period": period.astype(np.int32), "y": y})
    first_pos = s["period"].where(s["y"] == 1).groupby(s["addr"]).transform("min")
    return (first_pos + D < s["period"]).to_numpy(dtype=np.float32)


def load_dataset() -> Dataset:
    """Feature set fixed on the DEV era (scripts/s2_dev_features.py): the 11 base features are near chance forward in
    time (dev AUPRC 0.038, base rate 0.029); the matured blacklist lifts dev AUPRC to 0.35. Address-history counts were
    tried and EXCLUDED: legitimate addresses are subsampled (<= 1,000 per day), so repeat counts are a sampling artefact."""
    build()
    df = pd.read_parquet(PROC)
    parts, names = [], []
    for c in BASE:
        parts.append(df[c].to_numpy(dtype=np.float32)); names.append(c)
    for c in LOGS:
        parts.append(np.log1p(df[c].to_numpy(dtype=np.float64)).astype(np.float32)); names.append(f"log1p_{c}")
    parts.append(matured_blacklist(df["addr"].to_numpy(), df["period"].to_numpy(), df["y"].to_numpy(), SPLIT["D"])); names.append("addr_blacklisted_matured")
    assert_no_clock(names, "y")
    assert not {"year", "day", "addr", "address", "label"} & set(names)
    X = np.stack(parts, axis=1)
    fam = pd.Categorical(df["label"])
    return Dataset("bitcoinheist", X, df["period"].to_numpy(), df["y"].to_numpy(), names, {"F_all": np.arange(len(names))},
                   extra={"amount": df["income_btc"].to_numpy(dtype=np.float64), "family": fam.codes.astype(np.int16),
                          "families": list(fam.categories), "addr": df["addr"].to_numpy()})


def write_split() -> str:
    CFG.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(json.dumps(SPLIT, sort_keys=True).encode()).hexdigest()
    (CFG / "split.json").write_text(json.dumps(dict(SPLIT, sha256=h), indent=2, sort_keys=True), encoding="utf-8")
    return h


if __name__ == "__main__":
    build(force=True)
    ds = load_dataset()
    print(ds.X.shape, int(ds.period.min()), int(ds.period.max()), "pos", int(ds.y.sum()))
    per = pd.DataFrame({"p": ds.period, "y": ds.y}).groupby("p").y.agg(["size", "sum"])
    print("dev pos", int(per.loc[0:51, "sum"].sum()), "T1 pos", int(per.loc[52:64, "sum"].sum()), "T2 pos", int(per.loc[65:90, "sum"].sum()))
    print("test periods with <5 positives:", per.loc[52:90][per.loc[52:90, "sum"] < 5].index.tolist())
    print("split hash", write_split())

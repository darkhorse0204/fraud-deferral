"""In-memory dataset objects used by all scorers.

Dataset.X holds RAW feature values (BAF: NaN for sentinel-missing); any imputation/scaling is fitted by the
model wrapper on TRAINING rows only (see src.models.prep).  Row order is stable; `rid` == row index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.loaders import load_baf, load_elliptic
from src.features.featuresets import assert_no_clock, elliptic_sets

# Columns whose ONLY negative value is exactly -1 (verified on Base; M8 corrected). intended_balcon_amount,
# credit_risk_score and velocity_6h have legitimately negative continuous values and are NOT sentinels.
BAF_SENTINEL_COLS = ["prev_address_months_count", "current_address_months_count", "bank_months_count",
                     "session_length_in_minutes", "device_distinct_emails_8w"]
BAF_CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
BAF_DROP = ["device_fraud_count"]  # constant (audit)


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # (N, d) float32
    period: np.ndarray                 # (N,) int16, native periods
    y: np.ndarray                      # (N,) int8, -1 = unlabeled
    feat_names: list[str]
    sets: dict[str, np.ndarray]        # feature-set name -> column indices
    edge_index: np.ndarray | None = None   # (2, E) global row indices (graph datasets only)
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    def rows(self, lo: int, hi: int, labeled: bool = True) -> np.ndarray:
        m = (self.period >= lo) & (self.period <= hi)
        if labeled:
            m &= self.y >= 0
        return np.flatnonzero(m)


def load_elliptic_dataset() -> Dataset:
    nodes, edges = load_elliptic()
    sets = elliptic_sets()
    cols = sets["F_all"]
    for names in sets.values():
        assert_no_clock(names)
    X = nodes[cols].to_numpy(dtype=np.float32)
    idx = pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())
    ei = np.stack([idx.loc[edges["src_txId"]].to_numpy(), idx.loc[edges["dst_txId"]].to_numpy()])
    return Dataset("elliptic", X, nodes["period"].to_numpy(), nodes["label"].to_numpy(), cols,
                   {"F_all": np.arange(len(cols)), "F_local": np.arange(93)}, ei)


def load_baf_dataset(variant: str = "Base") -> Dataset:
    df = load_baf(variant)
    y = df["fraud_bool"].to_numpy(dtype=np.int8)
    period = df["month"].to_numpy(dtype=np.int16)
    num_cols = [c for c in df.columns if c not in ["fraud_bool", "month"] + BAF_CAT_COLS + BAF_DROP]
    parts, names = [], []
    for c in num_cols:
        v = df[c].to_numpy(dtype=np.float32).copy()
        if c in BAF_SENTINEL_COLS:
            miss = v < 0
            v[miss] = np.nan
            parts.append(v); names.append(c)
            parts.append(miss.astype(np.float32)); names.append(f"{c}__missing")
        else:
            parts.append(v); names.append(c)
    for c in BAF_CAT_COLS:  # categorical alphabets are schema-level (5 small columns), not data-fitted
        for lvl in sorted(df[c].unique()):
            parts.append((df[c] == lvl).to_numpy(dtype=np.float32)); names.append(f"{c}={lvl}")
    assert_no_clock(names, "fraud_bool")
    X = np.stack(parts, axis=1)
    return Dataset(f"baf:{variant}", X, period, y, names, {"F_all": np.arange(len(names))})

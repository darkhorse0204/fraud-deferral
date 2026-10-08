"""Raw -> processed parquet. Deterministic, no model fitting, no label-dependent transforms.

Elliptic processed schema (data/processed/elliptic_nodes.parquet):
    txId:int64, period:int16 (= time step), label:int8 (-1 unknown, 0 licit, 1 illicit),
    tx_f2..tx_f166:float32.   (tx_f1 is the time step itself; kept ONLY as `period`.)
Class coding follows the Kaggle description: class '1' = illicit (4,545), class '2' = licit (42,019).
The coding is asserted against these published counts in validate.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import ROOT, find_file

PROC = ROOT / "data" / "processed"
ELLIPTIC_LABEL_MAP = {"unknown": -1, "1": 1, "2": 0}


def _elliptic_raw_paths() -> tuple[Path, Path, Path]:
    raw = ROOT / "data" / "raw" / "elliptic"
    out = []
    for f in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"):
        p = find_file(raw, f)
        if p is None:
            raise FileNotFoundError(f"{f} missing; run: python -m src.data.download elliptic")
        out.append(p)
    return tuple(out)  # type: ignore[return-value]


def build_elliptic(force: bool = False) -> None:
    nodes_p, edges_p = PROC / "elliptic_nodes.parquet", PROC / "elliptic_edges.parquet"
    if nodes_p.exists() and edges_p.exists() and not force:
        return
    PROC.mkdir(parents=True, exist_ok=True)
    f_path, c_path, e_path = _elliptic_raw_paths()
    feat = pd.read_csv(f_path, header=None, engine="pyarrow")
    if feat.shape[1] != 167:
        raise ValueError(f"expected 167 columns (txId + 166 features), got {feat.shape[1]}")
    feat.columns = ["txId"] + [f"tx_f{k}" for k in range(1, 167)]
    cls = pd.read_csv(c_path, dtype={"txId": "int64", "class": "string"})
    unknown_codes = set(cls["class"].unique()) - set(ELLIPTIC_LABEL_MAP)
    if unknown_codes:
        raise ValueError(f"unexpected class codes: {unknown_codes}")
    cls["label"] = cls["class"].map(ELLIPTIC_LABEL_MAP).astype("int8")
    df = feat.merge(cls[["txId", "label"]], on="txId", how="left", validate="one_to_one")
    if df["label"].isna().any():
        raise ValueError("txIds in features without a class row")
    df["period"] = df["tx_f1"].round().astype("int16")
    if not np.allclose(df["tx_f1"], df["period"]):
        raise ValueError("time-step column is not integer valued")
    fcols = [f"tx_f{k}" for k in range(2, 167)]
    out = df[["txId", "period", "label"] + fcols].copy()
    out[fcols] = out[fcols].astype("float32")
    out = out.sort_values(["period", "txId"], kind="stable").reset_index(drop=True)
    out.to_parquet(nodes_p, index=False)
    edges = pd.read_csv(e_path, dtype="int64")
    edges.columns = ["src_txId", "dst_txId"]
    edges.to_parquet(edges_p, index=False)


def load_elliptic() -> tuple[pd.DataFrame, pd.DataFrame]:
    build_elliptic()
    return pd.read_parquet(PROC / "elliptic_nodes.parquet"), pd.read_parquet(PROC / "elliptic_edges.parquet")


def load_baf(variant: str = "Base") -> pd.DataFrame:
    p = find_file(ROOT / "data" / "raw" / "baf", f"{variant}.csv")
    if p is None:
        raise FileNotFoundError(f"{variant}.csv missing; run: python -m src.data.download baf")
    return pd.read_csv(p, engine="pyarrow")


if __name__ == "__main__":
    build_elliptic(force=True)
    n, e = load_elliptic()
    print(n.shape, e.shape)

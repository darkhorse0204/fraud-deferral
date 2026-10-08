"""Data validation + audit. Writes results/data_audit.json. Hard failures raise; findings are recorded.

Rules:
  * Anything that touches labels for exploratory statistics uses the DEV era only (no test peeking).
  * Structural checks (counts, ids, edges, schema) may use all periods: they involve no label->feature fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data import splits
from src.data.download import ROOT
from src.data.loaders import load_baf, load_elliptic
from src.data.periods import eval_groups
from src.features.featuresets import assert_no_clock, elliptic_sets

AUDIT = ROOT / "results" / "data_audit.json"
LEAK_AUROC = 0.95  # single-feature AUROC above this => drop-candidate (L10), pre-specified


class ValidationError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def _cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "data" / f"{name}.yaml").read_text(encoding="utf-8"))


def single_feature_auroc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    out = {}
    for c in X.columns:
        v = X[c].to_numpy(dtype="float64")
        if np.nanstd(v) < 1e-12:
            out[c] = 0.5
            continue
        a = roc_auc_score(y, np.nan_to_num(v))
        out[c] = max(a, 1 - a)  # direction-agnostic
    return pd.Series(out).sort_values(ascending=False)


def validate_elliptic() -> dict:
    cfg, sp = _cfg("elliptic"), splits.load("elliptic")
    nodes, edges = load_elliptic()
    exp = cfg["expected"]
    rep: dict = {}

    # --- structure -----------------------------------------------------------------------------
    _check(len(nodes) == exp["n_nodes"], f"node count {len(nodes)} != {exp['n_nodes']}")
    _check(len(edges) == exp["n_edges"], f"edge count {len(edges)} != {exp['n_edges']}")
    _check(nodes["txId"].is_unique, "duplicate txId")
    ids = set(nodes["txId"])
    _check(edges["src_txId"].isin(ids).all() and edges["dst_txId"].isin(ids).all(), "edge endpoint missing from nodes")
    _check(sorted(nodes["period"].unique()) == list(range(1, exp["n_timesteps"] + 1)), "time steps are not exactly 1..49")
    lab = nodes["label"].value_counts().to_dict()
    _check(lab.get(1, 0) == exp["n_illicit"], f"illicit count {lab.get(1)} != {exp['n_illicit']} (class coding wrong?)")
    _check(lab.get(0, 0) == exp["n_licit"], f"licit count {lab.get(0)} != {exp['n_licit']}")
    rep["n_nodes"], rep["n_edges"] = len(nodes), len(edges)
    rep["label_counts"] = {"illicit": int(lab[1]), "licit": int(lab[0]), "unknown": int(lab[-1])}

    # --- F5: no cross-period edges ---------------------------------------------------------------
    per = nodes.set_index("txId")["period"]
    sp_, dp_ = per.loc[edges["src_txId"]].to_numpy(), per.loc[edges["dst_txId"]].to_numpy()
    cross = int((sp_ != dp_).sum())
    rep["cross_period_edges"] = cross
    _check(cross == 0, f"{cross} edges cross time steps (spec flag F5 would be wrong; revisit graph design)")

    # --- features --------------------------------------------------------------------------------
    sets = elliptic_sets()
    for names in sets.values():
        assert_no_clock(names)
        _check(all(n in nodes.columns for n in names), "feature set refers to missing column")
    fcols = sets["F_all"]
    X = nodes[fcols]
    rep["feature_nan"] = int(X.isna().sum().sum())
    rep["feature_inf"] = int(np.isinf(X.to_numpy()).sum())
    _check(rep["feature_nan"] == 0 and rep["feature_inf"] == 0, "NaN/inf in features")
    rep["constant_features"] = [c for c in fcols if X[c].nunique() <= 1]

    # --- L14: duplicate feature rows (within and across periods) --------------------------------
    h = pd.util.hash_pandas_object(X, index=False)
    dup = h.duplicated(keep=False)
    cross_dup = nodes.assign(h=h)[dup].groupby("h")["period"].nunique()
    rep["duplicate_feature_rows"] = int(dup.sum())
    rep["duplicate_groups_spanning_periods"] = int((cross_dup > 1).sum())

    # --- per-period counts (+ L17 evaluation groups, nothing dropped) ---------------------------------
    lab_df = nodes[nodes["label"] >= 0]
    tab = nodes.groupby("period").agg(n=("txId", "size"), n_labeled=("label", lambda s: int((s >= 0).sum())),
                                      n_illicit=("label", lambda s: int((s == 1).sum())))
    rep["per_period"] = {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in tab.iterrows()}
    rep["periods_below_5_illicit"] = [int(p) for p in tab.index[tab["n_illicit"] < 5]]
    g = eval_groups(lab_df["period"].to_numpy(), lab_df["label"].to_numpy(), 5)
    rep["eval_groups"] = {int(k): int(v) for k, v in g.items()}
    t2 = sp["test_blocks"]["T2"]
    rep["T2_illicit_by_period"] = {p: int(tab.loc[p, "n_illicit"]) for p in range(t2[0], t2[1] + 1)}
    rep["T1_illicit_total"] = int(tab.loc[sp["test_blocks"]["T1"][0]:sp["test_blocks"]["T1"][1], "n_illicit"].sum())
    rep["T2_illicit_total"] = int(sum(rep["T2_illicit_by_period"].values()))

    # --- L10 / L1: single-feature AUROC on DEV era, labeled nodes only ---------------------------
    dev = lab_df[(lab_df["period"] >= sp["dev"][0]) & (lab_df["period"] <= sp["dev"][1])]
    au = single_feature_auroc(dev[fcols], dev["label"].to_numpy())
    rep["single_feature_auroc_dev_top10"] = {k: round(float(v), 4) for k, v in au.head(10).items()}
    rep["leak_candidates_auroc_gt_0.95"] = [k for k, v in au.items() if v > LEAK_AUROC]

    # --- L9: are labeled nodes distinguishable from unlabeled ones? (dev era, 60k sample) ---------
    d = nodes[(nodes["period"] <= sp["dev"][1])]
    d = d.sample(n=min(60_000, len(d)), random_state=0)
    yl = (d["label"] >= 0).astype(int).to_numpy()
    ntr = int(0.7 * len(d))  # sample is shuffled by .sample(); this is a diagnostic, not a model
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(d[fcols].iloc[:ntr], yl[:ntr])
    rep["labeled_vs_unlabeled_domain_auc_dev"] = round(float(roc_auc_score(yl[ntr:], clf.predict_proba(d[fcols].iloc[ntr:])[:, 1])), 4)

    # --- provider-side normalisation (L1/L3): features arrive pre-standardised; cannot be audited -----
    m = X.mean().abs()
    s = X.std()
    rep["provider_standardised"] = bool((m.median() < 0.05) and (abs(s.median() - 1) < 0.2))
    return rep


def validate_baf_variant(variant: str) -> dict:
    cfg, sp = _cfg("baf"), splits.load("baf")
    df = load_baf(variant)
    tgt, pcol = cfg["target"], cfg["period_col"]
    rep: dict = {"variant": variant, "rows": len(df), "columns": list(df.columns)}
    _check(len(df) == cfg["expected"]["n_rows_per_file"], f"{variant}: row count {len(df)}")
    _check(set(df[tgt].unique()) <= {0, 1}, f"{variant}: target not binary")
    _check(sorted(df[pcol].unique()) == cfg["expected"]["months"], f"{variant}: months {sorted(df[pcol].unique())}")
    rep["fraud_rate"] = round(float(df[tgt].mean()), 5)
    rep["per_month"] = {int(m): {"n": int(len(g)), "fraud": int(g[tgt].sum()), "rate": round(float(g[tgt].mean()), 5)}
                        for m, g in df.groupby(pcol)}
    feats = [c for c in df.columns if c not in (tgt, pcol)]
    assert_no_clock(feats, tgt)
    rep["n_features"] = len(feats)
    rep["nan_cells"] = int(df[feats].isna().sum().sum())
    rep["constant_features"] = [c for c in feats if df[c].nunique() <= 1]
    # pandas 3 uses a `str` dtype, so detect categoricals as "not numeric" rather than "object"
    rep["dtypes_object"] = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    # BAF encodes missing values as negative sentinels in some columns; record, don't assume.
    rep["columns_with_negative_values"] = {c: int((df[c] < 0).sum()) for c in feats
                                           if c not in rep["dtypes_object"] and pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any()}
    dups = int(df.duplicated(subset=feats).sum())
    rep["duplicate_rows_features_only"] = dups
    dev = df[(df[pcol] >= sp["dev"][0]) & (df[pcol] <= sp["dev"][1])]
    num = [c for c in feats if c not in rep["dtypes_object"]]
    au = single_feature_auroc(dev[num], dev[tgt].to_numpy())
    rep["single_feature_auroc_dev_top10"] = {k: round(float(v), 4) for k, v in au.head(10).items()}
    rep["leak_candidates_auroc_gt_0.95"] = [k for k, v in au.items() if v > LEAK_AUROC]
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["elliptic", "baf"])
    ap.add_argument("--baf-variants", nargs="+", default=["Base"])
    a = ap.parse_args()
    audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    if "elliptic" in a.datasets:
        audit["elliptic"] = validate_elliptic()
        print("elliptic: OK")
    if "baf" in a.datasets:
        audit.setdefault("baf", {})
        for v in a.baf_variants:
            audit["baf"][v] = validate_baf_variant(v)
            print(f"baf {v}: OK")
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", AUDIT)


if __name__ == "__main__":
    main()

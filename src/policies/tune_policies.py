"""Dev-era tuning of the decision layer (no test-era data). Staged so the CONTROL is tuned first:
  1. shared recalibration setup (calib kind, W, h) chosen by pi2's dev cost  -> used by EVERY calibrated policy
  2. MAD's own (lam, eps) chosen by dev cost relative to the tuned pi2
  3. conformal alpha
  4. pi0's raw-score threshold per scorer (F1-optimal on dev decision periods)
Objective everywhere: mean over (scorer, non-degenerate cost cell) of  cost_total / cost_total(reference).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import GRID
from src.policies.policies import PolicySpec, Table, run_suite
from src.utils import ROOT

DEV_PERIODS = {"elliptic": range(20, 35), "baf": range(3, 4)}
DEV_D = {"elliptic": 2, "baf": 1}
CELLS = [c for c in GRID if not c.degenerate]


def load_tables(dataset_dir: str, era: str, D: int, kinds=("main",)) -> dict[str, Table]:
    out = {}
    for f in sorted((ROOT / "results" / "scores" / dataset_dir).glob(f"*__{era}.parquet")):
        name = f.stem.replace(f"__{era}", "")
        if f"_D{D}_" not in name or "_Rinf" in name or "_shuf" in name:
            continue
        out[name] = Table(pd.read_parquet(f), D)
    return out


def total_cost(tabs: dict[str, Table], periods, spec: PolicySpec, seeds=None) -> dict[tuple[str, str], float]:
    res = {}
    for name, tab in tabs.items():
        df = run_suite(tab, periods, [spec], CELLS, seeds=seeds)
        for k, v in df.groupby("cost")["cost_sum"].sum().items():
            res[(name, k)] = float(v)
    return res


def ratio(a: dict, ref: dict) -> float:
    return float(np.mean([a[k] / ref[k] for k in a if ref.get(k, 0) > 0]))


def tune(dataset_key: str, dataset_dir: str, tune_calib: bool = True, fixed=None) -> dict:
    tabs = load_tables(dataset_dir, "dev", DEV_D[dataset_key])
    per = DEV_PERIODS[dataset_key]
    ref = total_cost(tabs, per, PolicySpec("pi1", W=5, calib="affine"))
    log = []
    # 1. shared recalibration
    if tune_calib:
        best = None
        for calib, W, h in itertools.product(["affine", "temp", "isotonic"], [3, 5, 8], [np.inf, 3.0]):
            r = ratio(total_cost(tabs, per, PolicySpec("pi2", W=W, h=h, calib=calib)), ref)
            log.append(dict(stage=1, calib=calib, W=W, h=h, ratio=r))
            if best is None or r < best[0]:
                best = (r, calib, W, h)
        _, calib, W, h = best
    else:
        calib, W, h = fixed["calib"], fixed["W"], fixed["h"]
    pi2_cost = total_cost(tabs, per, PolicySpec("pi2", W=W, h=h, calib=calib))
    # 2. MAD
    bestm = None
    for lam, eps, mg in itertools.product([0.0, 0.02, 0.1, 0.5], [0.5, np.inf], [0.0, 0.01, 0.03]):
        r = ratio(total_cost(tabs, per, PolicySpec("pi4", W=W, h=h, calib=calib, lam=lam, eps=eps, min_gain=mg)), pi2_cost)
        log.append(dict(stage=2, lam=lam, eps=eps, min_gain=mg, ratio_vs_pi2=r))
        if bestm is None or r < bestm[0]:
            bestm = (r, lam, eps, mg)
    _, lam, eps, mg = bestm
    # 3. conformal alpha
    besta = None
    for alpha in (0.05, 0.1, 0.2):
        r = ratio(total_cost(tabs, per, PolicySpec("pi6", W=W, h=h, calib=calib, alpha=alpha)), ref)
        log.append(dict(stage=3, alpha=alpha, ratio_vs_pi1=r))
        if besta is None or r < besta[0]:
            besta = (r, alpha)
    # 4. pi0 thresholds: F1-optimal raw-score threshold on dev decision periods, per scorer
    theta0 = {}
    for name, tab in tabs.items():
        ps, ys = [], []
        for seed in tab.seeds:
            for t in per:
                try:
                    cur = tab.current(seed, t)
                except KeyError:
                    continue
                ps.append(cur["p"]); ys.append(cur["y"])
        p, y = np.concatenate(ps), np.concatenate(ys)
        pr, rc, th = precision_recall_curve(y, p)
        f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
        theta0[name] = float(th[int(np.argmax(f1))]) if len(th) else 0.5
    out = dict(dataset=dataset_key, calib=calib, W=int(W), h=float(h) if np.isfinite(h) else "inf", lam=float(lam), min_gain=float(mg),
               eps=float(eps) if np.isfinite(eps) else "inf", alpha=float(besta[1]), theta0=theta0,
               dev_ratio_mad_vs_pi2=float(bestm[0]))
    return dict(hp=out, log=log)


if __name__ == "__main__":
    import json
    import sys

    key = sys.argv[1]
    d = "elliptic" if key == "elliptic" else "baf_Base"
    if key == "elliptic":
        res = tune("elliptic", "elliptic")
    else:
        ell = yaml.safe_load((ROOT / "configs" / "policies" / "policy_hp.yaml").read_text())["elliptic"]
        fixed = dict(calib=ell["calib"], W=int(ell["W"]), h=np.inf if ell["h"] == "inf" else float(ell["h"]))
        res = tune("baf", d, tune_calib=False, fixed=fixed)
        # BAF dev has ONE period: reuse the Elliptic-tuned decision-layer settings, keep only the BAF theta0 (pi0 baseline)
        for k in ("lam", "eps", "min_gain", "alpha"):
            res["hp"][k] = ell[k]
    path = ROOT / "configs" / "policies" / "policy_hp.yaml"
    cur = yaml.safe_load(path.read_text()) if path.exists() else {}
    cur[key] = res["hp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cur, sort_keys=True), encoding="utf-8")
    (ROOT / "results" / "tuning" / f"policy_tuning_{key}.json").write_text(json.dumps(res["log"], indent=1, default=str))
    print(json.dumps(res["hp"], indent=1, default=str))

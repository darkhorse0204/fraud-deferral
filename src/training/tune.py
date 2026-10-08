"""Dev-era random-search tuning. Criterion: mean walk-forward AUPRC over dev origins (score quality only; no
policy or cost is involved, so tuning cannot favour the proposed decision layer). Same budget per family.
Trial 0 is always the family default. Writes configs/models/<dataset>__<scorer>__<fs>.yaml and a trial log.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from src.data.dataset import load_baf_dataset, load_elliptic_dataset
from src.training.walkforward import WF, run
from src.utils import ROOT

DEV_SETUP = {  # D, Wc, R, dev origins, dev era
    "elliptic": dict(D=2, Wc=4, R=5, origins=(20, 25, 30), era=(1, 34)),
    "baf": dict(D=1, Wc=1, R=2, origins=(3,), era=(0, 3)),
}


def sample_hp(scorer: str, rng: np.random.Generator) -> dict:
    ch = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))  # noqa: E731
    if scorer == "lr":
        return dict(C=lu(1e-3, 10), balanced=ch([True, False]))
    if scorer == "rf":
        return dict(n_estimators=ch([200, 400]), max_depth=ch([None, 10, 20]), min_samples_leaf=ch([1, 2, 5]),
                    max_features=ch(["sqrt", 0.3]))
    if scorer in ("xgb", "delayed"):
        hp = dict(eta=lu(0.03, 0.3), max_depth=ch([3, 4, 6, 8]), rounds=ch([100, 200, 400]), subsample=ch([0.6, 0.8, 1.0]),
                  colsample=ch([0.5, 0.8, 1.0]), min_child_weight=ch([1, 5]), pos_weight_pow=ch([0.0, 0.5, 1.0]), M=3)
        if scorer == "delayed":
            hp["recent_periods"] = ch([2, 3, 5])
        return hp
    if scorer in ("mlp", "mcd"):
        return dict(widths=ch([(128, 64), (256, 128), (512, 256)]), drop=ch([0.1, 0.2, 0.3]) if scorer == "mcd" else ch([0.0, 0.1, 0.3]),
                    lr=lu(3e-4, 3e-3), wd=ch([1e-5, 1e-4, 1e-3]), epochs=ch([30, 60]), pos_weight_pow=ch([0.0, 0.5, 1.0]),
                    bs=ch([512, 1024]), M=3, patience=6)
    if scorer == "sage":
        return dict(hidden=ch([32, 64, 128]), drop=ch([0.1, 0.3]), lr=lu(1e-3, 1e-2), wd=ch([1e-5, 1e-4, 1e-3]),
                    epochs=ch([40, 80]), pos_weight_pow=ch([0.0, 0.5, 1.0]), M=3)
    if scorer == "evolve":
        return dict(hidden=ch([32, 64]), drop=ch([0.1, 0.3]), lr=lu(1e-3, 1e-2), epochs=ch([40, 80]), chunk=ch([6, 10]),
                    pos_weight_pow=ch([0.0, 0.5, 1.0]), M=2)
    raise ValueError(scorer)


DEFAULTS = {"lr": {}, "rf": {}, "xgb": {"M": 3}, "delayed": {"M": 3}, "mlp": {"M": 3}, "mcd": {}, "sage": {"M": 3},
            "evolve": {"M": 2}}
BUDGET = {"lr": 6, "rf": 12, "xgb": 12, "delayed": 8, "mlp": 12, "mcd": 8, "sage": 10, "evolve": 6}


def score_cfg(ds, dataset_key: str, scorer: str, fs: str, hp: dict) -> float:
    st = DEV_SETUP[dataset_key]
    cfg = WF(scorer, fs, st["D"], st["Wc"], st["R"], st["era"], (0,), hp, origins=st["origins"])
    df = run(ds, cfg, verbose=False)
    aps = []
    for tau in st["origins"]:
        d = df[(df.origin == tau) & (df.period >= tau) & (df.period <= tau + st["R"] - 1)]
        if d.y.sum() > 0:
            aps.append(average_precision_score(d.y, d.p))
    return float(np.mean(aps))


def tune(dataset_key: str, scorer: str, fs: str, budget: int | None = None, seed: int = 2024) -> dict:
    ds = load_elliptic_dataset() if dataset_key == "elliptic" else load_baf_dataset("Base")
    rng = np.random.default_rng(seed)
    n = budget or BUDGET[scorer]
    trials = []
    for i in range(n):
        hp = dict(DEFAULTS[scorer]) if i == 0 else sample_hp(scorer, rng)
        t0 = time.time()
        try:
            ap = score_cfg(ds, dataset_key, scorer, fs, hp)
        except Exception as e:  # noqa: BLE001  record and continue; a failing config is a bad config
            ap = float("nan"); print("   trial failed:", repr(e)[:120])
        trials.append(dict(trial=i, hp=hp, dev_auprc=ap, secs=round(time.time() - t0, 1)))
        print(f"[tune {dataset_key}/{scorer}/{fs}] trial {i:2d} AUPRC={ap:.4f} ({trials[-1]['secs']}s)", flush=True)
    ok = [t for t in trials if not np.isnan(t["dev_auprc"])]
    best = max(ok, key=lambda t: t["dev_auprc"])
    out = ROOT / "configs" / "models" / f"{dataset_key}__{scorer}__{fs}.yaml"
    out.write_text(yaml.safe_dump({"scorer": scorer, "fs": fs, "dataset": dataset_key, "dev_auprc": best["dev_auprc"],
                                   "hp": json.loads(json.dumps(best["hp"], default=list))}, sort_keys=True), encoding="utf-8")
    log = ROOT / "results" / "tuning"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"{dataset_key}__{scorer}__{fs}.json").write_text(json.dumps(trials, indent=1, default=list), encoding="utf-8")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["elliptic", "baf"])
    ap.add_argument("jobs", nargs="+", help="scorer:fs, e.g. xgb:F_all")
    ap.add_argument("--budget", type=int, default=None, help="override trials per family (same for all families in a run)")
    a = ap.parse_args()
    for j in a.jobs:
        s, f = j.split(":")
        b = tune(a.dataset, s, f, budget=a.budget)
        print("BEST", a.dataset, s, f, b["dev_auprc"], b["hp"], flush=True)

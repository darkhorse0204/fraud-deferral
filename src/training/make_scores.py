"""Generate cached score tables from frozen, dev-tuned hyperparameters.

Usage:  python -m src.training.make_scores <dataset> <era: dev|test> <suite> [jobs...]
Test-era runs REFUSE to start unless the pre-registration tag exists (scripts/lock_test.py) and log a touch.
Suites:  main | frozen | dsweep | shuffle | all
"""
from __future__ import annotations

import argparse
import sys

import yaml

from src.data.dataset import load_baf_dataset, load_elliptic_dataset
from src.training.walkforward import WF, run, save
from src.utils import ROOT, log_test_touch

ELL = dict(D=2, Wc=4, R=5, dev=dict(era=(20, 34), origins=(20, 25, 30), seeds=(0, 1, 2)),
           test=dict(era=(35, 49), origins=None, seeds=tuple(range(10))))
BAF = dict(D=1, Wc=1, R=2, dev=dict(era=(0, 3), origins=(3,), seeds=(0, 1, 2)),
           test=dict(era=(4, 7), origins=None, seeds=tuple(range(5))))

# (scorer, feature set); graph-model seeds are capped for cost
ELL_MAIN = [("lr", "F_all"), ("lr", "F_local"), ("rf", "F_all"), ("rf", "F_local"), ("xgb", "F_all"), ("xgb", "F_local"),
            ("mlp", "F_all"), ("mlp", "F_local"), ("mcd", "F_all"), ("sage", "F_local"), ("sage", "F_all"),
            ("evolve", "F_local"), ("delayed", "F_all")]
BAF_MAIN = [("lr", "F_all"), ("xgb", "F_all"), ("mlp", "F_all"), ("mcd", "F_all"), ("delayed", "F_all")]
SEED_CAP = {"evolve": 5}


def tuned_hp(dataset_key: str, scorer: str, fs: str) -> dict:
    p = ROOT / "configs" / "models" / f"{dataset_key}__{scorer}__{fs}.yaml"
    if not p.exists():                                    # F_all-tuned params are reused if a feature set was not tuned
        p = ROOT / "configs" / "models" / f"{dataset_key}__{scorer}__F_all.yaml"
    hp = dict(yaml.safe_load(p.read_text())["hp"])
    if "widths" in hp:
        hp["widths"] = tuple(hp["widths"])
    # tuning used small ensembles for speed; final scoring uses the full ensemble size
    if scorer in ("xgb", "mlp", "sage", "delayed"):
        hp["M"] = 5
    if scorer == "evolve":
        hp["M"] = 3
    return hp


def jobs_for(dataset_key: str, era: str, suite: str, only: tuple = ()) -> list[WF]:
    S = ELL if dataset_key == "elliptic" else BAF
    e = S[era]
    base = ELL_MAIN if dataset_key == "elliptic" else BAF_MAIN
    out = []

    def mk(scorer, fs, D=None, R="default", tag="", shuf=False, mask=False):
        D = S["D"] if D is None else D
        R = S["R"] if R == "default" else R
        seeds = e["seeds"][:SEED_CAP.get(scorer, 99)]
        origins = e["origins"] if R is not None else None
        if era == "dev" and R is None:
            return None
        cfg = WF(scorer, fs, D, S["Wc"], R, e["era"], tuple(seeds), {}, origins=origins, shuffle_edges=shuf, tag=tag, mask_agg=mask)
        if only and cfg.name() not in only:
            return None
        cfg.hp = tuned_hp(dataset_key, scorer, fs)           # loaded lazily so untuned jobs elsewhere do not block
        return cfg

    if suite in ("main", "all"):
        out += [mk(s, f) for s, f in base]
    if era == "test" and suite in ("frozen", "all"):
        fro = [("xgb", "F_all"), ("mlp", "F_all"), ("rf", "F_all")] if dataset_key == "elliptic" else [("xgb", "F_all"), ("mlp", "F_all")]
        out += [mk(s, f, R=None) for s, f in fro]
    if era == "test" and suite in ("dsweep", "all"):
        Ds = (0, 4) if dataset_key == "elliptic" else (0, 2)
        sweep = [("rf", "F_all"), ("xgb", "F_all"), ("mlp", "F_all")] if dataset_key == "elliptic" else [("xgb", "F_all"), ("mlp", "F_all")]
        out += [mk(s, f, D=d) for d in Ds for s, f in sweep]
    if era == "test" and dataset_key == "elliptic" and suite in ("shuffle", "all"):
        out += [mk("sage", "F_local", tag="shuf", shuf=True)]
    if era == "test" and dataset_key == "elliptic" and suite in ("robust", "all"):
        out += [mk(s, "F_all", tag="outage", mask=True) for s in ("rf", "xgb", "mlp")]
    return [j for j in out if j is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["elliptic", "baf_Base", "baf_Variant III"])
    ap.add_argument("era", choices=["dev", "test"])
    ap.add_argument("suite", choices=["main", "frozen", "dsweep", "shuffle", "robust", "all"])
    ap.add_argument("only", nargs="*", help="restrict to job names")
    a = ap.parse_args()
    if a.era == "test":
        lock = ROOT / "results" / "PREREG_LOCK.txt"
        if not lock.exists():
            sys.exit("REFUSED: test era is locked until scripts/lock_test.py has frozen configs (pre-registration).")
        log_test_touch("scores:" + a.dataset + ":" + a.suite, "generate test-era scores from frozen configs")
    key = "elliptic" if a.dataset == "elliptic" else "baf"
    ds = load_elliptic_dataset() if key == "elliptic" else load_baf_dataset(a.dataset.split("_", 1)[1])
    for cfg in jobs_for(key, a.era, a.suite, tuple(a.only)):
        print(">>", a.dataset, a.era, cfg.name(), flush=True)
        df = run(ds, cfg, verbose=False)
        print("   saved", save(df, a.dataset, cfg, a.era), len(df), flush=True)


if __name__ == "__main__":
    main()

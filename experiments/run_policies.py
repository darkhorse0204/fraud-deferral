"""Apply the frozen decision policies to cached score tables.

python -m experiments.run_policies <dataset_dir> <era> [--only name ...]
Outputs  results/policy/<dataset_dir>/<table>__<era>.parquet   (per seed/period/policy/cost counts + realised cost)
         results/diag/<dataset_dir>/<table>__<era>.parquet     (per seed/period score + calibration diagnostics)
Variants per table: base (9-cell grid, rho=1) ; rho0.9 (primary cell) ; cap0.005 / cap0.02 (primary cell).
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import yaml

from src.evaluation.diagnostics import period_diagnostics
from src.evaluation.metrics import GRID, PRIMARY, Costs
from src.policies.policies import PolicySpec, Table, run_suite
from src.utils import ROOT

ERA_PERIODS = {"elliptic": {"dev": (20, 34), "test": (35, 49)}, "baf": {"dev": (3, 3), "test": (4, 7)}}
POLICIES = ["pi0", "pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi4_noshrink", "pi5", "pi6", "pi7"]


def load_hp(key: str) -> dict:
    hp = yaml.safe_load((ROOT / "configs" / "policies" / "policy_hp.yaml").read_text())[key]
    hp = dict(hp)
    hp["h"] = np.inf if hp["h"] == "inf" else float(hp["h"])
    hp["eps"] = np.inf if hp["eps"] == "inf" else float(hp["eps"])
    return hp


def theta_for(hp: dict, name: str, D_main: int, R_main: int) -> float:
    key = re.sub(r"_D\d+_", f"_D{D_main}_", name)
    key = re.sub(r"_Rinf", f"_R{R_main}", key)
    key = re.sub(r"_R(\d+|inf)$", f"_R{R_main}", key)
    return hp["theta0"].get(key, 0.5)


def specs_for(hp: dict, name: str, D_main: int, R_main: int, which=POLICIES) -> list[PolicySpec]:
    base = dict(W=int(hp["W"]), h=hp["h"], calib=hp["calib"])
    out = []
    for n in which:
        kw = dict(base)
        if n.startswith("pi4"):
            kw.update(lam=hp["lam"], eps=hp["eps"], min_gain=hp.get("min_gain", 0.0))
        if n == "pi6":
            kw.update(alpha=hp["alpha"])
        if n == "pi0":
            kw.update(theta0=theta_for(hp, name, D_main, R_main))
        out.append(PolicySpec(n, **kw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("era", choices=["dev", "test"])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--variants", action="store_true", help="also run rho / capacity variants (main tables only)")
    a = ap.parse_args()
    key = "elliptic" if a.dataset_dir == "elliptic" else "baf"
    hp = load_hp(key)
    D_main, R_main = (2, 5) if key == "elliptic" else (1, 2)
    lo, hi = ERA_PERIODS[key][a.era]
    outp, outd = ROOT / "results" / "policy" / a.dataset_dir, ROOT / "results" / "diag" / a.dataset_dir
    outp.mkdir(parents=True, exist_ok=True); outd.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "results" / "scores" / a.dataset_dir).glob(f"*__{a.era}.parquet")):
        name = f.stem.replace(f"__{a.era}", "")
        if a.only and name not in a.only:
            continue
        D = int(re.search(r"_D(\d+)_", name).group(1))
        tab = Table(pd.read_parquet(f), D)
        specs = specs_for(hp, name, D_main, R_main)
        parts = [run_suite(tab, range(lo, hi + 1), specs, GRID).assign(variant="base")]
        is_main = (D == D_main) and ("_Rinf" not in name) and ("shuf" not in name)
        if is_main:   # A5 / H1b random-u control: u permuted within each (seed, origin) table, everything else identical
            raw = pd.read_parquet(f)
            raw["u"] = raw.groupby(["seed", "origin"])["u"].transform(lambda s_: s_.sample(frac=1.0, random_state=17).to_numpy())  # shuffle-ok
            ctl = run_suite(Table(raw, D), range(lo, hi + 1), specs_for(hp, name, D_main, R_main, ["pi3", "pi4"]), GRID)
            ctl["policy"] = ctl["policy"] + "_randu"
            parts.append(ctl.assign(variant="base"))
        if a.variants and is_main and a.era == "test":   # S6: prevalence shift by thinning events of ONE class from test periods on
            raw = pd.read_parquet(f)
            for tag, cls, keep in (("prev0.5x", 1, 0.5), ("prev2x", 0, 0.5)):
                rr = np.random.default_rng(99)
                drop = (raw["period"] >= lo) & (raw["y"] == cls) & (rr.random(len(raw)) > keep)
                res_s = run_suite(Table(raw[~drop], D), range(lo, hi + 1), specs_for(hp, name, D_main, R_main, ["pi1", "pi2", "pi3", "pi4"]), [PRIMARY])
                parts.append(res_s.assign(variant=tag))
        if a.variants and is_main:
            parts.append(run_suite(tab, range(lo, hi + 1), specs, [Costs(PRIMARY.C_FN, PRIMARY.C_FP, 1.0, 0.9)]).assign(variant="rho0.9"))
            for cf in (0.005, 0.02):
                parts.append(run_suite(tab, range(lo, hi + 1), specs, [PRIMARY], cap_frac=cf).assign(variant=f"cap{cf}"))
        res = pd.concat(parts, ignore_index=True)
        res.to_parquet(outp / f"{name}__{a.era}.parquet", index=False)
        diag = period_diagnostics(tab, range(lo, hi + 1), int(hp["W"]), hp["h"], hp["calib"])
        diag.to_parquet(outd / f"{name}__{a.era}.parquet", index=False)
        print("policy+diag", name, len(res), flush=True)


if __name__ == "__main__":
    main()

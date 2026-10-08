"""Study 2 (separately pre-registered): BitcoinHeist, amount-based costs, novelty-aware signals.

    python -m experiments.study2 tune <scorer>      dev-era random search, EQUAL budget (8 trials) for every family
    python -m experiments.study2 lock               freeze the addendum (hashes of configs AND of the policy/novelty code)
    python -m experiments.study2 score <scorer>     test-era score tables (refuses without the lock)
    python -m experiments.study2 novelty            unsupervised novelty signals for the primary scorer's score table
    python -m experiments.study2 policies           decision policies on the cached scores
    python -m experiments.study2 analyze            inference -> results/analysis_study2.json
Decision-layer settings are NOT tuned here: they are the ones frozen in Study 1 (configs/policies/policy_hp.yaml).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import bitcoinheist as BH
from src.policies import amount_policies as AP
from src.stats import inference as I
from src.training.tune import DEFAULTS, sample_hp
from src.training.walkforward import WF, run
from src.utils import ROOT, log_test_touch

S = dict(D=2, Wc=3, R=6, dev_origins=(30, 36, 42, 48), dev_era=(0, 51), test_era=(52, 90), seeds=(0, 1, 2, 3, 4),
         scorers=("lr", "rf", "xgb", "mlp"), budget=8,
         # fixed (not searched) settings; the RF is capacity-capped so that it fits in memory on a 16 GB machine
         fixed={"rf": {"max_samples": 100000, "max_depth": 16, "n_estimators": 200}, "mlp": {"bs": 4096, "patience": 4},
                "xgb": {"device": "cuda"}, "lr": {}})
CELLS = [AP.AmountCosts(cr, m) for cr in (0.05, 0.25, 1.0) for m in (0.05, 0.25, 1.0)]
PRIMARY = AP.AmountCosts(0.25, 0.25)
BLOCKS = {"T1": (52, 64), "T2": (65, 90), "ALL": (52, 90)}
CFG, SC, RES = ROOT / "configs" / "study2", ROOT / "results" / "scores" / "bitcoinheist", ROOT / "results" / "study2"
LOCK = ROOT / "results" / "PREREG_LOCK_STUDY2.txt"
FROZEN_CODE = ["src/policies/amount_policies.py", "src/policies/policies.py", "src/calibration/recal.py", "src/data/maturity.py",
               "src/data/bitcoinheist.py", "src/training/walkforward.py", "src/models/tabular.py", "src/models/nn.py", "experiments/study2.py"]


TAG = "test"


def spath(name: str):
    return SC / f"{name}__{TAG}.parquet"


def use_dev() -> None:
    """Dry-run mode (`--dev`): the same pipeline on DEV-era periods only, used to debug before the lock."""
    global TAG, RES
    TAG, RES = "dev", ROOT / "results" / "study2_dev"
    S.update(test_era=(36, 47), origins=(36, 42), seeds=(0, 1), scorers=("lr", "xgb"))
    BLOCKS.clear()
    BLOCKS.update({"T1": (36, 41), "T2": (42, 47), "ALL": (36, 47)})


def layer_hp() -> dict:
    h = yaml.safe_load((ROOT / "configs" / "policies" / "policy_hp.yaml").read_text())["elliptic"]
    return dict(calib=h["calib"], W=int(h["W"]), lam=float(h["lam"]), eps=np.inf if h["eps"] == "inf" else float(h["eps"]))


# ------------------------------------------------------------------------------------------------ tuning
def tune(scorer: str) -> None:
    ds = BH.load_dataset()
    rng = np.random.default_rng(2024)
    trials = []
    for i in range(S["budget"]):
        hp = dict(DEFAULTS[scorer]) if i == 0 else sample_hp(scorer, rng)
        hp.update(S["fixed"][scorer])
        t0 = time.time()
        df = run(ds, WF(scorer, "F_all", S["D"], S["Wc"], S["R"], S["dev_era"], (0,), hp, origins=S["dev_origins"]), verbose=False)
        aps = [average_precision_score(d.y, d.p) for tau in S["dev_origins"]
               for d in [df[(df.origin == tau) & (df.period >= tau) & (df.period <= tau + S["R"] - 1)]] if d.y.sum() > 0]
        trials.append(dict(trial=i, hp=hp, dev_auprc=float(np.mean(aps)), secs=round(time.time() - t0, 1)))
        print(f"[study2 tune {scorer}] trial {i} AUPRC={trials[-1]['dev_auprc']:.4f} ({trials[-1]['secs']}s)", flush=True)
    best = max(trials, key=lambda t: t["dev_auprc"])
    (CFG / "models").mkdir(parents=True, exist_ok=True)
    (CFG / "models" / f"{scorer}.yaml").write_text(yaml.safe_dump({"scorer": scorer, "dev_auprc": best["dev_auprc"],
                                                                    "hp": json.loads(json.dumps(best["hp"], default=list))}, sort_keys=True), encoding="utf-8")
    (ROOT / "results" / "tuning" / f"study2__{scorer}.json").write_text(json.dumps(trials, indent=1, default=list), encoding="utf-8")
    print("BEST", scorer, best["dev_auprc"], flush=True)


def tuned(scorer: str) -> dict:
    hp = dict(yaml.safe_load((CFG / "models" / f"{scorer}.yaml").read_text())["hp"])
    if "widths" in hp:
        hp["widths"] = tuple(hp["widths"])
    if scorer in ("xgb", "mlp"):
        hp["M"] = 5
    return hp


def primary() -> str:
    best = {s: yaml.safe_load((CFG / "models" / f"{s}.yaml").read_text())["dev_auprc"] for s in ("rf", "xgb", "mlp")}
    return max(best, key=best.get)


# ------------------------------------------------------------------------------------------------ lock
def digest() -> dict:
    out = {}
    for f in sorted(CFG.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # line-ending independent
    for rel in FROZEN_CODE + ["configs/policies/policy_hp.yaml"]:
        out[rel] = hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def lock(verify: bool = False) -> None:
    now = digest()
    if verify:
        rec = json.loads(LOCK.read_text())["files"]
        bad = [k for k in set(rec) | set(now) if rec.get(k) != now.get(k)]
        sys.exit(f"STUDY 2 FROZEN FILES CHANGED: {bad}") if bad else print("study-2 lock verified:", len(rec), "files unchanged")
        return
    if LOCK.exists():
        sys.exit("already locked")
    need = [CFG / "preregistration_study2.yaml", CFG / "split.json"] + [CFG / "models" / f"{s}.yaml" for s in S["scorers"]]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        sys.exit(f"cannot lock, missing {miss}")
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": now}, indent=1))
    print("STUDY 2 LOCKED", len(now), "files")


# ------------------------------------------------------------------------------------------------ scoring
def score(scorer: str) -> None:
    if TAG == "test":
        if not LOCK.exists():
            sys.exit("REFUSED: Study 2 test era is locked until `python -m experiments.study2 lock`.")
        log_test_touch(f"study2:scores:{scorer}", "generate Study 2 test-era scores from frozen configs")
    ds = BH.load_dataset()
    df = run(ds, WF(scorer, "F_all", S["D"], S["Wc"], S["R"], S["test_era"], S["seeds"], tuned(scorer), origins=S.get("origins")), verbose=True)
    SC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spath(scorer), index=False)
    print("saved", scorer, len(df), flush=True)


# ------------------------------------------------------------------------------------------------ novelty signals
LOGCOLS = ["length", "log1p_weight", "log1p_count", "log1p_looped", "log1p_neighbors", "log1p_income_btc"]


def novelty() -> None:
    """Three unsupervised signals, each fitted on the FEATURES of the scorer's training periods only (no labels):
    isolation forest (200 trees, defaults); Mahalanobis distance; mean distance to the 10 nearest of 50,000 pool events."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    ds = BH.load_dataset()
    sc = pd.read_parquet(spath(primary()), columns=["origin", "rid"]).drop_duplicates()
    idx = [ds.feat_names.index(c) for c in LOGCOLS]
    Z = ds.X[:, idx].astype(np.float64).copy()
    Z[:, 0] = np.log1p(Z[:, 0])
    parts = []
    for tau in sorted(sc.origin.unique()):
        cut = int(tau) - S["D"] - 1 - S["Wc"]
        pool = np.flatnonzero(ds.period <= cut)
        mu, sd = Z[pool].mean(0), Z[pool].std(0) + 1e-9
        Zp = (Z[pool] - mu) / sd
        rid = np.sort(sc[sc.origin == tau].rid.to_numpy())
        Zq = (Z[rid] - mu) / sd
        iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(ds.X[pool])
        u_if = -iso.score_samples(ds.X[rid])
        cov = np.cov(Zp, rowvar=False) + 1e-6 * np.eye(Zp.shape[1])
        u_maha = np.sqrt(np.einsum("ij,jk,ik->i", Zq, np.linalg.inv(cov), Zq))
        sub = np.random.default_rng(0).choice(len(Zp), size=min(50_000, len(Zp)), replace=False)
        nn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree").fit(Zp[sub])
        u_knn = nn.kneighbors(Zq)[0].mean(1)
        parts.append(pd.DataFrame({"origin": np.int16(tau), "rid": rid.astype(np.int32), "u_if": u_if, "u_maha": u_maha, "u_knn": u_knn}))
        print("novelty origin", tau, len(rid), flush=True)
    pd.concat(parts).to_parquet(spath("novelty"), index=False)


# ------------------------------------------------------------------------------------------------ policies
def _perm(df, col):
    out = df.copy()
    out[col] = out.groupby(["seed", "origin"])[col].transform(lambda s_: s_.sample(frac=1.0, random_state=17).to_numpy())  # shuffle-ok
    return out


def policies() -> None:
    ds = BH.load_dataset()
    A, hp, per = ds.extra["amount"], layer_hp(), range(S["test_era"][0], S["test_era"][1] + 1)
    RES.mkdir(parents=True, exist_ok=True)
    ps = primary()
    for sc in S["scorers"]:
        df = pd.read_parquet(spath(sc))
        cells = CELLS if sc == ps else [PRIMARY]
        parts = [AP.run_suite(AP.AmountTable(df, A, S["D"]), per, ["pi1", "pi2", "pi3", "pi4", "pi4_nodelta", "pi7"], cells, hp)]
        parts.append(AP.run_suite(AP.AmountTable(_perm(df, "u"), A, S["D"]), per, ["pi3", "pi4"], cells, hp, "_randu"))
        if sc == ps:
            nov = df.merge(pd.read_parquet(spath("novelty")), on=["origin", "rid"], how="left")
            assert nov[["u_if", "u_maha", "u_knn"]].notna().all().all()
            for col, tag in (("u_if", "_if"), ("u_maha", "_maha"), ("u_knn", "_knn")):
                parts.append(AP.run_suite(AP.AmountTable(nov, A, S["D"], ucol=col), per, ["pi3"], cells, hp, tag))
                parts.append(AP.run_suite(AP.AmountTable(_perm(nov, col), A, S["D"], ucol=col), per, ["pi3"], cells, hp, tag + "_rand"))
        pd.concat(parts, ignore_index=True).assign(scorer=sc).to_parquet(RES / f"policy__{sc}.parquet", index=False)
        print("policies", sc, flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def _pp(df, pol, cost):
    return df[(df.policy == pol) & (df.cost == cost)].groupby("period")[["n", "cost_sum", "n_review", "n_fn", "n_pos", "amt_missed", "amt_illicit"]].mean()


def _cmp(df, ctrl, new, lo, hi, cost, B=5000):
    a, b = _pp(df, ctrl, cost), _pp(df, new, cost)
    a, b = a[(a.index >= lo) & (a.index <= hi)], b[(b.index >= lo) & (b.index <= hi)]
    r = I.block_bootstrap_gain(a.cost_sum.to_numpy(), b.cost_sum.to_numpy(), a.n.to_numpy(), L=2, B=B, seed=0)
    r.update(I.effect_sizes(a.cost_sum, b.cost_sum, a.n))
    r.update(cost_ctrl=1000 * a.cost_sum.sum() / a.n.sum(), cost_new=1000 * b.cost_sum.sum() / b.n.sum(),
             review_ctrl=1000 * a.n_review.sum() / a.n.sum(), review_new=1000 * b.n_review.sum() / b.n.sum(),
             missed_amt_share_new=b.amt_missed.sum() / max(b.amt_illicit.sum(), 1e-12), missed_share_new=b.n_fn.sum() / max(b.n_pos.sum(), 1))
    r["tost"] = I.tost_noninferiority(a.cost_sum, b.cost_sum, a.n)
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}


def analyze() -> dict:
    ds = BH.load_dataset()
    ps, pk = primary(), PRIMARY.key()
    df = pd.read_parquet(RES / f"policy__{ps}.parquet")
    out = {"primary_scorer": ps, "blocks": BLOCKS, "primary_cell": pk}
    pols = ["pi1", "pi3", "pi4", "pi4_nodelta", "pi3_randu", "pi4_randu", "pi3_if", "pi3_maha", "pi3_knn", "pi3_if_rand", "pi3_maha_rand", "pi3_knn_rand", "pi7"]
    out["main"] = {b: {p: _cmp(df, "pi2", p, lo, hi, pk, B=3000) for p in pols} for b, (lo, hi) in BLOCKS.items()}
    lo, hi = BLOCKS["ALL"]
    conf = {"S2H1_pi2_vs_pi1": _cmp(df, "pi1", "pi2", lo, hi, pk), "S2H2_pi3_vs_pi2": _cmp(df, "pi2", "pi3", lo, hi, pk),
            "S2H3_pi4_vs_pi2": _cmp(df, "pi2", "pi4", lo, hi, pk), "S2H4_if_vs_pi2": _cmp(df, "pi2", "pi3_if", lo, hi, pk),
            "S2H4_maha_vs_pi2": _cmp(df, "pi2", "pi3_maha", lo, hi, pk), "S2H4_knn_vs_pi2": _cmp(df, "pi2", "pi3_knn", lo, hi, pk),
            "pi3_vs_randu": _cmp(df, "pi3_randu", "pi3", lo, hi, pk), "if_vs_rand": _cmp(df, "pi3_if_rand", "pi3_if", lo, hi, pk),
            "maha_vs_rand": _cmp(df, "pi3_maha_rand", "pi3_maha", lo, hi, pk), "knn_vs_rand": _cmp(df, "pi3_knn_rand", "pi3_knn", lo, hi, pk),
            "pi4_vs_nodelta": _cmp(df, "pi4_nodelta", "pi4", lo, hi, pk)}
    out["confirmatory"] = conf
    fam = {k: conf[k]["p"] for k in ("S2H1_pi2_vs_pi1", "S2H3_pi4_vs_pi2", "S2H4_if_vs_pi2", "S2H4_maha_vs_pi2", "S2H4_knn_vs_pi2")}
    out["holm"] = {k: dict(p=fam[k], p_adj=v[0], reject=bool(v[1])) for k, v in I.holm(fam).items()}
    out["grid"] = [dict(c_r=c.c_r, m=c.m, **{f"{p}_{k}": _cmp(df, "pi2", p, lo, hi, c.key(), B=1000)[k] for p in ("pi1", "pi3", "pi4", "pi3_if", "pi3_maha", "pi3_knn") for k in ("est", "lo", "hi")},
                        cost_pi2=_cmp(df, "pi2", "pi3", lo, hi, c.key(), B=200)["cost_ctrl"]) for c in CELLS]
    out["series"] = [dict(policy=p, period=int(t), cost_per_1000=1000 * r.cost_sum / r.n, reviews_per_1000=1000 * r.n_review / r.n)
                     for p in ("pi1", "pi2", "pi3", "pi4", "pi3_if", "pi3_maha", "pi3_knn", "pi3_randu") for t, r in _pp(df, p, pk).iterrows()]
    # all scorers, primary cell
    sc_rows = []
    for s in S["scorers"]:
        d = pd.read_parquet(RES / f"policy__{s}.parquet")
        raw = pd.read_parquet(spath(s))
        cur = raw[raw.period >= raw.origin]
        cur = cur[cur.groupby(["seed", "period"]).origin.transform("max") == cur.origin]
        ap = cur.groupby("seed").apply(lambda g: average_precision_score(g.y, g.p), include_groups=False).mean()
        for b, (l, h) in BLOCKS.items():
            r1, r3, r4 = _cmp(d, "pi1", "pi2", l, h, pk, B=500), _cmp(d, "pi2", "pi3", l, h, pk, B=500), _cmp(d, "pi2", "pi4", l, h, pk, B=500)
            sc_rows.append(dict(scorer=s, block=b, auprc_all=float(ap), cost_pi1=r1["cost_ctrl"], cost_pi2=r1["cost_new"], gain_pi2_vs_pi1=r1["est"],
                                gain_pi3=r3["est"], gain_pi3_lo=r3["lo"], gain_pi3_hi=r3["hi"], gain_pi4=r4["est"], gain_pi4_lo=r4["lo"], gain_pi4_hi=r4["hi"]))
    out["scorers"] = sc_rows
    # novel-family analysis (evaluation only): an illicit event is NOVEL if its family never appears in the scorer's training periods
    raw = pd.read_parquet(spath(ps)).merge(pd.read_parquet(spath("novelty")), on=["origin", "rid"], how="left")
    raw = raw[(raw.period >= raw.origin) & (raw.seed == 0)]
    raw = raw[raw.groupby("period").origin.transform("max") == raw.origin]
    fam_first = pd.DataFrame({"family": ds.extra["family"], "period": ds.period, "y": ds.y}).query("y == 1").groupby("family").period.min()
    raw["family"] = ds.extra["family"][raw.rid.to_numpy()]
    raw["cut"] = raw.origin - S["D"] - 1 - S["Wc"]
    raw["novel"] = (raw.y == 1) & (raw.family.map(fam_first) > raw.cut)
    pos = raw[raw.y == 1]
    nov = {}
    for name, d in (("novel", pos[pos.novel]), ("known", pos[~pos.novel])):
        nov[name] = dict(n=int(len(d)), mean_p=float(d.p.mean()), median_p=float(d.p.median()), mean_u=float(d.u.mean()),
                         mean_if=float(d.u_if.mean()), mean_maha=float(d.u_maha.mean()), mean_knn=float(d.u_knn.mean()))
    z = pos.novel.astype(int)
    nov["auroc_for_novel_family"] = {c: float(roc_auc_score(z, pos[c])) for c in ("u", "u_if", "u_maha", "u_knn")} if 0 < z.sum() < len(z) else None
    nov["auroc_score_for_novel_family_low_p"] = float(roc_auc_score(z, -pos.p)) if 0 < z.sum() < len(z) else None
    out["novel_family"] = nov
    addr = ds.extra["addr"]
    seen = pd.Series(ds.period, index=addr).groupby(level=0).min()
    te = raw.assign(first=seen.reindex(addr[raw.rid.to_numpy()]).to_numpy())
    out["share_test_events_from_unseen_addresses"] = float((te["first"] > te["cut"]).mean())
    out["per_period"] = [dict(period=int(p), n=int(len(g)), n_pos=int(g.y.sum()), n_novel=int(g.novel.sum())) for p, g in raw.groupby("period")]
    (ROOT / "results" / ("analysis_study2.json" if TAG == "test" else "analysis_study2_dev.json")).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    if "--dev" in sys.argv:
        sys.argv.remove("--dev")
        use_dev()
    cmd = sys.argv[1]
    if cmd == "tune":
        tune(sys.argv[2])
    elif cmd == "lock":
        lock(verify="--verify" in sys.argv)
    elif cmd == "score":
        score(sys.argv[2])
    elif cmd == "novelty":
        novelty()
    elif cmd == "policies":
        policies()
    elif cmd == "analyze":
        o = analyze()
        print("primary", o["primary_scorer"])
        for k, v in o["confirmatory"].items():
            print(f"{k:22s} gain={v['est']:9.2f} [{v['lo']:9.2f},{v['hi']:9.2f}] rel={100 * v['rel']:6.2f}% p={v['p']:.4f}")
        print(json.dumps(o["holm"], indent=1))
        print(json.dumps(o["novel_family"], indent=1))

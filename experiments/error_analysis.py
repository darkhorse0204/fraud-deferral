"""Error analysis on the FROZEN policies (primary scorer). Exploratory; no policy is changed.
 (1) Does the ensemble know when it is wrong?  among illicit events, compare u of events pi2 approves (missed) vs not.
 (2) Why does MAD hurt?  trajectory of the thresholds / epistemic cap it picks vs the Chow thresholds.
 (3) Where does cost come from? decomposition of pi2's cost by action/truth in T1 vs T2.
python -m experiments.error_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from experiments.analyze import MAIN, primary_scorer, table_name
from experiments.run_policies import load_hp, specs_for
from src.evaluation.metrics import APPROVE, BLOCK, PRIMARY, REVIEW, chow_thresholds
from src.policies.policies import Table, decide
from src.utils import ROOT


def run(dataset_dir="elliptic", D_main=2, R_main=5, blocks=None):
    blocks = blocks or {"T1": (35, 42), "T2": (43, 49)}
    key = "elliptic" if dataset_dir == "elliptic" else "baf"
    ps = primary_scorer(key)
    name = table_name(ps, "F_all", key)
    hp = load_hp(key)
    tab = Table(pd.read_parquet(ROOT / "results" / "scores" / dataset_dir / f"{name}__test.parquet"), D_main)
    spec2, spec4 = [s for s in specs_for(hp, name, D_main, R_main, ["pi2", "pi4"])]
    a0, b0 = chow_thresholds(PRIMARY)
    rows, traj = [], []
    for seed in tab.seeds:
        st2, st4 = {}, {}
        for t in range(35, 50):
            cur = tab.current(seed, t)
            w2, w4 = tab.window(seed, t, spec2.W, spec2.h), tab.window(seed, t, spec4.W, spec4.h)
            act2, pc, st2 = decide(spec2, PRIMARY, w2, cur, st2)
            act4, _, st4 = decide(spec4, PRIMARY, w4, cur, st4)
            la, lb = st4.get("mult", (0.0, 0.0))
            traj.append(dict(seed=seed, period=t, ma=float(np.exp(la)), mb=float(np.exp(lb)), review_rate_pi2=float((act2 == REVIEW).mean()),
                             review_rate_pi4=float((act4 == REVIEW).mean()), win_pos=int(w4["y"].sum()), win_n=len(w4["y"]),
                             mean_pc_cur=float(pc.mean()), prev_cur=float(cur["y"].mean()), prev_win=float(np.mean(w4["y"])) if len(w4["y"]) else np.nan))
            for i in range(len(pc)):
                pass
            rows.append(pd.DataFrame(dict(seed=seed, period=t, y=cur["y"], pc=pc, u=cur["u"], act2=act2, act4=act4)))
    ev = pd.concat(rows, ignore_index=True)
    tr = pd.DataFrame(traj)
    out = {"scorer": ps, "table": name, "chow": {"a": a0, "b": b0}}
    for bn, (lo, hi) in blocks.items():
        e = ev[ev.period.between(lo, hi)]
        pos = e[e.y == 1]
        missed = (pos.act2 == APPROVE).astype(int)
        out[bn] = dict(
            n_pos=int(len(pos) / max(len(tab.seeds), 1)), missed_rate_pi2=float(missed.mean()),
            auroc_u_for_missed=float(roc_auc_score(missed, pos.u)) if 0 < missed.sum() < len(missed) else None,
            auroc_pc_for_missed=float(roc_auc_score(missed, -pos.pc)) if 0 < missed.sum() < len(missed) else None,
            mean_u_missed=float(pos[missed == 1].u.mean()) if missed.sum() else None, mean_u_caught=float(pos[missed == 0].u.mean()) if (missed == 0).sum() else None,
            mean_u_negatives=float(e[e.y == 0].u.mean()), mean_pc_missed=float(pos[missed == 1].pc.mean()) if missed.sum() else None,
            cost_share={"fn_approved": float(30 * ((e.act2 == APPROVE) & (e.y == 1)).sum()),
                        "review": float((e.act2 == REVIEW).sum()), "fp_blocked": float(5 * ((e.act2 == BLOCK) & (e.y == 0)).sum())},
            mad_ma=float(tr[tr.period.between(lo, hi)].ma.mean()), mad_mb=float(tr[tr.period.between(lo, hi)].mb.mean()),
            pi2_review_rate=float(tr[tr.period.between(lo, hi)].review_rate_pi2.mean()), pi4_review_rate=float(tr[tr.period.between(lo, hi)].review_rate_pi4.mean()),
            prev_cur=float(tr[tr.period.between(lo, hi)].prev_cur.mean()), prev_win=float(tr[tr.period.between(lo, hi)].prev_win.mean()))
    out["trajectory"] = tr.groupby("period")[["ma", "mb", "review_rate_pi2", "review_rate_pi4", "prev_cur", "prev_win", "mean_pc_cur"]].mean().round(4).reset_index().to_dict("records")
    (ROOT / "results" / f"error_analysis_{dataset_dir}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = run()
    for b in ("T1", "T2"):
        print(b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o[b].items() if k != "cost_share"}, o[b]["cost_share"])
    print(pd.DataFrame(o["trajectory"]).to_string(index=False))

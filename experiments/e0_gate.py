"""E0 reproduction gate: default Random Forest under Weber et al.'s protocol (train steps 1-34, test 35-49).

Purpose: confirm the pipeline reproduces the published QUALITATIVE pattern (illicit-class F1 collapses after
step 43). No tuning, no use of the result for any design decision. Touches test-era labels -> logged.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import splits
from src.data.loaders import load_elliptic
from src.features.featuresets import elliptic_sets
from src.utils import ROOT, log_test_touch


def main() -> None:
    sp = splits.load("elliptic")
    nodes, _ = load_elliptic()
    lab = nodes[nodes["label"] >= 0]
    out = {}
    for fs, cols in elliptic_sets().items():
        tr = lab[lab["period"] <= sp["dev"][1]]
        rf = RandomForestClassifier(n_estimators=50, max_features=50, random_state=0, n_jobs=-1).fit(tr[cols], tr["label"])
        res = {}
        for name, (a, b) in {"T1": sp["test_blocks"]["T1"], "T2": sp["test_blocks"]["T2"]}.items():
            te = lab[lab["period"].between(a, b)]
            pr = rf.predict(te[cols])
            res[name] = {"illicit_precision": float(precision_score(te["label"], pr, zero_division=0)),
                         "illicit_recall": float(recall_score(te["label"], pr, zero_division=0)),
                         "illicit_f1": float(f1_score(te["label"], pr, zero_division=0)),
                         "n_illicit": int(te["label"].sum())}
        res["per_step_f1"] = {int(p): round(float(f1_score(g["label"], rf.predict(g[cols]), zero_division=0)), 3)
                              for p, g in lab[lab["period"] >= sp["test"][0]].groupby("period")}
        out[fs] = res
    log_test_touch("E0", "reproduction gate: default RF, Weber protocol; qualitative step-43 collapse check only")
    p = ROOT / "results" / "e0_gate.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for fs, r in out.items():
        print(fs, "T1 F1=%.3f  T2 F1=%.3f" % (r["T1"]["illicit_f1"], r["T2"]["illicit_f1"]))
        print("  per-step F1:", r["per_step_f1"])


if __name__ == "__main__":
    main()

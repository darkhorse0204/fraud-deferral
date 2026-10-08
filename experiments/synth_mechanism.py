"""E4 (synthetic): mechanism check of Proposition 1. NEVER used to rank methods or support headline claims.

Direct (p, u, y) generator, so that the only thing that differs between conditions is whether u carries label
information beyond the score p:
  * before the shift (t < T0): the scorer is perfectly calibrated, p = r = true risk, u is pure noise;
  * after the shift, with magnitude s:
      local         a random 20% subset S has true risk r*(1+3s) (clipped); scorer still reports p = r; u is high on S
                    (informative: P(y | p, u) depends on u)
      local_randu   identical risk shift on S, but u is independent noise (control: same marginal miscalibration)
      uniform       every event has risk r*(1+0.6s); u is noise (miscalibration absorbed by recalibration)
Prop. 1 predicts: gain from uncertainty-aware policies ~ 0 when ce_u ~ 0 (local_randu, uniform), > 0 when ce_u > 0.
The random-u control additionally tests the "hedging" confound: reviewing extra events can lower cost simply
because the stale model under-predicts risk, regardless of whether u is informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import PRIMARY, ce_u, ece
from src.policies.policies import PolicySpec, Table, run_suite
from src.utils import ROOT

T, T0, N, D, WC = 28, 16, 2500, 2, 3


def make_table(s: float, family: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, rid = [], 0
    for t in range(T0 - D - 1 - WC + 1, T + 1):          # scorer reserve periods onward (as in the real pipeline)
        r = rng.beta(0.5, 5.0, N)
        u = rng.random(N) * 0.01
        risk = r.copy()
        if t >= T0 and s > 0:
            if family in ("local", "local_randu"):
                S = rng.random(N) < 0.2
                risk = np.where(S, np.clip(r * (1 + 3 * s), 0, 1), r)
                if family == "local":
                    u = np.where(S, 0.02 + rng.random(N) * 0.01, u)
            elif family == "uniform":
                risk = np.clip(r * (1 + 0.6 * s), 0, 1)
        y = (rng.random(N) < risk).astype(int)
        rows.append(pd.DataFrame({"origin": T0, "seed": 0, "rid": np.arange(rid, rid + N), "period": t, "y": y, "p": r, "u": u}))
        rid += N
    return pd.concat(rows, ignore_index=True)


def main(reps: int = 30) -> pd.DataFrame:
    out = []
    specs = [PolicySpec("pi2", W=5), PolicySpec("pi3", W=5), PolicySpec("pi4", W=5, lam=0.02, min_gain=0.01)]
    for family in ("local", "local_randu", "uniform"):
        for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            g3, g4, cu, ec = [], [], [], []
            for r in range(reps):
                df = make_table(s, family, 7919 * r + int(10 * s))
                tab = Table(df, D)
                res = run_suite(tab, range(T0, T + 1), specs, [PRIMARY])
                c = res.groupby("policy")[["cost_sum", "n"]].sum()
                gain = lambda k: 1000 * (c.loc["pi2", "cost_sum"] - c.loc[k, "cost_sum"]) / c.loc["pi2", "n"]  # noqa: E731
                g3.append(gain("pi3")); g4.append(gain("pi4"))
                post = df[df.period >= T0]                    # evaluation-only diagnostics on the shifted periods
                cu.append(ce_u(post.p.to_numpy(), post.u.to_numpy(), post.y.to_numpy())); ec.append(ece(post.p, post.y))
            se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a)))  # noqa: E731
            out.append(dict(family=family, s=s, gain_pi3=float(np.mean(g3)), gain_pi3_se=se(g3), gain_pi4=float(np.mean(g4)),
                            gain_pi4_se=se(g4), ce_u=float(np.mean(cu)), ece=float(np.mean(ec))))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "synth_mechanism.csv", index=False)
    return df


if __name__ == "__main__":
    main()

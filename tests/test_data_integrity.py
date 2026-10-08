"""Real-data integrity + leakage checks. These run against data/processed and the frozen splits.

They are the Phase-0 gate: if any fails, no model code may be written.
"""
import numpy as np
import pytest

from src.data import splits
from src.data.download import ROOT, find_file
from src.data.graph import train_graph
from src.data.loaders import PROC, build_elliptic, load_baf, load_elliptic
from src.data.maturity import pool_mask
from src.features.featuresets import PoolScaler, assert_no_clock, elliptic_sets

ell_ok = find_file(ROOT / "data" / "raw" / "elliptic", "elliptic_txs_features.csv") is not None
baf_ok = find_file(ROOT / "data" / "raw" / "baf", "Base.csv") is not None


@pytest.fixture(scope="module")
def ell():
    build_elliptic()
    return load_elliptic()


@pytest.mark.skipif(not ell_ok, reason="Elliptic not downloaded")
class TestElliptic:
    def test_published_counts(self, ell):
        nodes, edges = ell
        assert len(nodes) == 203_769 and len(edges) == 234_355
        vc = nodes["label"].value_counts()
        assert vc[1] == 4_545 and vc[0] == 42_019  # fixes the class coding: '1' illicit, '2' licit

    def test_no_cross_period_edges(self, ell):
        nodes, edges = ell
        per = nodes.set_index("txId")["period"]
        assert (per.loc[edges["src_txId"]].to_numpy() == per.loc[edges["dst_txId"]].to_numpy()).all()

    def test_feature_sets_contain_no_clock_column(self, ell):
        nodes, _ = ell
        for names in elliptic_sets().values():
            assert_no_clock(names)
            assert set(names) <= set(nodes.columns)
            assert "tx_f1" not in nodes.columns  # clock stored only as `period`

    def test_eras_are_disjoint_and_ordered(self, ell):
        nodes, _ = ell
        s = splits.load("elliptic")
        dev = nodes["period"].between(*s["dev"])
        test = nodes["period"].between(*s["test"])
        assert not (dev & test).any() and (dev | test).all()
        assert nodes.loc[dev, "period"].max() < nodes.loc[test, "period"].min()

    @pytest.mark.parametrize("D", [0, 2, 4])
    def test_walkforward_training_pool_never_touches_test_period(self, ell, D):
        nodes, edges = ell
        s = splits.load("elliptic")
        per = nodes["period"].to_numpy()
        for t in range(s["test"][0], s["test"][1] + 1):
            m = pool_mask(per, t, D)
            assert per[m].max() <= t - D - 1 < t
            if t == s["test"][0]:  # the first test period may be trained on dev data only
                assert per[m].max() <= s["dev"][1]

    def test_train_graph_inductive_on_real_graph(self, ell):
        nodes, edges = ell
        idx = pd_index(nodes)
        src, dst = idx.loc[edges["src_txId"]].to_numpy(), idx.loc[edges["dst_txId"]].to_numpy()
        per = nodes["period"].to_numpy()
        for t, D in [(35, 0), (43, 2), (49, 4)]:
            ei, nodes_kept = train_graph(per, src, dst, t, D)
            assert (per[nodes_kept] + D < t).all()
            assert ei.shape[1] > 0

    def test_scaler_on_real_data_ignores_test_era(self, ell):
        nodes, _ = ell
        cols = elliptic_sets()["F_all"]
        X = nodes[cols].to_numpy(dtype="float64")
        per = nodes["period"].to_numpy()
        t, D = 35, 2
        sc = PoolScaler().fit(X, per, t, D)
        assert sc.max_period_ <= t - D - 1
        X2 = X.copy()
        X2[per >= t] = 1e6  # corrupt everything the model must not see
        sc2 = PoolScaler().fit(X2, per, t, D)
        np.testing.assert_array_equal(sc.mean_, sc2.mean_)
        np.testing.assert_array_equal(sc.std_, sc2.std_)


def pd_index(nodes):
    import pandas as pd
    return pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())


@pytest.mark.skipif(not baf_ok, reason="BAF not downloaded")
class TestBAF:
    def test_basic_schema(self):
        df = load_baf("Base")
        assert len(df) == 1_000_000
        assert set(df["fraud_bool"].unique()) <= {0, 1}
        assert sorted(df["month"].unique()) == list(range(8))

    def test_feature_set_excludes_month_and_target(self):
        df = load_baf("Base")
        feats = [c for c in df.columns if c not in ("fraud_bool", "month")]
        assert_no_clock(feats, "fraud_bool")

    def test_eras(self):
        s = splits.load("baf")
        assert s["dev"][1] + 1 == s["test"][0]

"""L2: the training graph must contain only matured nodes and edges between them."""
import numpy as np

from src.data.graph import period_graph, train_graph


def _toy(n_per=40, periods=range(1, 16), seed=0):
    rng = np.random.default_rng(seed)
    node_period = np.repeat(np.array(list(periods)), n_per)
    src, dst = [], []
    for p in periods:  # edges only within a period, as in Elliptic
        idx = np.flatnonzero(node_period == p)
        src.extend(rng.choice(idx, 80))
        dst.extend(rng.choice(idx, 80))
    return node_period, np.array(src), np.array(dst)


def test_train_graph_node_set_subset_of_pool():
    npd, s, d = _toy()
    for D in (0, 2, 4):
        for t in range(2, 20):
            ei, nodes = train_graph(npd, s, d, t, D)
            assert (npd[nodes] + D < t).all()
            assert ei.size == 0 or ei.max() < len(nodes)
            if ei.size:
                assert (npd[nodes[ei[0]]] + D < t).all()
                assert (npd[nodes[ei[1]]] + D < t).all()


def test_train_graph_drops_cross_boundary_edges():
    npd = np.array([1, 1, 5, 5])
    s, d = np.array([0, 1]), np.array([1, 2])  # edge (1,2) crosses periods 1 -> 5
    ei, nodes = train_graph(npd, s, d, t=4, D=0)  # pool = nodes of period 1 only
    assert nodes.tolist() == [0, 1]
    assert ei.shape[1] == 1


def test_period_graph_single_period_only():
    npd, s, d = _toy()
    ei, nodes = period_graph(npd, s, d, 7)
    assert (npd[nodes] == 7).all()
    assert ei.shape[1] == 80

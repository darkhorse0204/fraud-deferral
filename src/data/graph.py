"""Inductive per-period graphs for Elliptic (nodes = transactions).

Edges never cross periods in this dataset (asserted in validate.py). The training graph for a
decision at period t contains ONLY nodes in the matured pool and edges with both endpoints there.
"""
from __future__ import annotations

import numpy as np

from src.data.maturity import pool_mask


def _remap(edge_src: np.ndarray, edge_dst: np.ndarray, keep_nodes: np.ndarray, n_total: int):
    """Keep edges with both endpoints in keep_nodes (boolean over n_total); return local indices."""
    keep_e = keep_nodes[edge_src] & keep_nodes[edge_dst]
    idx = np.full(n_total, -1, dtype=np.int64)
    idx[np.flatnonzero(keep_nodes)] = np.arange(int(keep_nodes.sum()))
    return np.stack([idx[edge_src[keep_e]], idx[edge_dst[keep_e]]]), np.flatnonzero(keep_nodes)


def train_graph(node_period: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray, t: int, D: int):
    """Training graph at decision period t: (edge_index[2,E], global node ids)."""
    keep = pool_mask(node_period, t, D)
    ei, nodes = _remap(edge_src, edge_dst, keep, len(node_period))
    assert (node_period[nodes] + D < t).all(), "training graph contains non-matured node"
    return ei, nodes


def period_graph(node_period: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray, p: int):
    """Inference graph for a single period p (features observable at decision time)."""
    keep = node_period == p
    ei, nodes = _remap(edge_src, edge_dst, keep, len(node_period))
    return ei, nodes

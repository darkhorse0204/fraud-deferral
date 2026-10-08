"""Frozen splits are ordered, tile the test era, and are tamper-evident; eval grouping drops nothing."""
import json

import numpy as np
import pytest

from src.data import splits
from src.data.periods import eval_groups


def test_frozen_splits_load_and_are_ordered():
    splits.write_all()
    for name in ("elliptic", "baf"):
        s = splits.load(name)
        assert s["dev"][1] < s["test"][0]


def test_tamper_detected():
    splits.write_all()
    p = splits.SPLIT_DIR / "elliptic.json"
    original = p.read_text()
    rec = json.loads(original)
    try:
        rec["dev"] = [1, 36]  # leak test steps into dev
        p.write_text(json.dumps(rec))
        with pytest.raises(RuntimeError):
            splits.load("elliptic")
    finally:
        p.write_text(original)


def test_overlap_rejected():
    bad = {"dev": [1, 40], "test": [35, 49], "test_blocks": {"T": [35, 49]}}
    with pytest.raises(AssertionError):
        splits.validate_split(bad)


def test_eval_groups_cover_all_periods_and_reach_min_pos():
    rng = np.random.default_rng(3)
    periods = np.repeat(np.arange(1, 21), 100)
    labels = (rng.random(len(periods)) < 0.01).astype(int)
    g = eval_groups(periods, labels, min_pos=5)
    assert set(g) == set(range(1, 21))  # nothing dropped
    ids = sorted(set(g.values()))
    assert ids == list(range(len(ids)))
    for gi in ids[:-1]:
        members = [p for p, x in g.items() if x == gi]
        assert labels[np.isin(periods, members)].sum() >= 5
    for gi in ids:  # groups are contiguous in time
        members = sorted(p for p, x in g.items() if x == gi)
        assert members == list(range(members[0], members[-1] + 1))

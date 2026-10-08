"""Period assignment and evaluation grouping.

Native periods (Elliptic time step, BAF month) define maturity and splits. Sparse periods
are merged ONLY for evaluation/reporting groups, never for maturity logic.
"""
from __future__ import annotations

import numpy as np


def eval_groups(periods: np.ndarray, labels: np.ndarray, min_pos: int = 5) -> dict[int, int]:
    """Map native period -> evaluation group id.

    Deterministic greedy rule (fixed in advance): walk periods in order, accumulate consecutive
    periods until the group holds >= min_pos positives; a trailing under-filled group is merged
    into the previous one. Nothing is deleted (survivorship, leak L17).
    """
    periods = np.asarray(periods)
    labels = np.asarray(labels)
    uniq = np.sort(np.unique(periods))
    pos = {int(p): int(labels[periods == p].sum()) for p in uniq}
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_pos = 0
    for p in uniq:
        cur.append(int(p))
        cur_pos += pos[int(p)]
        if cur_pos >= min_pos:
            groups.append(cur)
            cur, cur_pos = [], 0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return {p: gi for gi, g in enumerate(groups) for p in g}

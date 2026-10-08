"""Static leakage guard (L4, L11). Fails CI if forbidden split/shuffle APIs appear in src/.

Run from anywhere: python scripts/check_leakage.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = [
    (r"\btrain_test_split\b", "random split (use src.data.splits)"),
    (r"\b(Stratified|Shuffle|Group)?KFold\b", "random K-fold on a time series"),
    (r"\bStratifiedShuffleSplit\b", "random split"),
    (r"shuffle\s*=\s*True", "shuffle=True in data handling (allowed only inside minibatch loaders: mark with '# minibatch-ok')"),
    (r"\.sample\(\s*frac\s*=\s*1", "full-frame shuffle"),
]


def main() -> int:
    bad = []
    for f in (ROOT / "src").rglob("*.py"):
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "# minibatch-ok" in line or line.lstrip().startswith("#"):
                continue
            for pat, why in BANNED:
                if re.search(pat, line):
                    bad.append(f"{f.relative_to(ROOT)}:{ln}: {why}: {line.strip()}")
    if bad:
        print("LEAKAGE GUARD FAILED:\n" + "\n".join(bad))
        return 1
    print("leakage guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

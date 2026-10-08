"""Frozen era splits. Period ranges are inclusive and written to a hashed JSON file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "configs" / "splits"

# Fixed BEFORE any model was run. Elliptic regime break (T2) taken from the literature
# (Weber et al. 2019: dark-market shutdown near step 43) -- disclosed as leak L19.
SPLITS = {
    "elliptic": {
        "dev": [1, 34],
        "test": [35, 49],
        "test_blocks": {"T1": [35, 42], "T2": [43, 49]},
        "delay_grid": [0, 2, 4],
    },
    "baf": {
        "dev": [0, 3],
        "test": [4, 7],
        "test_blocks": {"T1": [4, 5], "T2": [6, 7]},
        "delay_grid": [0, 1, 2],
    },
}


def validate_split(s: dict) -> None:
    d0, d1 = s["dev"]
    t0, t1 = s["test"]
    assert d0 <= d1 and t0 <= t1, "empty era"
    assert d1 < t0, "dev and test eras overlap or are out of order"
    assert t0 == d1 + 1, "gap between dev and test eras"
    blocks = sorted(s["test_blocks"].values())
    assert blocks[0][0] == t0 and blocks[-1][1] == t1, "test blocks must tile the test era"
    for a, b in zip(blocks, blocks[1:]):
        assert b[0] == a[1] + 1, "test blocks must be contiguous and disjoint"


def split_hash(s: dict) -> str:
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()


def write_all() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in SPLITS.items():
        validate_split(s)
        rec = dict(s, name=name, sha256=split_hash(s))
        (SPLIT_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")


def load(name: str) -> dict:
    rec = json.loads((SPLIT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in rec.items() if k not in ("name", "sha256")}
    if split_hash(body) != rec["sha256"]:
        raise RuntimeError(f"split file for {name} was modified after freezing")
    validate_split(body)
    return body


def era_of(period: int, s: dict) -> str:
    if s["dev"][0] <= period <= s["dev"][1]:
        return "dev"
    if s["test"][0] <= period <= s["test"][1]:
        return "test"
    raise ValueError(f"period {period} outside all eras")


if __name__ == "__main__":
    write_all()
    print("wrote", [p.name for p in SPLIT_DIR.glob('*.json')])

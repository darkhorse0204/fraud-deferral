"""Freeze the pre-registration. Run ONCE, after dev tuning and dev policy tuning are finished.

Writes results/PREREG_LOCK.txt containing SHA-256 hashes of every frozen config (splits, costs, model hps, policy hps,
preregistration.yaml). The test-era runner refuses to start without this file, and
`python -m scripts.lock_test --verify` re-checks that nothing has changed since (call it before analysing results).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

from src.utils import ROOT

FROZEN = ["configs/splits", "configs/models", "configs/policies", "configs/preregistration.yaml", "configs/costs.yaml"]
LOCK = ROOT / "results" / "PREREG_LOCK.txt"


def digest() -> dict:
    out = {}
    for rel in FROZEN:
        p = ROOT / rel
        files = [p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []
        for f in files:
            if f.is_file():
                # line endings are normalised so the lock verifies on any checkout (Windows CRLF or Unix LF)
                out[str(f.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    now = digest()
    if a.verify:
        rec = json.loads(LOCK.read_text())["files"]
        bad = [k for k in set(rec) | set(now) if rec.get(k) != now.get(k)]
        if bad:
            sys.exit(f"FROZEN CONFIGS CHANGED AFTER LOCK: {bad}")
        print("lock verified:", len(rec), "files unchanged")
        return
    if LOCK.exists():
        sys.exit("already locked; refusing to re-lock (delete deliberately and log the deviation)")
    missing = [f for f in ("configs/preregistration.yaml", "configs/costs.yaml", "configs/policies/policy_hp.yaml") if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"cannot lock, missing {missing}")
    LOCK.write_text(json.dumps({"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": now}, indent=1))
    print("LOCKED", len(now), "files")


if __name__ == "__main__":
    main()

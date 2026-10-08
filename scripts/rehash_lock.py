"""One-off: re-express the Study 1 lock with line-ending-normalised hashes.

The original lock hashed raw bytes, so a checkout with different line endings failed verification on identical content.
This script (1) REFUSES to run unless the original raw-byte lock still verifies, i.e. no frozen file has changed since
the lock was taken, then (2) rewrites the hashes with CRLF -> LF normalisation, keeping the original lock time and
recording the re-hash. No frozen file is modified.   python -m scripts.rehash_lock
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

from scripts.lock_test import FROZEN, LOCK, digest
from src.utils import ROOT


def raw_digest() -> dict:
    out = {}
    for rel in FROZEN:
        p = ROOT / rel
        for f in ([p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []):
            if f.is_file():
                out[str(f.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main() -> None:
    rec = json.loads(LOCK.read_text())
    if "rehashed_utc" in rec:
        sys.exit("lock is already line-ending normalised")
    raw = raw_digest()
    bad = [k for k in set(rec["files"]) | set(raw) if rec["files"].get(k) != raw.get(k)]
    if bad:
        sys.exit(f"REFUSED: frozen files differ from the original lock: {bad}")
    LOCK.write_text(json.dumps({"locked_utc": rec["locked_utc"], "rehashed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "note": "hashes re-expressed with CRLF->LF normalisation after the original raw-byte lock verified; no frozen file changed",
                                "files": digest()}, indent=1))
    print("re-hashed", len(rec["files"]), "files; original lock time kept:", rec["locked_utc"])


if __name__ == "__main__":
    main()

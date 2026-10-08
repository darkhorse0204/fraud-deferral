"""One-off: re-express the Study 2 lock with line-ending-normalised hashes for its config files.

The original lock normalised line endings for source files but hashed config files raw, so a fresh checkout failed
verification on identical content. The fix is one line in `digest()` of experiments/study2.py, which is itself a locked
file. This script therefore:
  (1) recomputes the ORIGINAL digest (config files raw) and REFUSES to run unless every locked file matches the original
      lock, except experiments/study2.py, whose only permitted difference is that one line (checked textually);
  (2) rewrites the lock with the new digest, keeping the original lock time and recording what changed.
python -m scripts.rehash_lock_study2
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

from experiments import study2 as S2
from src.utils import ROOT

OLD_LINE = 'hashlib.sha256(f.read_bytes()).hexdigest()'
NEW_LINE = 'hashlib.sha256(f.read_bytes().replace(b"\\r\\n", b"\\n")).hexdigest()   # line-ending independent'
RUNNER = "experiments/study2.py"


def original_digest() -> dict:
    out = {}
    for f in sorted(S2.CFG.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(f.read_bytes()).hexdigest()
    for rel in S2.FROZEN_CODE + ["configs/policies/policy_hp.yaml"]:
        src = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if rel == RUNNER:                                   # undo the single permitted edit before hashing
            txt = src.decode("utf-8")
            if txt.count(NEW_LINE) != 1:
                sys.exit("REFUSED: the runner does not contain exactly the one permitted edit")
            src = txt.replace(NEW_LINE, OLD_LINE).encode("utf-8")
        out[rel] = hashlib.sha256(src).hexdigest()
    return out


def main() -> None:
    rec = json.loads(S2.LOCK.read_text())
    if "rehashed_utc" in rec:
        sys.exit("lock is already line-ending normalised")
    old = original_digest()
    bad = [k for k in set(rec["files"]) | set(old) if rec["files"].get(k) != old.get(k)]
    if bad:
        sys.exit(f"REFUSED: locked files differ from the original lock: {bad}")
    S2.LOCK.write_text(json.dumps({"locked_utc": rec["locked_utc"], "rehashed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "note": "config hashes re-expressed with CRLF->LF normalisation after the original lock verified. The only locked file that "
                                           "changed is experiments/study2.py, by one line in digest() (line-ending normalisation of config files); "
                                           "scripts/rehash_lock_study2.py checks that this is its only difference from the locked version.",
                                   "files": S2.digest()}, indent=1))
    print("re-hashed", len(rec["files"]), "files; original lock time kept:", rec["locked_utc"])


if __name__ == "__main__":
    main()

"""Small shared helpers: seeding, config hashing, and the test-era touch log."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOUCH_LOG = ROOT / "results" / "test_touch_log.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cfg_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "no-commit"
    except Exception:  # noqa: BLE001
        return "no-git"


def log_test_touch(experiment: str, purpose: str, **extra) -> None:
    """Record every access to test-era labels. Append-only; honest bookkeeping for the paper."""
    TOUCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(TOUCH_LOG.read_text()) if TOUCH_LOG.exists() else []
    rows.append({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": experiment,
                 "purpose": purpose, "git": git_sha(), **extra})
    TOUCH_LOG.write_text(json.dumps(rows, indent=2), encoding="utf-8")

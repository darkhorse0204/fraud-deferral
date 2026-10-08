"""Download Kaggle datasets, extract safely, and maintain a SHA-256 manifest.

Trust-on-first-use: the first download writes hashes to data/MANIFEST.json; later runs
verify against it and abort on any mismatch. Raw data is never committed, only the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "MANIFEST.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> list[Path]:
    """Extract a zip, refusing any member that would land outside `dest` (zip-slip)."""
    dest = dest.resolve()
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {info.filename!r}")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            out.append(target)
    return out


def load_cfg(name: str) -> dict:
    with open(ROOT / "configs" / "data" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def find_file(raw_dir: Path, fname: str) -> Path | None:
    hits = list(raw_dir.rglob(fname))
    return hits[0] if hits else None


def fetch(name: str) -> None:
    cfg = load_cfg(name)
    raw = ROOT / cfg["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    missing = [f for f in cfg["expected_files"] if find_file(raw, f) is None]
    if missing:
        zips = list(raw.glob("*.zip"))
        if not zips:
            print(f"[{name}] downloading {cfg['kaggle_slug']} ...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download", "-d", cfg["kaggle_slug"], "-p", str(raw)],
                check=True,
            )
            zips = list(raw.glob("*.zip"))
        for z in zips:
            safe_extract(z, raw)
    manifest = read_manifest()
    entry = manifest.setdefault(name, {"license": cfg["license"], "slug": cfg["kaggle_slug"], "files": {}})
    for fname in cfg["expected_files"]:
        p = find_file(raw, fname)
        if p is None:
            raise FileNotFoundError(f"[{name}] expected file not found after extraction: {fname}")
        digest = sha256(p)
        rec = entry["files"].get(fname)
        if rec is None:
            entry["files"][fname] = {"sha256": digest, "bytes": p.stat().st_size}
            print(f"[{name}] recorded {fname} {digest[:12]}... ({p.stat().st_size:,} B)")
        elif rec["sha256"] != digest:
            raise RuntimeError(f"[{name}] CHECKSUM MISMATCH for {fname}: manifest {rec['sha256'][:12]} vs disk {digest[:12]}")
        else:
            print(f"[{name}] verified {fname}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=["elliptic", "baf"])
    for d in ap.parse_args().datasets:
        fetch(d)

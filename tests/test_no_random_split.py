"""L4: no random splitting APIs anywhere in src/, and the guard itself works."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_src_is_clean():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_leakage.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_guard_detects_violation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts" / "check_leakage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "bad.py").write_text("from sklearn.model_selection import train_test_split\n")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.main() == 1

"""Print generated paper macros (name=value), optionally filtered by prefixes or exact names:
    python -m scripts.list_macros EHa ETost          (prefix match)
    python -m scripts.list_macros =SHOneVerdict      (exact match)
"""
import re
import sys

from src.utils import ROOT

rows = []
for fn in ("numbers.tex", "numbers2.tex"):
    p = ROOT / "paper" / fn
    if not p.exists():
        continue
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("\\newcommand{"):
            rows.append((ln[len("\\newcommand{\\"):ln.index("}")], ln[ln.index("}{") + 2:-1]))
args = sys.argv[1:]
exact = {a[1:] for a in args if a.startswith("=")}
pre = tuple(a for a in args if not a.startswith("="))
sel = [(k, v) for k, v in rows if (not args) or k in exact or (pre and k.startswith(pre))]
bad = [k for k, _ in rows if re.search(r"\d", k)]
print(len(sel), "macros;", "names containing digits:", bad[:5])
print("  ".join(f"{k}={v}" for k, v in sel))

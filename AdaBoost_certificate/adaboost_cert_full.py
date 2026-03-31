from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "This certificate must not be run with Python -O / PYTHONOPTIMIZE, "
        "because optimized mode strips assert statements."
    )

import hashlib
import platform
import sys
from pathlib import Path

import adaboost_cert_core as core
import adaboost_cert_negcols as neg


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    core.core_report()
    print()
    neg.neg_report()
    print()
    print("Environment")
    print("-" * 72)
    print("  Interpreter :", sys.version.replace("\n", " "))
    print("  Platform    :", platform.platform())
    print("  SymPy       :", core.sp.__version__)
    print("  __debug__   :", __debug__)
    print()
    base = Path(__file__).resolve().parent
    files = [
        base / "adaboost_cert_core.py",
        base / "adaboost_cert_negcols.py",
        base / "adaboost_cert_full.py",
    ]
    print("SHA-256 digests")
    print("-" * 72)
    for path in files:
        if path.exists():
            print(f"  {path.name}: {sha256_of(path)}")
    print()
    print("Full certificate passed.")

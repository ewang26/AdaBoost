# AdaBoost counterexample full certificate — SageMath version
# Runs both core and negated-column certificates, prints environment info and digests.
# Run with: sage adaboost_cert_full.sage

if not __debug__:
    raise RuntimeError(
        "This certificate must not be run with Python -O / PYTHONOPTIMIZE, "
        "because optimized mode strips assert statements."
    )

import hashlib
import platform
import sys
from pathlib import Path

load("adaboost_cert_core.sage")
load("adaboost_cert_negcols.sage")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    core_report()
    print()
    neg_report()
    print()
    print("Environment")
    print("-" * 72)
    print("  Interpreter :", sys.version.replace("\n", " "))
    print("  Platform    :", platform.platform())
    print("  SageMath    :", version())
    print("  __debug__   :", __debug__)
    print()
    base = Path(__file__).resolve().parent if '__file__' in dir() else Path(".").resolve()
    files = [
        base / "adaboost_cert_core.sage",
        base / "adaboost_cert_negcols.sage",
        base / "adaboost_cert_full.sage",
    ]
    print("SHA-256 digests")
    print("-" * 72)
    for path in files:
        if path.exists():
            print(f"  {path.name}: {sha256_of(path)}")
    print()
    print("Full certificate passed.")

# Computational certificate for the AdaBoost counterexample

This directory contains the computer-verified certificate accompanying the
manuscript.  All proof-critical assertions are checked by exact arithmetic;
no floating-point computation is used in any assertion.

**Safety guard.**  Every certificate script refuses to run under
`python -O` (or with `PYTHONOPTIMIZE` set), because optimized mode
silently strips `assert` statements.

## Files

- `adaboost_cert_core.py` — Core certificate (Appendix A, items 1–14).
  Verifies the exact period-2 orbit, spectral data, Groebner-basis proofs
  of the adapted-basis transverse zeros, sector contraction bounds,
  branch-word and burst-lock certification, and the explicit rational
  starting point.  Also constructs M_0 = L_A ⊞ L_B explicitly and
  verifies γ(M_0) = 1/5 via both a row witness and a column witness
  (minimax theorem), and confirms that row-duplication yields the
  correct uniform-start equivalence.

- `adaboost_cert_negcols.py` — Negated-column certificate (Appendix A,
  items 15–16).  Verifies that negated columns are never selected in the
  `[M | -M]` formulation, and that the augmented margin equals 1/5.

- `adaboost_cert_full.py` — Runs both certificates in sequence and prints
  SHA-256 digests of all certificate files together with the interpreter
  and platform details for reproducibility.

## Requirements

- Python >= 3.12
- SymPy >= 1.14

Tested with Python 3.12.3 and SymPy 1.14.0.  No other dependencies are
required; the exact interval arithmetic uses only the Python standard
library (`fractions.Fraction`, `math.isqrt`).

## Usage

```bash
python adaboost_cert_core.py
python adaboost_cert_negcols.py
python adaboost_cert_full.py
```

Each script exits with status 0 and prints a report ending with
`... assertions passed.` when all checks succeed.

# AdaBoost Does Not Always Cycle

A computer-assisted counterexample to the open question (Rudin, Schapire, and Daubechies, COLT 2012) of whether exhaustive AdaBoost always converges to a finite cycle.

- **[AdaBoost_sol.pdf](AdaBoost_sol.pdf)** — Manuscript
- **[certificate/](certificate/)** — Computational certificate (exact rational arithmetic, no floating point)

## Running the certificate

Requires Python >= 3.12 and SymPy >= 1.14.

```bash
cd certificate
python adaboost_cert_full.py
```

See [cert_readme.md](certificate/cert_readme.md) for details.

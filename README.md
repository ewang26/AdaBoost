# AdaBoost Does Not Always Cycle

A computer-assisted counterexample to the open question (Rudin, Schapire, and Daubechies, COLT 2012) of whether exhaustive AdaBoost always converges to a finite cycle.

- **[arXiv:2604.07055](https://arxiv.org/abs/2604.07055)** — Paper
- **[AdaBoost_sol.pdf](AdaBoost_sol.pdf)** — Manuscript
- **[certificate/](certificate/)** — Computational certificate (exact rational arithmetic, no floating point)

## Running the certificate

Two independent implementations are provided: SymPy (Python) and SageMath.

### SymPy (requires Python >= 3.12 and SymPy >= 1.14)

```bash
cd certificate/sympy
python adaboost_cert_full.py
```

### SageMath (requires SageMath >= 10.8)

```bash
cd certificate/sagemath
sage adaboost_cert_full.sage
```

See [cert_readme.md](certificate/cert_readme.md) for details.

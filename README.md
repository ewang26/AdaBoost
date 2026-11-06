# AdaBoost Does Not Always Cycle

Computational certificates accompanying the computer-assisted counterexample to
Rudin, Schapire, and Daubechies’s COLT 2012 open problem.

- **[arXiv:2604.07055](https://arxiv.org/abs/2604.07055)** — Paper
- **[certificate/](certificate/)** — Exact-arithmetic certificates

## Running the certificate

The current certificate verifies the revised construction with 2,500 examples,
a uniform initial distribution, and weak-learning margin 1/5.
Run the following commands from the repository root, with assertions enabled.

### SymPy (Python 3.11+ and SymPy 1.14)

```bash
python3 -m pip install -r certificate/requirements.txt
python3 certificate/verify.py
python3 certificate/check_report.py
```

### SageMath (SageMath 10.8)

```bash
sage -python certificate/replication/replay_sage.py
python3 certificate/check_report.py
```

The Sage replay uses native Sage algebra and shares the geometric and entry
checks with the SymPy driver. The original drivers in `certificate/sympy/` and
`certificate/sagemath/` are retained for the earlier construction.

See [certificate/README.md](certificate/README.md) for verification details.

## Citation

In case you find this work interesting and use it, please cite:

```bibtex
@inproceedings{wang2026resolving,
  title     = {Resolving {AdaBoost} Cycling with {LLMs}: A Computer-Assisted Counterexample},
  author    = {Wang, Erik Y.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=4G0MC6XSqQ}
}
```

# Current exact certificate

From the repository root, use Python 3.11+ and SymPy 1.14:

~~~sh
python3 certificate/verify.py
python3 certificate/check_report.py
~~~

Assertions must remain enabled. The current finite checks derive maps from the
sign matrices, identify the algebraic quantities by exact identities and root
counts, certify two reflection-normalized cones, branch/locking/negation gaps,
the full twenty-weight 140-update strict prefix, the 2,500-row uniform encoding
and exact margin 1/5. Decimal conversions report accepted rational quantities.
The source hashes and results are in verification.json.

[CLAIM_MAP.md](CLAIM_MAP.md) connects all sixteen finite obligations to their
domains, consuming proof arguments and executable anchors. Software acceptance
does not replace the analytic deductions or the proof–specification audit.

## Current Sage replay

With SageMath 10.8 and assertions enabled:

~~~sh
sage -python certificate/replication/replay_sage.py
python3 certificate/check_report.py
~~~

Run from the repository root. The replay imports no SymPy and reads no saved
numerical result. It derives the maps, derivatives, elimination polynomials,
root counts and eigenvector identities with native Sage algebra. It reuses the
current rational geometric and entry acceptance sections from verify.py, with
explicit backend-interface adapters, and checks half-step polynomial identities,
irreducibility, isolators and norms natively. It checks the current 140 updates
and margin witnesses, not just the old start.

The retained Sage algebra module is loaded with an explicit convex-hull
root-identification repair before eigenvalue interval intersections. Its source,
the repaired-source digest, the shared driver and this replay are identified in
replication/verification_sage.json. Agreement is **cross-implementation
replication with shared mathematical specifications and acceptance logic**,
not independent formal verification of the proof.

The report checker validates both source-hash sets and agreement on the exact
140-column itinerary, burst word, encoding, margin and norms. Each backend may
produce different valid interval bounds; they need not be byte-identical.

On the macOS SageMath app, if its launcher is unavailable, invoke the app's
bundled Python directly with a writable DOT_SAGE. This is an environment
workaround; the portable command above is the normal entry point.

## Historical checks

The original sympy/ and sagemath/ full drivers remain unchanged for historical
reproduction; their default four-phase cones and large-denominator start are
not the current construction. Earlier recorded Sage reports in replication/
are retained and identified by their original filenames.

The optional legacy_verify.py separately checks the earlier four-phase
geometry and initialization with the shared SymPy algebra:

~~~sh
python3 certificate/legacy_verify.py
~~~

The active checker does not execute that regression or overwrite its outputs.
No current check requires a file outside this repository or network access.

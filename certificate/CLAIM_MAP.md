# Current finite obligations and proof alignment

The numbered obligations are specified in the manuscript's Certificate
documentation appendix. This index identifies executable anchors. Checking
that inputs, domains and acceptance conditions match the proof remains a human
task. Acceptance arithmetic is exact; division excludes zero; decimals report
already-certified rational quantities.

| Item | Claim and domain / acceptance condition | Analytic consumer | Active SymPy anchors |
|---|---|---|---|
| 1 | Five H0 updates on the rational affine charts; reduced fractions match | Branch maps and regularity | algebra.py: update_state, F0A_p, F0B_p, polynomial comparisons and gcd tests |
| 2 | B face e=0 agrees with A; fifth numerator divisible by e; row reflection swaps columns 3/4 | Face restriction; phase transport | algebra.py: face identities, rem0, rem1; verify.py: perm |
| 3 | Common-cycle equations and elimination polynomial; one root in the stated interval; nonzero denominators | Common-cycle isolation | algebra.py: N1, N2, f_poly, p_lo, p_hi, P_iv, d_iv |
| 4 | Substitute d=Dexpr into cycle equations; numerators vanish modulo f | Sufficiency beyond necessary elimination | algebra.py: Dexpr, polynomial remainder assertions |
| 5 | Irreducible degree-14 A return multiplier polynomial from characteristic elimination | Spectral data proposition | algebra.py: tr_RA, det_RA, P_lambda_poly |
| 6 | Irreducible degree-7 transverse B return multiplier polynomial | Block-triangular spectrum | algebra.py: kappa_expr, P_kappa_poly |
| 7 | Actual derivative roots identified by annihilation and one-root convex hulls with nonroot endpoints; 0<nu<lambda<kappa<1 | Actual eigenvalues and ordering | algebra.py: lam_from_char_iv, nu_from_char_iv, kap_eval_iv, hull loop and ordering assertion |
| 8 | Exact eigenvector equations, first component 1; transverse first-column entries vanish | Half-step directions; normalized derivative structure | algebra.py: thetaA_iv, thetaB0_iv, chiB_iv, ideals GA and GB |
| 9 | Positive half-step multipliers; polynomial identities, irreducibility, isolation, norms 1/2 and 1 | Analytic norm obstruction | verify.py: qa, qb, norms; algebra.py: cA_iv, cB_iv |
| 10 | Taylor estimates on physical boxes covering cones R=1/100000, rho=1/1000; shared bounds m,M,alpha,beta,Gamma,Delta | Sector invariance and summable errors | verify.py: abox, bbox, AH, BH, asector, bsector |
| 11 | Every H0 selected local edge exceeds competitors by >1/40; reflection gives H1 | Unique local five-step words | verify.py: wordA, wordB, branch |
| 12 | Later-edge surplus minus maximum boundary surplus >3/40 | Whole-burst locking | verify.py: lock |
| 13 | Sixteen edges on full20 start counts/2500; 140 strict winners, gaps >1e-8, outward dyadic rounding, exact chart landing, strict endpoint membership | Finite entry and continuous prefix neighborhood | verify.py: counts, W, prefix_gaps, seq, bursts, endpoint |
| 14 | Assertions enabled; rational acceptance, nonzero interval divisors; source hashes | Reproduction discipline | algebra.py: QInterval and optimization guard; verify.py; check_report.py |
| 15 | Tail selected edge exceeds all negated active/frozen edges by >1/4; prefix checks negatives too | Negation-closed corollary | verify.py: neggaps and sixteen-column M |
| 16 | Positive counts sum to 2500; exact primal/dual witnesses for both column sets | Row aggregation, uniform encoding, margin 1/5 | verify.py: counts, colweights, rowweights |

## Analytic arrows outside the software

Taylor bounds imply sector invariance. Summability gives the local logarithmic
law. The competition theorem covers arbitrary maximizing ties; the norm argument
makes its frequency irrational. Separated fourth A-marginal limits rule out
finite-cycle convergence. Row aggregation transfers to a uniform start.
Strict entry and continuity give an open neighborhood only within product states.
These deductions must be audited in the proof.

## Sage and historical files

Recorded local-cone and entry reports in replication/ used SageMath 10.8 and
shared portions of the acceptance logic. This is cross-implementation
replication, not independent specification verification.

The original sagemath/adaboost_cert_core.sage has the same map, eigenvector,
polynomial and interval-helper names, using native Sage rings. Its default start
and four-phase cones belong to the earlier example. The original sympy/ and
sagemath/ full drivers remain historical; executing them alone does not verify
the current 2,500-row result.

The current replication/replay_sage.py is a portable executable replay. It uses
the named native Sage algebra anchors (with actual-root identification repaired
before intersections), the shared current cone/entry sections in verify.py and
native half-step polynomial tests. Its report hashes the source inputs and its
own executable. Thus items 1–9 use native polynomial/field algebra and items
10–16 reuse the current rational acceptance logic and witnesses with the Sage
backend. The displayed outward direction intervals are checked by SymPy; Sage
uses its own valid isolators for the replicated inequalities.

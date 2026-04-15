# AdaBoost counterexample core certificate (SageMath version)
# Equivalent to adaboost_cert_core.py but using native SageMath exact arithmetic.
# Run with: sage adaboost_cert_core.sage

from dataclasses import dataclass
from fractions import Fraction as PyFraction
from math import isqrt
from decimal import Decimal, localcontext

# Sage's preparser converts integer literals (0, 1, 2, ...) to Sage Integers.
# Python's fractions.Fraction cannot compare/operate with Sage Integers because
# Sage Integer's .denominator is a method, not an attribute.
# We define Python-int constants here for use in all PyFraction contexts.
_py0 = int(0)
_py1 = int(1)
_py2 = int(2)

# Comparison-safe wrapper: convert Sage Integer to Python int before
# comparing with PyFraction.
def _pyint(x):
    """Convert to Python int if it's a Sage Integer, otherwise pass through."""
    if isinstance(x, PyFraction):
        return x
    return int(x)

# -----------------------------------------------------------------------------
# Exact matrices and one-step AdaBoost update
# -----------------------------------------------------------------------------

LA = matrix(QQ, [
    [1, 1, -1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
])

LB = matrix(QQ, [
    [1, 1, -1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
])

H0 = [0, 2, 3, 0, 1]   # (1,3,4,1,2)
H1 = [0, 3, 2, 0, 1]   # (1,4,3,1,2)

# We work in the fraction field of QQ[p,d,e] throughout.
R_poly = PolynomialRing(QQ, 'p, d, e')
p, d, e = R_poly.gens()
FF = R_poly.fraction_field()


def update_state(state, M, j):
    nrows = M.nrows()
    mu = sum(state[i] * M[i, j] for i in range(nrows))
    new = [state[i] / (FF(1) + mu * M[i, j]) for i in range(nrows)]
    return new, mu


def frac_equal(a, b):
    """Check equality of elements in the fraction field."""
    diff = a - b
    if hasattr(diff, 'numerator'):
        return diff.numerator() == 0
    return diff == 0


# -----------------------------------------------------------------------------
# Reduced branch maps on the affine manifolds
# -----------------------------------------------------------------------------

# A: Sigma_A
stateA = [FF(p - QQ(1)/2), FF(1 - p), FF(QQ(1)/2 - d), FF(d)]
state = list(stateA)
for j in H0:
    state, _ = update_state(state, LA, j)
F0A_p = FF(1) - state[1]
F0A_d = state[3]
F1A_p = F0A_p.subs({d: QQ(1)/2 - d})
F1A_d = FF(QQ(1)/2) - F0A_d.subs({d: QQ(1)/2 - d})

# B: Sigma_B
stateB = [FF(p - QQ(1)/2), FF(1 - p), FF(QQ(1)/2 - d - e), FF(d), FF(e)]
state = list(stateB)
for j in H0:
    state, _ = update_state(state, LB, j)
F0B_p = FF(1) - state[1]
F0B_d = state[3]
F0B_e = state[4]

dtil = QQ(1)/2 - d - e
F1B_p = F0B_p.subs({d: dtil})
F1B_d = FF(QQ(1)/2) - F0B_d.subs({d: dtil}) - F0B_e.subs({d: dtil})
F1B_e = F0B_e.subs({d: dtil})

# Derive H1 maps directly and verify they match the conjugated formulas
stateA1 = [FF(p - QQ(1)/2), FF(1 - p), FF(QQ(1)/2 - d), FF(d)]
state = list(stateA1)
for j in H1:
    state, _ = update_state(state, LA, j)
F1A_p_direct = FF(1) - state[1]
F1A_d_direct = state[3]
assert frac_equal(F1A_p_direct, F1A_p), "H1 A-map p mismatch"
assert frac_equal(F1A_d_direct, F1A_d), "H1 A-map d mismatch"

stateB1 = [FF(p - QQ(1)/2), FF(1 - p), FF(QQ(1)/2 - d - e), FF(d), FF(e)]
state = list(stateB1)
for j in H1:
    state, _ = update_state(state, LB, j)
F1B_p_direct = FF(1) - state[1]
F1B_d_direct = state[3]
F1B_e_direct = state[4]
assert frac_equal(F1B_p_direct, F1B_p), "H1 B-map p mismatch"
assert frac_equal(F1B_d_direct, F1B_d), "H1 B-map d mismatch"
assert frac_equal(F1B_e_direct, F1B_e), "H1 B-map e mismatch"

# Structural identities: face e=0 and exact divisibility by e
assert frac_equal(F0B_p.subs({e: 0}), F0A_p), "F0B_p|_{e=0} != F0A_p"
assert frac_equal(F0B_d.subs({e: 0}), F0A_d), "F0B_d|_{e=0} != F0A_d"
assert frac_equal(F1B_p.subs({e: 0}), F1A_p), "F1B_p|_{e=0} != F1A_p"
assert frac_equal(F1B_d.subs({e: 0}), F1A_d), "F1B_d|_{e=0} != F1A_d"

# Check e | numerator of F0B_e and F1B_e
num_e0 = R_poly(F0B_e.numerator())
num_e1 = R_poly(F1B_e.numerator())
assert num_e0 % R_poly(e) == 0, "e does not divide num(F0B_e)"
assert num_e1 % R_poly(e) == 0, "e does not divide num(F1B_e)"


# -----------------------------------------------------------------------------
# Match displayed manuscript branch-map polynomials (Section 3)
# -----------------------------------------------------------------------------

PA_manuscript = R_poly(
    4*d**4 - 6*d**3*p + 2*d**3 - 30*d**2*p**2 + 15*d**2*p - 2*d**2
    - 6*d*p**3 + 14*d*p**2 - 4*d*p + 6*p**4 + p**3 - 2*p**2
)
QA_manuscript = R_poly(
    8*d**4 - 8*d**3*p + 2*d**3 - 48*d**2*p**2 + 22*d**2*p - 3*d**2
    - 8*d*p**3 + 22*d*p**2 - 6*d*p + 8*p**4 + 2*p**3 - 3*p**2
)
DA_manuscript = R_poly(
    d*(2*d**3 + 2*d**2*p - d**2 - 6*d*p**2 + 2*p**3 + p**2)
)
RA_manuscript = R_poly(
    8*d**4 - 2*d**3 - 24*d**2*p**2 + 6*d**2*p - d**2
    + 10*d*p**2 - 2*d*p + 2*p**3 - p**2
)

_num_F0Ap = R_poly(F0A_p.numerator())
_den_F0Ap = R_poly(F0A_p.denominator())
_num_F0Ad = R_poly(F0A_d.numerator())
_den_F0Ad = R_poly(F0A_d.denominator())
assert R_poly(_num_F0Ap * QA_manuscript - _den_F0Ap * PA_manuscript) == 0, \
    "A-gadget p-component mismatch"
assert R_poly(_num_F0Ad * RA_manuscript - _den_F0Ad * DA_manuscript) == 0, \
    "A-gadget d-component mismatch"

# B-gadget
PB_manuscript = R_poly(
    4*d**4 + 16*d**3*e - 6*d**3*p + 2*d**3
    + 20*d**2*e**2 - 36*d**2*e*p + 12*d**2*e - 30*d**2*p**2 + 15*d**2*p - 2*d**2
    + 8*d*e**3 - 46*d*e**2*p + 18*d*e**2 - 48*d*e*p**2 + 30*d*e*p - 4*d*e
    - 6*d*p**3 + 14*d*p**2 - 4*d*p
    - 16*e**3*p + 8*e**3 - 22*e**2*p**2 + 15*e**2*p - 2*e**2
    + 8*e*p**2 - 4*e*p + 6*p**4 + p**3 - 2*p**2
)
QB_manuscript = R_poly(
    8*d**4 + 32*d**3*e - 8*d**3*p + 2*d**3
    + 40*d**2*e**2 - 56*d**2*e*p + 16*d**2*e - 48*d**2*p**2 + 22*d**2*p - 3*d**2
    + 16*d*e**3 - 72*d*e**2*p + 26*d*e**2 - 72*d*e*p**2 + 44*d*e*p - 6*d*e
    - 8*d*p**3 + 22*d*p**2 - 6*d*p
    - 24*e**3*p + 12*e**3 - 32*e**2*p**2 + 22*e**2*p - 3*e**2
    + 12*e*p**2 - 6*e*p + 8*p**4 + 2*p**3 - 3*p**2
)
DB_manuscript = R_poly(
    d*(
        2*d**3 + 8*d**2*e + 2*d**2*p - d**2
        + 10*d*e**2 - 2*d*e - 6*d*p**2
        + 4*e**3 - 2*e**2*p - e**2 - 4*e*p**2
        + 2*p**3 + p**2
    )
)
RB_manuscript = R_poly(
    8*d**4 + 32*d**3*e - 2*d**3
    + 40*d**2*e**2 - 24*d**2*e*p - 24*d**2*p**2 + 6*d**2*p - d**2
    + 16*d*e**3 - 32*d*e**2*p + 6*d*e**2 - 24*d*e*p**2 + 12*d*e*p - 2*d*e
    + 10*d*p**2 - 2*d*p
    - 8*e**3*p + 4*e**3 - 8*e**2*p**2 + 6*e**2*p - e**2
    + 4*e*p**2 - 2*e*p + 2*p**3 - p**2
)
EB_manuscript = R_poly(
    e*(
        2*d**3 + 4*d**2*e - 6*d**2*p + d**2
        + 2*d*e**2 - 8*d*e*p + 2*d*e + 2*d*p**2
        - 2*e**2*p + e**2 + 2*p**3 - p**2
    )
)

_num_F0Bp = R_poly(F0B_p.numerator())
_den_F0Bp = R_poly(F0B_p.denominator())
_num_F0Bd = R_poly(F0B_d.numerator())
_den_F0Bd = R_poly(F0B_d.denominator())
_num_F0Be = R_poly(F0B_e.numerator())
_den_F0Be = R_poly(F0B_e.denominator())
assert R_poly(_num_F0Bp * QB_manuscript - _den_F0Bp * PB_manuscript) == 0, \
    "B-gadget p-component mismatch"
assert R_poly(_num_F0Bd * RB_manuscript - _den_F0Bd * DB_manuscript) == 0, \
    "B-gadget d-component mismatch"
assert R_poly(_num_F0Be * RB_manuscript - _den_F0Be * EB_manuscript) == 0, \
    "B-gadget e-component mismatch"


# -----------------------------------------------------------------------------
# Exact period-2 orbit on the face e=0
# -----------------------------------------------------------------------------

# N1: numerator of (F0A_p - p), after removing content and dividing by -(2p-1)
_raw1 = R_poly(F0A_p.numerator()) - p * R_poly(F0A_p.denominator())
_raw1_red = R_poly(_raw1 // _raw1.content())
N1 = R_poly(_raw1_red // R_poly(-(2*p - 1)))

# N2: numerator of (F0A_d - (1/2-d)), after removing content and negating
_raw2 = R_poly(F0A_d.numerator()) - (QQ(1)/2 - d) * R_poly(F0A_d.denominator())
_raw2_red = R_poly(_raw2 // _raw2.content())
N2 = -_raw2_red

# Match displayed 2-cycle polynomials (Section 4)
N1_manuscript = R_poly(
    2*d**4 - 2*d**3*p + d**3 - 12*d**2*p**2 + 7*d**2*p - d**2
    - 2*d*p**3 + 6*d*p**2 - 2*d*p + 2*p**4 - p**2
)
N2_manuscript = R_poly(
    -16*d**5 + 8*d**4 + 48*d**3*p**2 - 16*d**3*p + 2*d**3
    - 32*d**2*p**2 + 10*d**2*p - d**2
    - 8*d*p**3 + 10*d*p**2 - 2*d*p + 2*p**3 - p**2
)
assert N1 == N1_manuscript, "N_1 polynomial mismatch"
assert N2 == N2_manuscript, "N_2 polynomial mismatch"

# Resultant to eliminate d: view N1, N2 as polynomials in d over QQ[p]
R_p_uni = PolynomialRing(QQ, 'pu')
pu = R_p_uni.gen()
R_d_over_p = PolynomialRing(R_p_uni, 'du')
du = R_d_over_p.gen()

# Map p->pu, d->du in N1 and N2
N1_d = R_d_over_p(N1.subs({p: pu, d: du, e: R_poly(0)}))
N2_d = R_d_over_p(N2.subs({p: pu, d: du, e: R_poly(0)}))
res = N1_d.resultant(N2_d)
res_poly = R_p_uni(res)

f_poly = R_p_uni(res_poly // (4096 * pu**10 * (2*pu - 1)**3))
f_expanded = f_poly

# Dexpr: the d-coordinate as a function of p on the cycle
Dexpr_poly = R_p_uni(1024*pu**6 - 2304*pu**5 + 1472*pu**4 - 48*pu**3 - 176*pu**2 + 38*pu - 9)
# This is 2*Dexpr as a polynomial; Dexpr = Dexpr_poly / 2

# Verify N1 and N2 vanish on the cycle: substitute d = Dexpr_poly/(2) into N1, N2
# and check divisibility by f in pu.
N1_sub_num = R_p_uni(R_poly(N1.subs({e: 0})).subs({p: pu, d: pu}).parent()(0))
# Actually, do this properly: substitute d -> Dexpr_poly/2 into N1(p=pu, d=?)
# N1 is a polynomial in p,d. We evaluate N1(pu, Dexpr_poly/2):
N1_at_cycle = R_p_uni(0)
N2_at_cycle = R_p_uni(0)
Dexpr_half = Dexpr_poly  # numerator; denominator is 2
# Evaluate via the .dict() of N1 in R_poly
for exp_tuple, coeff in N1.dict().items():
    # exp_tuple = (exp_p, exp_d, exp_e)
    ep, ed, ee = exp_tuple
    N1_at_cycle += R_p_uni(coeff) * pu**ep * Dexpr_half**ed
N1_at_cycle_num = N1_at_cycle  # This has denominator 2^(max_d_degree)
# Actually easier: multiply through by 2^deg_d
# Since N1 has d up to degree 4, the substitution d = Dexpr_poly/2 gives
# terms with denominator up to 2^4. Let's just work in QQ[pu] which handles fractions.
N1_at_cycle = R_p_uni(0)
for exp_tuple, coeff in N1.dict().items():
    ep, ed, ee = exp_tuple
    N1_at_cycle += QQ(coeff) * pu**ep * (Dexpr_poly / QQ(2))**int(ed)
assert N1_at_cycle % f_expanded == 0, "N1 does not vanish on the cycle"

N2_at_cycle = R_p_uni(0)
for exp_tuple, coeff in N2.dict().items():
    ep, ed, ee = exp_tuple
    N2_at_cycle += QQ(coeff) * pu**ep * (Dexpr_poly / QQ(2))**int(ed)
assert N2_at_cycle % f_expanded == 0, "N2 does not vanish on the cycle"

# Root isolation: exactly one root of f in the interval
p_lo = QQ(637116837818846) / 10**15
p_hi = QQ(637116837818848) / 10**15

# Sage provides built-in root counting via Sturm's theorem
def count_roots_in(poly, lo, hi):
    return poly.number_of_roots_in_interval(lo, hi)

assert count_roots_in(f_poly, p_lo, p_hi) == 1, \
    f"Expected 1 root of f in [{p_lo}, {p_hi}]"


# -----------------------------------------------------------------------------
# Exact rational interval arithmetic
# (Using Python's fractions.Fraction for exact bounds, same as original)
# -----------------------------------------------------------------------------

def qfrac(x):
    if isinstance(x, PyFraction):
        return x
    if isinstance(x, (int, Integer)):
        return PyFraction(int(x), int(1))
    if x in QQ:
        r = QQ(x)
        return PyFraction(int(r.numerator()), int(r.denominator()))
    raise TypeError(f"Cannot convert {type(x)} to Fraction")


@dataclass(frozen=True)
class QInterval:
    lo: PyFraction
    hi: PyFraction

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"Invalid interval [{self.lo}, {self.hi}]")

    @staticmethod
    def const(x):
        if isinstance(x, QInterval):
            return x
        return QInterval(qfrac(x), qfrac(x))

    def __add__(self, other):
        o = QInterval.const(other)
        return QInterval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, other):
        o = QInterval.const(other)
        return QInterval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other):
        o = QInterval.const(other)
        return QInterval(o.lo - self.hi, o.hi - self.lo)

    def __neg__(self):
        return QInterval(-self.hi, -self.lo)

    def __mul__(self, other):
        o = QInterval.const(other)
        vals = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return QInterval(min(vals), max(vals))

    __rmul__ = __mul__

    def reciprocal(self):
        _ZERO = PyFraction(int(0), int(1))
        if self.lo <= _ZERO <= self.hi:
            raise ZeroDivisionError(f"Interval crosses zero: [{self.lo}, {self.hi}]")
        _ONE = PyFraction(int(1), int(1))
        vals = [_ONE / self.lo, _ONE / self.hi]
        return QInterval(min(vals), max(vals))

    def __truediv__(self, other):
        o = QInterval.const(other)
        return self * o.reciprocal()

    def __rtruediv__(self, other):
        o = QInterval.const(other)
        return o * self.reciprocal()

    def __pow__(self, n):
        if not isinstance(n, (int, Integer)):
            raise TypeError("Only integer powers are supported")
        n = int(n)
        if n == 0:
            return QInterval.const(int(1))
        if n < 0:
            return (self ** (-n)).reciprocal()
        if n % 2 == 1:
            return QInterval(self.lo**n, self.hi**n)
        vals = [self.lo**n, self.hi**n]
        _ZERO = PyFraction(int(0), int(1))
        lo = _ZERO if self.lo <= _ZERO <= self.hi else min(vals)
        hi = max(vals)
        return QInterval(lo, hi)

    def contains_zero(self):
        _ZERO = PyFraction(int(0), int(1))
        return self.lo <= _ZERO <= self.hi

    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))

    def intersect(self, other):
        o = QInterval.const(other)
        lo = max(self.lo, o.lo)
        hi = min(self.hi, o.hi)
        if lo > hi:
            raise ValueError(f"Empty intersection of {self} and {o}")
        return QInterval(lo, hi)


def qiv(lo, hi=None):
    if hi is None:
        return QInterval.const(lo)
    return QInterval(qfrac(lo), qfrac(hi))


def sqrt_bounds_fraction(q, scale):
    _ZERO = PyFraction(int(0), int(1))
    if q < _ZERO:
        raise ValueError("sqrt requires a nonneg rational")
    S = q.numerator * int(scale) * int(scale)
    den = q.denominator
    m = isqrt(S // den)
    while (m + int(1)) * (m + int(1)) * den <= S:
        m += int(1)
    while m * m * den > S:
        m -= int(1)
    n = m if m * m * den == S else m + int(1)
    while (n - int(1)) >= int(0) and (n - int(1)) * (n - int(1)) * den >= S:
        n -= int(1)
    while n * n * den < S:
        n += int(1)
    return PyFraction(int(m), int(scale)), PyFraction(int(n), int(scale))


def sqrt_interval_q(iv, digits=int(30)):
    _ZERO = PyFraction(int(0), int(1))
    if iv.lo < _ZERO:
        raise ValueError("sqrt interval requires nonnegative lower bound")
    scale = int(10)**int(digits)
    lo, _ = sqrt_bounds_fraction(iv.lo, scale)
    _, hi = sqrt_bounds_fraction(iv.hi, scale)
    return QInterval(lo, hi)


# Interval evaluation for Sage polynomial/fraction-field expressions
def eval_expr_iv(expr, env):
    """Evaluate a rational expression over QInterval boxes.
    expr: element of R_poly or its fraction field, or a univariate poly in pu
    env: dict mapping ring generators to QInterval values
    """
    if isinstance(expr, QInterval):
        return expr
    if expr in QQ:
        return qiv(QQ(expr))

    # Get numerator and denominator
    num = expr.numerator()
    den = expr.denominator()

    def eval_poly(poly, env):
        """Evaluate a multivariate polynomial with QInterval arithmetic."""
        result = qiv(0)
        if not hasattr(poly, 'dict'):
            return qiv(QQ(poly))

        d_dict = poly.dict()
        gens = list(poly.parent().gens()) if hasattr(poly, 'parent') else []
        for exp_tuple, coeff in d_dict.items():
            term = qiv(QQ(coeff))
            # exp_tuple may be an ETuple, tuple, or integer (univariate)
            try:
                exp_len = len(exp_tuple)
                for i in range(exp_len):
                    exp = int(exp_tuple[i])
                    if exp != 0 and i < len(gens):
                        gen = gens[i]
                        if gen in env:
                            term = term * (env[gen] ** exp)
            except TypeError:
                # univariate case: exp_tuple is a plain integer
                exp = int(exp_tuple)
                if exp != 0 and len(gens) > 0 and gens[0] in env:
                    term = term * (env[gens[0]] ** exp)
            result = result + term
        return result

    # Try to cast to R_poly; if the expression involves only p (univariate),
    # it might live in a different ring.
    try:
        num_poly = R_poly(num)
        den_poly = R_poly(den)
    except (TypeError, ValueError):
        # Might be univariate in pu -- try direct evaluation
        num_poly = num
        den_poly = den

    num_val = eval_poly(num_poly, env)
    den_val = eval_poly(den_poly, env)
    if den_poly == 1 or (den_poly in QQ and QQ(den_poly) != 0):
        if den_poly in QQ:
            return num_val / qiv(QQ(den_poly))
    return num_val / den_val


def matmul_q(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    out = []
    for i in range(n):
        row = []
        for j in range(m):
            s = qiv(0)
            for t in range(k):
                s = s + A[i][t] * B[t][j]
            row.append(s)
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Reporting utilities
# -----------------------------------------------------------------------------

def fmt_frac(fr, digits=int(18)):
    with localcontext() as ctx:
        ctx.prec = int(digits) + int(8)
        dec = Decimal(int(fr.numerator)) / Decimal(int(fr.denominator))
        return format(dec, f".{int(digits)}g")


def fmt_iv(v, digits=18):
    return f"[{fmt_frac(v.lo, digits)}, {fmt_frac(v.hi, digits)}]"


# -----------------------------------------------------------------------------
# Root, cycle, and exact spectral intervals
# -----------------------------------------------------------------------------

P_iv = qiv(p_lo, p_hi)

# Evaluate Dexpr at P_iv -- Dexpr is a rational function of p alone
# Dexpr = (1024*p^6 - 2304*p^5 + 1472*p^4 - 48*p^3 - 176*p^2 + 38*p - 9) / 2
# Build it in R_poly so eval_expr_iv can handle it
Dexpr_sage = (1024*p**6 - 2304*p**5 + 1472*p**4 - 48*p**3 - 176*p**2 + 38*p - 9) / FF(2)
d_iv = eval_expr_iv(Dexpr_sage, {p: P_iv})
mu_star_iv = 2 * P_iv - 1

P0A_iv = [P_iv - qiv(QQ(1)/2), qiv(1) - P_iv, qiv(QQ(1)/2) - d_iv, d_iv]
P1A_iv = [P_iv - qiv(QQ(1)/2), qiv(1) - P_iv, d_iv, qiv(QQ(1)/2) - d_iv]

# Jacobians using manual quotient-rule differentiation
def diff_rational(expr, var):
    """Differentiate a fraction-field element using quotient rule."""
    num = R_poly(expr.numerator())
    den = R_poly(expr.denominator())
    dnum = num.derivative(var)
    dden = den.derivative(var)
    # d(n/d)/dv = (n'*d - n*d') / d^2
    return FF(dnum * den - num * dden) / FF(den**2)


def sage_jacobian(exprs, vars_list):
    """Compute Jacobian matrix of rational expressions as list of lists."""
    rows = []
    for expr in exprs:
        row = []
        for v in vars_list:
            row.append(diff_rational(expr, v))
        rows.append(row)
    return rows


J0A_rows = sage_jacobian([F0A_p, F0A_d], [p, d])
J1A_rows = sage_jacobian([F1A_p, F1A_d], [p, d])
J0B_rows = sage_jacobian([F0B_p, F0B_d, F0B_e], [p, d, e])
J1B_rows = sage_jacobian([F1B_p, F1B_d, F1B_e], [p, d, e])


# reduce_mod_f: substitute d=Dexpr (phase 0) or d=1/2-Dexpr (phase 1),
# optionally e=0, then reduce numerator and denominator mod f in univariate p.
#
# The reduced expression is a rational function of p alone; we keep it in
# R_poly's fraction field with d=0, e=0 effectively, i.e., the result is
# a univariate rational function in p stored as an element of FF.

# Build Dexpr in FF (fraction field of R_poly)
Dexpr_ff = FF(1024*p**6 - 2304*p**5 + 1472*p**4 - 48*p**3 - 176*p**2 + 38*p - 9) / FF(2)

# We need f as a polynomial in p (in R_poly). Since f_expanded is in R_p_uni (variable pu),
# convert it.
f_in_p = R_poly(f_expanded.subs({pu: p}))

def reduce_mod_f(expr, phase=0, set_e_zero=False):
    """Reduce a rational expression modulo the cycle polynomial f."""
    subs_dict = {}
    if set_e_zero:
        subs_dict[e] = FF(0)
    if phase == 0:
        subs_dict[d] = Dexpr_ff
    else:
        subs_dict[d] = FF(QQ(1)/2) - Dexpr_ff

    # Apply substitutions
    expr_sub = expr
    for var, val in subs_dict.items():
        expr_sub = expr_sub.subs({var: val})

    # Now expr_sub is a univariate rational function in p (in FF).
    # Extract numerator and denominator as polynomials in p.
    num = R_poly(expr_sub.numerator())
    den = R_poly(expr_sub.denominator())

    # These should be univariate in p (d and e gone). Reduce mod f_in_p.
    num_red = num % f_in_p
    den_red = den % f_in_p

    return FF(num_red) / FF(den_red)


def reduce_entry(expr):
    """Reduce an already-univariate-in-p rational expression modulo f."""
    num = R_poly(expr.numerator())
    den = R_poly(expr.denominator())
    num_red = num % f_in_p
    den_red = den % f_in_p
    return FF(num_red) / FF(den_red)


# Build reduced Jacobians
J0A_red = [[reduce_mod_f(J0A_rows[i][j], phase=0) for j in range(2)] for i in range(2)]
J1A_red = [[reduce_mod_f(J1A_rows[i][j], phase=1) for j in range(2)] for i in range(2)]
RA_red = [[reduce_entry(sum(J1A_red[i][k] * J0A_red[k][j] for k in range(2))) for j in range(2)] for i in range(2)]
tr_RA = reduce_entry(RA_red[0][0] + RA_red[1][1])
det_RA = reduce_entry(RA_red[0][0] * RA_red[1][1] - RA_red[0][1] * RA_red[1][0])

J0B_red = [[reduce_mod_f(J0B_rows[i][j], phase=0, set_e_zero=True) for j in range(3)] for i in range(3)]
J1B_red = [[reduce_mod_f(J1B_rows[i][j], phase=1, set_e_zero=True) for j in range(3)] for i in range(3)]
RB_red = [[reduce_entry(sum(J1B_red[i][k] * J0B_red[k][j] for k in range(3))) for j in range(3)] for i in range(3)]
kappa_expr = RB_red[2][2]

# Block-triangular structure of R_B
assert frac_equal(RB_red[2][0], 0), "R_B[2,0] should be zero (block-triangular)"
assert frac_equal(RB_red[2][1], 0), "R_B[2,1] should be zero (block-triangular)"
assert frac_equal(RB_red[0][0], RA_red[0][0]), "R_B upper-left block should equal R_A"
assert frac_equal(RB_red[0][1], RA_red[0][1]), "R_B upper-left block should equal R_A"
assert frac_equal(RB_red[1][0], RA_red[1][0]), "R_B upper-left block should equal R_A"
assert frac_equal(RB_red[1][1], RA_red[1][1]), "R_B upper-left block should equal R_A"

# Exact elimination polynomials for eigenvalues.
# We need resultant of the characteristic polynomial (in x) with f (in p),
# eliminating p.
# Build bivariate ring R[x][p] and compute resultant w.r.t. p.
R_for_x = PolynomialRing(QQ, 'x')
xvar = R_for_x.gen()
R_xp = PolynomialRing(R_for_x, 'pv')
pv = R_xp.gen()

# Convert tr_RA and det_RA to polynomials in pv
# These are univariate rational functions of p; extract num/den in R_poly,
# then map p -> pv.
tr_num_rp = R_poly(tr_RA.numerator())
tr_den_rp = R_poly(tr_RA.denominator())
det_num_rp = R_poly(det_RA.numerator())
det_den_rp = R_poly(det_RA.denominator())

def rpoly_to_pv(poly_rp):
    """Convert a univariate-in-p element of R_poly to R_xp (polynomial in pv over R_for_x)."""
    result = R_xp(0)
    for exp_tuple, coeff in poly_rp.dict().items():
        ep = exp_tuple[0]  # power of p
        result += R_xp(QQ(coeff)) * pv**int(ep)
    return result

tr_num_pv = rpoly_to_pv(tr_num_rp)
tr_den_pv = rpoly_to_pv(tr_den_rp)
det_num_pv = rpoly_to_pv(det_num_rp)
det_den_pv = rpoly_to_pv(det_den_rp)
f_pv = rpoly_to_pv(f_in_p)

# Characteristic polynomial: den_tr * den_det * x^2 - num_tr * den_det * x + num_det * den_tr
char_poly_pv = (tr_den_pv * det_den_pv * R_xp(xvar**2)
              - tr_num_pv * det_den_pv * R_xp(xvar)
              + det_num_pv * tr_den_pv)

P_lambda_raw = char_poly_pv.resultant(f_pv)
P_lambda_poly = R_for_x(P_lambda_raw)
P_lambda_factor = R_for_x(P_lambda_poly // P_lambda_poly.leading_coefficient())

# Similarly for kappa
kap_num_rp = R_poly(kappa_expr.numerator())
kap_den_rp = R_poly(kappa_expr.denominator())
kap_num_pv = rpoly_to_pv(kap_num_rp)
kap_den_pv = rpoly_to_pv(kap_den_rp)

kap_char_pv = kap_den_pv * R_xp(xvar) - kap_num_pv
P_kappa_raw = kap_char_pv.resultant(f_pv)
P_kappa_poly = R_for_x(P_kappa_raw)
P_kappa_factor = R_for_x(P_kappa_poly // P_kappa_poly.leading_coefficient())

assert P_lambda_factor.is_irreducible(), "P_lambda must be irreducible"
assert P_kappa_factor.is_irreducible(), "P_kappa must be irreducible"

# Match manuscript polynomials
P_cycle_manuscript = R_p_uni(
    1024*pu**7 - 3072*pu**6 + 3200*pu**5 - 1152*pu**4 - 144*pu**3 + 176*pu**2 - 40*pu + 7
)
P_lambda_manuscript = R_for_x(
    16*xvar**14 - 117232*xvar**13 - 8330040*xvar**12 - 3628852593*xvar**11
    + 766183444916*xvar**10 - 37440753370248*xvar**9 + 504085851461616*xvar**8
    + 57085439654130*xvar**7 + 18299281845941356*xvar**6 - 5676298472505780*xvar**5
    + 422743039128068*xvar**4 - 4789675432569*xvar**3 + 7468158864*xvar**2 - 2899292*xvar + 4
)
P_kappa_manuscript = R_for_x(
    xvar**7 - 35*xvar**6 + 385*xvar**5 - 1835*xvar**4 + 2315*xvar**3 - 625*xvar**2 + 51*xvar - 1
)
assert f_expanded == P_cycle_manuscript, "Cycle polynomial mismatch"
assert R_for_x(16 * P_lambda_factor) == P_lambda_manuscript, "P_lambda mismatch"
assert P_kappa_factor == P_kappa_manuscript, "P_kappa mismatch"

# Exact root isolation for eigenvalues
lam_int = (QQ(944920594891271) / 10**16, QQ(944920594891272) / 10**16)
nu_int = (QQ(11453882842312) / 10**16, QQ(11453882842313) / 10**16)
kap_int = (QQ(2038817328642625) / 10**16, QQ(2038817328642626) / 10**16)

assert count_roots_in(P_lambda_factor, lam_int[0], lam_int[1]) == 1, "lambda root isolation failed"
assert count_roots_in(P_lambda_factor, nu_int[0], nu_int[1]) == 1, "nu root isolation failed"
assert count_roots_in(P_kappa_factor, kap_int[0], kap_int[1]) == 1, "kappa root isolation failed"

# Exact interval evaluation of characteristic data at p_*
tr_iv = eval_expr_iv(tr_RA, {p: P_iv})
det_iv = eval_expr_iv(det_RA, {p: P_iv})
disc_iv = tr_iv**2 - 4 * det_iv
sqrt_disc_iv = sqrt_interval_q(disc_iv, digits=30)
lam_from_char_iv = (tr_iv + sqrt_disc_iv) / 2
nu_from_char_iv = (tr_iv - sqrt_disc_iv) / 2
kap_eval_iv = eval_expr_iv(kappa_expr, {p: P_iv})

lam_iv = qiv(lam_int[0], lam_int[1]).intersect(lam_from_char_iv)
nu_iv = qiv(nu_int[0], nu_int[1]).intersect(nu_from_char_iv)
kap_iv = qiv(kap_int[0], kap_int[1]).intersect(kap_eval_iv)

cA_iv = sqrt_interval_q(lam_iv, digits=30)
cB_iv = sqrt_interval_q(kap_iv, digits=30)
assert cA_iv.lo > _py0 and cA_iv.hi < _py1
assert cB_iv.lo > _py0 and cB_iv.hi < _py1


# -----------------------------------------------------------------------------
# Exact dominant eigenvectors normalized by p-component = 1
# -----------------------------------------------------------------------------

def eval_red_matrix(Mred):
    return [[eval_expr_iv(Mred[i][j], {p: P_iv}) for j in range(len(Mred[0]))] for i in range(len(Mred))]


J0A_iv = eval_red_matrix(J0A_red)
J1A_iv = eval_red_matrix(J1A_red)
J0B_iv = eval_red_matrix(J0B_red)
J1B_iv = eval_red_matrix(J1B_red)
RA_iv = eval_red_matrix(RA_red)
RB_iv = eval_red_matrix(RB_red)

# A: theta_A from dominant eigenvector of R_A
assert not RA_iv[0][1].contains_zero()
assert not (lam_iv - RA_iv[1][1]).contains_zero()
thetaA_from_row1 = (lam_iv - RA_iv[0][0]) / RA_iv[0][1]
thetaA_from_row2 = RA_iv[1][0] / (lam_iv - RA_iv[1][1])
thetaA_iv = thetaA_from_row1.intersect(thetaA_from_row2)
assert thetaA_iv.lo > _py0

# B: phase-0 dominant eigenvector of R_B, normalized by p=1
A11 = RB_iv[0][1]
A12 = RB_iv[0][2]
B1_rhs = kap_iv - RB_iv[0][0]
A21 = RB_iv[1][1] - kap_iv
A22 = RB_iv[1][2]
B2_rhs = -RB_iv[1][0]
Det = A11 * A22 - A12 * A21
assert not Det.contains_zero()
thetaB0_iv = (B1_rhs * A22 - A12 * B2_rhs) / Det
chiB_iv = (A11 * B2_rhs - B1_rhs * A21) / Det
thetaB1_iv = -(thetaB0_iv + chiB_iv)  # exact DS_B symmetry
assert thetaB0_iv.lo > _py0
assert chiB_iv.lo > _py0
assert thetaB1_iv.hi < _py0

# Half-step p-multipliers from the exact dominant directions
cA0_iv = J0A_iv[0][0] + J0A_iv[0][1] * thetaA_iv
cA1_iv = J1A_iv[0][0] - J1A_iv[0][1] * thetaA_iv
cB0_iv = J0B_iv[0][0] + J0B_iv[0][1] * thetaB0_iv + J0B_iv[0][2] * chiB_iv
cB1_iv = J1B_iv[0][0] + J1B_iv[0][1] * thetaB1_iv + J1B_iv[0][2] * chiB_iv

assert cA0_iv.lo > _py0 and cA1_iv.lo > _py0 and cA0_iv.hi < _py1 and cA1_iv.hi < _py1
assert cB0_iv.lo > _py0 and cB1_iv.lo > _py0 and cB0_iv.hi < _py1 and cB1_iv.hi < _py1

# Intersect with the exact square-root enclosures.
cA_iv = cA_iv.intersect(cA0_iv).intersect(cA1_iv)
cB_iv = cB_iv.intersect(cB0_iv).intersect(cB1_iv)


# -----------------------------------------------------------------------------
# Explicit phase sectors S_i^A and S_i^B
# -----------------------------------------------------------------------------

RADIUS = PyFraction(int(1), int(40000000))   # 2.5e-8 exactly
RHO = PyFraction(int(1), int(1000))          # sector slope parameter 10^{-3}

# exact-basis interval Jacobians for half-step maps
B0A = [[qiv(1), qiv(0)], [thetaA_iv, qiv(1)]]
B1A = [[qiv(1), qiv(0)], [-thetaA_iv, qiv(1)]]
Binv0A = [[qiv(1), qiv(0)], [-thetaA_iv, qiv(1)]]
Binv1A = [[qiv(1), qiv(0)], [thetaA_iv, qiv(1)]]
L0A_iv = matmul_q(Binv1A, matmul_q(J0A_iv, B0A))
L1A_iv = matmul_q(Binv0A, matmul_q(J1A_iv, B1A))

B0B = [[qiv(1), qiv(0), qiv(0)], [thetaB0_iv, qiv(1), qiv(0)], [chiB_iv, qiv(0), qiv(1)]]
B1B = [[qiv(1), qiv(0), qiv(0)], [thetaB1_iv, qiv(1), qiv(0)], [chiB_iv, qiv(0), qiv(1)]]
Binv0B = [[qiv(1), qiv(0), qiv(0)], [-thetaB0_iv, qiv(1), qiv(0)], [-chiB_iv, qiv(0), qiv(1)]]
Binv1B = [[qiv(1), qiv(0), qiv(0)], [-thetaB1_iv, qiv(1), qiv(0)], [-chiB_iv, qiv(0), qiv(1)]]
L0B_iv = matmul_q(Binv1B, matmul_q(J0B_iv, B0B))
L1B_iv = matmul_q(Binv0B, matmul_q(J1B_iv, B1B))


# Groebner-basis proof of exact transverse zeros.
# We work in a bivariate polynomial ring QQ[thA_s, p_s] with lex order
# (thA_s > p_s), so the Groebner basis can eliminate thA_s.

R_thA = PolynomialRing(QQ, 'thA_s, p_s', order='lex')
thA_s, p_s = R_thA.gens()

# f as polynomial in p_s
f_thA_ring = R_thA(f_in_p.subs({p: p_s}))

# Helper: convert a reduced Jacobian entry (univariate rational function of p in FF)
# to numerator and denominator in R_thA.
def get_entry_in_ring(expr, ring, p_target):
    """Convert FF element (univariate in p) to (num, den) in the given ring."""
    num_rp = R_poly(expr.numerator())
    den_rp = R_poly(expr.denominator())
    num_out = ring(0)
    den_out = ring(0)
    for exp_tuple, coeff in num_rp.dict().items():
        ep = exp_tuple[0]
        num_out += ring(QQ(coeff)) * p_target**int(ep)
    for exp_tuple, coeff in den_rp.dict().items():
        ep = exp_tuple[0]
        den_out += ring(QQ(coeff)) * p_target**int(ep)
    return num_out, den_out

# A-side eigenvector slope relation:
# R_A[0,1] * thA^2 + (R_A[0,0] - R_A[1,1]) * thA - R_A[1,0] = 0
# Clear denominators by multiplying through.
R01n, R01d = get_entry_in_ring(RA_red[0][1], R_thA, p_s)
R00n, R00d = get_entry_in_ring(RA_red[0][0], R_thA, p_s)
R11n, R11d = get_entry_in_ring(RA_red[1][1], R_thA, p_s)
R10n, R10d = get_entry_in_ring(RA_red[1][0], R_thA, p_s)

# The slope relation (cleared of denominators):
# R01n * (R00d*R11d*R10d) * thA^2
# + (R00n*R01d*R11d*R10d - R11n*R01d*R00d*R10d) * thA
# - R10n * R01d*R00d*R11d = 0
slope_poly_A = (R01n * R00d * R11d * R10d * thA_s**2 +
                (R00n * R01d * R11d * R10d - R11n * R01d * R00d * R10d) * thA_s -
                R10n * R01d * R00d * R11d)

I_A = R_thA.ideal([f_thA_ring, slope_poly_A])
GA = I_A.groebner_basis()

# Build L0A_sym[1,0] and L1A_sym[1,0] numerators and reduce modulo the ideal.
# For phase 0: L0A = Binv1A * J0A * B0A
#   Binv1A = [[1,0],[thA,1]], B0A = [[1,0],[thA,1]]
#   (J0A * B0A)[:,0] = [J0A[0,0] + J0A[0,1]*thA, J0A[1,0] + J0A[1,1]*thA]
#   L0A[1,0] = thA*(J0A[0,0]+J0A[0,1]*thA) + (J0A[1,0]+J0A[1,1]*thA)
J0A00n, J0A00d = get_entry_in_ring(J0A_red[0][0], R_thA, p_s)
J0A01n, J0A01d = get_entry_in_ring(J0A_red[0][1], R_thA, p_s)
J0A10n, J0A10d = get_entry_in_ring(J0A_red[1][0], R_thA, p_s)
J0A11n, J0A11d = get_entry_in_ring(J0A_red[1][1], R_thA, p_s)

# Clear denominators:
# thA * (J00n/J00d + J01n/J01d * thA) + (J10n/J10d + J11n/J11d * thA)
# = [thA * (J00n*J01d + J01n*J00d*thA) * J10d*J11d + (J10n*J00d*J01d*J11d + J11n*J00d*J01d*J10d*thA)] / (J00d*J01d*J10d*J11d)
# We only need the numerator to check it's zero mod the ideal.
L0A_10_num = (thA_s * (J0A00n * J0A01d + J0A01n * J0A00d * thA_s) * J0A10d * J0A11d +
              (J0A10n * J0A00d * J0A01d * J0A11d + J0A11n * J0A00d * J0A01d * J0A10d * thA_s))
assert I_A.reduce(L0A_10_num) == 0, "L0A[1,0] is not exactly zero (Groebner check)"

# For phase 1: L1A = Binv0A * J1A * B1A
#   Binv0A = [[1,0],[-thA,1]], B1A = [[1,0],[-thA,1]]
#   (J1A * B1A)[:,0] = [J1A[0,0]-J1A[0,1]*thA, J1A[1,0]-J1A[1,1]*thA]
#   L1A[1,0] = -thA*(J1A[0,0]-J1A[0,1]*thA) + (J1A[1,0]-J1A[1,1]*thA)
J1A00n, J1A00d = get_entry_in_ring(J1A_red[0][0], R_thA, p_s)
J1A01n, J1A01d = get_entry_in_ring(J1A_red[0][1], R_thA, p_s)
J1A10n, J1A10d = get_entry_in_ring(J1A_red[1][0], R_thA, p_s)
J1A11n, J1A11d = get_entry_in_ring(J1A_red[1][1], R_thA, p_s)

L1A_10_num = (-thA_s * (J1A00n * J1A01d - J1A01n * J1A00d * thA_s) * J1A10d * J1A11d +
              (J1A10n * J1A00d * J1A01d * J1A11d - J1A11n * J1A00d * J1A01d * J1A10d * thA_s))
assert I_A.reduce(L1A_10_num) == 0, "L1A[1,0] is not exactly zero (Groebner check)"


# B-side Groebner proof
R_thB = PolynomialRing(QQ, 'thB_s, chB_s, p_s3', order='lex')
thB_s, chB_s, p_s3 = R_thB.gens()

f_thB_ring = R_thB(f_in_p.subs({p: p_s3}))

def get_RB_entry_ps3(i, j):
    return get_entry_in_ring(RB_red[i][j], R_thB, p_s3)

RB01n, RB01d = get_RB_entry_ps3(0, 1)
RB02n, RB02d = get_RB_entry_ps3(0, 2)
RB00n, RB00d = get_RB_entry_ps3(0, 0)
RB10n, RB10d = get_RB_entry_ps3(1, 0)
RB11n, RB11d = get_RB_entry_ps3(1, 1)
RB12n, RB12d = get_RB_entry_ps3(1, 2)
kap_n, kap_d = get_RB_entry_ps3(2, 2)

# Eigenvector relation 1: R01*thB + R02*chB - (kappa - R00) = 0
B_eig_1 = (RB01n * RB02d * kap_d * RB00d * thB_s +
           RB02n * RB01d * kap_d * RB00d * chB_s -
           (kap_n * RB00d - RB00n * kap_d) * RB01d * RB02d)

# Eigenvector relation 2: (R11 - kappa)*thB + R12*chB + R10 = 0
B_eig_2 = ((RB11n * kap_d - kap_n * RB11d) * RB12d * RB10d * thB_s +
           RB12n * RB11d * kap_d * RB10d * chB_s +
           RB10n * RB11d * kap_d * RB12d)

I_B = R_thB.ideal([f_thB_ring, B_eig_1, B_eig_2])
GB = I_B.groebner_basis()

# Check L0B and L1B first-column transverse entries reduce to zero
def get_JB_entry_ps3(J_red, i, j):
    return get_entry_in_ring(J_red[i][j], R_thB, p_s3)


def build_LB_10_num(J_red, phase):
    entries = {}
    for i in range(3):
        for j in range(3):
            entries[(i,j)] = get_JB_entry_ps3(J_red, i, j)
    if phase == 0:
        # B0B = [[1,0,0],[thB,1,0],[chB,0,1]]
        # Binv1B = [[1,0,0],[thB+chB,1,0],[-chB,0,1]]
        b0, b1, b2 = R_thB(1), thB_s, chB_s
        binv10, binv20 = thB_s + chB_s, -chB_s
    else:
        # B1B = [[1,0,0],[-(thB+chB),1,0],[chB,0,1]]
        # Binv0B = [[1,0,0],[-thB,1,0],[-chB,0,1]]
        b0, b1, b2 = R_thB(1), -(thB_s + chB_s), chB_s
        binv10, binv20 = -thB_s, -chB_s

    # Clear all 9 denominators
    all_dens = [entries[(i,j)][1] for i in range(3) for j in range(3)]
    D = R_thB(1)
    for dd_v in all_dens:
        D = D * dd_v

    def J_scaled(i, j):
        return entries[(i,j)][0] * (D // entries[(i,j)][1])

    # (J*B)[:,0] = [J[i,0]*1 + J[i,1]*b1 + J[i,2]*b2 for i in range(3)]
    col0 = [J_scaled(i,0) + J_scaled(i,1)*b1 + J_scaled(i,2)*b2 for i in range(3)]

    L10_num = binv10 * col0[0] + col0[1]
    L20_num = binv20 * col0[0] + col0[2]
    return L10_num, L20_num

L0B_10_num, L0B_20_num = build_LB_10_num(J0B_red, 0)
L1B_10_num, L1B_20_num = build_LB_10_num(J1B_red, 1)

assert I_B.reduce(L0B_10_num) == 0, "L0B[1,0] Groebner check failed"
assert I_B.reduce(L0B_20_num) == 0, "L0B[2,0] Groebner check failed"
assert I_B.reduce(L1B_10_num) == 0, "L1B[1,0] Groebner check failed"
assert I_B.reduce(L1B_20_num) == 0, "L1B[2,0] Groebner check failed"


# Narrow interval sanity check for the numerically enclosed versions
DIAGNOSTIC_ZERO_TOL = PyFraction(int(1), int(50000000))  # 2e-8

for _name, _L in [("L0A", L0A_iv), ("L1A", L1A_iv)]:
    _entry = _L[1][0]
    assert _entry.contains_zero(), f"{_name}[1][0] should contain the exact zero"
    assert _entry.abs_upper() < DIAGNOSTIC_ZERO_TOL, f"{_name}[1][0] enclosure is unexpectedly wide"

for _name, _L in [("L0B", L0B_iv), ("L1B", L1B_iv)]:
    for _row in [1, 2]:
        _entry = _L[_row][0]
        assert _entry.contains_zero(), f"{_name}[{_row}][0] should contain the exact zero"
        assert _entry.abs_upper() < DIAGNOSTIC_ZERO_TOL, f"{_name}[{_row}][0] enclosure is unexpectedly wide"


# small phase boxes covering the sectors
half = qiv(QQ(1)/2)
one = qiv(1)

kA = thetaA_iv.hi + RHO
kB0d = thetaB0_iv.hi + RHO
kB1d = abs(thetaB1_iv.lo) + RHO
kBe = chiB_iv.hi + RHO

A0_box = [P_iv + qiv(0, RADIUS), d_iv + qiv(0, kA * RADIUS)]
A1_box = [P_iv + qiv(0, RADIUS), (half - d_iv) + qiv(-(kA * RADIUS), 0)]
B0_box = [P_iv + qiv(0, RADIUS), d_iv + qiv(0, kB0d * RADIUS), qiv(0, kBe * RADIUS)]
B1_box = [P_iv + qiv(0, RADIUS), (half - d_iv) + qiv(-(kB1d * RADIUS), 0), qiv(0, kBe * RADIUS)]


# Hessian sup norms on the phase boxes
def hessian_sup_matrix(expr, vars_tuple, box):
    env = {v: iv for v, iv in zip(vars_tuple, box)}
    out = {}
    for i in range(len(vars_tuple)):
        for j in range(len(vars_tuple)):
            dexpr = diff_rational(expr, vars_tuple[i])
            ddexpr = diff_rational(dexpr, vars_tuple[j])
            val = eval_expr_iv(ddexpr, env)
            out[(i, j)] = val.abs_upper()
    return out


A0_H = [hessian_sup_matrix(F0A_p, (p, d), A0_box), hessian_sup_matrix(F0A_d, (p, d), A0_box)]
A1_H = [hessian_sup_matrix(F1A_p, (p, d), A1_box), hessian_sup_matrix(F1A_d, (p, d), A1_box)]
B0_H = [
    hessian_sup_matrix(F0B_p, (p, d, e), B0_box),
    hessian_sup_matrix(F0B_d, (p, d, e), B0_box),
    hessian_sup_matrix(F0B_e, (p, d, e), B0_box),
]
B1_H = [
    hessian_sup_matrix(F1B_p, (p, d, e), B1_box),
    hessian_sup_matrix(F1B_d, (p, d, e), B1_box),
    hessian_sup_matrix(F1B_e, (p, d, e), B1_box),
]


def A_component_K(H, k):
    return PyFraction(int(1), int(2)) * (H[(0, 0)] + int(2) * k * H[(0, 1)] + (k**int(2)) * H[(1, 1)])


def B_component_K(H, kd, ke):
    return PyFraction(int(1), int(2)) * (
        H[(0, 0)]
        + int(2) * kd * H[(0, 1)]
        + int(2) * ke * H[(0, 2)]
        + (kd**int(2)) * H[(1, 1)]
        + int(2) * kd * ke * H[(1, 2)]
        + (ke**int(2)) * H[(2, 2)]
    )


Kx_A0 = A_component_K(A0_H[0], kA)
Kyphys_A0 = A_component_K(A0_H[1], kA)
Ky_A0 = thetaA_iv.hi * Kx_A0 + Kyphys_A0

Kx_A1 = A_component_K(A1_H[0], kA)
Kyphys_A1 = A_component_K(A1_H[1], kA)
Ky_A1 = thetaA_iv.hi * Kx_A1 + Kyphys_A1

Kx_B0 = B_component_K(B0_H[0], kB0d, kBe)
Kyphys_B0 = B_component_K(B0_H[1], kB0d, kBe)
Kzphys_B0 = B_component_K(B0_H[2], kB0d, kBe)
Ky_B0 = abs(thetaB1_iv.lo) * Kx_B0 + Kyphys_B0
Kz_B0 = chiB_iv.hi * Kx_B0 + Kzphys_B0

Kx_B1 = B_component_K(B1_H[0], kB1d, kBe)
Kyphys_B1 = B_component_K(B1_H[1], kB1d, kBe)
Kzphys_B1 = B_component_K(B1_H[2], kB1d, kBe)
Ky_B1 = thetaB0_iv.hi * Kx_B1 + Kyphys_B1
Kz_B1 = chiB_iv.hi * Kx_B1 + Kzphys_B1


def A_sector_constants(L, Kx, Ky):
    c = L[0][0]
    a = L[0][1]
    t = L[1][1]
    x_lo = c.lo - a.abs_upper() * RHO - Kx * RADIUS
    x_hi = c.hi + a.abs_upper() * RHO + Kx * RADIUS
    tau_hi = (t.abs_upper() * RHO + Ky * RADIUS) / x_lo
    alpha = t.abs_upper() / x_lo
    beta = Ky / x_lo
    gamma_val = a.abs_upper() / cA_iv.lo
    delta = Kx / cA_iv.lo
    return {
        "x_lo": x_lo, "x_hi": x_hi, "tau_hi": tau_hi,
        "alpha": alpha, "beta": beta, "gamma": gamma_val, "delta": delta,
    }


def B_sector_constants(L, Kx, Ky, Kz):
    c = L[0][0]
    ay = L[0][1]
    az = L[0][2]
    row_y = L[1][1].abs_upper() + L[1][2].abs_upper()
    row_z = L[2][1].abs_upper() + L[2][2].abs_upper()
    x_lo = c.lo - (ay.abs_upper() + az.abs_upper()) * RHO - Kx * RADIUS
    x_hi = c.hi + (ay.abs_upper() + az.abs_upper()) * RHO + Kx * RADIUS
    tau_hi = max((row_y * RHO + Ky * RADIUS) / x_lo, (row_z * RHO + Kz * RADIUS) / x_lo)
    alpha = max(row_y, row_z) / x_lo
    beta = max(Ky, Kz) / x_lo
    gamma_val = (ay.abs_upper() + az.abs_upper()) / cB_iv.lo
    delta = Kx / cB_iv.lo
    return {
        "x_lo": x_lo, "x_hi": x_hi, "tau_hi": tau_hi,
        "alpha": alpha, "beta": beta, "gamma": gamma_val, "delta": delta,
    }


A0_sec = A_sector_constants(L0A_iv, Kx_A0, Ky_A0)
A1_sec = A_sector_constants(L1A_iv, Kx_A1, Ky_A1)
B0_sec = B_sector_constants(L0B_iv, Kx_B0, Ky_B0, Kz_B0)
B1_sec = B_sector_constants(L1B_iv, Kx_B1, Ky_B1, Kz_B1)

for data in [A0_sec, A1_sec, B0_sec, B1_sec]:
    assert data["x_lo"] > _py0
    assert data["x_hi"] < _py1
    assert data["tau_hi"] < RHO
    assert data["alpha"] < _py1

mA = min(A0_sec["x_lo"], A1_sec["x_lo"])
MA = max(A0_sec["x_hi"], A1_sec["x_hi"])
mB = min(B0_sec["x_lo"], B1_sec["x_lo"])
MB = max(B0_sec["x_hi"], B1_sec["x_hi"])

assert MA / mB < _py1
assert mA / (MB**_py2) > _py1

errA_max = max(A0_sec["gamma"], A1_sec["gamma"]) * RHO + max(A0_sec["delta"], A1_sec["delta"]) * RADIUS
errB_max = max(B0_sec["gamma"], B1_sec["gamma"]) * RHO + max(B0_sec["delta"], B1_sec["delta"]) * RADIUS
assert errA_max < PyFraction(int(1), int(100))
assert errB_max < PyFraction(int(1), int(100))


# -----------------------------------------------------------------------------
# Local branch words on the sector boxes, and whole-burst locking
# -----------------------------------------------------------------------------

LA_list = [[int(LA[i, j]) for j in range(LA.ncols())] for i in range(LA.nrows())]
LB_list = [[int(LB[i, j]) for j in range(LB.ncols())] for i in range(LB.nrows())]


def build_state_A_from_box(box):
    pbox, dbox = box
    return [pbox - half, one - pbox, half - dbox, dbox]


def build_state_B_from_box(box):
    pbox, dbox, ebox = box
    return [pbox - half, one - pbox, half - dbox - ebox, dbox, ebox]


def edge_list_iv(state, M):
    out = []
    for j in range(len(M[0])):
        mu = qiv(0)
        for i, row in enumerate(M):
            mu = mu + state[i] * row[j]
        out.append(mu)
    return out


def update_iv(state, M, j):
    mu = qiv(0)
    for i, row in enumerate(M):
        mu = mu + state[i] * row[j]
    new = [state[i] / (1 + mu * row[j]) for i, row in enumerate(M)]
    return new, mu


def certify_word_on_box(state, M, word):
    min_gap = None
    later_min_surplus = None
    first_surplus_max = None
    selected_edges = []
    for step, j in enumerate(word, start=1):
        edges = edge_list_iv(state, M)
        sel = edges[j]
        selected_edges.append(sel)
        best_other = max(edges[k].hi for k in range(len(edges)) if k != j)
        gap = sel.lo - best_other
        min_gap = gap if min_gap is None else min(min_gap, gap)
        if step >= 2:
            surplus = sel.lo - mu_star_iv.hi
            later_min_surplus = surplus if later_min_surplus is None else min(later_min_surplus, surplus)
        else:
            surplus = sel.hi - mu_star_iv.lo
            first_surplus_max = surplus if first_surplus_max is None else max(first_surplus_max, surplus)
        state, _ = update_iv(state, M, j)
    return min_gap, later_min_surplus, first_surplus_max, selected_edges


word_data = {
    "A0": certify_word_on_box(build_state_A_from_box(A0_box), LA_list, H0),
    "A1": certify_word_on_box(build_state_A_from_box(A1_box), LA_list, H1),
    "B0": certify_word_on_box(build_state_B_from_box(B0_box), LB_list, H0),
    "B1": certify_word_on_box(build_state_B_from_box(B1_box), LB_list, H1),
}

g_br_box = min(word_data[name][0] for name in word_data)
later_surplus_box = min(word_data[name][1] for name in word_data)
first_surplus_box = max(word_data[name][2] for name in word_data)
burst_lock_margin = later_surplus_box - first_surplus_box

assert g_br_box > _py0
assert later_surplus_box > _py0
assert burst_lock_margin > _py0


# -----------------------------------------------------------------------------
# Explicit rational starting point and exact row-duplication data
# -----------------------------------------------------------------------------

a_num = [137116847818847, 362883152181153, 242004611239991, 257995388760009]
b_num = [137116857552466, 362883142447534, 242004593130510, 257995386869490, 20000000]
DEN = 10**15

a0 = [QQ(n) / DEN for n in a_num]
b0 = [QQ(n) / DEN for n in b_num]
assert sum(a0) == 1
assert sum(b0) == 1
assert all(w > 0 for w in a0 + b0)

a0p = qiv(a0[0] + QQ(1)/2)
a0d = qiv(a0[3])
b0p = qiv(b0[0] + QQ(1)/2)
b0d = qiv(b0[3])
b0e = qiv(b0[4])

xA_iv = a0p - P_iv
resA_iv = (a0d - d_iv) / xA_iv - thetaA_iv

xB_iv = b0p - P_iv
resBd_iv = (b0d - d_iv) / xB_iv - thetaB0_iv
resBe_iv = b0e / xB_iv - chiB_iv

assert xA_iv.lo > _py0 and xA_iv.hi < RADIUS
assert xB_iv.lo > _py0 and xB_iv.hi < RADIUS
assert resA_iv.lo > -RHO and resA_iv.hi < RHO
assert resBd_iv.lo > -RHO and resBd_iv.hi < RHO
assert resBe_iv.lo > -RHO and resBe_iv.hi < RHO
assert xA_iv.hi < xB_iv.lo


# -----------------------------------------------------------------------------
# Explicit M_0 construction, margin witness, and row-duplication verification
# -----------------------------------------------------------------------------

M0 = matrix(QQ, 20, 8)
for _r in range(4):
    for _s in range(5):
        for _j in range(4):
            M0[5 * _r + _s, _j] = LA[_r, _j]
            M0[5 * _r + _s, 4 + _j] = LB[_s, _j]

assert all(M0[i, j] in (-1, 1) for i in range(20) for j in range(8)), \
    "M_0 must be a {-1,+1}-valued matrix"

# --- Upper bound: gamma(M_0) <= 1/5 ---
uA_star = [QQ(1)/5, QQ(2)/5, QQ(1)/5, QQ(1)/5]
uB_star = [QQ(1)/5, QQ(2)/5, QQ(1)/5, QQ(1)/5, QQ(0)]
W_star = [uA_star[_r] * uB_star[_s] for _r in range(4) for _s in range(5)]
assert sum(W_star) == 1 and all(w >= 0 for w in W_star), \
    "W* must be a distribution on 20 rows"
M0_edges = [sum(W_star[i] * M0[i, j] for i in range(20)) for j in range(8)]
assert all(ej == QQ(1)/5 for ej in M0_edges), \
    "Product witness must give all eight M_0 edges exactly 1/5"

# --- Lower bound: gamma(M_0) >= 1/5 ---
_beta = [2, 1, 1, 1]
w_star = [QQ(b) / 10 for b in _beta] + [QQ(b) / 10 for b in _beta]
assert sum(w_star) == 1 and all(w > 0 for w in w_star), \
    "w* must be a column distribution"
M0_col_values = [sum(M0[i, j] * w_star[j] for j in range(8)) for i in range(20)]
assert min(M0_col_values) == QQ(1)/5, \
    "Column strategy must achieve min row value 1/5 for gamma(M_0) >= 1/5"

# Row duplication: M_0 -> M_tilde
_dup_total = sum(a_num[_r] * b_num[_s] for _r in range(4) for _s in range(5))
assert _dup_total == DEN ** 2, \
    "Row-duplication counts must total DEN^2"
for _r in range(4):
    for _s in range(5):
        assert QQ(a_num[_r] * b_num[_s]) / _dup_total == a0[_r] * b0[_s], \
            f"Uniform aggregation mismatch at row ({_r},{_s})"


# -----------------------------------------------------------------------------
# Core reporting
# -----------------------------------------------------------------------------

def core_report():
    print("AdaBoost counterexample core certificate -- SageMath version (items 1-14)")
    print("=" * 72)
    print("CAS backend: SageMath", version())
    print()
    print("Exact invariant-manifold identities:")
    print("  F0B|_{e=0} = F0A, F1B|_{e=0} = F1A.")
    print("  The B third-coordinate numerators are exactly divisible by e.")
    print("  Displayed branch-map polynomials (P_A,Q_A,D_A,R_A,P_B,Q_B,D_B,R_B,E_B)")
    print("    verified to match derived rational functions.")
    print()
    print("Cycle data:")
    print("  p_* in", fmt_iv(P_iv))
    print("  d_* in", fmt_iv(d_iv))
    print("  mu_* in", fmt_iv(mu_star_iv))
    print("  Displayed 2-cycle polynomials N_1, N_2 verified term-for-term.")
    print()
    print("Dominant full-return spectra:")
    print("  lambda in", fmt_iv(lam_iv))
    print("  nu     in", fmt_iv(nu_iv))
    print("  kappa  in", fmt_iv(kap_iv))
    print("  c_A = sqrt(lambda) in", fmt_iv(cA_iv))
    print("  (J0A v0A)_p in", fmt_iv(cA0_iv), ",  (J1A v1A)_p in", fmt_iv(cA1_iv))
    print("  c_B = sqrt(kappa)  in", fmt_iv(cB_iv))
    print("  (J0B v0B)_p in", fmt_iv(cB0_iv), ",  (J1B v1B)_p in", fmt_iv(cB1_iv))
    print()
    print("Dominant half-step directions (p-component normalized to 1):")
    print("  theta_A  in", fmt_iv(thetaA_iv))
    print("  theta_B0 in", fmt_iv(thetaB0_iv))
    print("  chi_B    in", fmt_iv(chiB_iv))
    print("  theta_B1 in", fmt_iv(thetaB1_iv))
    print()
    print("Explicit sectors: radius =", fmt_frac(RADIUS), ", rho =", fmt_frac(RHO))
    print("  A0: x'/x in", fmt_iv(qiv(A0_sec["x_lo"], A0_sec["x_hi"])),
          ",  |y'|/x' <=", fmt_frac(A0_sec["tau_hi"]))
    print("  A1: x'/x in", fmt_iv(qiv(A1_sec["x_lo"], A1_sec["x_hi"])),
          ",  |y'|/x' <=", fmt_frac(A1_sec["tau_hi"]))
    print("  B0: x'/x in", fmt_iv(qiv(B0_sec["x_lo"], B0_sec["x_hi"])),
          ",  max(|y'|,|z'|)/x' <=", fmt_frac(B0_sec["tau_hi"]))
    print("  B1: x'/x in", fmt_iv(qiv(B1_sec["x_lo"], B1_sec["x_hi"])),
          ",  max(|y'|,|z'|)/x' <=", fmt_frac(B1_sec["tau_hi"]))
    print()
    print("Local asymptotic constants:")
    print("  alpha_A <=", fmt_frac(max(A0_sec["alpha"], A1_sec["alpha"])),
          ", beta_A <=", fmt_frac(max(A0_sec["beta"], A1_sec["beta"])))
    print("  alpha_B <=", fmt_frac(max(B0_sec["alpha"], B1_sec["alpha"])),
          ", beta_B <=", fmt_frac(max(B0_sec["beta"], B1_sec["beta"])))
    print("  max relative x-error for A <=", fmt_frac(errA_max))
    print("  max relative x-error for B <=", fmt_frac(errB_max))
    print()
    print("Branch-word and burst-lock certificates:")
    print("  local branch-gap lower bound on the phase boxes =", fmt_frac(g_br_box))
    print("  later selected-edge surplus over mu_*            =", fmt_frac(later_surplus_box))
    print("  largest first-edge surplus on any phase box      =", fmt_frac(first_surplus_box))
    print("  whole-burst lock margin                           =", fmt_frac(burst_lock_margin))
    print()
    print("Run-length consequences:")
    print("  max A-run factor MA/mB   =", fmt_frac(MA / mB), "< 1  (so every A-run has length 1)")
    print("  min 2-step B-run factor mA/MB^2 =", fmt_frac(mA / (MB**_py2)), "> 1  (so every B-run has length at most 2)")
    print()
    print("Explicit rational start:")
    print("  a0 =", a0)
    print("  b0 =", b0)
    print("  x_A in", fmt_iv(xA_iv), ", residual in", fmt_iv(resA_iv))
    print("  x_B in", fmt_iv(xB_iv), ", d-residual in", fmt_iv(resBd_iv), ", e-residual in", fmt_iv(resBe_iv))
    print()
    print("Theorem-level M_0 verification:")
    print("  M_0 = L_A (+) L_B: 20x8 sign matrix constructed explicitly")
    print("  Row witness W* = u_A* x u_B*: all 8 edges exactly 1/5 (gamma <= 1/5)")
    print("  Column witness w* = (beta,beta)/10: min row value = 1/5 (gamma >= 1/5)")
    print("  Row duplication: total rows = DEN^2 =", DEN**2)
    print("  Uniform measure on M_tilde aggregates exactly to W_0 = a_0 x b_0")
    print()
    print("All core symbolic and exact-rational interval assertions passed.")

if __name__ == "__main__":
    core_report()

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt

import sympy as sp

if not __debug__:
    raise RuntimeError(
        "This certificate must not be run with Python -O / PYTHONOPTIMIZE, "
        "because optimized mode strips assert statements."
    )

# -----------------------------------------------------------------------------
# Exact matrices and one-step AdaBoost update
# -----------------------------------------------------------------------------

LA = sp.Matrix([
    [1, 1, -1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
])

LB = sp.Matrix([
    [1, 1, -1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
])

H0 = [0, 2, 3, 0, 1]   # (1,3,4,1,2)
H1 = [0, 3, 2, 0, 1]   # (1,4,3,1,2)


def update_state(state, M, j):
    mu = sp.together(sum(state[i] * M[i, j] for i in range(M.rows)))
    new = [sp.cancel(sp.together(state[i] / (1 + mu * M[i, j]))) for i in range(M.rows)]
    return new, sp.cancel(mu)


# -----------------------------------------------------------------------------
# Reduced branch maps on the affine manifolds
# -----------------------------------------------------------------------------

p, d, e = sp.symbols("p d e")

# A: Sigma_A
stateA = [p - sp.Rational(1, 2), 1 - p, sp.Rational(1, 2) - d, d]
state = stateA
for j in H0:
    state, _ = update_state(state, LA, j)
F0A_p = sp.together(1 - state[1])
F0A_d = sp.together(state[3])
F1A_p = sp.together(F0A_p.subs(d, sp.Rational(1, 2) - d))
F1A_d = sp.together(sp.Rational(1, 2) - F0A_d.subs(d, sp.Rational(1, 2) - d))

# B: Sigma_B
stateB = [p - sp.Rational(1, 2), 1 - p, sp.Rational(1, 2) - d - e, d, e]
state = stateB
for j in H0:
    state, _ = update_state(state, LB, j)
F0B_p = sp.together(1 - state[1])
F0B_d = sp.together(state[3])
F0B_e = sp.together(state[4])

dtil = sp.Rational(1, 2) - d - e
F1B_p = sp.together(F0B_p.subs(d, dtil))
F1B_d = sp.together(sp.Rational(1, 2) - F0B_d.subs(d, dtil) - F0B_e.subs(d, dtil))
F1B_e = sp.together(F0B_e.subs(d, dtil))

# Derive H1 maps directly and verify they match the conjugated formulas
stateA1 = [p - sp.Rational(1, 2), 1 - p, sp.Rational(1, 2) - d, d]
state = stateA1
for j in H1:
    state, _ = update_state(state, LA, j)
F1A_p_direct = sp.together(1 - state[1])
F1A_d_direct = sp.together(state[3])
assert sp.simplify(F1A_p_direct - F1A_p) == 0
assert sp.simplify(F1A_d_direct - F1A_d) == 0

stateB1 = [p - sp.Rational(1, 2), 1 - p, sp.Rational(1, 2) - d - e, d, e]
state = stateB1
for j in H1:
    state, _ = update_state(state, LB, j)
F1B_p_direct = sp.together(1 - state[1])
F1B_d_direct = sp.together(state[3])
F1B_e_direct = sp.together(state[4])
assert sp.simplify(F1B_p_direct - F1B_p) == 0
assert sp.simplify(F1B_d_direct - F1B_d) == 0
assert sp.simplify(F1B_e_direct - F1B_e) == 0

# Structural identities: face e=0 and exact divisibility by e
assert sp.simplify(F0B_p.subs(e, 0) - F0A_p) == 0
assert sp.simplify(F0B_d.subs(e, 0) - F0A_d) == 0
assert sp.simplify(F1B_p.subs(e, 0) - F1A_p) == 0
assert sp.simplify(F1B_d.subs(e, 0) - F1A_d) == 0

num_e0, den_e0 = sp.fraction(sp.together(F0B_e))
num_e1, den_e1 = sp.fraction(sp.together(F1B_e))
rem0 = sp.Poly(sp.expand(num_e0), e).div(sp.Poly(e, e))[1]
rem1 = sp.Poly(sp.expand(num_e1), e).div(sp.Poly(e, e))[1]
assert rem0 == 0
assert rem1 == 0

# -----------------------------------------------------------------------------
# Match displayed manuscript branch-map polynomials (Section 3)
# -----------------------------------------------------------------------------

# A-gadget: F_0^A(p,d) = (P_A/Q_A, D_A/R_A)
PA_manuscript = sp.expand(
    4*d**4 - 6*d**3*p + 2*d**3 - 30*d**2*p**2 + 15*d**2*p - 2*d**2
    - 6*d*p**3 + 14*d*p**2 - 4*d*p + 6*p**4 + p**3 - 2*p**2
)
QA_manuscript = sp.expand(
    8*d**4 - 8*d**3*p + 2*d**3 - 48*d**2*p**2 + 22*d**2*p - 3*d**2
    - 8*d*p**3 + 22*d*p**2 - 6*d*p + 8*p**4 + 2*p**3 - 3*p**2
)
DA_manuscript = sp.expand(
    d*(2*d**3 + 2*d**2*p - d**2 - 6*d*p**2 + 2*p**3 + p**2)
)
RA_manuscript = sp.expand(
    8*d**4 - 2*d**3 - 24*d**2*p**2 + 6*d**2*p - d**2
    + 10*d*p**2 - 2*d*p + 2*p**3 - p**2
)

_num_F0Ap, _den_F0Ap = sp.fraction(sp.together(F0A_p))
_num_F0Ad, _den_F0Ad = sp.fraction(sp.together(F0A_d))
assert sp.expand(_num_F0Ap * QA_manuscript - _den_F0Ap * PA_manuscript) == 0, \
    "A-gadget p-component: displayed P_A/Q_A differs from computed F_0^A"
assert sp.expand(_num_F0Ad * RA_manuscript - _den_F0Ad * DA_manuscript) == 0, \
    "A-gadget d-component: displayed D_A/R_A differs from computed F_0^A"

# B-gadget: F_0^B(p,d,e) = (P_B/Q_B, D_B/R_B, E_B/R_B)
PB_manuscript = sp.expand(
    4*d**4 + 16*d**3*e - 6*d**3*p + 2*d**3
    + 20*d**2*e**2 - 36*d**2*e*p + 12*d**2*e - 30*d**2*p**2 + 15*d**2*p - 2*d**2
    + 8*d*e**3 - 46*d*e**2*p + 18*d*e**2 - 48*d*e*p**2 + 30*d*e*p - 4*d*e
    - 6*d*p**3 + 14*d*p**2 - 4*d*p
    - 16*e**3*p + 8*e**3 - 22*e**2*p**2 + 15*e**2*p - 2*e**2
    + 8*e*p**2 - 4*e*p + 6*p**4 + p**3 - 2*p**2
)
QB_manuscript = sp.expand(
    8*d**4 + 32*d**3*e - 8*d**3*p + 2*d**3
    + 40*d**2*e**2 - 56*d**2*e*p + 16*d**2*e - 48*d**2*p**2 + 22*d**2*p - 3*d**2
    + 16*d*e**3 - 72*d*e**2*p + 26*d*e**2 - 72*d*e*p**2 + 44*d*e*p - 6*d*e
    - 8*d*p**3 + 22*d*p**2 - 6*d*p
    - 24*e**3*p + 12*e**3 - 32*e**2*p**2 + 22*e**2*p - 3*e**2
    + 12*e*p**2 - 6*e*p + 8*p**4 + 2*p**3 - 3*p**2
)
DB_manuscript = sp.expand(
    d*(
        2*d**3 + 8*d**2*e + 2*d**2*p - d**2
        + 10*d*e**2 - 2*d*e - 6*d*p**2
        + 4*e**3 - 2*e**2*p - e**2 - 4*e*p**2
        + 2*p**3 + p**2
    )
)
RB_manuscript = sp.expand(
    8*d**4 + 32*d**3*e - 2*d**3
    + 40*d**2*e**2 - 24*d**2*e*p - 24*d**2*p**2 + 6*d**2*p - d**2
    + 16*d*e**3 - 32*d*e**2*p + 6*d*e**2 - 24*d*e*p**2 + 12*d*e*p - 2*d*e
    + 10*d*p**2 - 2*d*p
    - 8*e**3*p + 4*e**3 - 8*e**2*p**2 + 6*e**2*p - e**2
    + 4*e*p**2 - 2*e*p + 2*p**3 - p**2
)
EB_manuscript = sp.expand(
    e*(
        2*d**3 + 4*d**2*e - 6*d**2*p + d**2
        + 2*d*e**2 - 8*d*e*p + 2*d*e + 2*d*p**2
        - 2*e**2*p + e**2 + 2*p**3 - p**2
    )
)

_num_F0Bp, _den_F0Bp = sp.fraction(sp.together(F0B_p))
_num_F0Bd, _den_F0Bd = sp.fraction(sp.together(F0B_d))
_num_F0Be, _den_F0Be = sp.fraction(sp.together(F0B_e))
assert sp.expand(_num_F0Bp * QB_manuscript - _den_F0Bp * PB_manuscript) == 0, \
    "B-gadget p-component: displayed P_B/Q_B differs from computed F_0^B"
assert sp.expand(_num_F0Bd * RB_manuscript - _den_F0Bd * DB_manuscript) == 0, \
    "B-gadget d-component: displayed D_B/R_B differs from computed F_0^B"
assert sp.expand(_num_F0Be * RB_manuscript - _den_F0Be * EB_manuscript) == 0, \
    "B-gadget e-component: displayed E_B/R_B differs from computed F_0^B"


# -----------------------------------------------------------------------------
# Exact period-2 orbit on the face e=0
# -----------------------------------------------------------------------------

N1 = sp.factor(
    sp.expand(sp.fraction(sp.together((F0A_p - p) * sp.denom(F0A_p)))[0] / (-2 * (2 * p - 1)))
)
N2 = sp.factor(
    sp.expand(sp.fraction(sp.together((F0A_d - (sp.Rational(1, 2) - d)) * sp.denom(F0A_d)))[0] * -2)
)

# Match displayed 2-cycle polynomials (Section 4)
N1_manuscript = sp.expand(
    2*d**4 - 2*d**3*p + d**3 - 12*d**2*p**2 + 7*d**2*p - d**2
    - 2*d*p**3 + 6*d*p**2 - 2*d*p + 2*p**4 - p**2
)
N2_manuscript = sp.expand(
    -16*d**5 + 8*d**4 + 48*d**3*p**2 - 16*d**3*p + 2*d**3
    - 32*d**2*p**2 + 10*d**2*p - d**2
    - 8*d*p**3 + 10*d*p**2 - 2*d*p + 2*p**3 - p**2
)
assert sp.expand(N1 - N1_manuscript) == 0, \
    "N_1: displayed polynomial differs from computed expression"
assert sp.expand(N2 - N2_manuscript) == 0, \
    "N_2: displayed polynomial differs from computed expression"

f = sp.factor(sp.resultant(N1, N2, d) / (4096 * p**10 * (2 * p - 1) ** 3))
f_expanded = sp.expand(f)
Dexpr = (
    1024 * p**6 - 2304 * p**5 + 1472 * p**4 - 48 * p**3 - 176 * p**2 + 38 * p - 9
) / 2

assert sp.rem(sp.expand(N1.subs(d, Dexpr)), f_expanded, p) == 0
assert sp.rem(sp.expand(N2.subs(d, Dexpr)), f_expanded, p) == 0

f_poly = sp.Poly(f_expanded, p)
p_lo = sp.Rational(637116837818846, 10**15)
p_hi = sp.Rational(637116837818848, 10**15)
assert f_poly.count_roots(p_lo, p_hi) == 1


# -----------------------------------------------------------------------------
# Exact rational interval arithmetic
# -----------------------------------------------------------------------------


def qfrac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, sp.Integer):
        return Fraction(int(x), 1)
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    raise TypeError(f"Cannot convert {type(x)} to Fraction")


@dataclass(frozen=True)
class QInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"Invalid interval [{self.lo}, {self.hi}]")

    @staticmethod
    def const(x) -> "QInterval":
        if isinstance(x, QInterval):
            return x
        return QInterval(qfrac(x), qfrac(x))

    def __add__(self, other) -> "QInterval":
        o = QInterval.const(other)
        return QInterval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, other) -> "QInterval":
        o = QInterval.const(other)
        return QInterval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other) -> "QInterval":
        o = QInterval.const(other)
        return QInterval(o.lo - self.hi, o.hi - self.lo)

    def __neg__(self) -> "QInterval":
        return QInterval(-self.hi, -self.lo)

    def __mul__(self, other) -> "QInterval":
        o = QInterval.const(other)
        vals = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return QInterval(min(vals), max(vals))

    __rmul__ = __mul__

    def reciprocal(self) -> "QInterval":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(f"Interval crosses zero: [{self.lo}, {self.hi}]")
        vals = [Fraction(1, 1) / self.lo, Fraction(1, 1) / self.hi]
        return QInterval(min(vals), max(vals))

    def __truediv__(self, other) -> "QInterval":
        o = QInterval.const(other)
        return self * o.reciprocal()

    def __rtruediv__(self, other) -> "QInterval":
        o = QInterval.const(other)
        return o * self.reciprocal()

    def __pow__(self, n: int) -> "QInterval":
        if not isinstance(n, int):
            raise TypeError("Only integer powers are supported")
        if n == 0:
            return QInterval.const(1)
        if n < 0:
            return (self ** (-n)).reciprocal()
        if n % 2 == 1:
            return QInterval(self.lo**n, self.hi**n)
        vals = [self.lo**n, self.hi**n]
        lo = Fraction(0, 1) if self.lo <= 0 <= self.hi else min(vals)
        hi = max(vals)
        return QInterval(lo, hi)

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def abs_upper(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))

    def intersect(self, other) -> "QInterval":
        o = QInterval.const(other)
        lo = max(self.lo, o.lo)
        hi = min(self.hi, o.hi)
        if lo > hi:
            raise ValueError(f"Empty intersection of {self} and {o}")
        return QInterval(lo, hi)



def qiv(lo, hi=None) -> QInterval:
    if hi is None:
        return QInterval.const(lo)
    return QInterval(qfrac(lo), qfrac(hi))


# Exact square-root enclosure by integer arithmetic.
# If q >= 0 and scale is a positive integer, this returns the exact floor and
# ceiling of scale * sqrt(q), hence rational lower/upper bounds with denominator
# "scale".  No floating point is used.

def sqrt_bounds_fraction(q: Fraction, scale: int) -> tuple[Fraction, Fraction]:
    if q < 0:
        raise ValueError("sqrt requires a nonnegative rational")
    S = q.numerator * scale * scale
    den = q.denominator
    m = isqrt(S // den)
    while (m + 1) * (m + 1) * den <= S:
        m += 1
    while m * m * den > S:
        m -= 1
    n = m if m * m * den == S else m + 1
    while (n - 1) >= 0 and (n - 1) * (n - 1) * den >= S:
        n -= 1
    while n * n * den < S:
        n += 1
    return Fraction(m, scale), Fraction(n, scale)



def sqrt_interval_q(iv: QInterval, digits: int = 30) -> QInterval:
    if iv.lo < 0:
        raise ValueError("sqrt interval requires nonnegative lower bound")
    scale = 10**digits
    lo, _ = sqrt_bounds_fraction(iv.lo, scale)
    _, hi = sqrt_bounds_fraction(iv.hi, scale)
    return QInterval(lo, hi)


# Exact interval evaluation of SymPy expressions over rational boxes.  The only
# supported operations are the ones actually occurring here: addition,
# multiplication, and integer powers (including negative powers).  All bounds
# are exact because the arithmetic is done in Fraction.

def eval_expr_iv(expr, env: dict[sp.Symbol, QInterval]) -> QInterval:
    expr = sp.sympify(expr)
    memo: dict[sp.Basic, QInterval] = {}

    def rec(node: sp.Basic) -> QInterval:
        if node in memo:
            return memo[node]
        if node in env:
            out = env[node]
        elif node.is_Integer:
            out = qiv(int(node))
        elif node.is_Rational:
            out = qiv(qfrac(node))
        elif node.is_Add:
            out = qiv(0)
            for arg in node.args:
                out = out + rec(arg)
        elif node.is_Mul:
            out = qiv(1)
            for arg in node.args:
                out = out * rec(arg)
        elif node.is_Pow:
            base, exp = node.as_base_exp()
            if not exp.is_Integer:
                raise TypeError(f"Non-integer power encountered: {node}")
            out = rec(base) ** int(exp)
        else:
            raise TypeError(f"Unsupported SymPy node in interval evaluator: {node!r}")
        memo[node] = out
        return out

    return rec(expr)



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
# Reporting utilities (placed here so that both this module and the negated-
# column module can import them)
# -----------------------------------------------------------------------------


def fmt_frac(fr: Fraction, digits: int = 18) -> str:
    with localcontext() as ctx:
        ctx.prec = digits + 8
        dec = Decimal(fr.numerator) / Decimal(fr.denominator)
        return format(dec, f".{digits}g")



def fmt_iv(v: QInterval, digits: int = 18) -> str:
    return f"[{fmt_frac(v.lo, digits)}, {fmt_frac(v.hi, digits)}]"


# -----------------------------------------------------------------------------
# Root, cycle, and exact spectral intervals
# -----------------------------------------------------------------------------

P_iv = qiv(p_lo, p_hi)
d_iv = eval_expr_iv(Dexpr, {p: P_iv})
mu_star_iv = 2 * P_iv - 1

P0A_iv = [P_iv - qiv(sp.Rational(1, 2)), qiv(1) - P_iv, qiv(sp.Rational(1, 2)) - d_iv, d_iv]
P1A_iv = [P_iv - qiv(sp.Rational(1, 2)), qiv(1) - P_iv, d_iv, qiv(sp.Rational(1, 2)) - d_iv]

J0A = sp.Matrix([F0A_p, F0A_d]).jacobian([p, d])
J1A = sp.Matrix([F1A_p, F1A_d]).jacobian([p, d])
J0B = sp.Matrix([F0B_p, F0B_d, F0B_e]).jacobian([p, d, e])
J1B = sp.Matrix([F1B_p, F1B_d, F1B_e]).jacobian([p, d, e])


def reduce_mod_f(expr, phase=0, set_e_zero=False):
    subs = {e: 0} if set_e_zero else {}
    subs[d] = Dexpr if phase == 0 else sp.Rational(1, 2) - Dexpr
    expr = sp.together(sp.cancel(expr.subs(subs)))
    num, den = sp.fraction(expr)
    num = sp.rem(sp.expand(num), f_expanded, p)
    den = sp.rem(sp.expand(den), f_expanded, p)
    return sp.cancel(sp.together(num / den))



def reduce_entry(expr):
    expr = sp.together(expr)
    num, den = sp.fraction(expr)
    num = sp.rem(sp.expand(num), f_expanded, p)
    den = sp.rem(sp.expand(den), f_expanded, p)
    return sp.cancel(sp.together(num / den))


J0A_red = sp.Matrix([[reduce_mod_f(J0A[i, j], phase=0) for j in range(2)] for i in range(2)])
J1A_red = sp.Matrix([[reduce_mod_f(J1A[i, j], phase=1) for j in range(2)] for i in range(2)])
RA_red = sp.Matrix([[reduce_entry(sum(J1A_red[i, k] * J0A_red[k, j] for k in range(2))) for j in range(2)] for i in range(2)])
tr_RA = reduce_entry(RA_red.trace())
det_RA = reduce_entry(RA_red.det())

J0B_red = sp.Matrix([[reduce_mod_f(J0B[i, j], phase=0, set_e_zero=True) for j in range(3)] for i in range(3)])
J1B_red = sp.Matrix([[reduce_mod_f(J1B[i, j], phase=1, set_e_zero=True) for j in range(3)] for i in range(3)])
RB_red = sp.Matrix([[reduce_entry(sum(J1B_red[i, k] * J0B_red[k, j] for k in range(3))) for j in range(3)] for i in range(3)])
kappa_expr = RB_red[2, 2]

# Block-triangular structure of R_B.
assert sp.simplify(RB_red[2, 0]) == 0, "R_B[2,0] should be zero (block-triangular)"
assert sp.simplify(RB_red[2, 1]) == 0, "R_B[2,1] should be zero (block-triangular)"
assert sp.simplify(RB_red[0, 0] - RA_red[0, 0]) == 0, "R_B upper-left block should equal R_A"
assert sp.simplify(RB_red[0, 1] - RA_red[0, 1]) == 0, "R_B upper-left block should equal R_A"
assert sp.simplify(RB_red[1, 0] - RA_red[1, 0]) == 0, "R_B upper-left block should equal R_A"
assert sp.simplify(RB_red[1, 1] - RA_red[1, 1]) == 0, "R_B upper-left block should equal R_A"

# exact elimination polynomials
x = sp.symbols("x")
tr_num, common_den = sp.fraction(tr_RA)
det_num, _ = sp.fraction(det_RA)
P_lambda = sp.factor(sp.resultant(sp.expand(common_den * x**2 - tr_num * x + det_num), f_expanded, p))
P_lambda_factor = sp.expand(P_lambda / sp.LC(sp.Poly(P_lambda, x)))

P_kappa = sp.factor(sp.resultant(sp.expand(sp.fraction(kappa_expr)[1] * x - sp.fraction(kappa_expr)[0]), f_expanded, p))
P_kappa_factor = sp.expand(P_kappa / sp.LC(sp.Poly(P_kappa, x)))

P_lambda_poly = sp.Poly(P_lambda_factor, x)
P_kappa_poly = sp.Poly(P_kappa_factor, x)
assert P_lambda_poly.is_irreducible
assert P_kappa_poly.is_irreducible

# Match the manuscript polynomials exactly (after clearing denominators for P_lambda).
P_cycle_manuscript = sp.expand(
    1024*p**7 - 3072*p**6 + 3200*p**5 - 1152*p**4 - 144*p**3 + 176*p**2 - 40*p + 7
)
P_lambda_manuscript = sp.expand(
    16*x**14 - 117232*x**13 - 8330040*x**12 - 3628852593*x**11
    + 766183444916*x**10 - 37440753370248*x**9 + 504085851461616*x**8
    + 57085439654130*x**7 + 18299281845941356*x**6 - 5676298472505780*x**5
    + 422743039128068*x**4 - 4789675432569*x**3 + 7468158864*x**2 - 2899292*x + 4
)
P_kappa_manuscript = sp.expand(
    x**7 - 35*x**6 + 385*x**5 - 1835*x**4 + 2315*x**3 - 625*x**2 + 51*x - 1
)
assert sp.expand(f_expanded - P_cycle_manuscript) == 0
assert sp.expand(16 * P_lambda_factor - P_lambda_manuscript) == 0
assert sp.expand(P_kappa_factor - P_kappa_manuscript) == 0

# exact root isolation
lam_int = (sp.Rational(944920594891271, 10**16), sp.Rational(944920594891272, 10**16))
nu_int = (sp.Rational(11453882842312, 10**16), sp.Rational(11453882842313, 10**16))
kap_int = (sp.Rational(2038817328642625, 10**16), sp.Rational(2038817328642626, 10**16))
assert P_lambda_poly.count_roots(*lam_int) == 1
assert P_lambda_poly.count_roots(*nu_int) == 1
assert P_kappa_poly.count_roots(*kap_int) == 1

# Exact interval evaluation of the characteristic data at p_*
tr_iv = eval_expr_iv(tr_RA, {p: P_iv})
det_iv = eval_expr_iv(det_RA, {p: P_iv})
disc_iv = tr_iv**2 - 4 * det_iv
sqrt_disc_iv = sqrt_interval_q(disc_iv, digits=30)
lam_from_char_iv = (tr_iv + sqrt_disc_iv) / 2
nu_from_char_iv = (tr_iv - sqrt_disc_iv) / 2
kap_eval_iv = eval_expr_iv(kappa_expr, {p: P_iv})

lam_iv = qiv(*lam_int).intersect(lam_from_char_iv)
nu_iv = qiv(*nu_int).intersect(nu_from_char_iv)
kap_iv = qiv(*kap_int).intersect(kap_eval_iv)

cA_iv = sqrt_interval_q(lam_iv, digits=30)
cB_iv = sqrt_interval_q(kap_iv, digits=30)
assert cA_iv.lo > 0 and cA_iv.hi < 1
assert cB_iv.lo > 0 and cB_iv.hi < 1


# -----------------------------------------------------------------------------
# Exact dominant eigenvectors normalized by p-component = 1
# -----------------------------------------------------------------------------


def eval_red_matrix(Mred):
    return [[eval_expr_iv(Mred[i, j], {p: P_iv}) for j in range(Mred.cols)] for i in range(Mred.rows)]


J0A_iv = eval_red_matrix(J0A_red)
J1A_iv = eval_red_matrix(J1A_red)
J0B_iv = eval_red_matrix(J0B_red)
J1B_iv = eval_red_matrix(J1B_red)
RA_iv = eval_red_matrix(RA_red)
RB_iv = eval_red_matrix(RB_red)

# A: theta_A from the dominant full-return eigenvector of R_A
# We take the intersection of the two row formulas, which certifies an interval
# containing the exact eigenvector slope.
assert not RA_iv[0][1].contains_zero()
assert not (lam_iv - RA_iv[1][1]).contains_zero()
thetaA_from_row1 = (lam_iv - RA_iv[0][0]) / RA_iv[0][1]
thetaA_from_row2 = RA_iv[1][0] / (lam_iv - RA_iv[1][1])
thetaA_iv = thetaA_from_row1.intersect(thetaA_from_row2)
assert thetaA_iv.lo > 0

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
assert thetaB0_iv.lo > 0
assert chiB_iv.lo > 0
assert thetaB1_iv.hi < 0

# half-step p-multipliers from the exact dominant directions
cA0_iv = J0A_iv[0][0] + J0A_iv[0][1] * thetaA_iv
cA1_iv = J1A_iv[0][0] - J1A_iv[0][1] * thetaA_iv
cB0_iv = J0B_iv[0][0] + J0B_iv[0][1] * thetaB0_iv + J0B_iv[0][2] * chiB_iv
cB1_iv = J1B_iv[0][0] + J1B_iv[0][1] * thetaB1_iv + J1B_iv[0][2] * chiB_iv

assert cA0_iv.lo > 0 and cA1_iv.lo > 0 and cA0_iv.hi < 1 and cA1_iv.hi < 1
assert cB0_iv.lo > 0 and cB1_iv.lo > 0 and cB0_iv.hi < 1 and cB1_iv.hi < 1

# Intersect with the exact square-root enclosures.  This certifies that the
# phase multipliers are compatible with c_A^2 = lambda and c_B^2 = kappa.
cA_iv = cA_iv.intersect(cA0_iv).intersect(cA1_iv)
cB_iv = cB_iv.intersect(cB0_iv).intersect(cB1_iv)


# -----------------------------------------------------------------------------
# Explicit phase sectors S_i^A and S_i^B
# -----------------------------------------------------------------------------

RADIUS = Fraction(1, 40000000)   # 2.5e-8 exactly
RHO = Fraction(1, 1000)          # sector slope parameter 10^{-3}

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

# Exact first-column structure in the adapted bases.
#
# The interval matrices L0A_iv, L1A_iv, L0B_iv, L1B_iv are built from interval
# enclosures of the exact dominant directions, so they only *contain* the exact
# zero transverse entries.  To make the certificate fully self-contained, we
# prove those entries are identically zero by symbolic reduction modulo the
# algebraic relations defining the eigen-directions.

def numerator_of(expr):
    return sp.expand(sp.fraction(sp.together(expr))[0])


# A-side: for any eigenvector slope th of R_A, J_i carries [1,th] into the
# conjugate eigendirection and the transformed first column is exactly aligned
# with e_1.
thA = sp.symbols("thA")
B0A_sym = sp.Matrix([[1, 0], [thA, 1]])
B1A_sym = sp.Matrix([[1, 0], [-thA, 1]])
Binv0A_sym = sp.Matrix([[1, 0], [-thA, 1]])
Binv1A_sym = sp.Matrix([[1, 0], [thA, 1]])
L0A_sym = sp.simplify(Binv1A_sym * J0A_red * B0A_sym)
L1A_sym = sp.simplify(Binv0A_sym * J1A_red * B1A_sym)
A_slope_relation = numerator_of(
    RA_red[0, 1] * thA**2 + (RA_red[0, 0] - RA_red[1, 1]) * thA - RA_red[1, 0]
)
GA = sp.groebner([f_expanded, A_slope_relation], thA, p, order="lex")
assert sp.expand(GA.reduce(numerator_of(L0A_sym[1, 0]))[1]) == 0
assert sp.expand(GA.reduce(numerator_of(L1A_sym[1, 0]))[1]) == 0


# B-side: kappa is the exact lower-right eigenvalue of R_B on the invariant
# face e=0, so the dominant direction (1,thB,chB) satisfies the two linear
# eigenvector relations below.  Reducing the transformed first-column entries by
# these relations proves the exact zeros used later in the sector estimates.
thB, chB = sp.symbols("thB chB")
B0B_sym = sp.Matrix([[1, 0, 0], [thB, 1, 0], [chB, 0, 1]])
B1B_sym = sp.Matrix([[1, 0, 0], [-(thB + chB), 1, 0], [chB, 0, 1]])
Binv0B_sym = sp.Matrix([[1, 0, 0], [-thB, 1, 0], [-chB, 0, 1]])
Binv1B_sym = sp.Matrix([[1, 0, 0], [thB + chB, 1, 0], [-chB, 0, 1]])
L0B_sym = sp.simplify(Binv1B_sym * J0B_red * B0B_sym)
L1B_sym = sp.simplify(Binv0B_sym * J1B_red * B1B_sym)
B_eig_relation_1 = numerator_of(
    RB_red[0, 1] * thB + RB_red[0, 2] * chB - (kappa_expr - RB_red[0, 0])
)
B_eig_relation_2 = numerator_of(
    (RB_red[1, 1] - kappa_expr) * thB + RB_red[1, 2] * chB + RB_red[1, 0]
)
GB = sp.groebner([f_expanded, B_eig_relation_1, B_eig_relation_2], thB, chB, p, order="lex")
assert sp.expand(GB.reduce(numerator_of(L0B_sym[1, 0]))[1]) == 0
assert sp.expand(GB.reduce(numerator_of(L0B_sym[2, 0]))[1]) == 0
assert sp.expand(GB.reduce(numerator_of(L1B_sym[1, 0]))[1]) == 0
assert sp.expand(GB.reduce(numerator_of(L1B_sym[2, 0]))[1]) == 0

# We also keep a narrow interval sanity check for the numerically enclosed
# versions built from interval slopes.
DIAGNOSTIC_ZERO_TOL = Fraction(1, 50000000)  # 2e-8, used only as a sanity check

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
half = qiv(sp.Rational(1, 2))
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
# Each derivative is evaluated by exact interval arithmetic on the explicit
# rational box.  No floating-point seeding is used anywhere.

def hessian_sup_matrix(expr, vars_tuple, box):
    env = {v: iv for v, iv in zip(vars_tuple, box)}
    out = {}
    for i in range(len(vars_tuple)):
        for j in range(len(vars_tuple)):
            dexpr = sp.diff(expr, vars_tuple[i], vars_tuple[j])
            val = eval_expr_iv(dexpr, env)
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
    return Fraction(1, 2) * (H[(0, 0)] + 2 * k * H[(0, 1)] + (k**2) * H[(1, 1)])



def B_component_K(H, kd, ke):
    return Fraction(1, 2) * (
        H[(0, 0)]
        + 2 * kd * H[(0, 1)]
        + 2 * ke * H[(0, 2)]
        + (kd**2) * H[(1, 1)]
        + 2 * kd * ke * H[(1, 2)]
        + (ke**2) * H[(2, 2)]
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


# The transverse x->y and x->(y,z) coefficients are omitted below because they
# are exactly zero for the exact dominant bases by the Groebner-basis proof
# above; the small interval widths in L[*][0] arise only from enclosing
# theta_A, theta_{B,*}, chi_B.

def A_sector_constants(L, Kx, Ky):
    c = L[0][0]
    a = L[0][1]
    t = L[1][1]
    x_lo = c.lo - a.abs_upper() * RHO - Kx * RADIUS
    x_hi = c.hi + a.abs_upper() * RHO + Kx * RADIUS
    tau_hi = (t.abs_upper() * RHO + Ky * RADIUS) / x_lo
    alpha = t.abs_upper() / x_lo
    beta = Ky / x_lo
    gamma = a.abs_upper() / cA_iv.lo
    delta = Kx / cA_iv.lo
    return {
        "x_lo": x_lo,
        "x_hi": x_hi,
        "tau_hi": tau_hi,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
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
    gamma = (ay.abs_upper() + az.abs_upper()) / cB_iv.lo
    delta = Kx / cB_iv.lo
    return {
        "x_lo": x_lo,
        "x_hi": x_hi,
        "tau_hi": tau_hi,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }


A0_sec = A_sector_constants(L0A_iv, Kx_A0, Ky_A0)
A1_sec = A_sector_constants(L1A_iv, Kx_A1, Ky_A1)
B0_sec = B_sector_constants(L0B_iv, Kx_B0, Ky_B0, Kz_B0)
B1_sec = B_sector_constants(L1B_iv, Kx_B1, Ky_B1, Kz_B1)

for data in [A0_sec, A1_sec, B0_sec, B1_sec]:
    assert data["x_lo"] > 0
    assert data["x_hi"] < 1
    assert data["tau_hi"] < RHO
    assert data["alpha"] < 1

mA = min(A0_sec["x_lo"], A1_sec["x_lo"])
MA = max(A0_sec["x_hi"], A1_sec["x_hi"])
mB = min(B0_sec["x_lo"], B1_sec["x_lo"])
MB = max(B0_sec["x_hi"], B1_sec["x_hi"])

# These two inequalities imply:
#   - every A-run has length exactly 1,
#   - every B-run has length at most 2.
assert MA / mB < 1
assert mA / (MB**2) > 1

# max relative x-errors used in the logarithmic asymptotics
errA_max = max(A0_sec["gamma"], A1_sec["gamma"]) * RHO + max(A0_sec["delta"], A1_sec["delta"]) * RADIUS
errB_max = max(B0_sec["gamma"], B1_sec["gamma"]) * RHO + max(B0_sec["delta"], B1_sec["delta"]) * RADIUS
assert errA_max < Fraction(1, 100)
assert errB_max < Fraction(1, 100)


# -----------------------------------------------------------------------------
# Local branch words on the sector boxes, and whole-burst locking
# -----------------------------------------------------------------------------

LA_list = [[int(LA[i, j]) for j in range(LA.cols)] for i in range(LA.rows)]
LB_list = [[int(LB[i, j]) for j in range(LB.cols)] for i in range(LB.rows)]


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

assert g_br_box > 0
assert later_surplus_box > 0
assert burst_lock_margin > 0


# -----------------------------------------------------------------------------
# Explicit rational starting point and exact row-duplication data
# -----------------------------------------------------------------------------

a_num = [137116847818847, 362883152181153, 242004611239991, 257995388760009]
b_num = [137116857552466, 362883142447534, 242004593130510, 257995386869490, 20000000]
DEN = 10**15

a0 = [sp.Rational(n, DEN) for n in a_num]
b0 = [sp.Rational(n, DEN) for n in b_num]
assert sum(a0) == 1
assert sum(b0) == 1
assert all(w > 0 for w in a0 + b0)

a0p = qiv(a0[0] + sp.Rational(1, 2))
a0d = qiv(a0[3])
b0p = qiv(b0[0] + sp.Rational(1, 2))
b0d = qiv(b0[3])
b0e = qiv(b0[4])

xA_iv = a0p - P_iv
resA_iv = (a0d - d_iv) / xA_iv - thetaA_iv

xB_iv = b0p - P_iv
resBd_iv = (b0d - d_iv) / xB_iv - thetaB0_iv
resBe_iv = b0e / xB_iv - chiB_iv

assert xA_iv.lo > 0 and xA_iv.hi < RADIUS
assert xB_iv.lo > 0 and xB_iv.hi < RADIUS
assert resA_iv.lo > -RHO and resA_iv.hi < RHO
assert resBd_iv.lo > -RHO and resBd_iv.hi < RHO
assert resBe_iv.lo > -RHO and resBe_iv.hi < RHO
assert xA_iv.hi < xB_iv.lo


# -----------------------------------------------------------------------------
# Explicit M_0 construction, margin witness, and row-duplication verification
# -----------------------------------------------------------------------------

# Construct M_0 = L_A ⊞ L_B as the 20×8 sign matrix defined in Section 1.
# Row (r,s) has A-block entries L_A[r,:] and B-block entries L_B[s,:].
M0 = sp.Matrix.zeros(20, 8)
for _r in range(4):
    for _s in range(5):
        for _j in range(4):
            M0[5 * _r + _s, _j] = LA[_r, _j]
            M0[5 * _r + _s, 4 + _j] = LB[_s, _j]

assert all(M0[i, j] in (-1, 1) for i in range(20) for j in range(8)), \
    "M_0 must be a {-1,+1}-valued matrix"

# --- Upper bound: gamma(M_0) <= 1/5 ---
# Row distribution W* = u_A* x u_B* on M_0.
# gamma(M_0) = min_D max_j (D^T M_0)_j, so exhibiting D = W* with
# max_j (W*^T M_0)_j = 1/5 proves gamma(M_0) <= 1/5.
uA_star = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 5), sp.Rational(1, 5)]
uB_star = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 5), sp.Rational(1, 5), sp.Rational(0)]
W_star = [uA_star[_r] * uB_star[_s] for _r in range(4) for _s in range(5)]
assert sum(W_star) == 1 and all(w >= 0 for w in W_star), \
    "W* must be a distribution on 20 rows"
M0_edges = [sum(W_star[i] * M0[i, j] for i in range(20)) for j in range(8)]
assert all(ej == sp.Rational(1, 5) for ej in M0_edges), \
    "Product witness must give all eight M_0 edges exactly 1/5"

# --- Lower bound: gamma(M_0) >= 1/5 ---
# Column distribution w* = (beta, beta) / 10 with beta = (2,1,1,1).
# By the minimax theorem gamma(M_0) = max_w min_i (M_0 w)_i, so exhibiting
# w = w* with min_i (M_0 w*)_i = 1/5 proves gamma(M_0) >= 1/5.
_beta = [2, 1, 1, 1]
w_star = [sp.Rational(b, 10) for b in _beta] + [sp.Rational(b, 10) for b in _beta]
assert sum(w_star) == 1 and all(w > 0 for w in w_star), \
    "w* must be a column distribution"
M0_col_values = [sum(M0[i, j] * w_star[j] for j in range(8)) for i in range(20)]
assert min(M0_col_values) == sp.Rational(1, 5), \
    "Column strategy must achieve min row value 1/5 for gamma(M_0) >= 1/5"

# Together: gamma(M_0) = 1/5.

# --- Row duplication: M_0 -> M_tilde ---
# M_tilde has a_num[r] * b_num[s] copies of row (r,s) of M_0 for a total of
# DEN^2 rows.  Uniform measure on M_tilde aggregates to W_0 = a_0 x b_0 on M_0,
# so AdaBoost on M_tilde from uniform start is equivalent to AdaBoost on M_0
# from W_0.  Row duplication preserves the margin, hence gamma(M_tilde) = 1/5.
_dup_total = sum(a_num[_r] * b_num[_s] for _r in range(4) for _s in range(5))
assert _dup_total == DEN ** 2, \
    "Row-duplication counts must total DEN^2"
for _r in range(4):
    for _s in range(5):
        assert sp.Rational(a_num[_r] * b_num[_s], _dup_total) == a0[_r] * b0[_s], \
            f"Uniform aggregation mismatch at row ({_r},{_s})"


# -----------------------------------------------------------------------------
# Core reporting
# -----------------------------------------------------------------------------

def core_report():
    print("AdaBoost counterexample core certificate (items 1-14)")
    print("=" * 72)
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
    print("  min 2-step B-run factor mA/MB^2 =", fmt_frac(mA / (MB**2)), "> 1  (so every B-run has length at most 2)")
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
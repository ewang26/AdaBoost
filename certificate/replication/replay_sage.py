"""Replay the current finite obligations using SageMath's symbolic backend.

Run: sage -python certificate/replication/replay_sage.py
Shares the current geometric/entry acceptance logic with verify.py; polynomial
derivation, differentiation, Groebner reduction and root counting use Sage.
No SymPy import, saved numeric result, or file outside this repository is used.
"""
from pathlib import Path
from fractions import Fraction as Q
import hashlib
import json
import sys
import time
import types

if not __debug__:
    raise RuntimeError("Run with assertions enabled")
sys.set_int_max_str_digits(0)
sys.dont_write_bytecode = True

from sage.all import QQ, PolynomialRing
from sage.all import *  # Required by the retained .sage algebra module.
from sage.repl.preparse import preparse
from sage.version import version as sage_version

started = time.time()
root = Path(__file__).resolve().parents[1]
source = (root / "sagemath/adaboost_cert_core.sage").read_text()
anchor = "lam_iv = qiv(lam_int[0], lam_int[1]).intersect(lam_from_char_iv)"
bridge = """
# Identify actual derivative roots BEFORE intersecting their isolators.
for _poly, _isolated, _actual in [
    (P_lambda_factor, lam_int, lam_from_char_iv),
    (P_lambda_factor, nu_int, nu_from_char_iv),
    (P_kappa_factor, kap_int, kap_eval_iv)]:
    _lo = min(qfrac(_isolated[0]), _actual.lo)
    _hi = max(qfrac(_isolated[1]), _actual.hi)
    _lq = QQ(_lo.numerator)/QQ(_lo.denominator)
    _hq = QQ(_hi.numerator)/QQ(_hi.denominator)
    assert _poly(_lq) != 0 and _poly(_hq) != 0
    assert count_roots_in(_poly, _lq, _hq) == 1
"""
assert source.count(anchor) == 1
repaired = source.replace(anchor, bridge + "\n" + anchor)
c = types.ModuleType("current_adaboost_sage_algebra")
c.__dict__.update(globals())
c.__name__ = "current_adaboost_sage_algebra"
sys.modules[c.__name__] = c
print("Deriving maps and algebraic data with SageMath", sage_version, flush=True)
exec(compile(preparse(repaired), "<sage-algebra-with-root-identification>", "exec"), c.__dict__)
assert 0 < c.nu_iv.lo <= c.nu_iv.hi < c.lam_iv.lo <= c.lam_iv.hi < c.kap_iv.lo <= c.kap_iv.hi < 1

# Explicit adapters between native backend interfaces; no numeric approximations.
a_sector = c.A_sector_constants
b_sector = c.B_sector_constants
def A_sector_constants(L, Kx, Ky, *, radius, rho):
    c.RADIUS, c.RHO = radius, rho
    return a_sector(L, Kx, Ky)
def B_sector_constants(L, Kx, Ky, Kz, *, radius, rho):
    c.RADIUS, c.RHO = radius, rho
    return b_sector(L, Kx, Ky, Kz)
c.A_sector_constants = A_sector_constants
c.B_sector_constants = B_sector_constants
c.Dexpr = c.Dexpr_sage

class RootPolynomial:
    def __init__(self, poly):
        self.poly = poly
    def count_roots(self, lo, hi):
        return c.count_roots_in(self.poly, QQ(lo), QQ(hi))
    def eval(self, value):
        return self.poly(QQ(value))
    def refine_root(self, lo, hi, *, eps):
        lo, hi = QQ(lo), QQ(hi)
        assert self.count_roots(lo, hi) == 1
        while hi - lo > eps:
            mid = (lo + hi) / 2
            assert self.poly(mid) != 0
            if self.count_roots(lo, mid) == 1:
                hi = mid
            else:
                lo = mid
        return lo, hi
c.f_poly = RootPolynomial(c.f_poly)

driver = (root / "verify.py").read_text()
# These shared sections perform only rational interval/finite-state operations.
local = "def mat(M):" + driver.split("def mat(M):", 1)[1].split(
    "# Direct half-burst multiplier polynomials and norm certificates.", 1)[0]
lo = local.index("# Check the outward enclosures printed in the solution.")
hi = local.index("# Row permutation", lo)
# Printed slope endpoints describe the SymPy isolators. Sage may have different,
# valid outward isolators, so it certifies the geometric tests with its own.
local = local[:lo] + "assert c.d_iv.lo > Q(1,4)\n\n" + local[hi:]
assert local.count("M.rows") == 2
local = local.replace("M.rows", "M.nrows()")
exec(compile(local, "<shared-current-cone-and-locking-tests>", "exec"), globals())

# Native Sage half-step polynomials, identities, isolators and norms.
ring = PolynomialRing(QQ, "z")
z = ring.gen()
qa = 4*z**14-268*z**13-5676*z**12-76559*z**11+61076*z**10+5403070*z**9+3714640*z**8-44707134*z**7+159688998*z**6-80404208*z**5+1806746*z**4+2355093*z**3+127106*z**2+1846*z+2
qb = z**7-9*z**6+23*z**5-3*z**4-45*z**3+13*z**2+5*z-1
pa = ring(c.P_lambda_manuscript.list())
pb = ring(c.P_kappa_manuscript.list())
assert pa(z*z) == qa*qa(-z)
assert pb(z*z) == -qb*qb(-z)
norms = {}
for name, poly, parent, interval, actual in [
    ("A",qa,pa,("0.30739560746557","0.30739560746558"),c.cA_iv),
    ("B",qb,pb,("0.45153264872461","0.45153264872462"),c.cB_iv)]:
    lo, hi = (QQ(Q(t).numerator)/QQ(Q(t).denominator) for t in interval)
    assert poly.is_irreducible()
    assert c.count_roots_in(poly, lo, hi) == 1
    assert c.count_roots_in(poly(-z), lo, hi) == 0
    assert Q(str(lo)) < actual.lo <= actual.hi < Q(str(hi))
    assert c.count_roots_in(parent, lo*lo, hi*hi) == 1
    norms[name] = str((-1)**poly.degree()*poly[0]/poly.leading_coefficient())
assert norms == {"A":"1/2", "B":"1"}

# Current 140-step full twenty-weight prefix and both margin witnesses.
s = types.SimpleNamespace(Rational=lambda a, b: QQ(a)/QQ(b))
entry = "# Full twenty-weight finite prefix;" + driver.split(
    "# Full twenty-weight finite prefix;", 1)[1].split("output={'passed':True", 1)[0]
exec(compile(entry, "<shared-current-entry-and-margin-tests>", "exec"), globals())

output = {
    "passed": True,
    "sage_version": sage_version,
    "elapsed_seconds": time.time()-started,
    "scope": "Current two normalized cones, half-step polynomials/norms, branch/locking/negation gaps, strict full20 140-step entry, 2500 rows, margin 1/5. Shared geometric and entry acceptance logic with verify.py; native Sage algebra.",
    "uniform_rows": 2500,
    "margin": "1/5",
    "strict_prefix_steps": 140,
    "prefix_gap_lower": str(min(prefix_gaps)),
    "selected_columns_zeroindexed": seq,
    "burst_word": "".join(bursts),
    "endpoint": endpoint,
    "norms": norms,
    "normalized_local_certificate": results,
    "repaired_sage_core_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
    "files_sha256": {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ["verify.py", "replication/replay_sage.py", "sagemath/adaboost_cert_core.sage"]
    },
}
(root/"replication/verification_sage.json").write_text(json.dumps(output, indent=2)+"\n")
print("PASS: current Sage algebra, normalized cones, 140 strict updates, 2,500 rows, margin 1/5", flush=True)

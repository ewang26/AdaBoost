"""Optional regression for the original four-sector, 10**30-row construction.

Run explicitly with python3 legacy_verify.py. These additional checks are
not imported or required by the normalized 2,500-row certificate.
"""

import sys

sys.dont_write_bytecode = True

from algebra import (
    A_component_K,
    A_sector_constants,
    B_component_K,
    B_sector_constants,
    F0A_d,
    F0A_p,
    F0B_d,
    F0B_e,
    F0B_p,
    F1A_d,
    F1A_p,
    F1B_d,
    F1B_e,
    F1B_p,
    Fraction,
    H0,
    H1,
    L0A_iv,
    L0B_iv,
    L1A_iv,
    L1B_iv,
    LA,
    LA_list,
    LB,
    LB_list,
    P_iv,
    build_state_A_from_box,
    build_state_B_from_box,
    cA0_iv,
    cA1_iv,
    cA_iv,
    cB0_iv,
    cB1_iv,
    cB_iv,
    certify_word_on_box,
    chiB_iv,
    d,
    d_iv,
    e,
    fmt_frac,
    fmt_iv,
    hessian_sup_matrix,
    kap_iv,
    lam_iv,
    mu_star_iv,
    nu_iv,
    p,
    qiv,
    sp,
    thetaA_iv,
    thetaB0_iv,
    thetaB1_iv,
)

RADIUS = Fraction(1, 40000000)   # 2.5e-8 exactly
RHO = Fraction(1, 1000)          # sector slope parameter 10^{-3}

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



A0_sec = A_sector_constants(L0A_iv, Kx_A0, Ky_A0, radius=RADIUS, rho=RHO)
A1_sec = A_sector_constants(L1A_iv, Kx_A1, Ky_A1, radius=RADIUS, rho=RHO)
B0_sec = B_sector_constants(L0B_iv, Kx_B0, Ky_B0, Kz_B0, radius=RADIUS, rho=RHO)
B1_sec = B_sector_constants(L1B_iv, Kx_B1, Ky_B1, Kz_B1, radius=RADIUS, rho=RHO)

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
    print("Optional legacy regression: four sectors and original initialization")
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
    print("All optional legacy assertions passed.")

if __name__ == "__main__":
    core_report()
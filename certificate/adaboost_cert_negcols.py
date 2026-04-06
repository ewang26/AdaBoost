from __future__ import annotations

import adaboost_cert_core as core
import sympy as sp

# -----------------------------------------------------------------------------
# Negated-column exclusion for the [M | -M] formulation (items 15-16)
# -----------------------------------------------------------------------------

# Frozen-gadget column edges on each sector box.  During an A-burst the B-state
# does not move, so its column edges are constant; we enclose them once over the
# full sector box.  Analogously for B-bursts with frozen A-state.

frozen_B_edges = {}
for _j in [0, 1]:
    _b_box = [core.B0_box, core.B1_box][_j]
    frozen_B_edges[_j] = core.edge_list_iv(core.build_state_B_from_box(_b_box), core.LB_list)

frozen_A_edges = {}
for _i in [0, 1]:
    _a_box = [core.A0_box, core.A1_box][_i]
    frozen_A_edges[_i] = core.edge_list_iv(core.build_state_A_from_box(_a_box), core.LA_list)


def certify_negated_A_burst(a_box, a_word, frozen_b_edges):
    """
    During an A-burst, at each step verify that the selected A-column edge
    strictly exceeds every negated column edge (both active and frozen).
    Returns the minimum certified gap.
    """
    state = core.build_state_A_from_box(a_box)
    min_gap = None
    for j in a_word:
        active_edges = core.edge_list_iv(state, core.LA_list)
        sel_lo = active_edges[j].lo
        max_neg_active = -min(e.lo for e in active_edges)
        max_neg_frozen = -min(e.lo for e in frozen_b_edges)
        gap = sel_lo - max(max_neg_active, max_neg_frozen)
        min_gap = gap if min_gap is None else min(min_gap, gap)
        state, _ = core.update_iv(state, core.LA_list, j)
    return min_gap


def certify_negated_B_burst(b_box, b_word, frozen_a_edges):
    """Analogous for B-bursts."""
    state = core.build_state_B_from_box(b_box)
    min_gap = None
    for j in b_word:
        active_edges = core.edge_list_iv(state, core.LB_list)
        sel_lo = active_edges[j].lo
        max_neg_active = -min(e.lo for e in active_edges)
        max_neg_frozen = -min(e.lo for e in frozen_a_edges)
        gap = sel_lo - max(max_neg_active, max_neg_frozen)
        min_gap = gap if min_gap is None else min(min_gap, gap)
        state, _ = core.update_iv(state, core.LB_list, j)
    return min_gap


neg_col_data = {}
for _i in [0, 1]:
    for _j in [0, 1]:
        neg_col_data[f"A{_i}_Bfrozen{_j}"] = certify_negated_A_burst(
            [core.A0_box, core.A1_box][_i], [core.H0, core.H1][_i], frozen_B_edges[_j]
        )
for _j in [0, 1]:
    for _i in [0, 1]:
        neg_col_data[f"B{_j}_Afrozen{_i}"] = certify_negated_B_burst(
            [core.B0_box, core.B1_box][_j], [core.H0, core.H1][_j], frozen_A_edges[_i]
        )

min_neg_col_gap = min(neg_col_data.values())
assert min_neg_col_gap > 0, f"Negated-column exclusion failed: min gap = {min_neg_col_gap}"

# Margin of [M_0 | -M_0].  The minimax distributions u_*^A = (1/5,2/5,1/5,1/5)
# on L_A and u_*^B = (1/5,2/5,1/5,1/5,0) on L_B each give all four column
# edges equal to exactly 1/5.  Hence every negated column edge is exactly -1/5,
# and gamma([M_0 | -M_0]) = gamma(M_0) = 1/5.
uA_star = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 5), sp.Rational(1, 5)]
uB_star = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 5), sp.Rational(1, 5), sp.Rational(0)]
assert sum(uA_star) == 1 and all(u >= 0 for u in uA_star), "u_A* must be a distribution"
assert sum(uB_star) == 1 and all(u >= 0 for u in uB_star), "u_B* must be a distribution"
eA_star = [sum(uA_star[i] * core.LA[i, j] for i in range(4)) for j in range(4)]
eB_star = [sum(uB_star[i] * core.LB[i, j] for i in range(5)) for j in range(4)]
assert all(e == sp.Rational(1, 5) for e in eA_star), "L_A minimax edges must all be 1/5"
assert all(e == sp.Rational(1, 5) for e in eB_star), "L_B minimax edges must all be 1/5"


def neg_report():
    print("AdaBoost counterexample negated-column certificate (items 15-16)")
    print("=" * 72)
    print("Negated-column exclusion for [M | -M] formulation:")
    for label in sorted(neg_col_data):
        print(f"  {label}: min gap = {core.fmt_frac(neg_col_data[label])}")
    print(f"  overall minimum negated-column gap = {core.fmt_frac(min_neg_col_gap)}")
    print("  Margin witness: u_A* edges on L_A =", [str(e) for e in eA_star])
    print("  Margin witness: u_B* edges on L_B =", [str(e) for e in eB_star])
    print("  gamma([M_0 | -M_0]) = gamma(M_0) = 1/5, hence by row duplication")
    print(r"  gamma([\widetilde M | -\widetilde M]) = gamma(\widetilde M) = 1/5 as well.")
    print()
    print("All negated-column and augmented-margin assertions passed.")

if __name__ == "__main__":
    neg_report()
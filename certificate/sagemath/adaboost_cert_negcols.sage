# AdaBoost counterexample negated-column certificate — SageMath version (items 15-16)
# Run with: sage adaboost_cert_negcols.sage

load("adaboost_cert_core.sage")

# -----------------------------------------------------------------------------
# Negated-column exclusion for the [M | -M] formulation (items 15-16)
# -----------------------------------------------------------------------------

frozen_B_edges = {}
for _j in [0, 1]:
    _b_box = [B0_box, B1_box][_j]
    frozen_B_edges[_j] = edge_list_iv(build_state_B_from_box(_b_box), LB_list)

frozen_A_edges = {}
for _i in [0, 1]:
    _a_box = [A0_box, A1_box][_i]
    frozen_A_edges[_i] = edge_list_iv(build_state_A_from_box(_a_box), LA_list)


def certify_negated_A_burst(a_box, a_word, frozen_b_edges):
    state = build_state_A_from_box(a_box)
    min_gap = None
    for j in a_word:
        active_edges = edge_list_iv(state, LA_list)
        sel_lo = active_edges[j].lo
        max_neg_active = -min(e_edge.lo for e_edge in active_edges)
        max_neg_frozen = -min(e_edge.lo for e_edge in frozen_b_edges)
        gap = sel_lo - max(max_neg_active, max_neg_frozen)
        min_gap = gap if min_gap is None else min(min_gap, gap)
        state, _ = update_iv(state, LA_list, j)
    return min_gap


def certify_negated_B_burst(b_box, b_word, frozen_a_edges):
    state = build_state_B_from_box(b_box)
    min_gap = None
    for j in b_word:
        active_edges = edge_list_iv(state, LB_list)
        sel_lo = active_edges[j].lo
        max_neg_active = -min(e_edge.lo for e_edge in active_edges)
        max_neg_frozen = -min(e_edge.lo for e_edge in frozen_a_edges)
        gap = sel_lo - max(max_neg_active, max_neg_frozen)
        min_gap = gap if min_gap is None else min(min_gap, gap)
        state, _ = update_iv(state, LB_list, j)
    return min_gap


neg_col_data = {}
for _i in [0, 1]:
    for _j in [0, 1]:
        neg_col_data[f"A{_i}_Bfrozen{_j}"] = certify_negated_A_burst(
            [A0_box, A1_box][_i], [H0, H1][_i], frozen_B_edges[_j]
        )
for _j in [0, 1]:
    for _i in [0, 1]:
        neg_col_data[f"B{_j}_Afrozen{_i}"] = certify_negated_B_burst(
            [B0_box, B1_box][_j], [H0, H1][_j], frozen_A_edges[_i]
        )

min_neg_col_gap = min(neg_col_data.values())
assert min_neg_col_gap > PyFraction(int(0)), f"Negated-column exclusion failed: min gap = {min_neg_col_gap}"

# Margin of [M_0 | -M_0]
uA_star_neg = [QQ(1)/5, QQ(2)/5, QQ(1)/5, QQ(1)/5]
uB_star_neg = [QQ(1)/5, QQ(2)/5, QQ(1)/5, QQ(1)/5, QQ(0)]
assert sum(uA_star_neg) == 1 and all(u >= 0 for u in uA_star_neg)
assert sum(uB_star_neg) == 1 and all(u >= 0 for u in uB_star_neg)
eA_star = [sum(uA_star_neg[i] * LA[i, j] for i in range(4)) for j in range(4)]
eB_star = [sum(uB_star_neg[i] * LB[i, j] for i in range(5)) for j in range(4)]
assert all(e_val == QQ(1)/5 for e_val in eA_star), "L_A minimax edges must all be 1/5"
assert all(e_val == QQ(1)/5 for e_val in eB_star), "L_B minimax edges must all be 1/5"


def neg_report():
    print("AdaBoost counterexample negated-column certificate — SageMath version (items 15-16)")
    print("=" * 72)
    print("Negated-column exclusion for [M | -M] formulation:")
    for label in sorted(neg_col_data):
        print(f"  {label}: min gap = {fmt_frac(neg_col_data[label])}")
    print(f"  overall minimum negated-column gap = {fmt_frac(min_neg_col_gap)}")
    print("  Margin witness: u_A* edges on L_A =", [str(e_val) for e_val in eA_star])
    print("  Margin witness: u_B* edges on L_B =", [str(e_val) for e_val in eB_star])
    print("  gamma([M_0 | -M_0]) = gamma(M_0) = 1/5, hence by row duplication")
    print(r"  gamma([\widetilde M | -\widetilde M]) = gamma(\widetilde M) = 1/5 as well.")
    print()
    print("All negated-column and augmented-margin assertions passed.")

if __name__ == "__main__":
    neg_report()

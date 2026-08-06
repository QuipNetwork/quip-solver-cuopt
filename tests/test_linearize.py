"""Linearization unit tests for J > 0, J < 0, J == 0 and a worked 3-node case."""

from quip_miner_cuopt.linearize import (
    brute_force_ising_ground,
    brute_force_milp_optimum,
    ising_energy,
    linearize,
    milp_objective,
    optimal_y_for_x,
    x_to_spins,
)


def test_j_positive_emits_one_fortet_lower_cut():
    m = linearize([0.0, 0.0], [1.0], [(0, 1)])
    assert len(m.y_edges) == 1
    assert m.num_constraints() == 1
    assert m.lower_bounds[0].kind == "finite"
    assert m.lower_bounds[0].value == -1.0
    assert m.upper_bounds[0].kind == "inf"
    # obj on y = 4 J = 4
    assert abs(m.obj[2] - 4.0) < 1e-12
    assert m.variable_types() == ["I", "I", "C"]


def test_j_negative_emits_two_upper_cuts():
    m = linearize([0.0, 0.0], [-1.0], [(0, 1)])
    assert len(m.y_edges) == 1
    assert m.num_constraints() == 2
    assert all(b.kind == "ninf" for b in m.lower_bounds)
    assert all(b.kind == "finite" and b.value == 0.0 for b in m.upper_bounds)
    # obj on y = 4 J = −4
    assert abs(m.obj[2] - (-4.0)) < 1e-12
    assert m.variable_types() == ["I", "I", "C"]


def test_j_zero_emits_no_y():
    m = linearize([0.5, -0.5], [0.0], [(0, 1)])
    assert m.y_edges == []
    assert m.num_constraints() == 0
    assert m.num_vars() == 2
    assert m.variable_types() == ["I", "I"]
    # constant = −h0 − h1 + J = −0.5 + 0.5 + 0 = 0
    assert abs(m.constant) < 1e-12


def test_worked_three_node_ferro():
    # h = [0,0,0], edges (0,1)=−1, (1,2)=−1, (0,2)=−1 (all ferro)
    h = [0.0, 0.0, 0.0]
    j = [-1.0, -1.0, -1.0]
    edges = [(0, 1), (1, 2), (0, 2)]
    m = linearize(h, j, edges)
    assert m.n == 3
    assert len(m.y_edges) == 3
    # Each negative J → 2 constraints → 6 total
    assert m.num_constraints() == 6
    # constant = 0 + (−1)*3 = −3
    assert abs(m.constant - (-3.0)) < 1e-12
    # ground state all-aligned energy = 3 * (−1) = −3
    _spins, e = brute_force_ising_ground(h, j, edges)
    assert abs(e - (-3.0)) < 1e-12
    x, milp_e = brute_force_milp_optimum(m)
    assert abs(milp_e - e) < 1e-9, f"milp={milp_e} ising={e}"
    spins = x_to_spins(x)
    scored = ising_energy(spins, h, j, edges)
    assert abs(scored - e) < 1e-9


def test_x_to_spins_threshold():
    assert x_to_spins([0.0, 0.49, 0.5, 1.0]) == [-1, -1, 1, 1]


def test_objective_cross_check_at_optimum():
    """MILP objective (with constant) equals spin energy at the optimum."""
    h = [0.5, -0.5, 0.0]
    j = [1.0, -2.0]
    edges = [(0, 1), (1, 2)]
    m = linearize(h, j, edges)
    x, milp_e = brute_force_milp_optimum(m)
    spins = x_to_spins(x)
    ising_e = ising_energy(spins, h, j, edges)
    y = optimal_y_for_x(m, x)
    obj = milp_objective(m, x, y)
    assert abs(milp_e - ising_e) < 1e-9
    assert abs(obj - ising_e) < 1e-9

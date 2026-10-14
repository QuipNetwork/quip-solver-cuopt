"""Brute-force parity: MILP optimum energy equals true Ising ground state."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import pytest

from quip_solver_cuopt.linearize import (
    brute_force_ising_ground,
    brute_force_milp_optimum,
    ising_energy,
    ising_energy_milli,
    linearize,
    milp_objective,
    optimal_y_for_x,
    x_to_spins,
)


def _assert_parity(
    h: Sequence[float],
    j: Sequence[float],
    edges: Sequence[Tuple[int, int]],
) -> None:
    model = linearize(h, j, edges)
    _gs, ground_e = brute_force_ising_ground(h, j, edges)
    x, milp_e = brute_force_milp_optimum(model)
    spins = x_to_spins(x)
    ising_e = ising_energy(spins, h, j, edges)
    y = optimal_y_for_x(model, x)
    obj = milp_objective(model, x, y)

    assert abs(milp_e - ground_e) < 1e-6, (
        f"MILP obj {milp_e} != ground {ground_e}; h={h} j={j} edges={edges} x={x}"
    )
    assert abs(ising_e - ground_e) < 1e-6, (
        f"decoded ising {ising_e} != ground {ground_e}"
    )
    # Objective cross-check: MILP objective (incl. constant) == spin energy.
    assert abs(obj - ising_e) < 1e-6, f"obj cross-check {obj} != ising {ising_e}"

    # Integer milli path agrees within 1 milli of float*1000 trunc.
    h_milli = [int(round(v * 1000)) for v in h]
    j_milli = [int(round(v * 1000)) for v in j]
    milli = ising_energy_milli(spins, h_milli, j_milli, edges)
    ground_milli = int(ground_e * 1000)  # toward-zero for positive; see note
    # Use round-trip from integer coeffs for a tight check.
    ground_from_milli = ising_energy_milli(_gs, h_milli, j_milli, edges)
    assert milli == ising_energy_milli(spins, h_milli, j_milli, edges)
    assert (
        abs(milli - ground_from_milli)
        <= abs(
            ising_energy_milli(_gs, h_milli, j_milli, edges)
            - ising_energy_milli(spins, h_milli, j_milli, edges)
        )
        or milli == ground_from_milli
    )
    # Spins from MILP must achieve ground energy under milli coeffs.
    assert milli == ground_from_milli, (
        f"milli {milli} vs ground_from_milli {ground_from_milli} "
        f"(float ground {ground_milli})"
    )


def _random_ising(
    rng: random.Random,
    n: int,
    j_sign: str,
) -> Tuple[List[float], List[float], List[Tuple[int, int]]]:
    coeffs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    h = [rng.choice(coeffs) for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    j: List[float] = []
    for i in range(n):
        for jj in range(i + 1, n):
            if rng.random() < 0.5:
                continue
            c = rng.choice(coeffs)
            if j_sign == "pos":
                c = abs(c) if c != 0 else 1.0
            elif j_sign == "neg":
                c = -abs(c) if c != 0 else -1.0
            edges.append((i, jj))
            j.append(c)
    if not edges and n >= 2:
        edges.append((0, 1))
        j.append(1.0 if j_sign != "neg" else -1.0)
    # Ensure all-pos / all-neg when required (drop zeros).
    if j_sign == "pos":
        j = [c if c > 0 else 1.0 for c in j]
    elif j_sign == "neg":
        j = [c if c < 0 else -1.0 for c in j]
    return h, j, edges


@pytest.mark.parametrize("seed", range(24))
def test_milp_matches_ising_mixed_j(seed: int):
    rng = random.Random(seed + 1000)
    n = rng.randint(5, 9)
    h, j, edges = _random_ising(rng, n, "mixed")
    _assert_parity(h, j, edges)


@pytest.mark.parametrize("seed", range(16))
def test_milp_matches_ising_all_positive_j(seed: int):
    rng = random.Random(seed + 2000)
    n = rng.randint(5, 9)
    h, j, edges = _random_ising(rng, n, "pos")
    assert all(c > 0 for c in j), j
    _assert_parity(h, j, edges)


@pytest.mark.parametrize("seed", range(16))
def test_milp_matches_ising_all_negative_j(seed: int):
    rng = random.Random(seed + 3000)
    n = rng.randint(5, 9)
    h, j, edges = _random_ising(rng, n, "neg")
    assert all(c < 0 for c in j), j
    _assert_parity(h, j, edges)


def test_fixed_all_positive_case():
    h = [0.5, -0.5, 0.0, 1.0, 0.0]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    j = [1.0, 2.0, 1.0, 1.0, 2.0]
    _assert_parity(h, j, edges)


def test_fixed_all_negative_case():
    h = [0.0, 0.0, 0.0, 0.0, 0.0]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    j = [-1.0, -1.0, -2.0, -1.0, -1.0]
    _assert_parity(h, j, edges)


def test_objective_cross_check_explicit():
    """Verbatim-friendly: MILP obj + constant equals spin energy."""
    h = [1.0, -1.0, 0.5]
    j = [0.5, -1.5, 1.0]
    edges = [(0, 1), (1, 2), (0, 2)]
    model = linearize(h, j, edges)
    x, milp_e = brute_force_milp_optimum(model)
    spins = x_to_spins(x)
    ising_e = ising_energy(spins, h, j, edges)
    y = optimal_y_for_x(model, x)
    obj = milp_objective(model, x, y)
    assert abs(milp_e - ising_e) < 1e-9
    assert abs(obj - ising_e) < 1e-9
    _gs, ground = brute_force_ising_ground(h, j, edges)
    assert abs(ising_e - ground) < 1e-9

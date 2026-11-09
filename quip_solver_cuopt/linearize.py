"""Sign-aware Ising → MILP linearization.

Substitute ``s_i = 2 x_i - 1`` with ``x_i ∈ {0,1}``, introduce continuous
``y_ij ≈ x_i x_j`` only for nonzero couplings, and emit the Fortet inequalities
required by the objective sign of each ``y_ij``.

Only ``x_i`` are integer variables. Every ``y_ij`` is continuous in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class YEdge:
    """One nonzero-coupling edge that owns a continuous ``y`` variable."""

    i: int
    j: int
    j_coup: float
    y_col: int


@dataclass
class Bound:
    """Finite or infinite constraint bound (cuOpt: number or ``inf`` / ``ninf``)."""

    kind: str  # "finite" | "inf" | "ninf"
    value: float = 0.0

    @staticmethod
    def finite(v: float) -> "Bound":
        return Bound("finite", v)

    @staticmethod
    def pos_inf() -> "Bound":
        return Bound("inf")

    @staticmethod
    def neg_inf() -> "Bound":
        return Bound("ninf")

    def to_json(self):
        if self.kind == "finite":
            return self.value
        if self.kind == "inf":
            return "inf"
        return "ninf"


@dataclass
class MilpModel:
    """Mixed-integer linear model of an Ising problem.

    Variable layout: ``[x_0 .. x_{n-1}, y_0 .. y_{m'-1}]`` where ``m'`` is the
    number of nonzero couplings. Only the ``x`` columns are integer.
    """

    n: int
    obj: List[float]
    constant: float
    y_edges: List[YEdge] = field(default_factory=list)
    csr_offsets: List[int] = field(default_factory=list)
    csr_indices: List[int] = field(default_factory=list)
    csr_values: List[float] = field(default_factory=list)
    lower_bounds: List[Bound] = field(default_factory=list)
    upper_bounds: List[Bound] = field(default_factory=list)

    def num_vars(self) -> int:
        return self.n + len(self.y_edges)

    def num_constraints(self) -> int:
        return max(0, len(self.csr_offsets) - 1)

    def variable_types(self) -> List[str]:
        return ["I"] * self.n + ["C"] * len(self.y_edges)

    def var_lower_bounds(self) -> List[float]:
        return [0.0] * self.num_vars()

    def var_upper_bounds(self) -> List[float]:
        return [1.0] * self.num_vars()


def linearize(
    h: Sequence[float],
    j: Sequence[float],
    edges: Sequence[Tuple[int, int]],
) -> MilpModel:
    """Build the sign-aware MILP for an Ising instance.

    Objective (minimize)::

        E = Σ_i 2 h_i x_i
          + Σ_{J≠0} (4 J y_ij − 2 J x_i − 2 J x_j)
          + constant
        constant = Σ_i (−h_i) + Σ_all J_ij

    Constraints on each continuous ``y_ij`` (never all three Fortet cuts):

    - ``J > 0``: ``y ≥ x_i + x_j − 1``  (CSR: ``y − x_i − x_j ≥ −1``)
    - ``J < 0``: ``y ≤ x_i`` and ``y ≤ x_j``
    - ``J == 0``: no ``y`` variable
    """
    n = len(h)
    obj = [0.0] * n
    constant = 0.0

    for i, hi in enumerate(h):
        # h_i * (2 x_i - 1) → 2 h_i x_i − h_i
        obj[i] += 2.0 * float(hi)
        constant -= float(hi)

    y_edges: List[YEdge] = []
    rows: List[List[Tuple[int, float]]] = []
    lower_bounds: List[Bound] = []
    upper_bounds: List[Bound] = []

    for (i, j_idx), j_coup in zip(edges, j):
        i = int(i)
        j_idx = int(j_idx)
        j_coup = float(j_coup)
        if i >= n or j_idx >= n:
            continue
        # Constant always includes every J (including self-loops / zeros).
        constant += j_coup
        if i == j_idx:
            # s_i^2 = 1; no bilinear term to linearize.
            continue
        if j_coup == 0.0:
            continue

        y_col = n + len(y_edges)
        # −2 J x_i − 2 J x_j + 4 J y
        obj[i] += -2.0 * j_coup
        obj[j_idx] += -2.0 * j_coup
        obj.append(4.0 * j_coup)

        if j_coup > 0.0:
            # y − x_i − x_j ≥ −1
            rows.append([(i, -1.0), (j_idx, -1.0), (y_col, 1.0)])
            lower_bounds.append(Bound.finite(-1.0))
            upper_bounds.append(Bound.pos_inf())
        else:
            # y − x_i ≤ 0
            rows.append([(i, -1.0), (y_col, 1.0)])
            lower_bounds.append(Bound.neg_inf())
            upper_bounds.append(Bound.finite(0.0))
            # y − x_j ≤ 0
            rows.append([(j_idx, -1.0), (y_col, 1.0)])
            lower_bounds.append(Bound.neg_inf())
            upper_bounds.append(Bound.finite(0.0))

        y_edges.append(YEdge(i=i, j=j_idx, j_coup=j_coup, y_col=y_col))

    csr_offsets = [0]
    csr_indices: List[int] = []
    csr_values: List[float] = []
    for row in rows:
        for col, val in row:
            csr_indices.append(col)
            csr_values.append(val)
        csr_offsets.append(len(csr_indices))

    return MilpModel(
        n=n,
        obj=obj,
        constant=constant,
        y_edges=y_edges,
        csr_offsets=csr_offsets,
        csr_indices=csr_indices,
        csr_values=csr_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def x_to_spins(x: Sequence[float]) -> List[int]:
    """Decode binary (or near-binary) ``x`` to spins ``s = 2x − 1`` in ``{-1,+1}``."""
    return [1 if float(v) >= 0.5 else -1 for v in x]


def milp_objective(model: MilpModel, x: Sequence[float], y: Sequence[float]) -> float:
    """Evaluate the MILP objective (including constant) at a fully specified point."""
    e = model.constant
    for i, xi in enumerate(x[: model.n]):
        e += model.obj[i] * float(xi)
    for k, ye in enumerate(model.y_edges):
        yk = float(y[k]) if k < len(y) else 0.0
        e += model.obj[ye.y_col] * yk
    return e


def optimal_y_for_x(model: MilpModel, x: Sequence[float]) -> List[float]:
    """Optimal continuous ``y`` for fixed binary ``x`` under sign-aware constraints."""
    out: List[float] = []
    for ye in model.y_edges:
        xi = float(x[ye.i]) if ye.i < len(x) else 0.0
        xj = float(x[ye.j]) if ye.j < len(x) else 0.0
        if ye.j_coup > 0.0:
            # push y down: y = max(0, x_i + x_j − 1)
            out.append(max(0.0, min(1.0, xi + xj - 1.0)))
        else:
            # push y up: y = min(x_i, x_j)
            out.append(max(0.0, min(1.0, min(xi, xj))))
    return out


def ising_energy(
    spins: Sequence[int],
    h: Sequence[float],
    j: Sequence[float],
    edges: Sequence[Tuple[int, int]],
) -> float:
    """True Ising energy for spins in ``{-1,+1}`` (float arithmetic)."""
    e = 0.0
    for i, s in enumerate(spins):
        if i < len(h):
            e += float(h[i]) * float(s)
    for (u, v), jv in zip(edges, j):
        su = float(spins[u]) if u < len(spins) else 0.0
        sv = float(spins[v]) if v < len(spins) else 0.0
        e += float(jv) * su * sv
    return e


def ising_energy_milli(
    spins: Sequence[int],
    h_milli: Sequence[int],
    j_milli: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> int:
    """Ising energy in integer milli units from decoded spins.

    ``E_milli = Σ h_milli_i s_i + Σ j_milli_ij s_i s_j``. This is the reported
    energy path: it never goes through the MILP objective or the basis change.
    """
    e = 0
    for i, s in enumerate(spins):
        if i < len(h_milli):
            e += int(h_milli[i]) * int(s)
    for (u, v), jm in zip(edges, j_milli):
        su = int(spins[u]) if u < len(spins) else 0
        sv = int(spins[v]) if v < len(spins) else 0
        e += int(jm) * su * sv
    return e


def brute_force_ising_ground(
    h: Sequence[float],
    j: Sequence[float],
    edges: Sequence[Tuple[int, int]],
) -> Tuple[List[int], float]:
    """Brute-force true ground-state energy over all ``2^n`` spin assignments."""
    n = len(h)
    if n > 20:
        raise ValueError(f"brute_force_ising_ground: n={n} exceeds 20")
    if n == 0:
        return [], 0.0
    best_spins = [1] * n
    best_e = float("inf")
    for mask in range(1 << n):
        spins = [-1] * n
        for i in range(n):
            if (mask >> i) & 1:
                spins[i] = 1
        e = ising_energy(spins, h, j, edges)
        if e < best_e:
            best_e = e
            best_spins = spins
    return best_spins, best_e


def brute_force_milp_optimum(model: MilpModel) -> Tuple[List[float], float]:
    """Enumerate all ``2^n`` binary assignments; return minimizing ``x`` and obj."""
    if model.n > 20:
        raise ValueError(f"brute_force_milp_optimum: n={model.n} exceeds 20")
    if model.n == 0:
        return [], model.constant
    best_x = [0.0] * model.n
    best_e = float("inf")
    for mask in range(1 << model.n):
        x = [0.0] * model.n
        for i in range(model.n):
            if (mask >> i) & 1:
                x[i] = 1.0
        y = optimal_y_for_x(model, x)
        e = milp_objective(model, x, y)
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, best_e

"""Shared problem formulations for Triangle Vertex Deletion (TVD).

One set of functions parameterized by (graph, weights, triangles); weighted and
unweighted TVD are the same code path, differing only in the weight vector
(weights defaults to all-ones).

Graph vertices are assumed to be labeled 0..n-1 contiguously (this is what
``instances.py`` produces after pruning) so that a vertex label doubles as its
qubit index.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

try:
    import dimod
except ImportError:  # pragma: no cover
    dimod = None

logger = logging.getLogger(__name__)

Triangle = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Triangle enumeration & graph-level pruning (spec §2, §4)
# ---------------------------------------------------------------------------


def enumerate_triangles(graph: nx.Graph) -> list[Triangle]:
    """List every triangle in ``graph`` as a sorted (u, v, w) tuple, u<v<w.

    Combinatorial listing: for every edge (u, v), any common neighbor w of
    both endpoints closes a triangle. Each triangle is discovered up to 3
    times (once per edge) and deduplicated via a set.
    """
    triangles: set[Triangle] = set()
    for u, v in graph.edges():
        common = set(graph.neighbors(u)) & set(graph.neighbors(v))
        for w in common:
            triangles.add(tuple(sorted((u, v, w))))
    return sorted(triangles)


def prune_zero_triangle_vertices(
    graph: nx.Graph, triangles: list[Triangle] | None = None
) -> tuple[nx.Graph, list[int]]:
    """Drop vertices that belong to zero triangles (spec §2, second rule).

    Such a vertex can never appear in the objective or in any hitting-set
    constraint, so it is unconstrained and forced to 0 in every optimum --
    removing it cannot change triangle membership among the remaining
    vertices, so a single pass suffices (no fixed-point iteration needed).

    Returns (pruned_graph, removed_vertices). Logs a warning if anything was
    removed, since instances generated per §3 are expected to already satisfy
    "every vertex is in >=1 triangle".
    """
    if triangles is None:
        triangles = enumerate_triangles(graph)
    in_triangle: set[int] = set()
    for t in triangles:
        in_triangle.update(t)
    removed = [v for v in graph.nodes() if v not in in_triangle]
    if not removed:
        return graph, []
    logger.warning(
        "prune_zero_triangle_vertices removed %d vertex/vertices with no "
        "triangle membership: %s",
        len(removed),
        removed,
    )
    pruned = graph.copy()
    pruned.remove_nodes_from(removed)
    return pruned, removed


def default_penalty(weights: dict[int, float]) -> float:
    """A penalty large enough that no infeasible assignment ever beats the
    best feasible one: strictly greater than the worst-case feasible
    objective (select every vertex)."""
    return float(sum(weights.values())) + 1.0


# ---------------------------------------------------------------------------
# PUBO formulation -- QAOA-facing, cubic, no ancillas (spec §2, §4)
# ---------------------------------------------------------------------------


def build_pubo(
    graph: nx.Graph,
    weights: dict[int, float],
    triangles: list[Triangle],
    penalty: float,
) -> dict[tuple[int, ...], float]:
    """Sparse PUBO cost dict: monomial (sorted qubit-index tuple) -> coeff.

    cost(x) = sum_v w_v x_v + penalty * sum_{(u,v,w) in triangles} (1-x_u)(1-x_v)(1-x_w)

    The second term is the exact penalty for violating the hitting-set
    constraint x_u + x_v + x_w >= 1 (it is 0 iff at least one of u, v, w is
    selected, and `penalty` otherwise). Expanding it produces the cubic
    monomial x_u x_v x_w directly -- no ancilla qubits.
    """
    pubo: dict[tuple[int, ...], float] = defaultdict(float)
    for v in graph.nodes():
        pubo[(v,)] += weights[v]
    for u, v, w in triangles:
        u, v, w = sorted((u, v, w))
        pubo[()] += penalty
        pubo[(u,)] -= penalty
        pubo[(v,)] -= penalty
        pubo[(w,)] -= penalty
        pubo[(u, v)] += penalty
        pubo[(u, w)] += penalty
        pubo[(v, w)] += penalty
        pubo[(u, v, w)] -= penalty
    return {k: c for k, c in pubo.items() if c != 0.0}


def evaluate_pubo(pubo: dict[tuple[int, ...], float], assignment: dict[int, int]) -> float:
    """Evaluate a PUBO dict at a 0/1 assignment (dict: vertex -> 0/1)."""
    total = 0.0
    for monomial, coeff in pubo.items():
        term = coeff
        for v in monomial:
            term *= assignment[v]
            if term == 0.0:
                break
        total += term
    return total


# ---------------------------------------------------------------------------
# QUBO formulation -- Rosenberg quadratization with ancilla reuse (spec §2, §4)
# ---------------------------------------------------------------------------


def _quadratize_pubo(
    pubo: dict[tuple[int, ...], float], n_vertices: int, ancilla_penalty: float
) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, int], int]]:
    """Reduce a degree-<=3 PUBO to degree-<=2 via Rosenberg substitution.

    For each distinct cubic monomial (u, v, w), introduce (or reuse) an
    ancilla y_uv = x_u * x_v for the pair (u, v) -- the two smallest indices
    of the sorted triple, so triangles sharing an edge share an ancilla --
    and substitute x_u x_v x_w -> y_uv * x_w. The substitution is enforced
    with the standard penalty:

        strength * (x_u x_v - 2 x_u y - 2 x_v y + 3 y)

    which is minimized (== 0) exactly when y == x_u * x_v. `strength` must
    exceed the total "reward" an incorrect y could offer across every cubic
    term that shares it (each has |coefficient| == penalty for our TVD PUBO),
    so it is `ancilla_penalty + sum(|coeff| for the terms using that pair)`,
    not a flat constant -- a pair reused by several triangles needs a
    proportionally larger enforcement penalty.
    """
    ancilla_map: dict[tuple[int, int], int] = {}
    next_ancilla = n_vertices
    quad: dict[tuple[int, ...], float] = defaultdict(float)
    cubic_by_pair: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)

    for monomial, coeff in pubo.items():
        if len(monomial) <= 2:
            quad[monomial] += coeff
            continue
        if len(monomial) != 3:
            raise ValueError(f"build_qubo only supports degree <= 3 PUBOs, got {monomial}")
        u, v, w = monomial
        cubic_by_pair[(u, v)].append((w, coeff))

    for pair, terms in cubic_by_pair.items():
        if pair not in ancilla_map:
            ancilla_map[pair] = next_ancilla
            next_ancilla += 1
        y = ancilla_map[pair]
        for w, coeff in terms:
            quad[(y, w) if y < w else (w, y)] += coeff

    for pair, y in ancilla_map.items():
        u, v = pair
        strength = ancilla_penalty + sum(abs(c) for _, c in cubic_by_pair[pair])
        quad[(u, v)] += strength
        quad[(u, y)] += -2 * strength
        quad[(v, y)] += -2 * strength
        quad[(y,)] += 3 * strength

    return {k: c for k, c in quad.items() if c != 0.0}, ancilla_map


def build_qubo(
    graph: nx.Graph,
    weights: dict[int, float],
    triangles: list[Triangle],
    penalty: float,
    ancilla_penalty: float | None = None,
):
    """Quadratic QUBO via Rosenberg quadratization, for the classical exact
    QUBO solver only (§2: QAOA never sees this -- it uses build_pubo directly).

    Returns (bqm, ancilla_map):
      - bqm: a dimod.BinaryQuadraticModel over integer variables
             0..n_vertices-1 (original vertices) and n_vertices.. (ancillas).
      - ancilla_map: dict (u, v) -> ancilla qubit index, so a solution can be
        decoded back to the original vertex variables (just drop the ancilla
        entries; the penalty guarantees they equal x_u * x_v at the optimum).
    """
    if dimod is None:  # pragma: no cover
        raise ImportError("dimod is required for build_qubo / the exact QUBO solver")

    n_vertices = graph.number_of_nodes()
    if ancilla_penalty is None:
        ancilla_penalty = penalty

    pubo = build_pubo(graph, weights, triangles, penalty)
    quad, ancilla_map = _quadratize_pubo(pubo, n_vertices, ancilla_penalty)

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for monomial, coeff in quad.items():
        if len(monomial) == 0:
            bqm.offset += coeff
        elif len(monomial) == 1:
            bqm.add_linear(monomial[0], coeff)
        else:
            i, j = monomial
            bqm.add_quadratic(i, j, coeff)

    # Constant term from the original PUBO (already folded into quad via
    # the () key if present).
    if () in pubo and () not in quad:
        bqm.offset += pubo[()]

    return bqm, ancilla_map


# ---------------------------------------------------------------------------
# ILP formulation -- native linear constraint, no product tricks (spec §2, §5)
# ---------------------------------------------------------------------------


@dataclass
class ILPFormulation:
    """Backend-agnostic description of the TVD ILP.

    min sum_v w_v x_v  s.t.  x_u + x_v + x_w >= 1 for every triangle (u,v,w),
    x_v in {0,1}.

    classical_solvers.py builds the actual gurobipy / docplex / pulp model
    from this (the formulation itself needs no per-backend logic since it has
    no binary-product tricks to encode).
    """

    vertices: list[int]
    weights: dict[int, float]
    triangles: list[Triangle] = field(default_factory=list)


def build_ilp(graph: nx.Graph, weights: dict[int, float], triangles: list[Triangle]) -> ILPFormulation:
    return ILPFormulation(vertices=sorted(graph.nodes()), weights=dict(weights), triangles=list(triangles))

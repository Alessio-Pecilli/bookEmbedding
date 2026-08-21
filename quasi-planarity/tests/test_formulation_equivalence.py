"""Cross-validation gate (spec §4): build_pubo, build_qubo (decoded), and
build_ilp must all agree on the optimal objective value and on the optimal
feasible-set structure, on small brute-forceable instances."""

from __future__ import annotations

from itertools import product

import pytest

from tvd_qaoa.classical_solvers import solve_ilp, solve_qubo_exact
from tvd_qaoa.formulation import (
    build_ilp,
    build_pubo,
    build_qubo,
    default_penalty,
    enumerate_triangles,
    evaluate_pubo,
)
from tvd_qaoa.instances import generate_instance


def _brute_force_pubo_optimum(pubo, n):
    best = None
    for bits in product((0, 1), repeat=n):
        assignment = {v: bits[v] for v in range(n)}
        cost = evaluate_pubo(pubo, assignment)
        if best is None or cost < best:
            best = cost
    return best


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_pubo_qubo_ilp_agree_on_optimum(seed, weighted):
    instance = generate_instance(n=8, seed=seed, weighted=weighted)
    graph, weights = instance.graph, instance.weights
    triangles = enumerate_triangles(graph)
    assert len(triangles) > 0

    penalty = default_penalty(weights)

    pubo = build_pubo(graph, weights, triangles, penalty)
    pubo_optimum = _brute_force_pubo_optimum(pubo, instance.n_vertices)

    ilp = build_ilp(graph, weights, triangles)
    ilp_result = solve_ilp(ilp)

    qubo_bqm, ancilla_map = build_qubo(graph, weights, triangles, penalty)
    qubo_result = solve_qubo_exact(qubo_bqm, ancilla_map, instance.n_vertices)
    assert qubo_result is not None

    assert pubo_optimum == pytest.approx(ilp_result.objective, abs=1e-6)
    assert qubo_result.objective == pytest.approx(ilp_result.objective, abs=1e-6)

    # Feasible-set structure: every triangle is hit by the ILP's selection.
    selected = set(ilp_result.selected_vertices)
    for u, v, w in triangles:
        assert {u, v, w} & selected


def test_reproducibility_same_seed_same_instance():
    a = generate_instance(n=10, seed=7, weighted=True)
    b = generate_instance(n=10, seed=7, weighted=True)
    assert sorted(a.graph.edges()) == sorted(b.graph.edges())
    assert a.weights == b.weights
    assert a.triangles == b.triangles


def test_every_vertex_in_at_least_one_triangle():
    instance = generate_instance(n=12, seed=99, weighted=False)
    covered = set()
    for t in instance.triangles:
        covered.update(t)
    assert covered == set(instance.graph.nodes())

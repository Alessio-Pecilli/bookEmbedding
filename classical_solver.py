from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from itertools import product
from typing import Dict, Iterable, Tuple

import config


@dataclass(frozen=True)
class ClassicalResult:
    assignment: Dict[int, int]  # edge_idx -> page
    weighted_cost: float
    solve_time_s: float
    status: str


def _normalize_crossings(
    weighted_crossings: Iterable[Tuple[int, int, float]],
    num_edges: int,
) -> Tuple[Tuple[int, int, float], ...]:
    crossings = tuple((int(e), int(f), float(w)) for e, f, w in weighted_crossings)
    for e, f, weight in crossings:
        if not (0 <= e < num_edges and 0 <= f < num_edges) or e == f:
            raise ValueError(f"Invalid crossing edge indices: {(e, f)}")
        if weight < 0:
            raise ValueError("Crossing weights must be non-negative")
    return crossings


def weighted_crossing_cost(
    assignment: Dict[int, int],
    weighted_crossings: Iterable[Tuple[int, int, float]],
) -> float:
    """Evaluate the original weighted objective for a page assignment."""
    total = 0.0
    for e, f, weight in weighted_crossings:
        if assignment.get(int(e), -1) == assignment.get(int(f), -2):
            page = assignment.get(int(e), -1)
            if page >= 0:
                total += float(weight)
    return float(total)


def is_valid_assignment(assignment: Dict[int, int], num_edges: int, num_pages: int) -> bool:
    return (
        num_edges >= 0
        and num_pages >= 1
        and set(assignment) == set(range(num_edges))
        and all(0 <= int(assignment[e]) < num_pages for e in range(num_edges))
    )


def solve_book_embedding_cpsat(
    num_edges: int,
    num_pages: int,
    weighted_crossings: Iterable[Tuple[int, int, float]],
    time_limit_s: float | None = None,
    num_workers: int | None = None,
    objective_scale: int | None = None,
) -> ClassicalResult:
    """
    Solve fixed-order book embedding page assignment.

    Variables:
      x[e,p] ∈ {0,1}  (edge e assigned to page p)

    Constraints:
      Σ_p x[e,p] = 1  for each e

    Objective (weighted):
      minimize Σ_(e,f,w) Σ_p w * (x[e,p] AND x[f,p])
    """
    from ortools.sat.python import cp_model

    if num_edges < 0 or num_pages < 1:
        raise ValueError("num_edges must be non-negative and num_pages must be positive")
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be positive")
    scale = int(objective_scale or config.CLASSICAL_OBJECTIVE_SCALE)
    if scale < 1:
        raise ValueError("objective_scale must be positive")
    weighted_crossings = _normalize_crossings(weighted_crossings, num_edges)

    model = cp_model.CpModel()

    x = {}
    for e in range(num_edges):
        for p in range(num_pages):
            x[(e, p)] = model.NewBoolVar(f"x_e{e}_p{p}")
        model.Add(sum(x[(e, p)] for p in range(num_pages)) == 1)

    obj_terms = []
    for (e, f, w) in weighted_crossings:
        for p in range(num_pages):
            y = model.NewBoolVar(f"y_e{e}_e{f}_p{p}")
            model.AddBoolAnd([x[(e, p)], x[(f, p)]]).OnlyEnforceIf(y)
            model.AddBoolOr([x[(e, p)].Not(), x[(f, p)].Not(), y])

            # CP-SAT objective needs integer coefficients.  The explicit large
            # scale preserves the weighted objective to sub-nanounit precision
            # for the generated instances instead of silently coarsening it.
            coeff = int(round(float(w) * scale))
            if coeff != 0:
                obj_terms.append(coeff * y)

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    if num_workers is not None:
        solver.parameters.num_search_workers = int(num_workers)

    t0 = perf_counter()
    status = solver.Solve(model)
    solve_time = perf_counter() - t0

    status_str = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT did not find a feasible assignment: {status_str}")

    assignment: Dict[int, int] = {}
    for e in range(num_edges):
        chosen = None
        for p in range(num_pages):
            if solver.Value(x[(e, p)]) == 1:
                chosen = p
                break
        assignment[e] = -1 if chosen is None else int(chosen)

    # Re-evaluate from the returned assignment.  This is the objective in the
    # original floating-point scale, independent of CP-SAT's integer scaling.
    weighted_cost = weighted_crossing_cost(assignment, weighted_crossings)
    return ClassicalResult(
        assignment=assignment,
        weighted_cost=weighted_cost,
        solve_time_s=float(solve_time),
        status=status_str,
    )


def solve_book_embedding_bruteforce(
    num_edges: int,
    num_pages: int,
    weighted_crossings: Iterable[Tuple[int, int, float]],
) -> ClassicalResult:
    """Exhaustively solve tiny instances; intended for tests and validation."""
    if num_edges < 0 or num_pages < 1:
        raise ValueError("num_edges must be non-negative and num_pages must be positive")

    weighted_crossings = _normalize_crossings(weighted_crossings, num_edges)
    t0 = perf_counter()
    best_assignment: Dict[int, int] | None = None
    best_cost = float("inf")

    for pages in product(range(num_pages), repeat=num_edges):
        assignment = {edge: int(page) for edge, page in enumerate(pages)}
        cost = weighted_crossing_cost(assignment, weighted_crossings)
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_assignment = assignment

    if best_assignment is None:
        best_assignment = {}
        best_cost = 0.0

    return ClassicalResult(
        assignment=best_assignment,
        weighted_cost=float(best_cost),
        solve_time_s=float(perf_counter() - t0),
        status="OPTIMAL_BRUTE_FORCE",
    )

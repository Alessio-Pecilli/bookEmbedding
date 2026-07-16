from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ClassicalResult:
    assignment: Dict[int, int]  # edge_idx -> page
    weighted_cost: float
    solve_time_s: float
    status: str


def solve_book_embedding_cpsat(
    num_edges: int,
    num_pages: int,
    weighted_crossings: Iterable[Tuple[int, int, float]],
    time_limit_s: float | None = None,
    num_workers: int | None = None,
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

            # CP-SAT objective needs integer coeffs: scale weights.
            # Use 1000x scaling by default; caller can pre-scale if desired.
            coeff = int(round(float(w) * 1000.0))
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

    assignment: Dict[int, int] = {}
    for e in range(num_edges):
        chosen = None
        for p in range(num_pages):
            if solver.Value(x[(e, p)]) == 1:
                chosen = p
                break
        assignment[e] = -1 if chosen is None else int(chosen)

    # Recover objective in original (float) scale.
    weighted_cost = float(solver.ObjectiveValue()) / 1000.0
    return ClassicalResult(
        assignment=assignment,
        weighted_cost=weighted_cost,
        solve_time_s=float(solve_time),
        status=status_str,
    )


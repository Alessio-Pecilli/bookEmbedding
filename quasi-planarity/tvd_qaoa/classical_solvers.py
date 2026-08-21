"""Classical baselines: greedy heuristic, exact ILP, exact QUBO (spec §5)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import networkx as nx

from .formulation import ILPFormulation, Triangle

logger = logging.getLogger(__name__)

# Exact QUBO brute force is exponential in (n_vertices + n_ancillas); refuse
# above this many total variables rather than hanging.
QUBO_EXACT_MAX_VARS = 22


@dataclass
class SolverResult:
    selected_vertices: list[int]
    objective: float
    wall_clock_seconds: float
    solver_name: str


# ---------------------------------------------------------------------------
# Heuristic: weighted greedy hitting set
# ---------------------------------------------------------------------------


def solve_heuristic(
    graph: nx.Graph, weights: dict[int, float], triangles: list[Triangle]
) -> SolverResult:
    """Repeatedly pick the vertex maximizing (uncovered triangles it's in) /
    w_v, delete it, mark its triangles covered, until none remain."""
    logger.info("heuristic: starting (%d triangles)", len(triangles))
    start = time.perf_counter()
    uncovered = set(triangles)
    vertex_triangles: dict[int, set[Triangle]] = {v: set() for v in graph.nodes()}
    for t in triangles:
        for v in t:
            vertex_triangles[v].add(t)

    selected: list[int] = []
    while uncovered:
        best_v, best_score = None, -1.0
        for v, v_tris in vertex_triangles.items():
            count = len(v_tris & uncovered)
            if count == 0:
                continue
            score = count / weights[v]
            if score > best_score:
                best_v, best_score = v, score
        if best_v is None:
            break  # should not happen if triangles cover only listed vertices
        selected.append(best_v)
        uncovered -= vertex_triangles[best_v]

    objective = sum(weights[v] for v in selected)
    elapsed = time.perf_counter() - start
    logger.info("heuristic: done, objective=%.4f (%.2fs)", objective, elapsed)
    return SolverResult(selected, objective, elapsed, "greedy_heuristic")


# ---------------------------------------------------------------------------
# Exact ILP ("PLI"): commercial solver first, PuLP+CBC fallback
# ---------------------------------------------------------------------------


def _solve_ilp_gurobi(ilp: ILPFormulation) -> SolverResult | None:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    try:
        start = time.perf_counter()
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = {v: model.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in ilp.vertices}
                model.setObjective(gp.quicksum(ilp.weights[v] * x[v] for v in ilp.vertices), GRB.MINIMIZE)
                for u, v, w in ilp.triangles:
                    model.addConstr(x[u] + x[v] + x[w] >= 1)
                model.optimize()
                if model.Status != GRB.OPTIMAL:
                    return None
                selected = [v for v in ilp.vertices if x[v].X > 0.5]
                elapsed = time.perf_counter() - start
                return SolverResult(selected, model.ObjVal, elapsed, "gurobi")
    except Exception as exc:  # license absent, etc. -- fall back, never hard-fail
        logger.info("Gurobi unavailable/failed (%s); falling back", exc)
        return None


def _solve_ilp_docplex(ilp: ILPFormulation) -> SolverResult | None:
    try:
        from docplex.mp.model import Model
    except ImportError:
        return None
    try:
        start = time.perf_counter()
        model = Model(name="tvd_ilp")
        x = {v: model.binary_var(name=f"x_{v}") for v in ilp.vertices}
        model.minimize(model.sum(ilp.weights[v] * x[v] for v in ilp.vertices))
        for u, v, w in ilp.triangles:
            model.add_constraint(x[u] + x[v] + x[w] >= 1)
        sol = model.solve()
        if sol is None:
            return None
        selected = [v for v in ilp.vertices if sol.get_value(x[v]) > 0.5]
        elapsed = time.perf_counter() - start
        return SolverResult(selected, sol.objective_value, elapsed, "cplex")
    except Exception as exc:
        logger.info("CPLEX unavailable/failed (%s); falling back", exc)
        return None


def _solve_ilp_pulp(ilp: ILPFormulation) -> SolverResult:
    import pulp

    start = time.perf_counter()
    model = pulp.LpProblem("tvd_ilp", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in ilp.vertices}
    model += pulp.lpSum(ilp.weights[v] * x[v] for v in ilp.vertices)
    for u, v, w in ilp.triangles:
        model += x[u] + x[v] + x[w] >= 1
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [v for v in ilp.vertices if x[v].value() > 0.5]
    objective = sum(ilp.weights[v] for v in selected)
    elapsed = time.perf_counter() - start
    return SolverResult(selected, objective, elapsed, "pulp_cbc")


def solve_ilp(ilp: ILPFormulation) -> SolverResult:
    """Try Gurobi, then CPLEX, then fall back to PuLP+CBC. Logs which
    backend actually ran; never hard-fails solely for lack of a commercial
    license."""
    logger.info("ILP: starting (%d vertices, %d triangles)", len(ilp.vertices), len(ilp.triangles))
    for solver_fn in (_solve_ilp_gurobi, _solve_ilp_docplex):
        result = solver_fn(ilp)
        if result is not None:
            logger.info(
                "ILP: done, backend=%s objective=%.4f (%.2fs)",
                result.solver_name, result.objective, result.wall_clock_seconds,
            )
            return result
    result = _solve_ilp_pulp(ilp)
    logger.info(
        "ILP: done, backend=%s objective=%.4f (%.2fs)",
        result.solver_name, result.objective, result.wall_clock_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# Exact QUBO: brute force via dimod.ExactSolver, decode ancillas away
# ---------------------------------------------------------------------------


def solve_qubo_exact(bqm, ancilla_map: dict, n_vertices: int) -> SolverResult | None:
    """Exhaustively solve the quadratized QUBO and decode back to original
    vertex variables (ancillas are dropped -- the penalty guarantees they
    equal x_u*x_v at the optimum). Returns None (and logs) if the instance is
    too large for brute force rather than hanging."""
    import dimod

    n_total = n_vertices + len(ancilla_map)
    if n_total > QUBO_EXACT_MAX_VARS:
        logger.warning(
            "solve_qubo_exact: skipping, %d total variables (%d vertices + "
            "%d ancillas) exceeds brute-force limit %d",
            n_total,
            n_vertices,
            len(ancilla_map),
            QUBO_EXACT_MAX_VARS,
        )
        return None

    logger.info("QUBO exact: starting brute force over %d variables", n_total)
    start = time.perf_counter()
    sampleset = dimod.ExactSolver().sample(bqm)
    best = sampleset.first
    selected = [v for v in range(n_vertices) if best.sample.get(v, 0) == 1]
    elapsed = time.perf_counter() - start
    logger.info("QUBO exact: done, objective=%.4f (%.2fs)", float(best.energy), elapsed)
    return SolverResult(selected, float(best.energy), elapsed, "dimod_exact")

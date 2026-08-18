from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np

import config
from book_viz import draw_book_embedding
from classical_solver import (
    is_valid_assignment,
    solve_book_embedding_cpsat,
    weighted_crossing_cost,
)
from graph_manager import assign_edge_weights, get_graph, precompute_crossings
from qaoa_solver import (
    CostModel,
    decode_logical_bitstring,
    build_cost_model,
    estimate_energy_from_counts,
    most_probable_bitstring,
    sample_qaoa_counts,
)


def decode_solution(bitstring: str, num_edges: int) -> Dict[int, int]:
    """Compatibility wrapper for the canonical logical decoder."""
    return decode_logical_bitstring(bitstring, num_edges, int(config.NUM_PAGES))


def count_violations(assignment: Dict[int, int]) -> int:
    return sum(1 for page in assignment.values() if int(page) < 0)


def optimize_qaoa(
    model: CostModel,
    layers: int,
    steps: int,
    shots: int,
    seed: int,
    backend: object | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Optimize QAOA parameters with deterministic seeded sampling."""
    from scipy.optimize import minimize
    if layers < 1 or steps < 1 or shots < 1:
        raise ValueError("layers, steps, and shots must be positive")
    if backend is None:
        from pytket.extensions.qiskit import AerBackend
        backend = AerBackend()

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-float(config.INIT_SCALE), float(config.INIT_SCALE), size=(2 * layers,))
    eval_counter = {"i": 0}

    def objective(x: np.ndarray) -> float:
        eval_counter["i"] += 1
        gammas = np.array(x[:layers], dtype=float)
        betas = np.array(x[layers:], dtype=float)
        counts = sample_qaoa_counts(
            model, gammas, betas, shots=shots, seed=seed + eval_counter["i"], backend=backend
        )
        return estimate_energy_from_counts(model, counts)

    t0 = perf_counter()
    result = minimize(
        objective,
        x0,
        method="COBYLA",
        # SciPy's COBYLA needs at least n_variables + 2 evaluations.  Treat a
        # smaller requested smoke-test budget as that minimum instead of
        # emitting its "Invalid MAXFUN" warning.
        options={"maxiter": max(int(steps), 2 * int(layers) + 2)},
    )
    elapsed = perf_counter() - t0
    parameters = np.asarray(result.x, dtype=float)
    gammas = parameters[:layers].copy()
    betas = parameters[layers:].copy()
    final_counts = sample_qaoa_counts(
        model,
        gammas,
        betas,
        shots=shots,
        seed=seed + 99_991,
        backend=backend,
    )
    return gammas, betas, estimate_energy_from_counts(model, final_counts), float(elapsed)


def evaluate_sampled_solutions(
    counts: Dict[str, int],
    num_edges: int,
    weighted_crossings: Iterable[Tuple[int, int, float]],
    optimal_cost: float,
    tolerance: float = 1e-8,
) -> Dict[str, object]:
    """Compute quality metrics for the complete sampled distribution."""
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("No sampled outcomes")
    weighted_crossings = tuple(weighted_crossings)

    valid_outcomes = []
    for bitstring, count in counts.items():
        assignment = decode_solution(bitstring, num_edges)
        if is_valid_assignment(assignment, num_edges, int(config.NUM_PAGES)):
            cost = weighted_crossing_cost(assignment, weighted_crossings)
            valid_outcomes.append((bitstring, int(count), cost))

    valid_shots = sum(count for _, count, _ in valid_outcomes)
    best_valid = min(
        valid_outcomes,
        key=lambda item: (item[2], -item[1], item[0]),
        default=None,
    )
    best_valid_cost = None if best_valid is None else best_valid[2]
    expected_valid_cost = (
        sum(count * cost for _, count, cost in valid_outcomes) / valid_shots
        if valid_shots
        else None
    )
    optimal_shots = sum(
        count for _, count, cost in valid_outcomes
        if abs(cost - optimal_cost) <= tolerance * max(1.0, abs(optimal_cost))
    )

    most_probable = most_probable_bitstring(counts)
    most_assignment = decode_solution(most_probable, num_edges)
    most_valid = is_valid_assignment(most_assignment, num_edges, int(config.NUM_PAGES))
    most_cost = (
        weighted_crossing_cost(most_assignment, weighted_crossings) if most_valid else None
    )
    gap = None if best_valid_cost is None else float(best_valid_cost - optimal_cost)
    ratio = (
        None
        if best_valid_cost is None
        or abs(optimal_cost) <= tolerance
        or abs(best_valid_cost) <= tolerance
        else float(optimal_cost / best_valid_cost)
    )

    return {
        "probability_valid_solution": valid_shots / shots,
        "probability_optimal_solution": optimal_shots / shots,
        "best_valid_weighted_cost_sampled": best_valid_cost,
        "best_valid_bitstring_sampled": None if best_valid is None else best_valid[0],
        "expected_weighted_cost_valid": expected_valid_cost,
        "most_probable_bitstring": most_probable,
        "most_probable_is_valid": most_valid,
        "most_probable_weighted_cost": most_cost,
        "approximation_gap_vs_classical": gap,
        "approximation_ratio": ratio,
        "valid_sample_count": valid_shots,
        "shots": shots,
    }


def main() -> None:
    print("\n" + "=" * 60)
    print("QAOA Solver — Fixed-Order Book Embedding (pytket + weighted crossings)")
    print("=" * 60)

    nodes, edges, node_order = get_graph()
    num_edges = len(edges)
    n_qubits = num_edges * int(config.NUM_PAGES)
    if n_qubits > int(config.MAX_QUBITS):
        raise ValueError(f"Too many qubits: {n_qubits} > MAX_QUBITS={config.MAX_QUBITS}")

    edge_weights = assign_edge_weights(edges, seed=int(config.SEED))
    weighted_crossings = precompute_crossings(edges, node_order, edge_weights=edge_weights)
    print(f"[INFO] Edges: {num_edges}, Pages: {config.NUM_PAGES}, Qubits: {n_qubits}")
    print(f"[INFO] Weighted crossings |C| = {len(weighted_crossings)}")
    print(f"[INFO] Seed: {config.SEED}")

    classical = solve_book_embedding_cpsat(
        num_edges=num_edges,
        num_pages=int(config.NUM_PAGES),
        weighted_crossings=weighted_crossings,
        time_limit_s=float(config.CLASSICAL_TIME_LIMIT_S),
        num_workers=int(config.CLASSICAL_NUM_WORKERS),
    )
    if classical.status not in {"OPTIMAL", "FEASIBLE"} or not is_valid_assignment(
        classical.assignment, num_edges, int(config.NUM_PAGES)
    ):
        raise RuntimeError(
            f"Classical baseline did not return a valid feasible solution: {classical.status}"
        )
    print(
        f"[CLASSICAL] status={classical.status} cost={classical.weighted_cost:.4f} "
        f"time={classical.solve_time_s:.3f}s"
    )

    model = build_cost_model(edges, weighted_crossings)
    print(f"[HAMILTONIAN] configured_alpha={config.ALPHA} effective_alpha={model.alpha}")
    from pytket.extensions.qiskit import AerBackend
    backend = AerBackend()

    def run_one(layers: int) -> Dict[str, object]:
        gammas, betas, expectation_energy, optimize_time = optimize_qaoa(
            model=model,
            layers=layers,
            steps=int(config.STEPS),
            shots=int(config.QAOA_OPTIMIZATION_SHOTS),
            seed=int(config.SEED),
            backend=backend,
        )
        counts = sample_qaoa_counts(
            model,
            gammas,
            betas,
            shots=int(config.QAOA_FINAL_SHOTS),
            seed=int(config.SEED) + 424_242,
            backend=backend,
        )
        metrics = evaluate_sampled_solutions(
            counts, num_edges, weighted_crossings, classical.weighted_cost
        )
        return {
            "layers": int(layers),
            "expectation_value_hamiltonian": float(expectation_energy),
            "gammas": gammas.tolist(),
            "betas": betas.tolist(),
            "counts": counts,
            "optimize_time_s": float(optimize_time),
            **metrics,
        }

    layers_to_run = (
        range(1, int(config.LAYERS) + 1)
        if bool(config.LAYER_SWEEP)
        else [int(config.LAYERS)]
    )
    results = []
    for layers in layers_to_run:
        print(f"[QAOA] Optimizing layers={layers} ...")
        results.append(run_one(layers))

    valid = [r for r in results if r["best_valid_weighted_cost_sampled"] is not None]
    best = min(
        valid,
        key=lambda r: (float(r["best_valid_weighted_cost_sampled"]), -float(r["probability_optimal_solution"])),
    ) if valid else min(results, key=lambda r: float(r["expectation_value_hamiltonian"]))

    print(
        f"[QAOA] best_layers={best['layers']} "
        f"best_valid_cost={best['best_valid_weighted_cost_sampled']} "
        f"p_valid={best['probability_valid_solution']:.4f} "
        f"p_opt={best['probability_optimal_solution']:.4f}"
    )

    if config.SHOW_PLOTS:
        draw_book_embedding(nodes, edges, node_order, classical.assignment)
        if best["best_valid_bitstring_sampled"] is not None:
            draw_book_embedding(
                nodes, edges, node_order,
                decode_solution(str(best["best_valid_bitstring_sampled"]), num_edges),
            )

    from results_io import save_run_json

    payload = {
        "config": {name: getattr(config, name) for name in (
            "USE_PLANAR_DEMO", "NUM_PAGES", "MAX_QUBITS", "NUM_NODES", "NUM_EDGES",
            "WEIGHT_LOW", "WEIGHT_HIGH", "ALPHA", "BETA", "LAYERS", "LAYER_SWEEP",
            "STEPS", "INIT_SCALE", "QAOA_OPTIMIZATION_SHOTS", "QAOA_FINAL_SHOTS",
            "CLASSICAL_TIME_LIMIT_S", "CLASSICAL_NUM_WORKERS", "SEED",
            "CLASSICAL_OBJECTIVE_SCALE", "SHOW_PLOTS",
        )},
        "graph": {
            "nodes": nodes, "edges": edges, "node_order": node_order,
            "edge_weights": {int(k): float(v) for k, v in edge_weights.items()},
        },
        "crossings": [[int(e), int(f), float(w)] for e, f, w in weighted_crossings],
        "n_qubits": n_qubits,
        "hamiltonian": {"configured_alpha": float(config.ALPHA), "effective_alpha": model.alpha},
        "classical": asdict(classical),
        "qaoa": {"best": best, "all_runs": results},
    }
    out_path = save_run_json(payload)
    print(f"[RESULTS] Saved: {out_path}")


if __name__ == "__main__":
    main()

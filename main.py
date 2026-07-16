from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np

import config
from book_viz import draw_book_embedding
from classical_solver import solve_book_embedding_cpsat
from graph_manager import assign_edge_weights, get_graph, precompute_crossings
from qaoa_solver import (
    CostModel,
    build_cost_model,
    estimate_energy_from_counts,
    most_probable_bitstring,
    sample_qaoa_counts,
)


def decode_solution(bitstring: str, num_edges: int) -> Dict[int, int]:
    """
    Decode a bitstring into an assignment edge_idx -> page.
    Encoding: qubit = edge_idx * NUM_PAGES + page
    """
    assignment: Dict[int, int] = {}
    k = int(config.NUM_PAGES)

    for e_idx in range(num_edges):
        start = e_idx * k
        end = start + k
        bits = bitstring[start:end]
        active = [p for p in range(k) if bits[p] == "1"]
        assignment[e_idx] = active[0] if len(active) == 1 else -1
    return assignment


def weighted_crossing_cost(assignment: Dict[int, int], weighted_crossings) -> float:
    total = 0.0
    for (e, f, w) in weighted_crossings:
        pe = assignment.get(int(e), -1)
        pf = assignment.get(int(f), -1)
        if pe >= 0 and pe == pf:
            total += float(w)
    return float(total)


def count_violations(assignment: Dict[int, int]) -> int:
    return sum(1 for v in assignment.values() if int(v) < 0)


def best_decode_with_optional_reverse(bitstring: str, num_edges: int) -> Dict[int, int]:
    """
    Depending on backend conventions, bit order may be reversed.
    Try both and pick the one with fewer one-hot violations.
    """
    a1 = decode_solution(bitstring, num_edges)
    a2 = decode_solution(bitstring[::-1], num_edges)
    if count_violations(a2) < count_violations(a1):
        return a2
    return a1


def optimize_qaoa(
    model: CostModel,
    layers: int,
    steps: int,
    shots: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Optimize QAOA parameters via a derivative-free method (COBYLA).
    Returns (best_gammas, best_betas, best_energy, optimize_time_s).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-float(config.INIT_SCALE), float(config.INIT_SCALE), size=(2 * layers,))

    eval_counter = {"i": 0}

    def objective(x: np.ndarray) -> float:
        eval_counter["i"] += 1
        gammas = np.array(x[:layers], dtype=float)
        betas = np.array(x[layers:], dtype=float)
        # Keep sampling deterministic enough for optimizer by fixing seed per-eval.
        counts = sample_qaoa_counts(model, gammas, betas, shots=shots, seed=seed + eval_counter["i"])
        return estimate_energy_from_counts(model, counts)

    t0 = perf_counter()
    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": int(steps)},
    )
    dt = perf_counter() - t0

    x_best = np.array(res.x, dtype=float)
    gammas_best = x_best[:layers].copy()
    betas_best = x_best[layers:].copy()

    # Re-evaluate with the final parameters (fresh seed) for reporting.
    counts_final = sample_qaoa_counts(model, gammas_best, betas_best, shots=shots, seed=seed + 99991)
    energy_final = estimate_energy_from_counts(model, counts_final)

    return gammas_best, betas_best, float(energy_final), float(dt)


def main():
    print("\n" + "=" * 60)
    print("QAOA Solver — Fixed-Order Book Embedding (pytket + weighted crossings)")
    print("=" * 60)

    nodes, edges, node_order = get_graph()
    num_edges = len(edges)
    n_qubits = num_edges * int(config.NUM_PAGES)

    if n_qubits > int(config.MAX_QUBITS):
        raise ValueError(
            f"Too many qubits: {n_qubits} > MAX_QUBITS={config.MAX_QUBITS}. "
            f"Reduce NUM_EDGES or NUM_PAGES."
        )

    edge_weights = assign_edge_weights(edges, seed=config.SEED)
    weighted_crossings = precompute_crossings(edges, node_order, edge_weights=edge_weights)

    print(f"[INFO] Edges: {num_edges}, Pages: {config.NUM_PAGES}, Qubits: {n_qubits}")
    print(f"[INFO] Weighted crossings |C| = {len(weighted_crossings)}")

    # --- Classical baseline (CP-SAT) ---
    classical = solve_book_embedding_cpsat(
        num_edges=num_edges,
        num_pages=int(config.NUM_PAGES),
        weighted_crossings=weighted_crossings,
        time_limit_s=10.0,
        num_workers=8,
    )
    print(f"[CLASSICAL] status={classical.status} cost={classical.weighted_cost:.4f} time={classical.solve_time_s:.3f}s")

    # --- QAOA model ---
    model = build_cost_model(edges, weighted_crossings)

    def run_one(layers: int):
        gammas, betas, best_energy, opt_time = optimize_qaoa(
            model=model,
            layers=layers,
            steps=int(config.STEPS),
            shots=5000,
            seed=int(config.SEED),
        )

        counts = sample_qaoa_counts(model, gammas, betas, shots=20000, seed=int(config.SEED) + 424242)
        bs = most_probable_bitstring(counts)
        assignment = best_decode_with_optional_reverse(bs, num_edges=num_edges)
        q_cost = weighted_crossing_cost(assignment, weighted_crossings)
        violations = count_violations(assignment)

        return {
            "layers": int(layers),
            "best_energy": float(best_energy),
            "gammas": gammas.tolist(),
            "betas": betas.tolist(),
            "bitstring": bs,
            "assignment": assignment,
            "weighted_cost": float(q_cost),
            "violations": int(violations),
            "optimize_time_s": float(opt_time),
        }

    results = []
    if bool(config.LAYER_SWEEP):
        for l in range(1, int(config.LAYERS) + 1):
            print(f"[QAOA] Optimizing layers={l} ...")
            results.append(run_one(l))
    else:
        print(f"[QAOA] Optimizing layers={config.LAYERS} ...")
        results.append(run_one(int(config.LAYERS)))

    # Pick best by weighted cost among valid solutions, else by energy.
    valid = [r for r in results if r["violations"] == 0]
    if valid:
        best = min(valid, key=lambda r: r["weighted_cost"])
    else:
        best = min(results, key=lambda r: r["best_energy"])

    gap = best["weighted_cost"] - classical.weighted_cost
    print(
        f"[QAOA] best_layers={best['layers']} cost={best['weighted_cost']:.4f} "
        f"violations={best['violations']} gap_vs_classical={gap:.4f}"
    )

    # --- Visualization ---
    print("[INFO] Visualize classical assignment...")
    draw_book_embedding(nodes, edges, node_order, classical.assignment)
    print("[INFO] Visualize QAOA assignment...")
    draw_book_embedding(nodes, edges, node_order, best["assignment"])

    # --- Save results (implemented in results_io.py) ---
    try:
        from results_io import save_run_json

        payload = {
            "config": {
                "USE_PLANAR_DEMO": bool(config.USE_PLANAR_DEMO),
                "NUM_PAGES": int(config.NUM_PAGES),
                "MAX_QUBITS": int(config.MAX_QUBITS),
                "WEIGHT_LOW": float(config.WEIGHT_LOW),
                "WEIGHT_HIGH": float(config.WEIGHT_HIGH),
                "ALPHA": float(config.ALPHA),
                "BETA": float(config.BETA),
                "LAYERS": int(config.LAYERS),
                "LAYER_SWEEP": bool(config.LAYER_SWEEP),
                "STEPS": int(config.STEPS),
                "SEED": int(config.SEED),
            },
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "node_order": node_order,
                "edge_weights": {int(k): float(v) for k, v in edge_weights.items()},
            },
            "crossings": [[int(e), int(f), float(w)] for (e, f, w) in weighted_crossings],
            "n_qubits": int(n_qubits),
            "classical": asdict(classical),
            "qaoa": {
                "best": best,
                "all_runs": results,
                "gap_vs_classical": float(gap),
            },
        }

        out_path = save_run_json(payload)
        print(f"[RESULTS] Saved: {out_path}")
    except Exception as ex:
        print(f"[RESULTS] Skipped saving (results_io.py not ready): {ex}")


if __name__ == "__main__":
    main()


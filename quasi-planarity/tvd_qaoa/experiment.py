"""Experiment orchestration, driven by a config (spec §7, §8).

Per instance, in order: generate/load -> prune (§2) -> build all three
formulations (§4) -> run heuristic + exact ILP + exact QUBO (§5) -> run the
QAOA layer-count sweep (§6) -> assemble one result record.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import classical_solvers as cs
from . import io_utils
from .formulation import (
    build_ilp,
    build_pubo,
    build_qubo,
    default_penalty,
    prune_zero_triangle_vertices,
)
from .instances import Instance, generate_sweep
from .qaoa import BackendConfig, qaoa_p_sweep

logger = logging.getLogger(__name__)


class _Timer:
    """Context manager that logs a stage's start/end and records its elapsed
    wall-clock time on `self.elapsed`, so every pipeline stage's timing shows
    up both in the live log stream and in the results row."""

    def __init__(self, instance_id: str, stage: str):
        self.instance_id = instance_id
        self.stage = stage
        self.elapsed = 0.0

    def __enter__(self) -> "_Timer":
        logger.info("[%s] %s: starting", self.instance_id, self.stage)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.perf_counter() - self._start
        if exc_type is None:
            logger.info("[%s] %s: done (%.2fs)", self.instance_id, self.stage, self.elapsed)
        return False


@dataclass
class ExperimentConfig:
    problem_type: str  # "weighted" or "unweighted"
    max_vertices: int
    n_min: int = 4
    step: int = 2
    n_seeds_per_size: int = 1
    base_seed: int = 42
    p_max: int = 3
    penalty: float | None = None  # None -> default_penalty(weights) per instance
    approx_ratio_threshold: float = 0.99
    n_restarts: int = 3
    n_shots: int = 1024
    n_shots_optimization: int | None = None
    weight_low: int = 1
    weight_high: int = 10
    # QAOA simulation backend: "aer_statevector" (exact, default, small n
    # only), "aer" (shot-sampled, CPU or GPU), "aer_mps" (shot-sampled
    # matrix-product-state / tensor-network method -- what a 40-qubit-scale
    # server would use). See qaoa.BackendConfig / qaoa.get_backend.
    qaoa_backend: str = "aer_statevector"
    qaoa_device: str = "CPU"
    qaoa_mps_max_bond_dimension: int | None = None
    qaoa_mps_truncation_threshold: float | None = None
    instances_dir: Path = field(default_factory=lambda: io_utils.INSTANCES_DIR)
    results_csv: Path = field(default_factory=lambda: io_utils.RESULTS_DIR / "results.csv")
    angles_dir: Path = field(default_factory=lambda: io_utils.ANGLES_DIR)

    @property
    def weighted(self) -> bool:
        return self.problem_type == "weighted"

    @property
    def backend_config(self) -> BackendConfig:
        return BackendConfig(
            name=self.qaoa_backend,
            device=self.qaoa_device,
            mps_max_bond_dimension=self.qaoa_mps_max_bond_dimension,
            mps_truncation_threshold=self.qaoa_mps_truncation_threshold,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        for key in ("instances_dir", "results_csv", "angles_dir"):
            if key in payload:
                payload[key] = Path(payload[key])
        return cls(**payload)


def run_instance(instance: Instance, config: ExperimentConfig) -> dict:
    """Run the full pipeline for one instance and return its result row."""
    iid = instance.instance_id
    pipeline_start = time.perf_counter()
    logger.info(
        "[%s] === starting pipeline (problem_type=%s, n_vertices=%d) ===",
        iid, instance.problem_type, instance.n_vertices,
    )

    from .formulation import enumerate_triangles

    with _Timer(iid, "prune + enumerate_triangles") as t:
        pruned_graph, removed = prune_zero_triangle_vertices(instance.graph, instance.triangles)
        if removed:
            logger.warning("[%s] pruning removed %s", iid, removed)
            instance.graph = pruned_graph
            instance.weights = {v: w for v, w in instance.weights.items() if v in pruned_graph.nodes()}
        triangles = enumerate_triangles(instance.graph)
    prune_wall_clock = t.elapsed

    n_qubits_used = instance.graph.number_of_nodes()
    penalty = config.penalty if config.penalty is not None else default_penalty(instance.weights)

    # --- formulations (§4) ---
    with _Timer(iid, "build_pubo") as t:
        pubo = build_pubo(instance.graph, instance.weights, triangles, penalty)
    build_pubo_wall_clock = t.elapsed

    with _Timer(iid, "build_qubo") as t:
        qubo_bqm, ancilla_map = build_qubo(instance.graph, instance.weights, triangles, penalty)
    build_qubo_wall_clock = t.elapsed

    with _Timer(iid, "build_ilp") as t:
        ilp = build_ilp(instance.graph, instance.weights, triangles)
    build_ilp_wall_clock = t.elapsed

    # --- classical solvers (§5) ---
    heuristic_result = cs.solve_heuristic(instance.graph, instance.weights, triangles)

    ilp_start = time.perf_counter()
    ilp_result = cs.solve_ilp(ilp)
    ilp_wall_clock = time.perf_counter() - ilp_start

    qubo_start = time.perf_counter()
    qubo_result = cs.solve_qubo_exact(qubo_bqm, ancilla_map, n_qubits_used)
    qubo_wall_clock = time.perf_counter() - qubo_start

    # --- QAOA layer-count sweep (§6) ---
    qaoa_start = time.perf_counter()
    qaoa_result = qaoa_p_sweep(
        pubo,
        n_qubits_used,
        config.p_max,
        classical_optimal=ilp_result.objective,
        approx_ratio_threshold=config.approx_ratio_threshold,
        n_restarts=config.n_restarts,
        n_shots=config.n_shots,
        seed=instance.seed,
        backend_config=config.backend_config,
        n_shots_optimization=config.n_shots_optimization,
        instance_id=iid,
    )
    qaoa_wall_clock = time.perf_counter() - qaoa_start

    io_utils.save_angles_artifact(
        instance,
        {
            "instance_id": instance.instance_id,
            "optimal_p": qaoa_result.optimal_p,
            "optimal_betas": qaoa_result.optimal_betas,
            "optimal_gammas": qaoa_result.optimal_gammas,
            "approx_ratio_at_optimal_p": qaoa_result.approx_ratio_at_optimal_p,
            "threshold_met": qaoa_result.threshold_met,
            "sweep": [
                {
                    "p": r.p,
                    "betas": r.betas,
                    "gammas": r.gammas,
                    "expected_cost": r.expected_cost,
                    "best_sampled_objective": r.best_sampled_objective,
                    "approx_ratio": r.approx_ratio,
                    "wall_clock_seconds": r.wall_clock_seconds,
                }
                for r in qaoa_result.sweep
            ],
        },
        base_dir=config.angles_dir,
    )

    total_wall_clock = time.perf_counter() - pipeline_start
    logger.info(
        "[%s] === pipeline complete in %.2fs (heuristic=%.4g ilp=%.4g qaoa=%.4g "
        "optimal_p=%d ratio=%.4f threshold_met=%s) ===",
        iid, total_wall_clock, heuristic_result.objective, ilp_result.objective,
        qaoa_result.best_sampled_objective_at_optimal_p, qaoa_result.optimal_p,
        qaoa_result.approx_ratio_at_optimal_p, qaoa_result.threshold_met,
    )

    row = {
        "instance_id": instance.instance_id,
        "problem_type": instance.problem_type,
        "n_vertices": instance.n_vertices,
        "n_qubits_used": n_qubits_used,
        "n_triangles": len(triangles),
        "seed": instance.seed,
        "heuristic_value": heuristic_result.objective,
        "ilp_optimal_value": ilp_result.objective,
        "ilp_solver_backend": ilp_result.solver_name,
        "qubo_optimal_value": qubo_result.objective if qubo_result else None,
        "qaoa_value": qaoa_result.best_sampled_objective_at_optimal_p,
        "optimal_p": qaoa_result.optimal_p,
        "approx_ratio_at_optimal_p": qaoa_result.approx_ratio_at_optimal_p,
        "threshold_met": qaoa_result.threshold_met,
        "prune_wall_clock_seconds": prune_wall_clock,
        "build_pubo_wall_clock_seconds": build_pubo_wall_clock,
        "build_qubo_wall_clock_seconds": build_qubo_wall_clock,
        "build_ilp_wall_clock_seconds": build_ilp_wall_clock,
        "heuristic_wall_clock_seconds": heuristic_result.wall_clock_seconds,
        "ilp_wall_clock_seconds": ilp_wall_clock,
        "qubo_wall_clock_seconds": qubo_wall_clock,
        "qaoa_wall_clock_seconds": qaoa_wall_clock,
        "total_wall_clock_seconds": total_wall_clock,
    }
    io_utils.append_result_row(row, csv_path=config.results_csv)
    return row


def run_experiment(config: ExperimentConfig) -> list[dict]:
    instances = generate_sweep(
        n_min=config.n_min,
        max_vertices=config.max_vertices,
        step=config.step,
        weighted=config.weighted,
        base_seed=config.base_seed,
        n_seeds_per_size=config.n_seeds_per_size,
        weight_low=config.weight_low,
        weight_high=config.weight_high,
    )
    logger.info(
        "Experiment: %d instance(s) to run (problem_type=%s, n_min=%d, max_vertices=%d, step=%d)",
        len(instances), config.problem_type, config.n_min, config.max_vertices, config.step,
    )
    experiment_start = time.perf_counter()
    rows = []
    for i, instance in enumerate(instances, start=1):
        io_utils.save_instance(instance, base_dir=config.instances_dir)
        logger.info(
            "Experiment progress: instance %d/%d (%s)",
            i, len(instances), instance.instance_id,
        )
        rows.append(run_instance(instance, config))
    logger.info(
        "Experiment done: %d instance(s) in %.2fs",
        len(instances), time.perf_counter() - experiment_start,
    )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the TVD-QAOA experiment pipeline")
    parser.add_argument("--config", type=str, help="Path to a JSON experiment config")
    parser.add_argument("--problem-type", choices=["weighted", "unweighted"], default="unweighted")
    parser.add_argument("--max-vertices", type=int, default=10)
    parser.add_argument("--n-min", type=int, default=4)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--n-seeds-per-size", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--p-max", type=int, default=3)
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_json(args.config)
    else:
        config = ExperimentConfig(
            problem_type=args.problem_type,
            max_vertices=args.max_vertices,
            n_min=args.n_min,
            step=args.step,
            n_seeds_per_size=args.n_seeds_per_size,
            base_seed=args.base_seed,
            p_max=args.p_max,
        )
    run_experiment(config)


if __name__ == "__main__":
    main()

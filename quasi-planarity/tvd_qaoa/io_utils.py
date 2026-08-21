"""Save/load instances and results (spec §3 persistence, §7 results schema)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from .instances import Instance

DATA_DIR = Path("data")
INSTANCES_DIR = DATA_DIR / "instances"
RESULTS_DIR = DATA_DIR / "results"
ANGLES_DIR = DATA_DIR / "angles"


def instance_path(instance: Instance, base_dir: Path = INSTANCES_DIR) -> Path:
    return base_dir / instance.problem_type / f"{instance.instance_id}.json"


def save_instance(instance: Instance, base_dir: Path = INSTANCES_DIR) -> Path:
    path = instance_path(instance, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance.instance_id,
        "problem_type": instance.problem_type,
        "n_vertices": instance.n_vertices,
        "seed": instance.seed,
        "generator_version": instance.generator_version,
        "generation_params": instance.generation_params,
        "vertices": sorted(instance.graph.nodes()),
        "edges": sorted(tuple(sorted(e)) for e in instance.graph.edges()),
        "weights": {str(v): w for v, w in instance.weights.items()},
        "triangles": [list(t) for t in instance.triangles],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_instance(path: Path) -> Instance:
    payload = json.loads(Path(path).read_text())
    graph = nx.Graph()
    graph.add_nodes_from(payload["vertices"])
    graph.add_edges_from(payload["edges"])
    weights = {int(v): w for v, w in payload["weights"].items()}
    triangles = [tuple(t) for t in payload["triangles"]]
    return Instance(
        instance_id=payload["instance_id"],
        problem_type=payload["problem_type"],
        n_vertices=payload["n_vertices"],
        seed=payload["seed"],
        graph=graph,
        weights=weights,
        triangles=triangles,
        generation_params=payload["generation_params"],
        generator_version=payload["generator_version"],
    )


def save_angles_artifact(instance: Instance, qaoa_result: dict, base_dir: Path = ANGLES_DIR) -> Path:
    """Per-instance JSON artifact: optimized (beta*, gamma*) at the chosen p,
    plus the full per-p sweep, keyed by the same instance_id as §3's naming."""
    path = base_dir / instance.problem_type / f"{instance.instance_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qaoa_result, indent=2))
    return path


RESULTS_COLUMNS = [
    "instance_id",
    "problem_type",
    "n_vertices",
    "n_qubits_used",
    "n_triangles",
    "seed",
    "heuristic_value",
    "ilp_optimal_value",
    "ilp_solver_backend",
    "qubo_optimal_value",
    "qaoa_value",
    "optimal_p",
    "approx_ratio_at_optimal_p",
    "threshold_met",
    "prune_wall_clock_seconds",
    "build_pubo_wall_clock_seconds",
    "build_qubo_wall_clock_seconds",
    "build_ilp_wall_clock_seconds",
    "heuristic_wall_clock_seconds",
    "ilp_wall_clock_seconds",
    "qubo_wall_clock_seconds",
    "qaoa_wall_clock_seconds",
    "total_wall_clock_seconds",
]


def append_result_row(row: dict, csv_path: Path = RESULTS_DIR / "results.csv") -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([{c: row.get(c) for c in RESULTS_COLUMNS}])
    if csv_path.exists():
        existing_header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if existing_header != RESULTS_COLUMNS:
            raise ValueError(
                f"{csv_path} has header {existing_header}, which no longer "
                f"matches the current RESULTS_COLUMNS {RESULTS_COLUMNS} "
                "(the results schema changed). Move/delete the stale file "
                "before appending, rather than silently misaligning columns."
            )
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def load_results(csv_path: Path = RESULTS_DIR / "results.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

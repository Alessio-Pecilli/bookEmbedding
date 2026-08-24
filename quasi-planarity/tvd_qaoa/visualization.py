"""Per-instance visualization: the original graph next to the same graph
with its best classical (exact ILP) TVD solution highlighted, and -- when a
QAOA result is available -- a third panel with the best QAOA-sampled
solution highlighted.

Standalone usage (visualizes every saved instance under `data/instances`,
solving each exactly via ILP; no QAOA panel, since that requires actually
running the QAOA sweep):

    python -m tvd_qaoa.visualization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from . import io_utils
from .instances import Instance

logger = logging.getLogger(__name__)

VISUALIZATIONS_DIR = Path("data") / "visualizations"

_NODE_COLOR = "#6fa8dc"
_SELECTED_COLOR = "#e06666"
_EDGE_COLOR = "#999999"


def _draw_panel(ax, graph: nx.Graph, pos: dict, weights: dict[int, float], selected: list[int], title: str) -> None:
    selected_set = set(selected)
    node_colors = [_SELECTED_COLOR if v in selected_set else _NODE_COLOR for v in graph.nodes()]
    show_weights = any(w != 1 for w in weights.values())
    labels = {v: (f"{v} ({weights[v]:g})" if show_weights else str(v)) for v in graph.nodes()}

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=_EDGE_COLOR)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_instance_solution(
    instance: Instance,
    classical_vertices: list[int],
    classical_objective: float | None = None,
    classical_solver_name: str | None = None,
    qaoa_vertices: list[int] | None = None,
    qaoa_objective: float | None = None,
    qaoa_p: int | None = None,
    qaoa_approx_ratio: float | None = None,
    output_dir: Path = VISUALIZATIONS_DIR,
) -> Path:
    """Draw ``instance.graph`` on a shared node layout across panels: the
    original graph, the best classical (exact ILP) TVD solution, and --
    when ``qaoa_vertices`` is given -- the best QAOA-sampled solution, each
    with its selected vertex set highlighted. Saved as one PNG at
    ``output_dir/{problem_type}/{instance_id}.png``."""
    graph = instance.graph
    pos = nx.spring_layout(graph, seed=instance.seed)

    n_panels = 3 if qaoa_vertices is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    _draw_panel(axes[0], graph, pos, instance.weights, [], "Original graph")

    classical_title = "Best classical (ILP)"
    if classical_solver_name:
        classical_title += f" [{classical_solver_name}]"
    _draw_panel(axes[1], graph, pos, instance.weights, classical_vertices, classical_title)

    subtitle_parts = [f"classical selected={sorted(classical_vertices)}"]
    if classical_objective is not None:
        subtitle_parts.append(f"classical obj={classical_objective:g}")

    if qaoa_vertices is not None:
        qaoa_title = "Best QAOA" + (f" (p={qaoa_p})" if qaoa_p is not None else "")
        _draw_panel(axes[2], graph, pos, instance.weights, qaoa_vertices, qaoa_title)
        subtitle_parts.append(f"qaoa selected={sorted(qaoa_vertices)}")
        if qaoa_objective is not None:
            subtitle_parts.append(f"qaoa obj={qaoa_objective:g}")
        if qaoa_approx_ratio is not None:
            subtitle_parts.append(f"approx ratio={qaoa_approx_ratio:.3f}")

    fig.suptitle(
        f"{instance.instance_id}  ({instance.problem_type}, n={instance.n_vertices})\n"
        + "  |  ".join(subtitle_parts),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = output_dir / instance.problem_type / f"{instance.instance_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[%s] visualization saved to %s", instance.instance_id, out_path)
    return out_path


def visualize_saved_instances(
    instances_dir: Path = io_utils.INSTANCES_DIR,
    output_dir: Path = VISUALIZATIONS_DIR,
) -> list[Path]:
    """Load every instance JSON under ``instances_dir``, solve it exactly via
    ILP, and save its side-by-side visualization. Returns the list of PNG
    paths written."""
    from . import classical_solvers as cs
    from .formulation import build_ilp

    paths = []
    for instance_path in sorted(Path(instances_dir).glob("*/*.json")):
        instance = io_utils.load_instance(instance_path)
        ilp = build_ilp(instance.graph, instance.weights, instance.triangles)
        result = cs.solve_ilp(ilp)
        paths.append(
            plot_instance_solution(
                instance,
                result.selected_vertices,
                classical_objective=result.objective,
                classical_solver_name=result.solver_name,
                output_dir=output_dir,
            )
        )
    return paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Visualize saved TVD instances and their exact ILP solution")
    parser.add_argument("--instances-dir", type=Path, default=io_utils.INSTANCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=VISUALIZATIONS_DIR)
    args = parser.parse_args()
    paths = visualize_saved_instances(args.instances_dir, args.output_dir)
    logger.info("Wrote %d visualization(s) under %s", len(paths), args.output_dir)


if __name__ == "__main__":
    main()

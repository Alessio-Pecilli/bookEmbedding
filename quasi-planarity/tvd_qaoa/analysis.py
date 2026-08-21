"""Derived tables/plots from experiment results (spec §7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import io_utils


def qubits_vs_size_table(results: pd.DataFrame) -> pd.DataFrame:
    """n_qubits_used vs n_vertices, grouped by problem_type."""
    return (
        results[["problem_type", "n_vertices", "n_qubits_used"]]
        .drop_duplicates()
        .sort_values(["problem_type", "n_vertices"])
        .reset_index(drop=True)
    )


def plot_qubits_vs_size(results: pd.DataFrame, out_path: Path) -> Path:
    table = qubits_vs_size_table(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    for problem_type, group in table.groupby("problem_type"):
        ax.plot(group["n_vertices"], group["n_qubits_used"], marker="o", label=problem_type)
    ax.plot(
        [table["n_vertices"].min(), table["n_vertices"].max()],
        [table["n_vertices"].min(), table["n_vertices"].max()],
        linestyle="--",
        color="gray",
        label="n_qubits = n_vertices",
    )
    ax.set_xlabel("Graph size (n_vertices)")
    ax.set_ylabel("Qubits used (post-pruning)")
    ax.set_title("QAOA qubits used vs. graph size")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_analysis(
    results_csv: Path = io_utils.RESULTS_DIR / "results.csv",
    out_dir: Path = io_utils.RESULTS_DIR / "analysis",
) -> dict[str, Path]:
    results = io_utils.load_results(results_csv)
    table = qubits_vs_size_table(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "qubits_vs_size.csv"
    table.to_csv(table_path, index=False)
    plot_path = plot_qubits_vs_size(results, out_dir / "qubits_vs_size.png")
    return {"table": table_path, "plot": plot_path}


if __name__ == "__main__":
    paths = run_analysis()
    print(paths)

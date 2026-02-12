from src.pipeline import run_pipeline
from src.visualization import visualize


if __name__ == "__main__":

    G, C, C_reduced, pos, edge_labels, full_solution = run_pipeline(n_nodes=6, edge_prob=1.0)

    if full_solution is not None:
        visualize(G, C, C_reduced, pos, edge_labels, full_solution)
        print("Vertices kept in crossings graph:", sum(full_solution.values()))
    else:
        print("No QAOA was run because there were no triangles.")



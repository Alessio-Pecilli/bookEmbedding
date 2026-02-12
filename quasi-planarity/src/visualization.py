import matplotlib.pyplot as plt
import networkx as nx
import itertools
from .triangle_detection import find_triangles
from .utils import stage, done


def visualize(G, C, C_reduced, pos, edge_labels, full_solution=None):
    # Create a figure with subplots for all visualizations
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()  # Flatten to 1D for easier indexing
    plot_idx = 0

    # ---------------------------
    # Original graph
    # ---------------------------
    ax = axes[plot_idx]
    plot_idx += 1
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("Original Graph with Edge Labels")

    # ---------------------------
    # Original graph after QAOA
    # ---------------------------
    if full_solution is not None:
        edges_to_keep = [e for idx, e in enumerate(G.edges()) if full_solution.get(idx, 1) == 1]
        edges_removed = [e for idx, e in enumerate(G.edges()) if full_solution.get(idx, 1) == 0]

        if len(edges_removed) > 0:
            removed_labels = [edge_labels[e] for e in edges_removed]
            print("Edges removed by QAOA:", removed_labels)
        else:
            print("No edges were removed by QAOA.")

        G_qaoa = nx.Graph()
        G_qaoa.add_nodes_from(G.nodes())
        G_qaoa.add_edges_from(edges_to_keep)

        ax = axes[plot_idx]
        plot_idx += 1
        nx.draw(G_qaoa, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500)
        kept_labels = {e: edge_labels[e] for e in edges_to_keep}
        nx.draw_networkx_edge_labels(G_qaoa, pos, edge_labels=kept_labels, ax=ax)
        ax.set_title("Original Graph After QAOA (Removed edges omitted)")

    # ---------------------------
    # Crossings graph before QAOA
    # ---------------------------
    C_noniso = C.subgraph([n for n in C.nodes() if C.degree[n] > 0]).copy()
    if len(C_noniso) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        posC = nx.spring_layout(C_noniso, seed=42)
        labelsC = {n: C_noniso.nodes[n]["label"] for n in C_noniso.nodes()}

        # Highlight triangle nodes
        triangles = find_triangles(C_noniso)
        triangle_nodes = set(itertools.chain.from_iterable(triangles))
        node_colors = ['red' if n in triangle_nodes else 'orange' for n in C_noniso.nodes()]

        nx.draw(C_noniso, posC, ax=ax, with_labels=False, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(C_noniso, posC, labelsC, ax=ax)
        ax.set_title("Crossings Graph Before QAOA (Red = triangle nodes)")

        # Print all triangles and indicate removed vertices
        if full_solution is not None:
            print("\nTriangles and removed vertices:")
            for tri in triangles:
                removed_in_tri = [v for v in tri if full_solution.get(v, 1) == 0]
                kept_in_tri = [v for v in tri if full_solution.get(v, 1) == 1]
                tri_labels = [C.nodes[v]["label"] for v in tri]
                removed_labels = [C.nodes[v]["label"] for v in removed_in_tri]
                kept_labels_tri = [C.nodes[v]["label"] for v in kept_in_tri]
                print(f"Triangle {tri_labels} -> kept: {kept_labels_tri}, removed: {removed_labels}")
    else:
        print("No non-isolated nodes to display in crossings graph.")

    # ---------------------------
    # Reduced crossings graph BEFORE QAOA
    # ---------------------------
    if C_reduced is not None:
        ax = axes[plot_idx]
        plot_idx += 1
        # Keep all nodes (even isolated) in reduced graph
        posC_reduced_before = nx.spring_layout(C_reduced, seed=42)
        labelsC_reduced = {n: C_reduced.nodes[n]["label"] for n in C_reduced.nodes()}
        # Triangle nodes are all nodes in C_reduced
        node_colors_reduced_before = ['red' for n in C_reduced.nodes()]
        nx.draw(C_reduced, posC_reduced_before, ax=ax, with_labels=False, node_color=node_colors_reduced_before, node_size=500)
        nx.draw_networkx_labels(C_reduced, posC_reduced_before, labelsC_reduced, ax=ax)
        ax.set_title("Reduced Crossings Graph BEFORE QAOA (Only triangle vertices)")

    # ---------------------------
    # Reduced crossings graph AFTER QAOA
    # ---------------------------
    if C_reduced is not None and full_solution is not None:
        kept_nodes = [n for n in C_reduced.nodes() if full_solution.get(n, 1) == 1]
        removed_nodes = [n for n in C_reduced.nodes() if full_solution.get(n, 1) == 0]
        if len(kept_nodes + removed_nodes) == 0:
            print("No nodes in reduced crossings graph to display.")
        else:
            C_red_kept = C_reduced.subgraph(kept_nodes + removed_nodes).copy()
            ax = axes[plot_idx]
            plot_idx += 1
            posC2 = nx.spring_layout(C_red_kept, seed=42)
            labelsC2 = {n: C_red_kept.nodes[n]["label"] for n in C_red_kept.nodes()}

            triangles_reduced = find_triangles(C_red_kept)
            triangle_nodes_reduced = set(itertools.chain.from_iterable(triangles_reduced))

            # Color: red = triangle node kept, gray = triangle node removed, green = non-triangle node kept
            node_colors2 = []
            for n in C_red_kept.nodes():
                if n in triangle_nodes_reduced:
                    if full_solution.get(n, 1) == 1:
                        node_colors2.append('red')
                    else:
                        node_colors2.append('gray')
                else:
                    node_colors2.append('green')

            nx.draw(C_red_kept, posC2, ax=ax, with_labels=False, node_color=node_colors2, node_size=500)
            nx.draw_networkx_labels(C_red_kept, posC2, labelsC2, ax=ax)
            ax.set_title("Reduced Crossings Graph After QAOA (Red=kept triangle nodes, Gray=removed triangle nodes)")

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()




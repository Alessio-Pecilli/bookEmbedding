import itertools
import networkx as nx

from .graph_generation import generate_nonplanar_graph
from .triangle_detection import find_triangles
from .qaoa import run_qaoa_component
from .utils import stage, done

def run_pipeline(n_nodes=10, edge_prob=0.4, layout_type='random', lam=3.0):
    stage_time = stage("Generating graph and crossings graph")
    G, C, pos, edge_labels = generate_nonplanar_graph(n_nodes, edge_prob, layout_type)
    done(stage_time)

    triangles = find_triangles(C)
    triangle_nodes = set(itertools.chain.from_iterable(triangles))

    if len(triangle_nodes) == 0:
        print("No triangles in crossings graph. Routine stopped.")
        return G, C, None, pos, edge_labels, None

    stage_time2 = stage("Reducing crossings graph to triangle vertices")
    C_reduced = C.subgraph(triangle_nodes).copy()
    done(stage_time2)

    # Recompute connected components on reduced graph
    stage_time3 = stage("Finding connected components on reduced graph")
    components = list(nx.connected_components(C_reduced))
    triangles_per_component = []
    components_with_triangles = []
    for comp in components:
        tri_in_comp = [t for t in triangles if set(t).issubset(comp)]
        if len(tri_in_comp) > 0:
            components_with_triangles.append(comp)
            triangles_per_component.append(tri_in_comp)
    done(stage_time3)

    stage_time4 = stage("Running QAOA on components")
    full_solution = {n:1 for n in triangle_nodes}  # only triangle vertices exist now
    for comp_idx, comp_nodes in enumerate(components_with_triangles):
        subgraph = C_reduced.subgraph(comp_nodes).copy()
        node_map = {old: new for new, old in enumerate(subgraph.nodes())}
        subgraph = nx.relabel_nodes(subgraph, node_map)
        comp_triangles = [[node_map[v] for v in t] for t in triangles_per_component[comp_idx]]
        original_nodes = list(components_with_triangles[comp_idx])

        best_sub, energy_sub = run_qaoa_component(
            subgraph,
            comp_triangles,
            original_nodes,
            edge_labels,
            lam=lam
        )

        for sub_idx, old_node in enumerate(components_with_triangles[comp_idx]):
            full_solution[old_node] = best_sub[sub_idx]
    done(stage_time4)

    return G, C, C_reduced, pos, edge_labels, full_solution
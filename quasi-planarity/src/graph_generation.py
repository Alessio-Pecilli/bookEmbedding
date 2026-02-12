import networkx as nx
import numpy as np
import itertools

def generate_nonplanar_graph(n_nodes=10, edge_prob=0.4, layout_type='random', random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if layout_type == 'random':
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=random_seed)
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        positions = nx.spring_layout(G, seed=random_seed)
    elif layout_type == 'bipartite':
        n_left = n_nodes // 2
        n_right = n_nodes - n_left
        G = nx.complete_bipartite_graph(n_left, n_right)
        positions = nx.spring_layout(G, seed=random_seed)
    else:
        raise ValueError("layout_type must be 'random' or 'bipartite'")

    nx.set_node_attributes(G, positions, 'pos')
    edges = list(G.edges())
    edge_labels = {e: f"e{i}" for i, e in enumerate(edges)}

    # Build crossings graph
    def segments_intersect(p1,p2,p3,p4):
        def orient(a,b,c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        o1 = orient(p1,p2,p3)
        o2 = orient(p1,p2,p4)
        o3 = orient(p3,p4,p1)
        o4 = orient(p3,p4,p2)
        return o1*o2 <0 and o3*o4 <0

    C = nx.Graph()
    for idx, e in enumerate(edges):
        C.add_node(idx, label=f"e{idx}")

    crossings = 0
    for (i,e1),(j,e2) in itertools.combinations(list(enumerate(edges)), 2):
        u1,v1 = e1
        u2,v2 = e2
        if len({u1,v1,u2,v2}) < 4:
            continue
        p1,p2 = positions[u1], positions[v1]
        p3,p4 = positions[u2], positions[v2]
        if segments_intersect(p1,p2,p3,p4):
            C.add_edge(i,j)
            crossings += 1

    print(f"Graph generated with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Crossings graph has {C.number_of_nodes()} nodes, {crossings} edges")
    return G, C, positions, edge_labels

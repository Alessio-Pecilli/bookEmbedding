"""Quasi-planarity instance generation: caterpillar -> two-layer/one-page
drawing -> crossing graph -> auxiliary TVD graph.

TVD is a reformulation of restoring quasi-planarity to a drawn graph: a
drawing is 3-quasi-planar iff no 3 of its edges pairwise cross, so the
minimum edge set whose removal makes it quasi-planar is exactly a minimum
hitting set over every triple of pairwise-crossing edges. Model that as TVD
by building an auxiliary graph AG: one AG-vertex per drawn edge that takes
part in >=1 triple of pairwise-crossing edges, one AG-edge per pair of drawn
edges that cross as part of such a triple. A triangle in AG is exactly a
triple of pairwise-crossing edges, so "delete a minimum-weight vertex set to
make AG triangle-free" is exactly "delete a minimum-weight edge set to make
the drawing quasi-planar" -- the existing TVD pipeline (formulation.py,
classical_solvers.py, qaoa.py) applies unchanged once AG is built.

Pipeline (this module):
  1. generate_caterpillar    -- random spine + random legs per spine vertex
  2. two_layer_bipartition / random_one_page_position -- random drawing
  3. build_crossing_graph_*  -- pairwise edge-crossing test for that drawing
  4. build_auxiliary_graph   -- restrict the crossing graph to edges that lie
                                 in >=1 triangle (spec above), relabel to
                                 0..n-1 contiguous vertex ids
  5. generate_quasi_planarity_instance / _sweep -- wrap into Instance objects,
     same shape instances.py produces, so experiment.py needs no changes to
     consume them.
"""

from __future__ import annotations

import logging
from itertools import combinations

import networkx as nx
import numpy as np

from .formulation import Triangle, enumerate_triangles
from .instances import GENERATOR_VERSION, Instance, _generate_weights

logger = logging.getLogger(__name__)

_MAX_DRAWING_RETRIES = 25

CaterpillarEdge = tuple[int, int]


# ---------------------------------------------------------------------------
# 1. Caterpillar generation
# ---------------------------------------------------------------------------


def generate_caterpillar(
    spine_length: int,
    legs_low: int,
    legs_high: int,
    rng: np.random.Generator,
) -> tuple[nx.Graph, list[int], dict]:
    """A random caterpillar: a `spine_length`-vertex path (the spine), each
    spine vertex given a random number of pendant leaves (legs) drawn
    uniformly from [legs_low, legs_high]. Every vertex is within distance 1
    of the spine by construction."""
    if spine_length < 2:
        raise ValueError("a caterpillar needs a spine of at least 2 vertices")
    if legs_low < 0 or legs_high < legs_low:
        raise ValueError(f"invalid legs range [{legs_low}, {legs_high}]")

    graph = nx.Graph()
    spine = list(range(spine_length))
    graph.add_nodes_from(spine)
    graph.add_edges_from(zip(spine[:-1], spine[1:]))

    next_id = spine_length
    legs_per_spine: dict[int, int] = {}
    for s in spine:
        n_legs = int(rng.integers(legs_low, legs_high + 1))
        legs_per_spine[s] = n_legs
        for _ in range(n_legs):
            graph.add_edge(s, next_id)
            next_id += 1

    params = {
        "spine_length": spine_length,
        "legs_low": legs_low,
        "legs_high": legs_high,
        "legs_per_spine": legs_per_spine,
        "n_vertices": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }
    return graph, spine, params


# ---------------------------------------------------------------------------
# 2. Random drawing assignment
# ---------------------------------------------------------------------------


def two_layer_bipartition(graph: nx.Graph, rng: np.random.Generator) -> dict[int, int]:
    """Assign every vertex to layer 0 or 1 via the tree's natural
    bipartition (BFS 2-coloring from an arbitrary root), so every edge
    always goes between the two layers -- the only randomness is which side
    of the bipartition lands on layer 0 vs layer 1."""
    root = next(iter(graph.nodes()))
    color = nx.bipartite.color(graph) if nx.is_connected(graph) else {}
    if not color:
        # disconnected fallback (shouldn't happen for a caterpillar): BFS per component
        color = {}
        for component in nx.connected_components(graph):
            sub_root = next(iter(component))
            for v, d in nx.single_source_shortest_path_length(graph, sub_root).items():
                color[v] = d % 2
    flip = bool(rng.integers(0, 2))
    return {v: (c if not flip else 1 - c) for v, c in color.items()}


def random_layer_positions(
    graph: nx.Graph, layer: dict[int, int], rng: np.random.Generator
) -> dict[int, int]:
    """Independent random left-to-right order (position index) of the
    vertices within each layer."""
    position: dict[int, int] = {}
    for layer_id in (0, 1):
        verts = [v for v in graph.nodes() if layer[v] == layer_id]
        order = rng.permutation(len(verts))
        for pos, idx in enumerate(order):
            position[verts[idx]] = pos
    return position


def random_one_page_position(graph: nx.Graph, rng: np.random.Generator) -> dict[int, int]:
    """A uniformly random spine order (permutation) of every vertex, for the
    one-page (single spine, arcs above it) drawing."""
    verts = list(graph.nodes())
    order = rng.permutation(len(verts))
    return {verts[idx]: pos for pos, idx in enumerate(order)}


# ---------------------------------------------------------------------------
# 3. Pairwise edge-crossing tests -> crossing graph
# ---------------------------------------------------------------------------


def _edges_cross_two_layer(
    edge_a: CaterpillarEdge,
    edge_b: CaterpillarEdge,
    layer: dict[int, int],
    position: dict[int, int],
) -> bool:
    """Two edges in a 2-layer drawing (each edge has one endpoint per layer,
    guaranteed by the bipartition assignment) cross iff their layer-0/layer-1
    endpoint order is reversed. Edges sharing an endpoint never cross (the
    shared endpoint gives equal position on that layer, so the product is
    0, not negative)."""

    def endpoints(edge: CaterpillarEdge) -> tuple[int, int]:
        u, v = edge
        return (u, v) if layer[u] == 0 else (v, u)  # (layer-0 endpoint, layer-1 endpoint)

    a0, a1 = endpoints(edge_a)
    b0, b1 = endpoints(edge_b)
    return (position[a0] - position[b0]) * (position[a1] - position[b1]) < 0


def _edges_cross_one_page(
    edge_a: CaterpillarEdge, edge_b: CaterpillarEdge, position: dict[int, int]
) -> bool:
    """Standard book-embedding crossing rule: two spine chords cross iff
    exactly one endpoint of one lies strictly between the endpoints of the
    other. Edges sharing an endpoint never cross."""
    lo1, hi1 = sorted((position[edge_a[0]], position[edge_a[1]]))
    lo2, hi2 = sorted((position[edge_b[0]], position[edge_b[1]]))
    if len({lo1, hi1, lo2, hi2}) < 4:
        return False
    return (lo1 < lo2 < hi1 < hi2) or (lo2 < lo1 < hi2 < hi1)


def build_crossing_graph_two_layer(
    graph: nx.Graph, layer: dict[int, int], position: dict[int, int]
) -> nx.Graph:
    """Crossing graph: one node per caterpillar edge, one link per pair of
    caterpillar edges that cross under this 2-layer drawing."""
    edges = [tuple(sorted(e)) for e in graph.edges()]
    crossing_graph = nx.Graph()
    crossing_graph.add_nodes_from(edges)
    for edge_a, edge_b in combinations(edges, 2):
        if _edges_cross_two_layer(edge_a, edge_b, layer, position):
            crossing_graph.add_edge(edge_a, edge_b)
    return crossing_graph


def build_crossing_graph_one_page(graph: nx.Graph, position: dict[int, int]) -> nx.Graph:
    edges = [tuple(sorted(e)) for e in graph.edges()]
    crossing_graph = nx.Graph()
    crossing_graph.add_nodes_from(edges)
    for edge_a, edge_b in combinations(edges, 2):
        if _edges_cross_one_page(edge_a, edge_b, position):
            crossing_graph.add_edge(edge_a, edge_b)
    return crossing_graph


# ---------------------------------------------------------------------------
# 4. Auxiliary TVD graph: restrict the crossing graph to triangle-edges only
# ---------------------------------------------------------------------------


def build_auxiliary_graph(crossing_graph: nx.Graph) -> tuple[nx.Graph, list[Triangle]]:
    """One AG-vertex per crossing-graph vertex (= drawn edge) that lies in
    >=1 triangle (= 3 pairwise-crossing drawn edges), one AG-edge per
    crossing-graph edge that itself lies in >=1 such triangle -- i.e. only
    crossings that are part of an actual 3-pairwise-crossing triple, not
    every crossing between two triangle-participating edges."""
    triangles = enumerate_triangles(crossing_graph)
    vertices_in_triangle: set = set()
    edges_in_triangle: set[tuple] = set()
    for u, v, w in triangles:
        vertices_in_triangle.update((u, v, w))
        edges_in_triangle.update(
            (tuple(sorted((u, v))), tuple(sorted((u, w))), tuple(sorted((v, w))))
        )
    aux = nx.Graph()
    aux.add_nodes_from(vertices_in_triangle)
    aux.add_edges_from(edges_in_triangle)
    return aux, triangles


# ---------------------------------------------------------------------------
# 5. Full instance assembly
# ---------------------------------------------------------------------------


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_quasi_planarity_instance(
    spine_length: int,
    seed: int,
    weighted: bool,
    drawing: str,
    legs_low: int = 1,
    legs_high: int = 3,
    weight_low: int = 1,
    weight_high: int = 10,
    max_retries: int = _MAX_DRAWING_RETRIES,
) -> Instance:
    """Generate one TVD instance via the quasi-planarity reformulation:
    random caterpillar -> random 2-layer/1-page drawing -> crossing graph ->
    auxiliary graph (spec at module top). `drawing` is "two_layer" or
    "one_page".

    Same (spine_length, seed, weighted, drawing, legs_low, legs_high,
    weight_low, weight_high) always yields a byte-identical instance. If a
    drawn caterpillar happens to already be quasi-planar (no 3
    pairwise-crossing edges at all -- possible for small/sparse
    caterpillars), a fresh random drawing of the *same* caterpillar is
    retried up to `max_retries` times before giving up.
    """
    if drawing not in ("two_layer", "one_page"):
        raise ValueError(f"drawing must be 'two_layer' or 'one_page', got {drawing!r}")

    ss = np.random.SeedSequence(seed)
    cat_seed, drawing_seed, weight_seed = ss.spawn(3)
    cat_rng = np.random.default_rng(cat_seed)
    drawing_rng = np.random.default_rng(drawing_seed)
    weight_rng = np.random.default_rng(weight_seed)

    caterpillar, spine, cat_params = generate_caterpillar(spine_length, legs_low, legs_high, cat_rng)

    aux = None
    triangles_cg: list[Triangle] = []
    drawing_params: dict = {}
    for attempt in range(1, max_retries + 1):
        if drawing == "two_layer":
            layer = two_layer_bipartition(caterpillar, drawing_rng)
            position = random_layer_positions(caterpillar, layer, drawing_rng)
            crossing_graph = build_crossing_graph_two_layer(caterpillar, layer, position)
            drawing_params = {
                "drawing": "two_layer",
                "layer": {str(v): l for v, l in layer.items()},
                "position": {str(v): p for v, p in position.items()},
            }
        else:
            position = random_one_page_position(caterpillar, drawing_rng)
            crossing_graph = build_crossing_graph_one_page(caterpillar, position)
            drawing_params = {
                "drawing": "one_page",
                "position": {str(v): p for v, p in position.items()},
            }

        candidate_aux, triangles_cg = build_auxiliary_graph(crossing_graph)
        if candidate_aux.number_of_nodes() > 0:
            aux = candidate_aux
            drawing_params["attempts"] = attempt
            break
        logger.info(
            "generate_quasi_planarity_instance: attempt %d/%d drawing has no "
            "3-pairwise-crossing edges, retrying",
            attempt, max_retries,
        )

    if aux is None:
        raise RuntimeError(
            f"generate_quasi_planarity_instance: no drawing of the "
            f"spine_length={spine_length} caterpillar (seed={seed}, "
            f"drawing={drawing}) produced a 3-pairwise-crossing edge triple "
            f"after {max_retries} retries -- try a larger spine/legs range."
        )

    relabeled = nx.convert_node_labels_to_integers(
        aux, ordering="sorted", label_attribute="caterpillar_edge"
    )
    triangles = enumerate_triangles(relabeled)
    vertices = sorted(relabeled.nodes())

    if weighted:
        weights = _generate_weights(vertices, weight_rng, weight_low, weight_high)
    else:
        weights = {v: 1.0 for v in vertices}

    edge_labels = {
        str(v): list(relabeled.nodes[v]["caterpillar_edge"]) for v in relabeled.nodes()
    }
    for v in relabeled.nodes():
        del relabeled.nodes[v]["caterpillar_edge"]  # keep the graph plain (matches instances.py)

    problem_type = "weighted" if weighted else "unweighted"
    instance_id = f"cat_spine{spine_length:03d}_{drawing}_seed{seed}"
    gen_params = {
        "source": "quasi_planarity",
        "caterpillar": cat_params,
        **drawing_params,
        "aux_vertex_to_caterpillar_edge": edge_labels,
    }
    if weighted:
        gen_params.update({"weight_low": weight_low, "weight_high": weight_high})

    return Instance(
        instance_id=instance_id,
        problem_type=problem_type,
        n_vertices=len(vertices),
        seed=seed,
        graph=relabeled,
        weights=weights,
        triangles=triangles,
        generation_params=gen_params,
        generator_version=GENERATOR_VERSION,
    )


def generate_quasi_planarity_sweep(
    spine_lengths: list[int],
    drawing: str,
    weighted: bool,
    base_seed: int,
    n_seeds_per_size: int = 1,
    legs_low: int = 1,
    legs_high: int = 3,
    weight_low: int = 1,
    weight_high: int = 10,
) -> list[Instance]:
    """Reproducible sweep across `spine_lengths`, `n_seeds_per_size` random
    caterpillar+drawing seeds per spine length. Per-instance seeds are
    derived deterministically from `base_seed` (same scheme as
    `instances.generate_sweep`), so the whole sweep reproduces from
    `base_seed` alone."""
    n_seeds_needed = len(spine_lengths) * n_seeds_per_size
    child_seeds = np.random.SeedSequence(base_seed).spawn(n_seeds_needed)
    seed_ints = [int(s.generate_state(1)[0]) for s in child_seeds]

    instances = []
    idx = 0
    for spine_length in spine_lengths:
        for _ in range(n_seeds_per_size):
            seed = seed_ints[idx]
            idx += 1
            instances.append(
                generate_quasi_planarity_instance(
                    spine_length,
                    seed,
                    weighted,
                    drawing,
                    legs_low=legs_low,
                    legs_high=legs_high,
                    weight_low=weight_low,
                    weight_high=weight_high,
                )
            )
    return instances

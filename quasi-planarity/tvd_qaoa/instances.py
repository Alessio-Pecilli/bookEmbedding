"""Reproducible instance generation for TVD (spec §3).

Both problem variants (weighted / unweighted) reuse the same graph generator;
only the weight vector differs. Every stochastic step is driven by an
explicit seed (via ``numpy.random.SeedSequence``/``default_rng``) passed
through the call chain -- no global RNG state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .formulation import enumerate_triangles, prune_zero_triangle_vertices

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "1.1"

# Retry/generation defaults for the K3-cluster-stitching triangle-rich model.
_DEFAULT_EXTRA_EDGE_PROB = 0.3   # density of extra "stitching" edges between clusters
_MAX_RETRIES = 25
_EXTRA_EDGE_GROWTH = 1.2         # bump factor applied to extra_edge_prob on each retry


@dataclass
class Instance:
    instance_id: str
    problem_type: str          # "weighted" or "unweighted"
    n_vertices: int
    seed: int
    graph: nx.Graph
    weights: dict[int, float]
    triangles: list[tuple[int, int, int]]
    generation_params: dict = field(default_factory=dict)
    generator_version: str = GENERATOR_VERSION


def _make_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent, reproducible RNG streams for graph topology and weights,
    both deterministically derived from a single instance seed."""
    ss = np.random.SeedSequence(seed)
    graph_seed, weight_seed = ss.spawn(2)
    return np.random.default_rng(graph_seed), np.random.default_rng(weight_seed)


def _generate_candidate_graph(n: int, rng: np.random.Generator, extra_edge_prob: float) -> nx.Graph:
    """Triangle-rich random model: partition vertices into K3 clusters (any
    leftover 1-2 vertices are attached onto an existing triangle's edge, so
    they too close a triangle), then stitch the clusters together with extra
    random edges at density `extra_edge_prob`.

    Unlike ``networkx.powerlaw_cluster_graph`` -- whose first `m` seed nodes
    start with *no* edges among themselves, which leaves nodes triangle-free
    whenever n is small relative to m -- this construction guarantees every
    vertex is in >=1 triangle by construction, for any n >= 3.
    """
    if n < 3:
        raise ValueError("TVD instances need at least 3 vertices to admit a triangle")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    i = 0
    while i + 3 <= n:
        u, v, w = i, i + 1, i + 2
        graph.add_edges_from([(u, v), (v, w), (u, w)])
        i += 3
    remainder = n - i
    if remainder > 0:
        anchor_u, anchor_v = i - 3, i - 2  # an edge from the last-built triangle
        for extra in range(i, n):
            graph.add_edges_from([(extra, anchor_u), (extra, anchor_v)])

    for a in range(n):
        for b in range(a + 1, n):
            if not graph.has_edge(a, b) and rng.random() < extra_edge_prob:
                graph.add_edge(a, b)
    return graph


def generate_graph_with_all_vertices_in_triangles(
    n: int,
    rng: np.random.Generator,
    max_retries: int = _MAX_RETRIES,
) -> tuple[nx.Graph, list[tuple[int, int, int]], dict]:
    """Generate an n-vertex graph in which every vertex participates in at
    least one triangle, via K3-cluster stitching (see
    `_generate_candidate_graph`).

    Retry logic (documented per spec §3, kept as a defensive fallback even
    though the K3-cluster construction guarantees the property by itself):
    start from extra_edge_prob=0.3; after each attempt whose pruned vertex
    count is < n, bump extra_edge_prob by 20% (capped at 1.0) and retry.
    Raise if no attempt within `max_retries` yields a graph where all n
    vertices survive pruning -- the generator never silently returns a
    smaller graph.
    """
    extra_edge_prob = _DEFAULT_EXTRA_EDGE_PROB
    last_seen = -1
    for attempt in range(1, max_retries + 1):
        candidate = _generate_candidate_graph(n, rng, extra_edge_prob)
        triangles = enumerate_triangles(candidate)
        pruned, removed = prune_zero_triangle_vertices(candidate, triangles)
        if pruned.number_of_nodes() == n:
            relabeled = nx.convert_node_labels_to_integers(pruned, ordering="sorted")
            triangles = enumerate_triangles(relabeled)
            params = {
                "extra_edge_prob": extra_edge_prob,
                "attempts": attempt,
                "model": "k3_cluster_stitched",
            }
            return relabeled, triangles, params
        last_seen = pruned.number_of_nodes()
        extra_edge_prob = min(1.0, extra_edge_prob * _EXTRA_EDGE_GROWTH)
    raise RuntimeError(
        f"generate_graph_with_all_vertices_in_triangles: could not produce an "
        f"n={n} graph with every vertex in >=1 triangle after {max_retries} "
        f"retries (last attempt retained {last_seen}/{n} vertices after pruning)."
    )


def _generate_weights(
    vertices: list[int], rng: np.random.Generator, low: int = 1, high: int = 10
) -> dict[int, float]:
    draws = rng.integers(low, high + 1, size=len(vertices))
    return {v: float(w) for v, w in zip(vertices, draws)}


def generate_instance(
    n: int,
    seed: int,
    weighted: bool,
    weight_low: int = 1,
    weight_high: int = 10,
    max_retries: int = _MAX_RETRIES,
) -> Instance:
    """Generate a single TVD instance. Same (n, seed, weighted, weight_low,
    weight_high) always yields a byte-identical instance.

    The unweighted variant reuses the exact same graph topology draw as the
    weighted variant would for the same seed (both derive the graph from the
    same `graph_seed` sub-stream); only whether the weight sub-stream is
    consumed differs.
    """
    graph_rng, weight_rng = _make_rngs(seed)
    graph, triangles, gen_params = generate_graph_with_all_vertices_in_triangles(
        n, graph_rng, max_retries=max_retries
    )
    vertices = sorted(graph.nodes())
    if weighted:
        weights = _generate_weights(vertices, weight_rng, weight_low, weight_high)
        gen_params = {**gen_params, "weight_low": weight_low, "weight_high": weight_high}
    else:
        weights = {v: 1.0 for v in vertices}

    problem_type = "weighted" if weighted else "unweighted"
    instance_id = f"n{n:03d}_seed{seed}"
    return Instance(
        instance_id=instance_id,
        problem_type=problem_type,
        n_vertices=len(vertices),
        seed=seed,
        graph=graph,
        weights=weights,
        triangles=triangles,
        generation_params=gen_params,
        generator_version=GENERATOR_VERSION,
    )


def generate_sweep(
    n_min: int,
    max_vertices: int,
    step: int,
    weighted: bool,
    base_seed: int,
    n_seeds_per_size: int = 1,
    weight_low: int = 1,
    weight_high: int = 10,
) -> list[Instance]:
    """Generate a reproducible sweep of instances across n in
    range(n_min, max_vertices + 1, step), `n_seeds_per_size` seeds per size.

    Per-instance seeds are derived deterministically from `base_seed` via
    `numpy.random.SeedSequence(base_seed).spawn(...)`, so the whole sweep is
    reproducible from `base_seed` alone.
    """
    sizes = list(range(n_min, max_vertices + 1, step))
    n_seeds_needed = len(sizes) * n_seeds_per_size
    child_seeds = np.random.SeedSequence(base_seed).spawn(n_seeds_needed)
    seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]

    instances = []
    idx = 0
    for n in sizes:
        for _ in range(n_seeds_per_size):
            seed = seed_ints[idx]
            idx += 1
            instances.append(
                generate_instance(
                    n, seed, weighted, weight_low=weight_low, weight_high=weight_high
                )
            )
    return instances

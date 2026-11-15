"""Generate four candidate paths per source-destination pair (P1 Section 5.1).

P1: "for each node pair, we precompute 4 shortest paths as candidate paths".
That is Yen's k-shortest simple paths with hop count as the distance, not the
edge-disjoint subset used in the public TEAL pickle (which only has 2 paths on B4).
"""

from __future__ import annotations

import pickle
from itertools import islice
from pathlib import Path
from sys import maxsize

import networkx as nx

from config import NUM_PATHS, PATH_DIR
from .load_topology import Topology


def _weighted_copy(graph: nx.DiGraph, metric: str = "min-hop") -> nx.DiGraph:
    weighted = graph.copy()
    for u, v, cap in weighted.edges.data("capacity"):
        cap = 0.0 if cap is None else float(cap)
        if metric == "min-hop":
            weighted[u][v]["weight"] = maxsize if cap <= 0.0 else 1.0
        elif metric == "inv-cap":
            weighted[u][v]["weight"] = maxsize if cap <= 0.0 else 1.0 / cap
        else:
            raise ValueError(f"Unknown path metric: {metric}")
    return weighted


def _k_shortest(graph: nx.DiGraph, src, dst, k: int) -> list[list]:
    try:
        return list(islice(nx.shortest_simple_paths(graph, src, dst, weight="weight"), k))
    except nx.NetworkXNoPath:
        return []


def _pad_paths(paths: list[list], k: int) -> list[list]:
    if not paths:
        raise ValueError("Demand has no path")
    if len(paths) >= k:
        return paths[:k]
    return [paths[0]] * (k - len(paths)) + paths


def generate_candidate_paths(
    topology: Topology,
    k: int = NUM_PATHS,
    metric: str = "min-hop",
    cache: bool = True,
) -> dict[tuple, list[list]]:
    """Return { (s, t): [path, ...] } with exactly k paths per ordered pair."""
    cache_path = PATH_DIR / f"{topology.name}-{k}-paths_{metric}.pkl"
    if cache and cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    weighted = _weighted_copy(topology.graph, metric)
    path_dict = {}
    for src in topology.nodes:
        for dst in topology.nodes:
            if src == dst:
                continue
            paths = _k_shortest(weighted, src, dst, k)
            path_dict[(src, dst)] = _pad_paths(paths, k)

    if cache:
        PATH_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(path_dict, handle)
    return path_dict

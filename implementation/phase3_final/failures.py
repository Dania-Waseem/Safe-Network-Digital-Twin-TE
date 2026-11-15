"""Apply a directed link-failure case to a ready topology (E3 link-failure experiment).

Failure cases come from ``data/scenarios/link_failures/<topology>.json`` (P1 §5.5
sampler, seeds 100..1000, 0-3 failed directed edges). Their ``edge_indices`` and
``edges`` follow the link order of ``data/processed/graph_data/<topology>_graph.npz``
(= capacity / bipartite order); the file's ``edge_order`` description (topology JSON
order) is inaccurate for B4 and UsCarrier. Edges are matched by their (u, v) tuple, so
the order question cannot affect the result.

Rebuild rule:
1. failed links get capacity 0;
2. candidate paths that traverse a failed link are dropped;
3. a pair left with no candidate path gets Yen k=4 (hop count) recomputed on the
   surviving graph (``preprocessing.paths.yen_unit``);
4. a pair with no path at all in the surviving graph is *unservable*: its original
   paths are kept as placeholders (needed by the padded policy graph) and its demand
   is removed from the routed matrix; it counts as undelivered in satisfaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from preprocessing.paths import yen_unit
from twin.data import PROCESSED, ROOT, TopologyTensors

SCENARIOS = ROOT / "data" / "scenarios" / "link_failures"


@dataclass
class FailureCase:
    topology: str
    seed: int
    n_failed: int
    edges: list[tuple[int, int]]

    @property
    def key(self) -> str:
        return f"{self.topology}#s{self.seed}f{self.n_failed}"


@dataclass
class FailedTopology:
    topo: TopologyTensors
    case: FailureCase
    failed_links: np.ndarray
    unservable_pairs: np.ndarray
    n_pairs_rerouted: int
    n_paths_dropped: int


def load_cases(topology: str) -> list[FailureCase]:
    payload = json.loads((SCENARIOS / f"{topology}.json").read_text(encoding="utf-8"))
    return [FailureCase(topology, int(c["seed"]), int(c["n_failed"]), [tuple(e) for e in c["edges"]])
            for c in payload["cases"]]


def _edge_list(topology: str) -> list[tuple[int, int]]:
    g = np.load(PROCESSED / "graph_data" / f"{topology}_graph.npz")
    return list(zip(g["edge_src"].tolist(), g["edge_dst"].tolist()))


def _pair_index(n: int, s: int, t: int) -> int:
    return s * (n - 1) + (t if t < s else t - 1)


def apply_failure(base: TopologyTensors, case: FailureCase) -> FailedTopology:
    edges = _edge_list(base.name)
    if len(edges) != base.n_links:
        raise RuntimeError("graph.npz edge list does not match topology link count")
    eidx = {e: i for i, e in enumerate(edges)}
    failed = np.array(sorted(eidx[tuple(e)] for e in case.edges), dtype=np.int64)
    if case.n_failed == 0:
        return FailedTopology(replace(base, name=case.key), case, failed, np.zeros(0, dtype=np.int64), 0, 0)

    n_nodes = int(max(max(e) for e in edges)) + 1
    counts = np.asarray(base.n_paths, dtype=np.int64)
    n_pairs = counts.size
    # links per path
    order = np.argsort(base.incidence_path, kind="mergesort")
    ip, il = base.incidence_path[order], base.incidence_link[order]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(ip)) + 1, [ip.size]])
    links_of_path = {int(ip[a]): il[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    failed_set = set(failed.tolist())
    dead = np.array([bool(failed_set.intersection(links_of_path[p].tolist())) for p in range(int(counts.sum()))])

    succ: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, (u, v) in enumerate(edges):
        if i not in failed_set and base.capacity[i] > 0:
            succ[u].append(v)

    starts = np.cumsum(counts) - counts
    new_links: list[np.ndarray] = []
    new_hops: list[int] = []
    new_counts = np.zeros(n_pairs, dtype=np.int64)
    unservable: list[int] = []
    rerouted = 0
    pairs = [(s, t) for s in range(n_nodes) for t in range(n_nodes) if s != t]
    for d, (s, t) in enumerate(pairs):
        assert _pair_index(n_nodes, s, t) == d
        own = range(int(starts[d]), int(starts[d] + counts[d]))
        alive = [p for p in own if not dead[p]]
        if alive:
            for p in alive:
                new_links.append(links_of_path[p])
                new_hops.append(int(base.hops[p]))
            new_counts[d] = len(alive)
            continue
        found = yen_unit(succ, s, t, 4)
        if found:
            rerouted += 1
            for path in found:
                new_links.append(np.array([eidx[(a, b)] for a, b in zip(path[:-1], path[1:])], dtype=np.int64))
                new_hops.append(len(path) - 1)
            new_counts[d] = len(found)
        else:  # disconnected: keep placeholders, demand removed by routable_matrix()
            unservable.append(d)
            for p in own:
                new_links.append(links_of_path[p])
                new_hops.append(int(base.hops[p]))
            new_counts[d] = counts[d]
    inc_path = np.concatenate([np.full(len(l), i, dtype=np.int64) for i, l in enumerate(new_links)])
    inc_link = np.concatenate(new_links).astype(np.int64)
    capacity = base.capacity.copy()
    capacity[failed] = 0.0
    topo = TopologyTensors(
        name=case.key, n_links=base.n_links, denominator=base.denominator, capacity=capacity,
        n_paths=new_counts, hops=np.asarray(new_hops, dtype=np.int64),
        demand_row=np.repeat(np.arange(n_pairs, dtype=np.int64), new_counts),
        incidence_path=inc_path, incidence_link=inc_link,
    )
    return FailedTopology(topo, case, failed, np.asarray(unservable, dtype=np.int64), rerouted, int(dead.sum()))


def routable_matrix(ft: FailedTopology, matrix: np.ndarray) -> np.ndarray:
    """Matrix with unservable pairs removed (their demand cannot be delivered by any routing)."""
    if ft.unservable_pairs.size == 0:
        return matrix
    m = np.array(matrix, dtype=np.float64, copy=True)
    n = m.shape[0]
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    for d in ft.unservable_pairs:
        s, t = pairs[int(d)]
        m[s, t] = 0.0
    return m

"""Load a directed topology and record problems without dropping them silently.

Path cost and link direction follow the frozen P1 loader:
``baseline_p1_reproduction/preprocessing/load_topology.py`` and
``generate_paths._weighted_copy`` (min-hop). This module does not import
that tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Topology:
    name: str
    source: str
    node_ids: list[int]
    edges: list[tuple[int, int]]
    capacities: np.ndarray
    graph: nx.DiGraph
    warnings: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    imputed_capacity_count: int = 0
    capacity_note: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def _dense_ids(ids: list[int], name: str) -> list[int]:
    expected = list(range(len(ids)))
    if sorted(ids) != expected:
        raise ValueError(
            f"{name}: node ids are not exactly 0..{len(ids) - 1}. "
            "They were not renumbered."
        )
    return expected


def from_node_link(name: str, data: dict, source: str, capacity_note: str) -> Topology:
    """Build a simple directed graph. Duplicates and self-loops are recorded."""
    raw_ids = [int(node["id"]) for node in data["nodes"]]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError(f"{name}: duplicate node ids {raw_ids}. Not dropped.")
    node_ids = _dense_ids(raw_ids, name)
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    edges: list[tuple[int, int]] = []
    capacities: list[float] = []
    warnings: list[str] = []
    excluded: list[dict] = []
    seen: dict[tuple[int, int], float] = {}
    for index, link in enumerate(data["links"]):
        if "source" not in link or "target" not in link:
            excluded.append({"index": index, "reason": "missing endpoint", "link": dict(link)})
            warnings.append(f"link {index} is missing an endpoint and was not added")
            continue
        u, v = int(link["source"]), int(link["target"])
        if u not in graph or v not in graph:
            excluded.append({"index": index, "reason": "endpoint not in node set", "source": u, "target": v})
            warnings.append(f"link {index} ({u}->{v}) references an unknown node and was not added")
            continue
        if u == v:
            excluded.append({"index": index, "reason": "self-loop", "source": u, "target": v})
            warnings.append(f"self-loop {u}->{v} was not added as a routing link")
            continue
        if "capacity" not in link:
            excluded.append({"index": index, "reason": "missing capacity", "source": u, "target": v})
            warnings.append(f"link {index} ({u}->{v}) has no capacity and was not given an invented value")
            continue
        try:
            cap = float(link["capacity"])
        except (TypeError, ValueError):
            excluded.append({"index": index, "reason": "capacity not numeric", "source": u, "target": v})
            warnings.append(f"link {index} ({u}->{v}) has a non-numeric capacity and was not added")
            continue
        if not np.isfinite(cap) or cap <= 0.0:
            excluded.append(
                {"index": index, "reason": "non-positive or non-finite capacity", "source": u, "target": v, "capacity": cap}
            )
            warnings.append(f"link {index} ({u}->{v}) capacity {cap} is not a positive finite value and was not added")
            continue
        if (u, v) in seen:
            excluded.append(
                {
                    "index": index,
                    "reason": "duplicate link",
                    "source": u,
                    "target": v,
                    "capacity": cap,
                    "kept_capacity": seen[(u, v)],
                }
            )
            warnings.append(
                f"duplicate link {u}->{v} (capacity {cap}); the first capacity {seen[(u, v)]} is the one in the graph"
            )
            continue
        seen[(u, v)] = cap
        graph.add_edge(u, v, capacity=cap)
        edges.append((u, v))
        capacities.append(cap)
    return Topology(
        name=name,
        source=source,
        node_ids=node_ids,
        edges=edges,
        capacities=np.asarray(capacities, dtype=np.float64),
        graph=graph,
        warnings=warnings,
        excluded=excluded,
        capacity_note=capacity_note,
    )


def from_zoo_gml(name: str, text: str, source: str, placeholder_capacity: float) -> Topology:
    """Topology Zoo GML. Cogentco has no link speed; the placeholder is explicit."""
    graph = nx.Graph()
    for block in text.split("node ["):
        node_id = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("id "):
                node_id = int(line.split()[1])
                break
        if node_id is not None:
            graph.add_node(node_id)
    raw_edges = []
    for block in text.split("edge ["):
        tokens = block.replace("[", " ").replace("]", " ").split()
        src = tgt = None
        for index, token in enumerate(tokens[:-1]):
            if token == "source":
                src = int(tokens[index + 1])
            elif token == "target":
                tgt = int(tokens[index + 1])
        if src is None or tgt is None:
            continue
        raw_edges.append((src, tgt))
    node_ids = _dense_ids(sorted(graph.nodes()), name)
    directed = nx.DiGraph()
    directed.add_nodes_from(node_ids)
    edges: list[tuple[int, int]] = []
    capacities: list[float] = []
    warnings = [
        f"{name} GML has no link speed. Each undirected Zoo edge is stored as two "
        f"directed edges with placeholder capacity {placeholder_capacity:g}. "
        "This is not a measured capacity."
    ]
    excluded: list[dict] = []
    seen: set[tuple[int, int]] = set()
    undirected_seen: set[tuple[int, int]] = set()
    for index, (u, v) in enumerate(raw_edges):
        if u == v:
            excluded.append({"index": index, "reason": "self-loop", "source": u, "target": v})
            warnings.append(f"self-loop {u}->{v} was not added")
            continue
        if u not in directed or v not in directed:
            excluded.append({"index": index, "reason": "endpoint not in node set", "source": u, "target": v})
            warnings.append(f"edge {u}-{v} references an unknown node and was not added")
            continue
        key = (min(u, v), max(u, v))
        if key in undirected_seen:
            excluded.append({"index": index, "reason": "duplicate undirected edge", "source": u, "target": v})
            warnings.append(f"duplicate undirected edge {u}-{v} was not added again")
            continue
        undirected_seen.add(key)
        for a, b in ((u, v), (v, u)):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            directed.add_edge(a, b, capacity=float(placeholder_capacity))
            edges.append((a, b))
            capacities.append(float(placeholder_capacity))
    return Topology(
        name=name,
        source=source,
        node_ids=node_ids,
        edges=edges,
        capacities=np.asarray(capacities, dtype=np.float64),
        graph=directed,
        warnings=warnings,
        excluded=excluded,
        imputed_capacity_count=len(edges),
        capacity_note=(
            f"placeholder {placeholder_capacity:g} on every directed edge because the "
            "Zoo file has no LinkSpeed"
        ),
    )


def connectivity_report(topology: Topology) -> dict:
    weak_sets = list(nx.weakly_connected_components(topology.graph))
    undirected = nx.Graph()
    undirected.add_nodes_from(topology.node_ids)
    undirected.add_edges_from(topology.edges)
    undirected_sets = list(nx.connected_components(undirected))
    strong = nx.is_strongly_connected(topology.graph) if topology.num_nodes else False
    missing = 0
    if topology.num_nodes:
        for source in topology.node_ids:
            missing += (topology.num_nodes - 1) - len(nx.descendants(topology.graph, source))
    return {
        "nodes": topology.num_nodes,
        "directed_edges": topology.num_edges,
        "undirected_edges": undirected.number_of_edges(),
        "weakly_connected": len(weak_sets) == 1,
        "weak_component_sizes": sorted((len(c) for c in weak_sets), reverse=True),
        "undirected_connected": len(undirected_sets) == 1,
        "undirected_component_sizes": sorted((len(c) for c in undirected_sets), reverse=True),
        "strongly_connected": strong,
        "ordered_pairs_without_a_directed_path": missing,
        "excluded_link_records": len(topology.excluded),
        "warnings": list(topology.warnings),
    }

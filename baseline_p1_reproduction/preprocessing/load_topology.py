"""Load a WAN topology and directed link capacities (P1 Section 4.1, 5.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from config import TOPOLOGY_DIR


@dataclass
class Topology:
    name: str
    graph: nx.DiGraph
    nodes: list
    edges: list
    capacities: np.ndarray
    edge_to_idx: dict
    idx_to_edge: list

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def load_topology(name: str = "B4", topology_dir: Path | None = None) -> Topology:
    """Load a TEAL-format node-link JSON graph (directed links with capacity)."""
    topology_dir = Path(topology_dir or TOPOLOGY_DIR)
    path = topology_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Topology not found: {path}")

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    graph = json_graph.node_link_graph(data, directed=True, edges="links")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Empty topology: {path}")

    # Node ids are 0..n-1 in the TEAL B4 dump; keep that order for TM indexing.
    nodes = list(range(graph.number_of_nodes()))
    if set(graph.nodes()) != set(nodes):
        nodes = sorted(graph.nodes())

    edges = list(graph.edges())
    capacities = np.array(
        [float(graph[u][v].get("capacity", 0.0)) for u, v in edges],
        dtype=np.float64,
    )
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    return Topology(
        name=name,
        graph=graph,
        nodes=nodes,
        edges=edges,
        capacities=capacities,
        edge_to_idx=edge_to_idx,
        idx_to_edge=edges,
    )

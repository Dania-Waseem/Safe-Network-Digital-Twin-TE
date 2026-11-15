"""Load Topology-Zoo GraphML/GML into the TEAL-like Topology dataclass."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from config import TOPOLOGY_DIR
from .load_topology import Topology


def _parse_zoo_gml(path: Path) -> nx.Graph:
    """Robust Topology-Zoo GML reader (NetworkX rejects duplicate edges/labels)."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    g = nx.Graph()
    for block in text.split("node ["):
        if "id " not in block:
            continue
        nid = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("id "):
                nid = int(line.split()[1])
                break
        if nid is not None:
            g.add_node(nid)
    for block in text.split("edge ["):
        src = tgt = None
        cap = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("source "):
                src = int(line.split()[1])
            elif line.startswith("target "):
                tgt = int(line.split()[1])
            elif line.startswith("LinkSpeedRaw "):
                try:
                    cap = float(line.split()[1])
                except ValueError:
                    cap = None
        if src is not None and tgt is not None and src != tgt:
            if g.has_edge(src, tgt):
                continue
            g.add_edge(src, tgt, capacity=cap)
    return g


def graphml_to_topology(
    path: Path,
    name: str,
    default_capacity: float = 10000.0,
) -> Topology:
    path = Path(path)
    if path.suffix.lower() == ".gml":
        raw = _parse_zoo_gml(path)
    else:
        raw = nx.read_graphml(path)
    # Make a simple directed graph with integer nodes 0..n-1
    und = raw.to_undirected()
    mapping = {n: i for i, n in enumerate(sorted(und.nodes(), key=str))}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(mapping)))
    for u, v, data in und.edges(data=True):
        iu, iv = mapping[u], mapping[v]
        if iu == iv:
            continue
        cap = data.get("LinkSpeedRaw") or data.get("capacity") or default_capacity
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            cap = default_capacity
        if cap <= 0:
            cap = default_capacity
        g.add_edge(iu, iv, capacity=float(cap))
        g.add_edge(iv, iu, capacity=float(cap))
    edges = list(g.edges())
    capacities = np_caps(g, edges)
    return Topology(
        name=name,
        graph=g,
        nodes=list(range(g.number_of_nodes())),
        edges=edges,
        capacities=capacities,
        edge_to_idx={e: i for i, e in enumerate(edges)},
        idx_to_edge=edges,
    )


def np_caps(g, edges):
    import numpy as np

    return np.array([float(g[u][v].get("capacity", 0.0)) for u, v in edges], dtype=np.float64)


def save_teal_json(topology: Topology, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or TOPOLOGY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(topology.graph, edges="links")
    dest = out_dir / f"{topology.name}.json"
    dest.write_text(json.dumps(data), encoding="utf-8")
    return dest

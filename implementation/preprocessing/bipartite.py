"""Link-path bipartite graph in the layout used by frozen P1.

``build_bipartite_graph`` places link nodes first and path nodes after them.
An undirected bipartite edge exists when a path traverses a link, stored once
in each direction so a later GIN layer can pass messages both ways.

P1 requires exactly four paths per demand because it pads. This graph keeps
one path node per generated path, so a demand can have 1, 2, 3, or 4 path nodes.

The structure does not depend on the traffic matrix. Demand is a path-node
feature filled for one matrix at a time by ``tensors_for_matrix``.
"""

from __future__ import annotations

import numpy as np
import torch

from .topology import Topology


def incidence(topology: Topology, paths: dict) -> dict[str, np.ndarray]:
    edge_index = {edge: i for i, edge in enumerate(topology.edges)}
    counts = paths["n_paths"]
    ptr = paths["path_node_ptr"]
    nodes = paths["path_nodes"]
    hops = paths["hop_count"]
    n_paths = int(counts.astype(np.int64).sum())
    n_inc = int(hops.astype(np.int64).sum())
    path_index = np.empty(n_inc, dtype=np.int32)
    link_index = np.empty(n_inc, dtype=np.int32)
    demand_row = np.empty(n_paths, dtype=np.int32)
    cursor = 0
    slot = 0
    for demand_i, n_found in enumerate(counts.tolist()):
        for _ in range(int(n_found)):
            seq = nodes[int(ptr[cursor]) : int(ptr[cursor + 1])]
            demand_row[cursor] = demand_i
            for a, b in zip(seq[:-1].tolist(), seq[1:].tolist()):
                link = edge_index.get((int(a), int(b)))
                if link is None:
                    raise KeyError(f"Path uses {a}->{b}, which is not a topology link")
                path_index[slot] = cursor
                link_index[slot] = link
                slot += 1
            cursor += 1
    if slot != n_inc:
        raise RuntimeError(f"Incidence fill {slot} does not match hop sum {n_inc}")
    return {"path_index": path_index, "link_index": link_index, "demand_row": demand_row}


def od_vector(matrix: np.ndarray, n: int) -> np.ndarray:
    """Off-diagonal demands in the same (s, t) order as the path table."""
    values = np.empty(n * (n - 1), dtype=np.float64)
    cursor = 0
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            values[cursor] = matrix[src, dst]
            cursor += 1
    return values


def tensors_for_matrix(
    topology: Topology,
    paths: dict,
    links: dict[str, np.ndarray],
    demand_original: np.ndarray,
    demand_normalised: np.ndarray,
) -> dict[str, torch.Tensor]:
    """One PyG-ready graph for a single traffic matrix.

    Column 0 is the normalised value (P1 ``initial_features(scale=True)``).
    Column 1 is the original value (P1's frozen unscaled h^0).
    """
    n_links = topology.num_edges
    n_paths = int(paths["n_paths"].astype(np.int64).sum())
    path_x = np.zeros((n_paths, 5), dtype=np.float32)
    rows = links["demand_row"]
    path_x[:, 0] = demand_normalised[rows]
    path_x[:, 1] = demand_original[rows]
    path_x[:, 2] = paths["hop_count"].astype(np.float32)
    path_x[:, 3] = paths["demand_src"][rows].astype(np.float32)
    path_x[:, 4] = paths["demand_dst"][rows].astype(np.float32)
    link_x = np.zeros((n_links, 5), dtype=np.float32)
    link_x[:, 0] = topology.capacities / float(np.max(topology.capacities))
    link_x[:, 1] = topology.capacities
    src = np.asarray([u for u, _ in topology.edges], dtype=np.float32)
    dst = np.asarray([v for _, v in topology.edges], dtype=np.float32)
    link_x[:, 3] = src
    link_x[:, 4] = dst
    x = np.vstack([link_x, path_x])
    # Path nodes are offset by n_links, matching P1's bipartite index.
    path_end = links["path_index"].astype(np.int64) + n_links
    link_end = links["link_index"].astype(np.int64)
    forward = np.vstack([path_end, link_end])
    backward = np.vstack([link_end, path_end])
    edge_index = np.hstack([forward, backward])
    return {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(edge_index),
        "num_link_nodes": torch.tensor(n_links),
        "num_path_nodes": torch.tensor(n_paths),
    }


FEATURE_COLUMNS = (
    "0 normalised capacity (link) or normalised demand (path); P1 scale=True",
    "1 original capacity or original demand; P1 frozen h^0",
    "2 hop count on path nodes, 0 on link nodes",
    "3 link source or demand source",
    "4 link destination or demand destination",
)

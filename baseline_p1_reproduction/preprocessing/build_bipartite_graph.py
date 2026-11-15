"""Build the link-path bipartite graph used by TopoFlowNet (P1 Section 4.2.1).

Link nodes: physical directed WAN links, feature = capacity.
Path nodes: candidate paths, feature = demand of the owning pair.
An undirected bipartite edge exists iff path p traverses link e.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import NUM_PATHS
from .load_topology import Topology


@dataclass
class TEProblem:
    """Static topology + path incidence, plus helpers for a single TM."""

    name: str
    num_nodes: int
    num_edges: int
    num_demands: int
    num_paths: int
    capacities: torch.Tensor
    demand_pairs: list[tuple]
    paths: list[list]
    p2e: torch.Tensor
    bipartite_index: torch.Tensor
    hops: torch.Tensor
    edge_to_idx: dict

    def to(self, device: torch.device) -> "TEProblem":
        self.capacities = self.capacities.to(device)
        self.p2e = self.p2e.to(device)
        self.bipartite_index = self.bipartite_index.to(device)
        self.hops = self.hops.to(device)
        return self

    def demand_vector(self, tm: np.ndarray, device=None) -> torch.Tensor:
        values = [float(tm[s, t]) for s, t in self.demand_pairs]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def initial_features(self, demands: torch.Tensor, scale: bool = False) -> torch.Tensor:
        """h^0 in P1 eq. (5): raw capacities then per-path raw demand.

        P1 writes h^0_Ve = [c(e)] and h^0_Vp = [d_p]. Default is unscaled.
        scale=True is kept only for ablation / debugging, not for reproduction.
        """
        cap = self.capacities
        path_demand = demands.repeat_interleave(self.num_paths)
        if scale:
            denom = cap.max().clamp_min(1.0)
            cap = cap / denom
            path_demand = path_demand / denom
        return torch.cat([cap, path_demand], dim=0).unsqueeze(-1)


def build_bipartite_graph(
    topology: Topology,
    path_dict: dict,
    k: int = NUM_PATHS,
    device: torch.device | None = None,
) -> TEProblem:
    demand_pairs = [
        (s, t) for s in topology.nodes for t in topology.nodes if s != t
    ]
    paths = []
    src, dst = [], []
    hops = []
    path_id = 0
    for s, t in demand_pairs:
        pair_paths = path_dict[(s, t)]
        if len(pair_paths) != k:
            raise ValueError(f"Expected {k} paths for {(s, t)}, got {len(pair_paths)}")
        for path in pair_paths:
            paths.append(path)
            hop_count = 0
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) not in topology.edge_to_idx:
                    raise KeyError(f"Path edge {(u, v)} is not in the topology")
                link_idx = topology.edge_to_idx[(u, v)]
                src.append(topology.num_edges + path_id)
                dst.append(link_idx)
                hop_count += 1
            hops.append(max(hop_count, 1))
            path_id += 1

    # Undirected bipartite edges so GINConv passes information both ways.
    bipartite = torch.tensor([src + dst, dst + src], dtype=torch.long)
    p2e = torch.tensor(
        [[p - topology.num_edges for p in src], dst],
        dtype=torch.long,
    )
    problem = TEProblem(
        name=topology.name,
        num_nodes=topology.num_nodes,
        num_edges=topology.num_edges,
        num_demands=len(demand_pairs),
        num_paths=k,
        capacities=torch.tensor(topology.capacities, dtype=torch.float32),
        demand_pairs=demand_pairs,
        paths=paths,
        p2e=p2e,
        bipartite_index=bipartite,
        hops=torch.tensor(hops, dtype=torch.float32),
        edge_to_idx=topology.edge_to_idx,
    )
    if device is not None:
        problem.to(device)
    return problem

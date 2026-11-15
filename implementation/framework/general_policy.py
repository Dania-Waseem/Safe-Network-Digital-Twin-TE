"""A new P1-style actor. The frozen B4 checkpoint is never loaded or written.

Features are capacity and demand divided by the topology's maximum capacity,
so a change of units does not change the network input. Path padding is the
same four-slot table used by the frozen actor; fractions on repeated paths
are added back onto the real path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from twin.data import ROOT, TopologyTensors

from .policy_adapter import NUM_PATHS, PolicyGraph, _grl_model_class, build_policy_graph, collapse_fractions

CHECKPOINT = ROOT / "models" / "policy" / "general_grl.pt"
# Extended training writes here; the baseline CHECKPOINT above stays untouched.
IMPROVED_CHECKPOINT = ROOT / "models" / "policy" / "improved" / "best_checkpoint.pt"


def normalized_inputs(capacity: torch.Tensor, demand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = capacity.max().clamp_min(1.0)
    return capacity / denom, demand / denom


class GeneralPolicy:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.model = _grl_model_class()().to(self.device)
        self._graphs: dict[str, PolicyGraph] = {}
        self.last_proposal: np.ndarray | None = None

    def graph_for(self, topo: TopologyTensors) -> PolicyGraph:
        cached = self._graphs.get(topo.name)
        if cached is None:
            cached = build_policy_graph(topo)
            cached.edge_index = cached.edge_index.to(self.device)
            self._graphs[topo.name] = cached
        return cached

    def release(self, name: str | None = None) -> None:
        if name is None:
            self._graphs.clear()
        else:
            self._graphs.pop(name, None)

    def _h0(self, topo: TopologyTensors, demand: torch.Tensor) -> torch.Tensor:
        capacity = torch.as_tensor(topo.capacity, dtype=torch.float32, device=self.device)
        cap_n, dem_n = normalized_inputs(capacity, demand)
        return torch.cat([cap_n, dem_n.repeat_interleave(NUM_PATHS)], dim=0).unsqueeze(-1)

    def fractions(self, topo: TopologyTensors, demand_per_demand: np.ndarray) -> torch.Tensor:
        graph = self.graph_for(topo)
        demand = torch.as_tensor(demand_per_demand, dtype=torch.float32, device=self.device)
        if demand.shape != (graph.n_demands,):
            raise ValueError(f"{topo.name}: expected {graph.n_demands} demands, got {tuple(demand.shape)}")
        out = self.model.act(self._h0(topo, demand), graph.edge_index, graph.n_links, deterministic=True)
        return out["fractions"]

    def propose(self, topo: TopologyTensors, demand_per_demand: np.ndarray) -> np.ndarray:
        fractions = self.fractions(topo, demand_per_demand).detach().cpu().numpy()
        action = collapse_fractions(fractions, self.graph_for(topo))
        self.last_proposal = action
        return action

    def utilisation(self, topo: TopologyTensors, matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        from twin.physics import od_demands

        graph = self.graph_for(topo)
        demand_np = od_demands(matrix)
        demand = torch.as_tensor(demand_np, dtype=torch.float32, device=self.device)
        fractions = self.fractions(topo, demand_np)
        flat = fractions.reshape(-1)
        slot = torch.as_tensor(graph.pad_slot_real, dtype=torch.long, device=self.device)
        action = torch.zeros(int(graph.counts.sum()), dtype=torch.float32, device=self.device)
        action = action.scatter_add(0, slot, flat)
        demand_path = demand[torch.as_tensor(topo.demand_row, dtype=torch.long, device=self.device)]
        flow = action * demand_path
        incidence_path = torch.as_tensor(topo.incidence_path, dtype=torch.long, device=self.device)
        incidence_link = torch.as_tensor(topo.incidence_link, dtype=torch.long, device=self.device)
        load = torch.zeros(topo.n_links, dtype=torch.float32, device=self.device)
        load = load.scatter_add(0, incidence_link, flow[incidence_path])
        capacity = torch.as_tensor(topo.capacity, dtype=torch.float32, device=self.device).clamp_min(1e-6)
        return load / capacity, fractions

    def save(self, path: Path, extra: dict) -> None:
        path = Path(path)
        if "baseline_p1_reproduction" in path.parts or "baseline_p1" in path.parts:
            raise RuntimeError("refusing to write a checkpoint into the frozen repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), **extra}, path)

    def load(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device)
        return payload

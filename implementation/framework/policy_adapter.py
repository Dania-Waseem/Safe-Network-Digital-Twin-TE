"""Read-only adapter around the frozen P1 actor.

The checkpoint and baseline_p1_reproduction sources are not modified.
The actor always sees four path slots. When a demand has fewer than four
simple paths, the missing slots repeat the minimum-hop path, matching P1's
padding. Fractions on those repeated slots are added back onto the real path
so the twin and the gate keep the unpadded Phase 3 path table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from twin.data import BASELINE, MODELS, ROOT, TopologyTensors

GRL_ROOT = BASELINE
DEFAULT_CHECKPOINT = (
    MODELS / "baseline_p1" / "b4_freeze" / "grl_te_b4_freeze_seed100.pt"
)
NUM_PATHS = 4


def _grl_model_class():
    grl = str(GRL_ROOT)
    if grl not in sys.path:
        sys.path.insert(0, grl)
    from models.a2c import GRLTEModel

    return GRLTEModel


@dataclass
class PolicyGraph:
    edge_index: torch.Tensor
    n_links: int
    n_demands: int
    real_to_slot: np.ndarray
    pad_slot_real: np.ndarray
    counts: np.ndarray


def _shortest_real_index(counts: np.ndarray, hops: np.ndarray) -> np.ndarray:
    starts = np.cumsum(counts) - counts
    chosen = np.zeros(counts.shape[0], dtype=np.int64)
    for count in np.unique(counts):
        if int(count) <= 0:
            continue
        selected = np.flatnonzero(counts == count)
        block = hops[starts[selected, None] + np.arange(int(count))[None, :]]
        chosen[selected] = starts[selected] + np.argmin(block, axis=1)
    return chosen


def build_policy_graph(topo: TopologyTensors) -> PolicyGraph:
    counts = np.asarray(topo.n_paths, dtype=np.int64)
    n_demands = int(counts.shape[0])
    n_real = int(counts.sum())
    starts = np.cumsum(counts) - counts
    local = np.arange(n_real) - np.repeat(starts, counts)
    real_to_slot = np.repeat(np.arange(n_demands) * NUM_PATHS, counts) + local
    shortest = _shortest_real_index(counts, topo.hops)

    pad_path = real_to_slot[topo.incidence_path]
    pad_link = np.asarray(topo.incidence_link, dtype=np.int64)
    extra_path = []
    extra_link = []
    # Group existing incidences by real path for the few padded demands.
    if np.any(counts < NUM_PATHS):
        order = np.argsort(topo.incidence_path, kind="mergesort")
        grouped_path = topo.incidence_path[order]
        grouped_link = topo.incidence_link[order]
        cuts = np.flatnonzero(np.diff(grouped_path)) + 1
        bounds = np.concatenate([[0], cuts, [grouped_path.size]])
        links_of = {}
        for start, stop in zip(bounds[:-1], bounds[1:]):
            links_of[int(grouped_path[start])] = grouped_link[start:stop]
        needs = np.flatnonzero((counts > 0) & (counts < NUM_PATHS))
        for demand in needs:
            donor = int(shortest[demand])
            donor_links = links_of[donor]
            for slot in range(int(counts[demand]), NUM_PATHS):
                pad_id = int(demand) * NUM_PATHS + slot
                extra_path.append(np.full(donor_links.shape, pad_id, dtype=np.int64))
                extra_link.append(np.asarray(donor_links, dtype=np.int64))
    if extra_path:
        pad_path = np.concatenate([pad_path, np.concatenate(extra_path)])
        pad_link = np.concatenate([pad_link, np.concatenate(extra_link)])

    path_node = topo.n_links + pad_path
    src = np.concatenate([path_node, pad_link])
    dst = np.concatenate([pad_link, path_node])
    edge_index = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
    pad_slot_real = np.full(n_demands * NUM_PATHS, -1, dtype=np.int64)
    pad_slot_real[real_to_slot] = np.arange(n_real)
    for demand in np.flatnonzero((counts > 0) & (counts < NUM_PATHS)):
        donor = int(shortest[demand])
        for slot in range(int(counts[demand]), NUM_PATHS):
            pad_slot_real[int(demand) * NUM_PATHS + slot] = donor
    return PolicyGraph(
        edge_index=edge_index,
        n_links=topo.n_links,
        n_demands=n_demands,
        real_to_slot=real_to_slot,
        pad_slot_real=pad_slot_real,
        counts=counts,
    )


def collapse_fractions(fractions: np.ndarray, graph: PolicyGraph) -> np.ndarray:
    """Map the actor's [D, 4] softmax onto the unpadded path table."""
    flat = np.asarray(fractions, dtype=np.float64).reshape(-1)
    if flat.size != graph.pad_slot_real.size:
        raise ValueError("actor fractions do not match the padded path table")
    action = np.zeros(int(graph.counts.sum()), dtype=np.float64)
    np.add.at(action, graph.pad_slot_real, flat)
    return action


class FrozenP1Policy:
    """Deterministic frozen B4 actor. Weights are loaded, never written."""

    def __init__(self, checkpoint: Path | None = None, device: torch.device | None = None) -> None:
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device or torch.device("cpu")
        model_cls = _grl_model_class()
        self.model = model_cls()
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
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

    @torch.no_grad()
    def propose(self, topo: TopologyTensors, demand_per_demand: np.ndarray) -> np.ndarray:
        graph = self.graph_for(topo)
        demands = torch.as_tensor(demand_per_demand, dtype=torch.float32, device=self.device)
        if demands.shape != (graph.n_demands,):
            raise ValueError(f"{topo.name}: expected {graph.n_demands} demands, got {demands.shape}")
        path_demand = demands.repeat_interleave(NUM_PATHS)
        capacity = torch.as_tensor(topo.capacity, dtype=torch.float32, device=self.device)
        h0 = torch.cat([capacity, path_demand], dim=0).unsqueeze(-1)
        out = self.model.act(h0, graph.edge_index, graph.n_links, deterministic=True)
        fractions = out["fractions"].detach().cpu().numpy()
        action = collapse_fractions(fractions, graph)
        self.last_proposal = action
        return action

    @torch.no_grad()
    def inspect(self, topo: TopologyTensors, demand_per_demand: np.ndarray) -> dict:
        """Read-only view of the frozen actor. Does not change its weights."""
        graph = self.graph_for(topo)
        demands = torch.as_tensor(demand_per_demand, dtype=torch.float32, device=self.device)
        path_demand = demands.repeat_interleave(NUM_PATHS)
        capacity = torch.as_tensor(topo.capacity, dtype=torch.float32, device=self.device)
        h0 = torch.cat([capacity, path_demand], dim=0).unsqueeze(-1)
        out = self.model.act(h0, graph.edge_index, graph.n_links, deterministic=True)
        fractions = out["fractions"].detach().cpu().numpy()
        raw = out["raw"].detach().cpu().numpy()
        clipped = np.clip(fractions, 1e-12, 1.0)
        entropy = -np.sum(clipped * np.log(clipped), axis=1)
        counts = graph.counts
        return {
            "action": collapse_fractions(fractions, graph),
            "h0_max": float(h0.max()),
            "h0_mean": float(h0.mean()),
            "capacity_max": float(np.max(topo.capacity)),
            "capacity_min": float(np.min(topo.capacity)),
            "demand_max": float(np.max(demand_per_demand)),
            "demand_mean": float(np.mean(demand_per_demand)),
            "raw_abs_mean": float(np.mean(np.abs(raw))),
            "raw_abs_max": float(np.max(np.abs(raw))),
            "mean_entropy": float(np.mean(entropy)),
            "max_entropy": float(np.log(NUM_PATHS)),
            "peaked_fraction": float(np.mean(np.max(fractions, axis=1) > 0.99)),
            "demands": int(counts.shape[0]),
            "demands_with_fewer_than_4_paths": int(np.sum(counts < NUM_PATHS)),
            "uniform_capacity": bool(np.allclose(np.min(topo.capacity), np.max(topo.capacity))),
        }

"""Topology-isolated PolicyGraph cache for reproducible multi-topology eval.

Does not modify ``framework.policy_adapter``. Graphs are keyed by a fingerprint
of topology identity (name + structural sizes + capacity/hops checksums) so
incompatible topologies never share tensors accidentally.
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from framework.policy_adapter import PolicyGraph, build_policy_graph
from twin.data import TopologyTensors


def topology_fingerprint(topo: TopologyTensors) -> str:
    """Stable key: name + shape signatures + content digests."""
    cap = np.asarray(topo.capacity, dtype=np.float64)
    hops = np.asarray(topo.hops, dtype=np.int64)
    n_paths = np.asarray(topo.n_paths, dtype=np.int64)
    hasher = hashlib.sha256()
    hasher.update(str(topo.name).encode("utf-8"))
    hasher.update(np.asarray([topo.n_links, float(topo.denominator)], dtype=np.float64).tobytes())
    hasher.update(cap.tobytes())
    hasher.update(hops.tobytes())
    hasher.update(n_paths.tobytes())
    hasher.update(np.asarray(topo.demand_row, dtype=np.int64).tobytes())
    return f"{topo.name}::{hasher.hexdigest()[:16]}"


@dataclass(frozen=True)
class CachedGraphRef:
    fingerprint: str
    topology_name: str
    n_links: int
    n_demands: int
    n_real_paths: int
    device: str


class TopologyGraphCache:
    """Build-once graph store with strict topology isolation."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self._graphs: dict[str, PolicyGraph] = {}
        self._meta: dict[str, CachedGraphRef] = {}

    def __len__(self) -> int:
        return len(self._graphs)

    def clear(self, topology_name: str | None = None) -> None:
        if topology_name is None:
            self._graphs.clear()
            self._meta.clear()
            gc.collect()
            return
        drop = [k for k, m in self._meta.items() if m.topology_name == topology_name]
        for key in drop:
            self._graphs.pop(key, None)
            self._meta.pop(key, None)
        gc.collect()

    def get(self, topo: TopologyTensors) -> PolicyGraph:
        key = topology_fingerprint(topo)
        cached = self._graphs.get(key)
        if cached is not None:
            meta = self._meta[key]
            if meta.topology_name != topo.name:
                raise RuntimeError(
                    f"graph cache key collision: fingerprint {key} maps to "
                    f"{meta.topology_name!r} but requested {topo.name!r}"
                )
            if meta.n_links != int(topo.n_links) or meta.n_demands != int(topo.n_paths.shape[0]):
                raise RuntimeError(
                    f"stale graph cache entry for {topo.name}: "
                    f"links {meta.n_links}/{topo.n_links}, "
                    f"demands {meta.n_demands}/{topo.n_paths.shape[0]}"
                )
            return cached

        gc.collect()
        graph = build_policy_graph(topo)
        graph.edge_index = graph.edge_index.to(self.device)
        self._graphs[key] = graph
        self._meta[key] = CachedGraphRef(
            fingerprint=key,
            topology_name=topo.name,
            n_links=int(topo.n_links),
            n_demands=int(graph.n_demands),
            n_real_paths=int(graph.counts.sum()),
            device=str(self.device),
        )
        gc.collect()
        return graph

    def install_on_policy(self, policy, topo: TopologyTensors) -> PolicyGraph:
        """Place the cached graph into ``policy._graphs`` under ``topo.name``."""
        graph = self.get(topo)
        # GeneralPolicy / FrozenP1Policy both use topo.name as their local key.
        existing = getattr(policy, "_graphs", None)
        if existing is None:
            raise AttributeError("policy has no _graphs dict")
        # Refuse to overwrite a different object for the same name without clear.
        prior = existing.get(topo.name)
        if prior is not None and prior is not graph:
            # Replace with the isolated cache entry (deterministic source of truth).
            existing[topo.name] = graph
        else:
            existing[topo.name] = graph
        return graph

    def metadata(self) -> list[dict]:
        return [
            {
                "fingerprint": m.fingerprint,
                "topology_name": m.topology_name,
                "n_links": m.n_links,
                "n_demands": m.n_demands,
                "n_real_paths": m.n_real_paths,
                "device": m.device,
            }
            for m in self._meta.values()
        ]

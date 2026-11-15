"""Isolated RL policy backend for Safety Gate evaluation.

Wraps ``framework.general_policy.GeneralPolicy`` without modifying it.
Uses ``TopologyGraphCache`` for topology-isolated graph reuse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from repo_paths import REPO
from twin.data import TopologyTensors
from twin.physics import od_demands

from .graph_cache import TopologyGraphCache


KNOWN_CHECKPOINTS = {
    "improved_stable": "models/policy/improved/recommended_stable.pt",
    "improved_best": "models/policy/improved/best_checkpoint.pt",
    "general_3epoch": "models/policy/general_grl.pt",
}


@dataclass
class PolicyAction:
    """A proposed routing action with identity metadata for paired eval."""

    action: np.ndarray
    checkpoint_id: str
    checkpoint_path: str
    topology: str
    n_real_paths: int
    n_demands: int
    graph_fingerprint: str
    deterministic: bool
    demand_sum: float

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("action", None)
        payload["action_shape"] = list(np.asarray(self.action).shape)
        payload["action_sum"] = float(np.sum(self.action))
        payload["action_min"] = float(np.min(self.action))
        payload["action_max"] = float(np.max(self.action))
        return payload


class PolicyBackend:
    """Load a generalized RL checkpoint and propose deterministic actions."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        checkpoint_id: str | None = None,
        device: torch.device | None = None,
        graph_cache: TopologyGraphCache | None = None,
    ) -> None:
        path = Path(checkpoint)
        # relative checkpoint paths are relative to the repository root (models/...)
        self.checkpoint_path = path if path.is_absolute() else REPO / path
        self.checkpoint_id = checkpoint_id or self.checkpoint_path.stem
        self.device = device or torch.device("cpu")
        self.graph_cache = graph_cache or TopologyGraphCache(self.device)
        self.policy = GeneralPolicy(self.device)
        self._payload: dict[str, Any] | None = None
        self._loaded = False

    def load(self) -> dict[str, Any]:
        if self._loaded:
            assert self._payload is not None
            return self._payload
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        payload = self.policy.load(self.checkpoint_path)
        self.policy.model.eval()
        self._payload = payload if isinstance(payload, dict) else {"model": "raw"}
        self._loaded = True
        return self._payload

    @property
    def payload(self) -> dict[str, Any]:
        return self.load()

    def propose_from_demand(
        self,
        topo: TopologyTensors,
        demand_per_demand: np.ndarray,
    ) -> PolicyAction:
        self.load()
        graph = self.graph_cache.install_on_policy(self.policy, topo)
        demand = np.asarray(demand_per_demand, dtype=np.float64)
        if demand.shape != (graph.n_demands,):
            raise ValueError(
                f"{topo.name}: expected {graph.n_demands} demands, got {demand.shape}"
            )
        with torch.no_grad():
            action = self.policy.propose(topo, demand)
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] != int(graph.counts.sum()):
            raise RuntimeError(
                f"path identity mismatch on {topo.name}: "
                f"action={action.shape[0]} vs graph real paths={int(graph.counts.sum())}"
            )
        from .graph_cache import topology_fingerprint

        return PolicyAction(
            action=action,
            checkpoint_id=self.checkpoint_id,
            checkpoint_path=str(self.checkpoint_path),
            topology=topo.name,
            n_real_paths=int(graph.counts.sum()),
            n_demands=int(graph.n_demands),
            graph_fingerprint=topology_fingerprint(topo),
            deterministic=True,
            demand_sum=float(np.sum(demand)),
        )

    def propose(self, topo: TopologyTensors, matrix: np.ndarray) -> PolicyAction:
        """Propose from a full traffic matrix (OD flattened via physics.od_demands)."""
        return self.propose_from_demand(topo, od_demands(matrix))


def policy_backend_from_config(
    cfg: dict[str, Any],
    *,
    checkpoint_id: str | None = None,
    device: torch.device | None = None,
    graph_cache: TopologyGraphCache | None = None,
) -> PolicyBackend:
    policy_cfg = cfg.get("policy", {})
    cid = checkpoint_id or policy_cfg.get("checkpoint_id", "improved_stable")
    path = policy_cfg.get("default_checkpoint")
    catalog = policy_cfg.get("checkpoints", {})
    if cid in catalog:
        path = catalog[cid]
    elif path is None and cid in KNOWN_CHECKPOINTS:
        path = KNOWN_CHECKPOINTS[cid]
    if path is None:
        raise KeyError(f"no checkpoint path for policy id {cid!r}")
    device = device or torch.device(str(cfg.get("evaluation", {}).get("device", "cpu")))
    return PolicyBackend(
        path,
        checkpoint_id=str(cid),
        device=device,
        graph_cache=graph_cache or TopologyGraphCache(device),
    )

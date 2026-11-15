"""One-step A2C with a shared TopoFlowNet (P1 Sections 4.3 and 4.3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from config import NUM_LAYERS, NUM_PATHS, RAW_ACTION_CLAMP
from .actor import Actor
from .critic import Critic
from .topoflownet import TopoFlowNet


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


class GRLTEModel(nn.Module):
    def __init__(self, num_layers: int = NUM_LAYERS, num_paths: int = NUM_PATHS):
        super().__init__()
        self.num_paths = num_paths
        self.topoflownet = TopoFlowNet(num_layers=num_layers, num_paths=num_paths)
        in_dim = num_paths * (num_layers + 1)
        self.actor = Actor(in_dim=in_dim, out_dim=num_paths)
        self.critic = Critic(in_dim=in_dim)
        self.apply(_init_linear)
        nn.init.uniform_(self.critic.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.critic.net[-1].bias)

    def encode(self, h0, edge_index, num_edges: int) -> torch.Tensor:
        path_emb = self.topoflownet(h0, edge_index, num_edges)
        num_demands = path_emb.size(0) // self.num_paths
        return path_emb.reshape(num_demands, self.num_paths * path_emb.size(-1))

    def value(self, demand_features: torch.Tensor) -> torch.Tensor:
        """Scalar V(s) for r_t = F_total (eq. 30).

        Critic maps each demand's 28-d row to a scalar (P1 §5.3). Aggregation
        to one state value is not stated; we sum so V(s) lives in the same
        units as total allocated traffic.
        """
        return self.critic(demand_features).sum()

    def act(
        self,
        h0,
        edge_index,
        num_edges: int,
        deterministic: bool = False,
    ) -> dict:
        demand_features = self.encode(h0, edge_index, num_edges)
        mean, std = self.actor(demand_features)
        dist = Normal(mean, std)
        if deterministic:
            raw = mean
            log_prob = None
        else:
            raw = dist.rsample()
            log_prob = dist.log_prob(raw).sum(dim=-1)
        raw_clamped = raw.clamp(-RAW_ACTION_CLAMP, RAW_ACTION_CLAMP)
        fractions = torch.softmax(raw_clamped, dim=-1)
        value = self.value(demand_features)
        return {
            "raw": raw,
            "fractions": fractions,
            "log_prob": log_prob,
            "value": value,
            "mean": mean,
            "std": std,
            "demand_features": demand_features,
        }

"""Gaussian actor heads (P1 eqs. 14-19)."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ACTOR_IN_DIM, ACTOR_OUT_DIM, LOG_STD_MAX, LOG_STD_MIN


class Actor(nn.Module):
    """Two linear heads: mean and log-std, both 28 -> 4."""

    def __init__(self, in_dim: int = ACTOR_IN_DIM, out_dim: int = ACTOR_OUT_DIM):
        super().__init__()
        self.mean_head = nn.Linear(in_dim, out_dim)
        self.log_std_head = nn.Linear(in_dim, out_dim)

    def forward(self, demand_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(demand_features)
        log_std = self.log_std_head(demand_features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

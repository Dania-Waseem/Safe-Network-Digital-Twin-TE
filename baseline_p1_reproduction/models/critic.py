"""State-value critic (P1 Section 5.3): 28 -> 256 -> 256 -> 1."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ACTOR_IN_DIM, CRITIC_HIDDEN


class Critic(nn.Module):
    def __init__(self, in_dim: int = ACTOR_IN_DIM, hidden: int = CRITIC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, demand_features: torch.Tensor) -> torch.Tensor:
        """Per-demand values; the A2C wrapper reduces them to V(s)."""
        return self.net(demand_features).squeeze(-1)

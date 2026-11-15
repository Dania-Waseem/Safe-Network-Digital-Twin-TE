"""Ensemble of path-wise multipliers. Link load is the routed flow times the multiplier.

Each member sees path features (normalised demand, routing fraction, hop count)
and outputs a positive multiplier. Link load is the sum of
``demand * action * multiplier`` over paths that use the link. Members that
disagree produce a larger predictive variance. The scatter step is the same
for every topology, so GEANT and Kdl do not need a separate model.
"""

from __future__ import annotations

import torch
from torch import nn


# softplus(0.541324854) ≈ 1, so a zero-weight last layer starts at the physics model.
_UNIT_BIAS = 0.541324854


class TwinMember(nn.Module):
    def __init__(self, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, _UNIT_BIAS)

    def multiplier(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.head(self.backbone(features))).squeeze(-1)


class EnsembleTwin(nn.Module):
    def __init__(self, n_members: int = 5, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [TwinMember(hidden=hidden, dropout=dropout) for _ in range(n_members)]
        )

    def member_loads(
        self,
        features: torch.Tensor,
        flow: torch.Tensor,
        incidence_path: torch.Tensor,
        incidence_link: torch.Tensor,
        n_links: int,
        member: TwinMember,
    ) -> torch.Tensor:
        mult = member.multiplier(features)
        contrib = flow * mult
        load = torch.zeros(n_links, dtype=features.dtype, device=features.device)
        load = load.scatter_add(0, incidence_link, contrib[incidence_path])
        return load

    def predict(
        self,
        features: torch.Tensor,
        flow: torch.Tensor,
        incidence_path: torch.Tensor,
        incidence_link: torch.Tensor,
        n_links: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean load, variance, and the per-member stack [M, E]."""
        loads = [
            self.member_loads(features, flow, incidence_path, incidence_link, n_links, member)
            for member in self.members
        ]
        stack = torch.stack(loads, dim=0)
        mean = stack.mean(dim=0)
        var = stack.var(dim=0, unbiased=False)
        return mean, var, stack

"""Demand and capacity adjustment (P1 eqs. 23-30) plus evaluation metrics."""

from __future__ import annotations

import torch

from config import DEMAND_EPS
from preprocessing.build_bipartite_graph import TEProblem


def _edge_load(flow: torch.Tensor, problem: TEProblem) -> torch.Tensor:
    load = torch.zeros(problem.num_edges, device=flow.device, dtype=flow.dtype)
    load.index_add_(0, problem.p2e[1], flow[problem.p2e[0]])
    return load


def _path_max_util(util: torch.Tensor, problem: TEProblem) -> torch.Tensor:
    """max_{e in p} util_e for every path."""
    n_path = problem.num_demands * problem.num_paths
    path_util = torch.zeros(n_path, device=util.device, dtype=util.dtype)
    incident = util[problem.p2e[1]]
    path_util.scatter_reduce_(0, problem.p2e[0], incident, reduce="amax", include_self=True)
    return path_util


def adjust_allocation(
    flow: torch.Tensor,
    demands: torch.Tensor,
    problem: TEProblem,
    eps: float = DEMAND_EPS,
) -> torch.Tensor:
    """Return a feasible path-flow vector (P1 eqs. 23-28).

    `flow` has shape [D*k] in traffic units (F_d(p) * d).
    """
    k = problem.num_paths
    flow = flow.reshape(-1, k).clone()
    total = flow.sum(dim=-1)
    ratio = total / demands.clamp_min(eps)
    overflow = ratio > 1.0 + eps
    if overflow.any():
        flow[overflow] = flow[overflow] / ratio[overflow].unsqueeze(-1)
    flow = flow.reshape(-1)

    load = _edge_load(flow, problem)
    util = load / problem.capacities.clamp_min(eps)
    path_scale = _path_max_util(util, problem).clamp_min(1.0)
    return flow / path_scale


def evaluate_allocation(
    flow: torch.Tensor,
    demands: torch.Tensor,
    problem: TEProblem,
    eps: float = DEMAND_EPS,
) -> dict:
    """Metrics used in P1: satisfaction, MLU, capacity violations."""
    k = problem.num_paths
    allocated = flow.reshape(-1, k).sum(dim=-1)
    total_demand = demands.sum().clamp_min(eps)
    satisfaction = (allocated.sum() / total_demand).item()
    load = _edge_load(flow, problem)
    cap = problem.capacities.clamp_min(eps)
    util = load / cap
    mlu = util.max().item()
    n_viol = int((load > cap * 1.001 + 1e-3).sum().item())
    viol_rate = n_viol / problem.num_edges
    return {
        "demand_satisfaction": satisfaction,
        "total_allocated": allocated.sum().item(),
        "total_demand": total_demand.item(),
        "max_link_utilization": mlu,
        "capacity_violations": n_viol,
        "violation_rate": viol_rate,
        "link_util": util.detach().cpu(),
    }

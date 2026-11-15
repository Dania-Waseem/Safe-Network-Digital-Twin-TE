"""Forward allocation and held-out evaluation (P1 training vs deployment)."""

from __future__ import annotations

import time

import numpy as np
import torch

from config import ADMM_ITERS
from models.a2c import GRLTEModel
from optimization.admm import admm_refine
from optimization.constraint_adjustment import adjust_allocation, evaluate_allocation
from preprocessing.build_bipartite_graph import TEProblem


@torch.no_grad()
def allocate(
    model: GRLTEModel,
    problem: TEProblem,
    tm: np.ndarray,
    use_admm: bool = True,
    num_admm_iters: int | None = None,
    admm_variant: str = "p1",
) -> dict:
    """Deployment path: deterministic actor -> ADMM -> constraint adjustment."""
    device = problem.capacities.device
    demands = problem.demand_vector(tm, device=device)
    h0 = problem.initial_features(demands)
    t0 = time.perf_counter()
    out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
    flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
    if use_admm:
        iters = ADMM_ITERS.get(problem.name, 2) if num_admm_iters is None else num_admm_iters
        flow = admm_refine(
            out["fractions"], demands, problem, num_iters=iters, variant=admm_variant
        )
    flow = adjust_allocation(flow, demands, problem)
    runtime = time.perf_counter() - t0
    metrics = evaluate_allocation(flow, demands, problem)
    metrics["runtime_s"] = runtime
    metrics["flow"] = flow
    metrics["fractions"] = out["fractions"]
    return metrics


@torch.no_grad()
def evaluate_model(
    model: GRLTEModel,
    problem: TEProblem,
    tms: list[np.ndarray],
    use_admm: bool = True,
    num_admm_iters: int | None = None,
    admm_variant: str = "p1",
) -> dict:
    model.eval()
    rows = []
    for tm in tms:
        rows.append(
            allocate(
                model,
                problem,
                tm,
                use_admm=use_admm,
                num_admm_iters=num_admm_iters,
                admm_variant=admm_variant,
            )
        )
    keys = [
        "demand_satisfaction",
        "max_link_utilization",
        "capacity_violations",
        "violation_rate",
        "runtime_s",
        "total_allocated",
        "total_demand",
    ]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["n"] = len(rows)
    summary["per_tm"] = rows
    return summary

"""Baselines run on the same candidate paths as GRL-TE (P1 Section 5.2)."""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.optimize import linprog

from optimization.constraint_adjustment import adjust_allocation, evaluate_allocation
from preprocessing.build_bipartite_graph import TEProblem


def _incidence(problem: TEProblem) -> np.ndarray:
    n_path = problem.num_demands * problem.num_paths
    mat = np.zeros((problem.num_edges, n_path), dtype=np.float64)
    p2e = problem.p2e.cpu().numpy()
    for path_idx, edge_idx in zip(p2e[0], p2e[1]):
        mat[edge_idx, path_idx] = 1.0
    return mat


def lp_allocate(problem: TEProblem, tm: np.ndarray) -> dict:
    """Path-formulation LP: max throughput s.t. demand and capacity (P1 eqs. 1-4)."""
    demands = problem.demand_vector(tm).cpu().numpy()
    k = problem.num_paths
    n_path = problem.num_demands * k
    incidence = _incidence(problem)
    cap = problem.capacities.cpu().numpy()

    c = -np.ones(n_path)
    a_demand = np.zeros((problem.num_demands, n_path))
    for d in range(problem.num_demands):
        a_demand[d, d * k : (d + 1) * k] = 1.0
    a_ub = np.vstack([a_demand, incidence])
    b_ub = np.concatenate([demands, cap])
    bounds = [(0.0, None)] * n_path

    t0 = time.perf_counter()
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    runtime = time.perf_counter() - t0
    if not result.success or result.x is None:
        flow = torch.zeros(n_path)
    else:
        flow = torch.tensor(result.x, dtype=torch.float32)
    flow = adjust_allocation(flow.to(problem.capacities.device), problem.demand_vector(tm, problem.capacities.device), problem)
    metrics = evaluate_allocation(flow, problem.demand_vector(tm, problem.capacities.device), problem)
    metrics["runtime_s"] = runtime
    metrics["lp_status"] = result.message
    return metrics


def _split_allocate(problem: TEProblem, tm: np.ndarray, fractions: torch.Tensor) -> dict:
    demands = problem.demand_vector(tm, device=problem.capacities.device)
    flow = (fractions * demands.unsqueeze(-1)).reshape(-1)
    t0 = time.perf_counter()
    flow = adjust_allocation(flow, demands, problem)
    runtime = time.perf_counter() - t0
    metrics = evaluate_allocation(flow, demands, problem)
    metrics["runtime_s"] = runtime
    return metrics


def ecmp_allocate(problem: TEProblem, tm: np.ndarray) -> dict:
    fractions = torch.full(
        (problem.num_demands, problem.num_paths),
        1.0 / problem.num_paths,
        device=problem.capacities.device,
    )
    return _split_allocate(problem, tm, fractions)


def shortest_path_allocate(problem: TEProblem, tm: np.ndarray) -> dict:
    fractions = torch.zeros(problem.num_demands, problem.num_paths, device=problem.capacities.device)
    fractions[:, 0] = 1.0
    return _split_allocate(problem, tm, fractions)


def evaluate_baseline(fn, problem: TEProblem, tms: list[np.ndarray]) -> dict:
    rows = [fn(problem, tm) for tm in tms]
    summary = {
        "demand_satisfaction": float(np.mean([r["demand_satisfaction"] for r in rows])),
        "max_link_utilization": float(np.mean([r["max_link_utilization"] for r in rows])),
        "capacity_violations": float(np.mean([r["capacity_violations"] for r in rows])),
        "runtime_s": float(np.mean([r["runtime_s"] for r in rows])),
        "n": len(rows),
        "per_tm": rows,
    }
    return summary

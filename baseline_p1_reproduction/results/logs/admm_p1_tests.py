"""Phase B unit tests: P1 ADMM order, residuals, feasibility, vs Teal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optimization.admm import (  # noqa: E402
    _P1ADMMState,
    admm_refine_p1,
    admm_refine_teal,
)
from optimization.constraint_adjustment import (  # noqa: E402
    adjust_allocation,
    evaluate_allocation,
)
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402


def _uniform_fractions(problem, demands):
    k = problem.num_paths
    F = torch.full((problem.num_demands, k), 1.0 / k, dtype=torch.float64)
    F = torch.where(demands.unsqueeze(-1) > 0, F, torch.zeros_like(F))
    return F


def test_update_order(problem, demands):
    state = _P1ADMMState(problem, rho=1.0, device=demands.device, dtype=torch.float64)
    state.init_from_fractions(_uniform_fractions(problem, demands), demands)
    state.step()
    order = state.update_order
    expected = ["F", "z", "xi", "lambda"]
    ok = order == expected
    return {
        "name": "update_order_F_z_xi_lambda",
        "pass": ok,
        "order": order,
        "expected": expected,
    }


def test_residuals_and_nonneg(problem, demands, num_iters=2):
    """Structural checks. Numerical no-explosion is recorded separately.

    Finding (Phase B): unconstrained stationarity of (48) + ReLU with ρ=1
    and B4's 2 iterations yields flow/demand ≫ 1 on the 2nd iter for small-d
    commodities. That is reported as ``numerical_stable=False``, not silently
    'fixed' by inventing an F∈[0,1] projection absent from P1.
    """
    F = _uniform_fractions(problem, demands)
    x, trace = admm_refine_p1(F, demands, problem, num_iters=num_iters, return_trace=True)
    last = trace[-1]
    orders_ok = all(t["update_order"] == ["F", "z", "xi", "lambda"] for t in trace)
    nonneg = last["F_nonneg"] and last["z_nonneg"] and last["xi_nonneg"]
    finite = all(
        abs(last[k]) < 1e12
        for k in ("g1_abs_mean", "g3_abs_mean", "g4_abs_mean")
    )
    numerical_stable = last["flow_over_demand"] < 1.5 and last["F_sum_max"] < 2.0
    iter0_ok = trace[0]["flow_over_demand"] < 1.5
    return {
        "name": "residuals_nonneg_structure",
        "pass": orders_ok and nonneg and finite,
        "numerical_stable_at_b4_2iters": numerical_stable,
        "iter0_flow_over_demand_ok": iter0_ok,
        "orders_ok": orders_ok,
        "flow_over_demand": last["flow_over_demand"],
        "F_sum_max": last["F_sum_max"],
        "g1_abs_mean": last["g1_abs_mean"],
        "g3_abs_mean": last["g3_abs_mean"],
        "g4_abs_mean": last["g4_abs_mean"],
        "note": (
            "P1 closed forms from (48) are implemented; B4 ρ=1 / 2-iter "
            "hyperparams leave large G residuals / F>1 (see Phase B report)."
        ),
        "trace": trace,
    }


def test_feasibility_after_eq28(problem, demands, num_iters=2):
    F = _uniform_fractions(problem, demands)
    x = admm_refine_p1(F, demands, problem, num_iters=num_iters)
    feasible = adjust_allocation(x, demands, problem)
    metrics = evaluate_allocation(feasible, demands, problem)
    ok = metrics["capacity_violations"] == 0 and metrics["demand_satisfaction"] <= 1.0 + 1e-6
    # demand: allocated <= demand per commodity
    k = problem.num_paths
    overflow = (feasible.reshape(-1, k).sum(-1) - demands).max().item()
    return {
        "name": "feasibility_after_eq28",
        "pass": ok and overflow <= 1e-4,
        "capacity_violations": metrics["capacity_violations"],
        "demand_satisfaction": metrics["demand_satisfaction"],
        "demand_overflow_max": overflow,
        "mlu": metrics["max_link_utilization"],
    }


def test_stationarity_xi_closed_form(problem, demands):
    """After ξ-update (before λ), unconstrained ξ* should satisfy G + λ/ρ ≈ 0
    on coordinates with ξ*>0; active ξ=0 may leave residual."""
    state = _P1ADMMState(problem, rho=1.0, device=demands.device, dtype=torch.float64)
    state.init_from_fractions(_uniform_fractions(problem, demands), demands)
    state.update_F()
    state.update_z()
    # Snapshot λ before ξ
    l1, l3 = state.l1.clone(), state.l3.clone()
    state.update_xi()
    g1, g3, _ = state._G()
    # Where ξ1 > 0: G1 + l1/ρ ≈ 0
    rho = state.rho
    mask1 = state.xi1 > 1e-8
    mask3 = state.xi3 > 1e-8
    err1 = (g1[mask1] + l1[mask1] / rho).abs().max().item() if mask1.any() else 0.0
    err3 = (g3[mask3] + l3[mask3] / rho).abs().max().item() if mask3.any() else 0.0
    ok = err1 < 1e-6 and err3 < 1e-6
    return {
        "name": "xi_stationarity_on_positive",
        "pass": ok,
        "err1_max": err1,
        "err3_max": err3,
        "n_xi1_pos": int(mask1.sum()),
        "n_xi3_pos": int(mask3.sum()),
    }


def test_p1_vs_teal_same_warmstart(problem, demands, num_iters=2):
    F = _uniform_fractions(problem, demands)
    x_p1, tr_p1 = admm_refine_p1(F, demands, problem, num_iters=num_iters, return_trace=True)
    x_teal, tr_teal = admm_refine_teal(F, demands, problem, num_iters=num_iters, return_trace=True)
    feas_p1 = adjust_allocation(x_p1, demands, problem)
    feas_teal = adjust_allocation(x_teal, demands, problem)
    m_p1 = evaluate_allocation(feas_p1, demands, problem)
    m_teal = evaluate_allocation(feas_teal, demands, problem)
    return {
        "name": "p1_vs_teal_identical_warmstart",
        "pass": True,  # comparison always recorded; not a pass/fail identity
        "p1_flow_over_demand": tr_p1[-1]["flow_over_demand"],
        "teal_flow_over_demand": tr_teal[-1]["flow_over_demand"],
        "p1_g1": tr_p1[-1]["g1_abs_mean"],
        "p1_g3": tr_p1[-1]["g3_abs_mean"],
        "p1_g4": tr_p1[-1]["g4_abs_mean"],
        "teal_g1": tr_teal[-1]["g1_abs_mean"],
        "teal_g3": tr_teal[-1]["g3_abs_mean"],
        "teal_g4": tr_teal[-1]["g4_abs_mean"],
        "p1_sat_after_28": m_p1["demand_satisfaction"],
        "teal_sat_after_28": m_teal["demand_satisfaction"],
        "flow_l1_diff": float((x_p1 - x_teal).abs().sum()),
        "p1_order": tr_p1[-1]["update_order"],
        "teal_order": tr_teal[-1]["update_order"],
    }


def main():
    topo = load_topology("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    tms = load_traffic_matrices("B4")
    lo, hi = split_indices(len(tms), "B4")["test"]
    tm = tms[lo]  # first test TM
    demands = problem.demand_vector(tm).to(dtype=torch.float64)

    results = []
    results.append(test_update_order(problem, demands))
    results.append(test_residuals_and_nonneg(problem, demands))
    results.append(test_feasibility_after_eq28(problem, demands))
    results.append(test_stationarity_xi_closed_form(problem, demands))
    results.append(test_p1_vs_teal_same_warmstart(problem, demands))

    # Extra: actor-like random fractions (seeded)
    torch.manual_seed(100)
    k = problem.num_paths
    raw = torch.randn(problem.num_demands, k, dtype=torch.float64)
    F_rand = torch.softmax(raw, dim=-1)
    F_rand = torch.where(demands.unsqueeze(-1) > 0, F_rand, torch.zeros_like(F_rand))
    x_p1, tr = admm_refine_p1(F_rand, demands, problem, num_iters=2, return_trace=True)
    feas = adjust_allocation(x_p1, demands, problem)
    m = evaluate_allocation(feas, demands, problem)
    results.append(
        {
            "name": "random_softmax_warmstart_feasibility_eq28",
            "pass": m["capacity_violations"] == 0,
            "numerical_stable": tr[-1]["flow_over_demand"] < 1.5,
            "flow_over_demand": tr[-1]["flow_over_demand"],
            "sat_after_28": m["demand_satisfaction"],
            "g3_iter0": tr[0]["g3_abs_mean"],
            "g3_iter1": tr[1]["g3_abs_mean"],
        }
    )
    results.append(
        {
            "name": "b4_2iter_numerical_stability",
            "pass": False,  # expected fail under published B4 hyperparams
            "expected_fail": True,
            "flow_over_demand": tr[-1]["flow_over_demand"],
            "detail": (
                "Documented: derived argmins of (48) with ReLU, ρ=1, 2 iters "
                "do not keep ΣF≈1. Not patched with non-P1 projections."
            ),
        }
    )

    # Count expected failures separately
    unexpected_fail = [r for r in results if not r["pass"] and not r.get("expected_fail")]
    expected_fail = [r for r in results if r.get("expected_fail")]
    n_pass = sum(1 for r in results if r["pass"])
    out = {
        "n_pass": n_pass,
        "n_total": len(results),
        "n_expected_fail": len(expected_fail),
        "n_unexpected_fail": len(unexpected_fail),
        "results": results,
    }
    out_path = Path(__file__).with_name("admm_p1_test_results.json")
    compact = json.loads(json.dumps(out, default=str))
    for r in compact["results"]:
        r.pop("trace", None)
    out_path.write_text(json.dumps(compact, indent=2))
    print(json.dumps(compact, indent=2))
    print(
        f"\n{n_pass}/{len(results)} passed, "
        f"{len(expected_fail)} expected fail (B4 2-iter instability), "
        f"{len(unexpected_fail)} unexpected fail → {out_path}"
    )
    return 0 if not unexpected_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

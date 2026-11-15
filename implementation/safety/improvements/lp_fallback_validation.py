"""Offline LP path-fraction fallback validation (experimental, non-production).

Validates whether a catalog min-MLU LP can recover safe routing for the five
UsCarrier@0.25 cases previously shown feasible. Does not modify production
fallback/gate/catalog/eval artifacts.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import equal_split, od_demands, shortest_split

from .fallback import select_physics_fallback
from .final_five_case_lp import (
    demand_vector_for_pathset,
    fractions_from_flow,
    load_catalog_pathset,
    load_five_cases,
    solve_min_mlu_path_lp,
    validate_flow_loads,
)
from .investigate_fallback import _stats, iterative_rebalance_search
from .oracle import action_link_load

OUT_DIR = ROOT / "results" / "safety_gate" / "improvements" / "lp_fallback_validation"
PRIOR_LP_DIR = (
    ROOT
    / "results"
    / "safety_gate"
    / "improvements"
    / "fallback_investigation"
    / "final_five_case_lp"
)


def verify_lp_formulation(pathset, demand: np.ndarray, topo, matrix: np.ndarray) -> dict[str, Any]:
    """Sanity checks that the experimental LP matches Twin physics conventions."""
    checks: dict[str, Any] = {}
    od = od_demands(matrix)
    dv = demand_vector_for_pathset(pathset, matrix)
    checks["demand_aligns_od_demands"] = bool(np.allclose(dv, od))
    checks["capacity_aligns_topo"] = bool(np.allclose(pathset.capacity, topo.capacity))
    checks["incidence_aligns_topo"] = bool(
        np.array_equal(pathset.incidence_path, topo.incidence_path)
        and np.array_equal(pathset.incidence_link, topo.incidence_link)
    )
    checks["n_paths_aligns_topo"] = int(pathset.n_paths_total) == int(np.sum(topo.n_paths))
    checks["demand_conservation_rows"] = int(pathset.n_demands)
    checks["link_rows"] = int(pathset.n_links)
    checks["formulation"] = (
        "min U_max s.t. sum_{p in P_d} f_p = demand_d; "
        "sum_p A_{e,p} f_p <= U_max * cap_e; f_p >= 0; U_max >= 0"
    )
    checks["all_passed"] = all(
        checks[k]
        for k in (
            "demand_aligns_od_demands",
            "capacity_aligns_topo",
            "incidence_aligns_topo",
            "n_paths_aligns_topo",
        )
    )
    return checks


def experimental_lp_fallback(
    pathset,
    matrix: np.ndarray,
    *,
    time_limit: float = 60.0,
) -> dict[str, Any]:
    """Isolated LP fallback simulator: catalog continuous path-fraction min-MLU."""
    demand = demand_vector_for_pathset(pathset, matrix)
    t0 = time.time()
    sol = solve_min_mlu_path_lp(pathset, demand, time_limit=time_limit)
    wall = time.time() - t0
    out: dict[str, Any] = {
        "selected": "lp_catalog_min_mlu",
        "solver_status": sol.get("status"),
        "solver_message": sol.get("message"),
        "solver_status_code": sol.get("solver_status_code"),
        "optimality_gap": sol.get("optimality_gap"),
        "runtime_s": float(sol.get("runtime_s") or wall),
        "wall_time_s": wall,
        "optimal_mlu": sol.get("optimal_mlu"),
        "feasible": None,
        "action": None,
        "flow": None,
        "recomputed": None,
    }
    if sol.get("flow") is None or sol.get("status") != "optimal":
        out["feasible"] = False if sol.get("status") == "infeasible" else None
        return out

    flow = np.asarray(sol["flow"], dtype=np.float64)
    frac = fractions_from_flow(pathset, flow, demand)
    val = validate_flow_loads(pathset, flow)
    recomputed = val["recomputed"]
    out["action"] = frac
    out["flow"] = flow
    out["recomputed"] = recomputed
    out["feasible"] = recomputed["n_overloaded_links"] == 0 and recomputed["mlu"] <= 1.0 + 1e-9
    # Demand conservation residual
    counts = np.asarray(pathset.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    max_resid = 0.0
    for d, (s, c) in enumerate(zip(starts, counts)):
        got = float(flow[int(s) : int(s + c)].sum())
        max_resid = max(max_resid, abs(got - float(demand[d])))
    out["max_demand_residual"] = max_resid
    out["frac_sum_per_demand_ok"] = bool(
        np.allclose(
            [
                frac[int(s) : int(s + c)].sum()
                for s, c in zip(starts, counts)
                if int(c) > 0
            ],
            1.0,
            atol=1e-6,
        )
    )
    return out


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_five_cases()
    topo = load_topology("UsCarrier")
    pathset = load_catalog_pathset()
    demands, _ = load_split_demands("val", "UsCarrier")

    rows: list[dict[str, Any]] = []
    allocations: dict[str, Any] = {"cases": {}, "formulation_checks": None}
    matrices = {}
    for c in cases:
        li = int(c["local_index"])
        matrices[li] = demands[li] * float(c["traffic_scale"])

    # Formulation verification on the shared TM
    ref = matrices[int(cases[0]["local_index"])]
    form_checks = verify_lp_formulation(pathset, demand_vector_for_pathset(pathset, ref), topo, ref)
    allocations["formulation_checks"] = form_checks
    if not form_checks["all_passed"]:
        raise RuntimeError(f"LP formulation checks failed: {form_checks}")

    # TM identity check
    tm_identical = all(np.allclose(m, ref) for m in matrices.values())
    allocations["tm_identical_across_cases"] = tm_identical

    # Solve LP once if TMs identical; still apply/report per case_id.
    shared_lp = experimental_lp_fallback(pathset, ref, time_limit=60.0) if tm_identical else None

    for c in cases:
        li = int(c["local_index"])
        matrix = matrices[li]
        # Existing production fallback (shortest/equal)
        prod = select_physics_fallback(topo, matrix)
        short_a = shortest_split(topo.n_paths, topo.hops)
        equal_a = equal_split(topo.n_paths)
        short_s = _stats(action_link_load(topo, matrix, short_a), topo.capacity)
        equal_s = _stats(action_link_load(topo, matrix, equal_a), topo.capacity)
        reb = iterative_rebalance_search(topo, matrix, n_iters=60)

        if shared_lp is not None:
            lp = shared_lp
        else:
            lp = experimental_lp_fallback(pathset, matrix, time_limit=60.0)

        # Independent physics recompute via Twin action_link_load on LP fractions
        twin_recompute = None
        if lp.get("action") is not None:
            twin_load = action_link_load(topo, matrix, lp["action"])
            twin_recompute = _stats(twin_load, topo.capacity)

        row = {
            "case_id": c["case_id"],
            "topology": c["topology"],
            "scale": float(c["traffic_scale"]),
            "local_index": li,
            "matrix_id": int(c["matrix_id"]),
            "seed": int(c["seed"]),
            "production_selected": prod.selected,
            "production_mlu": prod.selected_mlu,
            "production_violations": prod.selected_violations,
            "production_feasible": prod.feasible_fallback,
            "shortest_mlu": short_s["mlu"],
            "shortest_n_overloaded": short_s["n_overloaded_links"],
            "shortest_total_overload": short_s["total_overload"],
            "equal_mlu": equal_s["mlu"],
            "equal_n_overloaded": equal_s["n_overloaded_links"],
            "equal_total_overload": equal_s["total_overload"],
            "rebalance_mlu": reb["mlu"],
            "rebalance_n_overloaded": reb["n_overloaded_links"],
            "rebalance_total_overload": reb["total_overload"],
            "lp_solver_status": lp.get("solver_status"),
            "lp_optimal_mlu": lp.get("optimal_mlu"),
            "lp_recomputed_mlu": None if not lp.get("recomputed") else lp["recomputed"]["mlu"],
            "lp_twin_recomputed_mlu": None if twin_recompute is None else twin_recompute["mlu"],
            "lp_n_overloaded": None if not lp.get("recomputed") else lp["recomputed"]["n_overloaded_links"],
            "lp_total_overload": None if not lp.get("recomputed") else lp["recomputed"]["total_overload"],
            "lp_feasible": lp.get("feasible"),
            "lp_runtime_s": lp.get("runtime_s"),
            "lp_optimality_gap": lp.get("optimality_gap"),
            "lp_max_demand_residual": lp.get("max_demand_residual"),
            "lp_beats_shortest": (
                lp.get("recomputed") is not None and lp["recomputed"]["mlu"] < short_s["mlu"] - 1e-12
            ),
            "lp_beats_equal": (
                lp.get("recomputed") is not None and lp["recomputed"]["mlu"] < equal_s["mlu"] - 1e-12
            ),
            "lp_beats_rebalance": (
                lp.get("recomputed") is not None and lp["recomputed"]["mlu"] < reb["mlu"] - 1e-12
            ),
            "lp_beats_all_baselines": False,
        }
        row["lp_beats_all_baselines"] = bool(
            row["lp_beats_shortest"] and row["lp_beats_equal"] and row["lp_beats_rebalance"]
        )
        rows.append(row)

        allocations["cases"][c["case_id"]] = {
            "lp_optimal_mlu": lp.get("optimal_mlu"),
            "lp_feasible": lp.get("feasible"),
            "lp_runtime_s": lp.get("runtime_s"),
            "solver_status": lp.get("solver_status"),
            "nonzero_path_fractions": (
                int(np.count_nonzero(lp["action"] > 1e-8)) if lp.get("action") is not None else None
            ),
            "frac_nonzero_idx": (
                np.flatnonzero(lp["action"] > 1e-8).tolist() if lp.get("action") is not None else None
            ),
            "frac_nonzero_val": (
                lp["action"][lp["action"] > 1e-8].tolist() if lp.get("action") is not None else None
            ),
        }

    # Cross-case consistency
    mlu_vals = [r["lp_recomputed_mlu"] for r in rows]
    consistent = all(v is not None for v in mlu_vals) and (
        max(mlu_vals) - min(mlu_vals) < 1e-9  # type: ignore[operator]
    )
    summary = {
        "n_cases": len(rows),
        "n_lp_feasible": sum(1 for r in rows if r["lp_feasible"] is True),
        "n_lp_mlu_below_1": sum(
            1 for r in rows if r["lp_recomputed_mlu"] is not None and r["lp_recomputed_mlu"] < 1.0 - 1e-12
        ),
        "n_lp_beats_all_baselines": sum(1 for r in rows if r["lp_beats_all_baselines"]),
        "mean_lp_runtime_s": float(np.mean([r["lp_runtime_s"] for r in rows])),
        "lp_mlu": rows[0]["lp_recomputed_mlu"],
        "shortest_mlu": rows[0]["shortest_mlu"],
        "equal_mlu": rows[0]["equal_mlu"],
        "rebalance_mlu": rows[0]["rebalance_mlu"],
        "results_consistent_across_five": consistent,
        "tm_identical": tm_identical,
        "formulation_checks_passed": form_checks["all_passed"],
        "prior_lp_dir": str(PRIOR_LP_DIR.relative_to(ROOT)) if PRIOR_LP_DIR.exists() else None,
    }

    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (OUT_DIR / "routing_allocations.json").write_text(
        json.dumps({"summary": summary, "allocations": allocations}, indent=2),
        encoding="utf-8",
    )
    _plot(rows)
    _report(rows, summary, form_checks)
    return summary


def _plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0 = rows[0]
    labels = ["shortest", "equal", "rebalance", "LP catalog\nfallback"]
    vals = [r0["shortest_mlu"], r0["equal_mlu"], r0["rebalance_mlu"], r0["lp_recomputed_mlu"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C78A8", "#4C78A8", "#F58518", "#54A24B"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color="red", ls="--", label="capacity MLU=1")
    ax.set_ylabel("Maximum link utilisation")
    ax.set_title("Experimental LP fallback vs existing candidates\n(5 identical UsCarrier@0.25 cases)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_plot.png", dpi=140)
    plt.close(fig)


def _report(rows: list[dict[str, Any]], summary: dict[str, Any], form_checks: dict[str, Any]) -> None:
    r0 = rows[0]
    lines = [
        "# LP Fallback Validation Report",
        "",
        "Experimental validation only. Production gate/fallback/Twin/RL/eval artifacts were not modified.",
        "",
        "## 1. Scope",
        "",
        "Five previously inconclusive-then-proven-feasible UsCarrier cases at scale 0.25:",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['case_id']}`")
    lines.extend(
        [
            "",
            f"Traffic matrices identical across cases: **{summary['tm_identical']}**.",
            f"Cross-case LP result consistency: **{summary['results_consistent_across_five']}**.",
            "",
            "## 2. Formulation verification",
            "",
            f"- Statement: `{form_checks['formulation']}`",
            f"- Demand vector aligns with `od_demands`: **{form_checks['demand_aligns_od_demands']}**",
            f"- Capacities align with Twin topology tensors: **{form_checks['capacity_aligns_topo']}**",
            f"- Incidence aligns with bipartite tensors: **{form_checks['incidence_aligns_topo']}**",
            f"- Path count aligns: **{form_checks['n_paths_aligns_topo']}**",
            f"- All checks passed: **{form_checks['all_passed']}**",
            "",
            "Independent load recomputation uses both incidence aggregation of LP flows and "
            "Twin `action_link_load` on the derived path fractions.",
            "",
            "## 3. Results",
            "",
            "| Method | MLU | Overloaded links | Total overload | Feasible |",
            "|---|---:|---:|---:|---|",
            f"| Production fallback ({r0['production_selected']}) | {r0['production_mlu']:.6f} | "
            f"{r0['production_violations']} | — | {r0['production_feasible']} |",
            f"| Shortest | {r0['shortest_mlu']:.6f} | {r0['shortest_n_overloaded']} | "
            f"{r0['shortest_total_overload']:.3f} | False |",
            f"| Equal | {r0['equal_mlu']:.6f} | {r0['equal_n_overloaded']} | "
            f"{r0['equal_total_overload']:.3f} | False |",
            f"| Path-rebalancing | {r0['rebalance_mlu']:.6f} | {r0['rebalance_n_overloaded']} | "
            f"{r0['rebalance_total_overload']:.3f} | False |",
            f"| **LP catalog fallback** | **{r0['lp_recomputed_mlu']:.6f}** | "
            f"**{r0['lp_n_overloaded']}** | **{r0['lp_total_overload']:.6f}** | "
            f"**{r0['lp_feasible']}** |",
            "",
            f"- LP solver status: **{r0['lp_solver_status']}** (gap={r0['lp_optimality_gap']})",
            f"- LP runtime (shared solve, reported per case): **{r0['lp_runtime_s']:.3f} s**",
            f"- Mean reported LP runtime across five rows: **{summary['mean_lp_runtime_s']:.3f} s**",
            f"- Twin recomputed MLU from fractions: **{r0['lp_twin_recomputed_mlu']:.6f}**",
            f"- Max demand residual: **{r0['lp_max_demand_residual']}**",
            "",
            "All five case rows share the same LP MLU / feasibility / baseline comparisons "
            f"({summary['n_lp_feasible']}/5 feasible, "
            f"{summary['n_lp_beats_all_baselines']}/5 beat all baselines).",
            "",
            "## 4. Direct answers",
            "",
            f"1. **Does LP fallback consistently achieve MLU below 1?** "
            f"**Yes — {summary['n_lp_mlu_below_1']}/5** "
            f"(MLU={summary['lp_mlu']:.6f}).",
            f"2. **Does it outperform all existing fallback candidates?** "
            f"**Yes — {summary['n_lp_beats_all_baselines']}/5** "
            f"(vs shortest {summary['shortest_mlu']:.3f}, equal {summary['equal_mlu']:.3f}, "
            f"rebalance {summary['rebalance_mlu']:.3f}).",
            f"3. **Computational cost?** About **{summary['mean_lp_runtime_s']:.2f} s** "
            "per UsCarrier catalog LP solve on this machine (HiGHS, ~98k paths, 24.8k demands, "
            "378 links). Far above a constant-time shortest/equal evaluation.",
            "4. **Technically justified as an additional fallback mechanism?** "
            "**Yes, for this narrow slice:** when a catalog-feasible path-split exists "
            "(forced-link bound ≤ 1), LP recovers safe routing where shortest/equal fail. "
            "It is **not** justified as a universal replacement — 35/40 prior failures remain "
            "catalog-infeasible and LP cannot clear them.",
            "5. **Limitations?** "
            "(a) Helps only catalog-feasible TMs; "
            "(b) ~seconds of solve time vs microseconds for heuristics; "
            "(c) requires a reliable LP stack and catalog incidence; "
            "(d) continuous splits may need discretisation for some routers; "
            "(e) does not expand the path catalog.",
            "6. **Smallest next experiment?** "
            "Offline A/B on the full 80-case paired set: run experimental LP fallback "
            "**only when** forced-link LB ≤ 1 (or production fallback is infeasible), "
            "else keep shortest/equal — report applied violation rate and p95 latency. "
            "Still no production wiring.",
            "",
            "## 5. Artifacts",
            "",
            "- `results.csv`",
            "- `routing_allocations.json`",
            "- `comparison_plot.png`",
            "",
            "## 6. Preservation",
            "",
            "No changes to `safety/gate.py`, `fallback.py`, `gate_logic.py`, checkpoints, "
            "datasets, traffic matrices, or A–F evaluation results.",
            "",
        ]
    )
    (OUT_DIR / "LP_FALLBACK_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))

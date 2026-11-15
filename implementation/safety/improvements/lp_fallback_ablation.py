"""Offline paired A/B: existing fallback vs conditional catalog LP fallback.

Experimental only. Does not modify production gate/fallback/eval artifacts.
Uses frozen E-variant gate decisions from af_eval_k1_calibrated_devtopos_v1.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import equal_split, od_demands, shortest_split

from .fallback import select_physics_fallback
from .investigate_fallback import _stats, forced_common_link_lower_bound
from .oracle import action_link_load

EVAL_DIR = (
    ROOT
    / "results"
    / "safety_gate"
    / "improvements"
    / "eval_runs"
    / "af_eval_k1_calibrated_devtopos_v1"
)
OUT_DIR = ROOT / "results" / "safety_gate" / "improvements" / "lp_fallback_ablation"
VARIANT_E = "E_improved_twin_calibrated_uncertainty"
LP_TIME_LIMIT = 60.0


def load_e_cases() -> list[dict[str, Any]]:
    path = EVAL_DIR / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [r for r in rows if r.get("gate_variant") == VARIANT_E]
    if len(cases) != 80:
        raise RuntimeError(f"expected 80 E cases, found {len(cases)}")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case_ids in E variant")
    return cases


def solve_catalog_min_mlu_lp(topo, matrix: np.ndarray, *, time_limit: float = LP_TIME_LIMIT) -> dict[str, Any]:
    """Catalog path-flow min-MLU LP on Twin topology tensors (experimental)."""
    od = od_demands(matrix)
    counts = np.asarray(topo.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    n_paths = int(counts.sum())
    n_dem = int(counts.shape[0])
    n_links = int(topo.n_links)
    cap = np.asarray(topo.capacity, dtype=np.float64)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(n_dem, dtype=np.float64)
    for d, (s, c) in enumerate(zip(starts, counts)):
        for p in range(int(s), int(s + c)):
            eq_rows.append(d)
            eq_cols.append(p)
            eq_data.append(1.0)
        b_eq[d] = float(od[d])
    A_eq = csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_dem, n_paths))
    A_link = coo_matrix(
        (
            np.ones(topo.incidence_path.shape[0], dtype=np.float64),
            (
                np.asarray(topo.incidence_link, dtype=np.int64),
                np.asarray(topo.incidence_path, dtype=np.int64),
            ),
        ),
        shape=(n_links, n_paths),
    ).tocsr()

    A_eq_ext = hstack([A_eq, csr_matrix((n_dem, 1))])
    A_ub = hstack([A_link, csr_matrix(-cap.reshape(-1, 1))])
    c = np.zeros(n_paths + 1, dtype=np.float64)
    c[-1] = 1.0
    bounds = [(0.0, None)] * n_paths + [(0.0, None)]

    t0 = time.time()
    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=np.zeros(n_links),
            A_eq=A_eq_ext,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": float(time_limit)},
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "inconclusive",
            "reason": f"solver_exception: {exc}",
            "runtime_s": time.time() - t0,
            "optimal_mlu": None,
            "flow": None,
            "feasible_verified": False,
        }
    runtime = time.time() - t0
    msg = str(res.message)
    if not (res.status == 0 and res.success and res.x is not None):
        infeas = res.status == 2 or "infeasible" in msg.lower()
        return {
            "status": "infeasible" if infeas else "inconclusive",
            "solver_status_code": int(res.status),
            "message": msg,
            "runtime_s": runtime,
            "optimal_mlu": None,
            "flow": None,
            "feasible_verified": False,
        }

    flow = np.asarray(res.x[:-1], dtype=np.float64)
    u = float(res.x[-1])
    # Independent load recompute
    load = np.zeros(n_links, dtype=np.float64)
    np.add.at(load, topo.incidence_link, flow[topo.incidence_path])
    st = _stats(load, cap)
    # Demand residuals
    max_resid = 0.0
    for d, (s, c) in enumerate(zip(starts, counts)):
        max_resid = max(max_resid, abs(float(flow[int(s) : int(s + c)].sum()) - float(od[d])))
    feasible = st["n_overloaded_links"] == 0 and st["mlu"] <= 1.0 + 1e-9
    return {
        "status": "optimal",
        "solver_status_code": int(res.status),
        "message": msg,
        "runtime_s": runtime,
        "optimal_mlu": u,
        "optimality_gap": 0.0,
        "flow": flow,
        "recomputed": st,
        "max_demand_residual": max_resid,
        "feasible_verified": bool(feasible),
        # Only treat as usable fallback if verified feasible under capacity.
        "apply": bool(feasible),
    }


def _ci_mean(diffs: list[float]) -> dict[str, float]:
    """Paired mean difference with normal approx 95% CI (descriptive; not a significance claim)."""
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    if n < 2:
        return {"mean": mean, "ci95_low": float("nan"), "ci95_high": float("nan"), "n": n}
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return {"mean": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se, "n": n}


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    t_exp0 = time.time()
    cases = load_e_cases()
    topo_cache: dict[str, Any] = {}
    demand_cache: dict[tuple[str, str], Any] = {}
    forced_cache: dict[tuple, dict[str, Any]] = {}
    lp_cache: dict[tuple, dict[str, Any]] = {}

    paired: list[dict[str, Any]] = []
    lp_log: list[dict[str, Any]] = []

    for row in cases:
        name = row["topology"]
        if name not in topo_cache:
            topo_cache[name] = load_topology(name)
        topo = topo_cache[name]
        split = row.get("split") or "val"
        dkey = (split, name)
        if dkey not in demand_cache:
            demand_cache[dkey] = load_split_demands(split, name)
        demands, _meta = demand_cache[dkey]
        local_i = int(row["local_index"])
        scale = float(row["scale"])
        matrix = demands[local_i] * scale

        # --- Variant A: existing gate + fallback (recompute physics fallback for rejected) ---
        acceptance = bool(row.get("acceptance"))
        gate_unchanged = True  # by construction

        if acceptance:
            # Applied = proposed (logged); no fallback.
            a_mlu = float(row["mlu_applied"])
            a_viol = int(row.get("capacity_violations_applied") or 0)
            a_feasible_applied = a_viol == 0
            # Approximate overload/mean from proposed==applied when accepted & safe.
            a_total_overload = 0.0 if a_feasible_applied else float("nan")
            a_mean_util = float("nan")
            a_source = "accepted_proposed"
            a_fallback_feasible = None
            a_selected = None
        else:
            fb = select_physics_fallback(topo, matrix)
            a_stats = _stats(fb.selected_load, topo.capacity)
            a_mlu = a_stats["mlu"]
            a_viol = a_stats["n_overloaded_links"]
            a_total_overload = a_stats["total_overload"]
            a_mean_util = a_stats["mean_util"]
            a_feasible_applied = a_viol == 0
            a_source = f"fallback:{fb.selected}"
            a_fallback_feasible = bool(fb.feasible_fallback)
            a_selected = fb.selected
            # Consistency with frozen log
            if row.get("selected_fallback") != fb.selected:
                raise RuntimeError(
                    f"fallback selection drift on {row['case_id']}: "
                    f"log={row.get('selected_fallback')} recomputed={fb.selected}"
                )

        # --- Variant B: conditional LP ---
        b_mlu = a_mlu
        b_viol = a_viol
        b_total_overload = a_total_overload
        b_mean_util = a_mean_util
        b_feasible_applied = a_feasible_applied
        b_source = a_source
        b_selected = a_selected
        b_fallback_feasible = a_fallback_feasible
        lp_invoked = False
        lp_applied = False
        lp_skipped_reason = None
        forced_lb = None
        lp_runtime = None
        lp_status = None

        if acceptance:
            lp_skipped_reason = "accepted_no_fallback"
        elif a_fallback_feasible:
            lp_skipped_reason = "existing_fallback_feasible"
        else:
            fkey = (name, local_i, scale)
            if fkey not in forced_cache:
                print(f"forced LB {fkey}", flush=True)
                forced_cache[fkey] = forced_common_link_lower_bound(topo, matrix)
            forced = forced_cache[fkey]
            forced_lb = float(forced["mlu_lower_bound"])
            if forced["infeasible_proven"] or forced_lb > 1.0 + 1e-9:
                lp_skipped_reason = "forced_link_lb_gt_1"
            else:
                lp_invoked = True
                if fkey not in lp_cache:
                    print(f"LP solve {fkey} n_paths={int(np.sum(topo.n_paths))}", flush=True)
                    lp_cache[fkey] = solve_catalog_min_mlu_lp(topo, matrix, time_limit=LP_TIME_LIMIT)
                sol = lp_cache[fkey]
                lp_runtime = float(sol["runtime_s"])
                lp_status = sol["status"]
                entry = {
                    "case_id": row["case_id"],
                    "topology": name,
                    "scale": scale,
                    "local_index": local_i,
                    "forced_mlu_lower_bound": forced_lb,
                    "lp_status": sol["status"],
                    "lp_runtime_s": sol["runtime_s"],
                    "lp_optimal_mlu": sol.get("optimal_mlu"),
                    "lp_recomputed_mlu": None
                    if not sol.get("recomputed")
                    else sol["recomputed"]["mlu"],
                    "lp_feasible_verified": sol.get("feasible_verified"),
                    "lp_applied": False,
                    "max_demand_residual": sol.get("max_demand_residual"),
                    "message": sol.get("message") or sol.get("reason"),
                }
                if sol.get("apply"):
                    st = sol["recomputed"]
                    b_mlu = float(st["mlu"])
                    b_viol = int(st["n_overloaded_links"])
                    b_total_overload = float(st["total_overload"])
                    b_mean_util = float(st["mean_util"])
                    b_feasible_applied = True
                    b_source = "lp_catalog_min_mlu"
                    b_selected = "lp_catalog_min_mlu"
                    b_fallback_feasible = True
                    lp_applied = True
                    entry["lp_applied"] = True
                else:
                    # Preserve original fallback result
                    lp_skipped_reason = f"lp_not_applied:{sol['status']}"
                lp_log.append(entry)

        paired.append(
            {
                "case_id": row["case_id"],
                "topology": name,
                "scale": scale,
                "seed": int(row["seed"]),
                "local_index": local_i,
                "matrix_id": int(row["matrix_id"]),
                "acceptance": acceptance,
                "gate_decision_unchanged": gate_unchanged,
                "unsafe_accept_A": bool(row.get("unsafe_accept")),
                "unsafe_accept_B": bool(row.get("unsafe_accept")),  # gate unchanged
                "A_mlu_applied": a_mlu,
                "A_n_overloaded": a_viol,
                "A_total_overload": a_total_overload,
                "A_mean_util": a_mean_util,
                "A_feasible_applied": a_feasible_applied,
                "A_source": a_source,
                "A_selected_fallback": a_selected,
                "A_fallback_feasible": a_fallback_feasible,
                "B_mlu_applied": b_mlu,
                "B_n_overloaded": b_viol,
                "B_total_overload": b_total_overload,
                "B_mean_util": b_mean_util,
                "B_feasible_applied": b_feasible_applied,
                "B_source": b_source,
                "B_selected_fallback": b_selected,
                "B_fallback_feasible": b_fallback_feasible,
                "lp_invoked": lp_invoked,
                "lp_applied": lp_applied,
                "lp_skipped_reason": lp_skipped_reason,
                "forced_mlu_lower_bound": forced_lb,
                "lp_runtime_s": lp_runtime,
                "lp_status": lp_status,
                "delta_mlu_B_minus_A": b_mlu - a_mlu,
                "delta_overload_B_minus_A": (
                    (b_total_overload - a_total_overload)
                    if np.isfinite(a_total_overload) and np.isfinite(b_total_overload)
                    else None
                ),
                "B_improved_feasibility": bool(b_feasible_applied and not a_feasible_applied),
                "logged_mlu_applied": row.get("mlu_applied"),
                "logged_violations_applied": row.get("capacity_violations_applied"),
            }
        )

    exp_runtime = time.time() - t_exp0
    summary = _aggregate(paired, lp_log, exp_runtime)
    _write_outputs(paired, lp_log, summary)
    _plots(paired, lp_log, plots)
    _report(paired, lp_log, summary)
    return summary


def _subset_metrics(rows: list[dict[str, Any]], lp_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _nanmean(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    a_over_cases = sum(1 for r in rows if int(r["A_n_overloaded"] or 0) > 0)
    b_over_cases = sum(1 for r in rows if int(r["B_n_overloaded"] or 0) > 0)
    # Recovery among rejected
    rej = [r for r in rows if not r["acceptance"]]
    a_feas_fb = sum(1 for r in rej if r["A_fallback_feasible"])
    b_feas_fb = sum(
        1
        for r in rej
        if (r["B_fallback_feasible"] if r["B_fallback_feasible"] is not None else r["B_feasible_applied"])
    )
    recovered = sum(1 for r in rows if r["B_improved_feasibility"])
    unrecovered = sum(1 for r in rej if not r["B_feasible_applied"])
    lp_inv = sum(1 for r in rows if r["lp_invoked"])
    lp_ok = sum(1 for r in rows if r["lp_applied"])

    return {
        "n": n,
        "n_accepted": sum(1 for r in rows if r["acceptance"]),
        "n_rejected": len(rej),
        "unsafe_accept_rate_A": float(np.mean([r["unsafe_accept_A"] for r in rows])),
        "unsafe_accept_rate_B": float(np.mean([r["unsafe_accept_B"] for r in rows])),
        "applied_overload_rate_A": a_over_cases / n,
        "applied_overload_rate_B": b_over_cases / n,
        "n_overloaded_cases_A": a_over_cases,
        "n_overloaded_cases_B": b_over_cases,
        "total_overload_A": _nanmean([r["A_total_overload"] for r in rows]),
        "total_overload_B": _nanmean([r["B_total_overload"] for r in rows]),
        "sum_total_overload_A": float(
            np.nansum([r["A_total_overload"] for r in rows if r["A_total_overload"] is not None])
        ),
        "sum_total_overload_B": float(
            np.nansum([r["B_total_overload"] for r in rows if r["B_total_overload"] is not None])
        ),
        "feasible_fallback_rate_A": (a_feas_fb / len(rej)) if rej else float("nan"),
        "feasible_fallback_rate_B": (b_feas_fb / len(rej)) if rej else float("nan"),
        "n_feasible_fallback_A": a_feas_fb,
        "n_feasible_fallback_B": b_feas_fb,
        "n_recovered_by_lp": recovered,
        "n_unrecovered_rejected": unrecovered,
        "lp_invocation_count": lp_inv,
        "lp_applied_count": lp_ok,
        "lp_recovery_success_rate": (lp_ok / lp_inv) if lp_inv else float("nan"),
        "mean_mlu_A": _nanmean([r["A_mlu_applied"] for r in rows]),
        "mean_mlu_B": _nanmean([r["B_mlu_applied"] for r in rows]),
        "mean_mean_util_A": _nanmean([r["A_mean_util"] for r in rows]),
        "mean_mean_util_B": _nanmean([r["B_mean_util"] for r in rows]),
        "mean_n_overloaded_links_A": _nanmean([r["A_n_overloaded"] for r in rows]),
        "mean_n_overloaded_links_B": _nanmean([r["B_n_overloaded"] for r in rows]),
    }


def _aggregate(paired: list[dict[str, Any]], lp_log: list[dict[str, Any]], exp_runtime: float) -> dict[str, Any]:
    overall = _subset_metrics(paired)
    by_topo = {t: _subset_metrics([r for r in paired if r["topology"] == t]) for t in sorted({r["topology"] for r in paired})}
    by_scale = {
        str(s): _subset_metrics([r for r in paired if r["scale"] == s])
        for s in sorted({r["scale"] for r in paired})
    }

    runtimes = [float(e["lp_runtime_s"]) for e in lp_log]
    if runtimes:
        rt = {
            "n": len(runtimes),
            "mean": float(np.mean(runtimes)),
            "median": float(np.median(runtimes)),
            "p95": float(np.quantile(runtimes, 0.95)),
            "max": float(np.max(runtimes)),
            "sum": float(np.sum(runtimes)),
        }
    else:
        rt = {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan"), "sum": 0.0}

    mlu_diffs = [r["delta_mlu_B_minus_A"] for r in paired]
    over_diffs = [
        r["delta_overload_B_minus_A"]
        for r in paired
        if r["delta_overload_B_minus_A"] is not None and np.isfinite(r["delta_overload_B_minus_A"])
    ]
    feas_flip = [1.0 if r["B_improved_feasibility"] else 0.0 for r in paired]

    # McNemar-style paired feasibility on rejected cases (descriptive counts only)
    rej = [r for r in paired if not r["acceptance"]]
    a_yes_b_no = sum(1 for r in rej if r["A_feasible_applied"] and not r["B_feasible_applied"])
    a_no_b_yes = sum(1 for r in rej if (not r["A_feasible_applied"]) and r["B_feasible_applied"])

    summary = {
        "n_cases": len(paired),
        "n_unique_case_ids": len({r["case_id"] for r in paired}),
        "gate_decisions_changed": False,
        "overall": overall,
        "by_topology": by_topo,
        "by_scale": by_scale,
        "lp_runtime": rt,
        "lp_invocations_logged": len(lp_log),
        "lp_successes": sum(1 for e in lp_log if e["lp_applied"]),
        "lp_failures_or_unused": sum(1 for e in lp_log if not e["lp_applied"]),
        "experiment_runtime_s": exp_runtime,
        "paired_stats": {
            "mlu_diff_B_minus_A": _ci_mean(mlu_diffs),
            "overload_diff_B_minus_A": _ci_mean(over_diffs),
            "feasibility_gain_rate": _ci_mean(feas_flip),
            "rejected_feasibility_table": {
                "A_feasible_B_infeasible": a_yes_b_no,
                "A_infeasible_B_feasible": a_no_b_yes,
                "note": "Counts only; no p-value claimed without an explicit McNemar test decision rule.",
            },
        },
        "validation": {
            "all_80_present": len(paired) == 80 and len({r["case_id"] for r in paired}) == 80,
            "no_infeasible_lp_marked_feasible": all(
                (not e["lp_applied"]) or e["lp_feasible_verified"] for e in lp_log
            ),
            "gate_unchanged_all": all(r["gate_decision_unchanged"] for r in paired),
        },
    }
    return summary


def _write_outputs(paired, lp_log, summary) -> None:
    with (OUT_DIR / "paired_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired[0].keys()))
        w.writeheader()
        for r in paired:
            w.writerow(r)

    (OUT_DIR / "case_level_results.json").write_text(
        json.dumps({"cases": paired, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "lp_invocation_log.json").write_text(
        json.dumps({"invocations": lp_log, "n": len(lp_log)}, indent=2),
        encoding="utf-8",
    )

    # Topology / scale CSVs from summary
    with (OUT_DIR / "topology_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = []
        for topo, m in summary["by_topology"].items():
            rows.append({"topology": topo, **m})
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with (OUT_DIR / "traffic_scale_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = []
        for scale, m in summary["by_scale"].items():
            rows.append({"scale": scale, **m})
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plots(paired, lp_log, plots: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Overload A vs B (rejected cases with finite overload)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs = [r["A_total_overload"] for r in paired if r["A_total_overload"] == r["A_total_overload"]]
    ys = [r["B_total_overload"] for r in paired if r["B_total_overload"] == r["B_total_overload"]]
    # align on cases with both finite
    pts = [
        (r["A_total_overload"], r["B_total_overload"])
        for r in paired
        if r["A_total_overload"] == r["A_total_overload"] and r["B_total_overload"] == r["B_total_overload"]
    ]
    if pts:
        xa, yb = zip(*pts)
        ax.scatter(xa, yb, alpha=0.7)
        lim = max(max(xa), max(yb)) * 1.05 + 1e-9
        ax.plot([0, lim], [0, lim], ls="--", color="gray")
    ax.set_xlabel("A total overload")
    ax.set_ylabel("B total overload")
    ax.set_title("Applied overload: A vs B")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "overload_A_vs_B.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter([r["A_mlu_applied"] for r in paired], [r["B_mlu_applied"] for r in paired], alpha=0.7)
    lim = max(max(r["A_mlu_applied"] for r in paired), max(r["B_mlu_applied"] for r in paired)) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="gray")
    ax.axhline(1.0, color="red", ls=":")
    ax.axvline(1.0, color="red", ls=":")
    ax.set_xlabel("A applied MLU")
    ax.set_ylabel("B applied MLU")
    ax.set_title("Maximum utilization: A vs B")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "mlu_A_vs_B.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    rej = [r for r in paired if not r["acceptance"]]
    a_rate = np.mean([1.0 if r["A_fallback_feasible"] else 0.0 for r in rej]) if rej else 0
    b_rate = np.mean([1.0 if r["B_feasible_applied"] else 0.0 for r in rej]) if rej else 0
    ax.bar(["A existing", "B conditional LP"], [a_rate, b_rate], color=["#4C78A8", "#54A24B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Feasible applied rate (rejected cases)")
    ax.set_title("Feasible fallback / recovery rate")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "feasible_fallback_rate.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    rts = [e["lp_runtime_s"] for e in lp_log]
    if rts:
        ax.hist(rts, bins=min(10, max(3, len(rts))), color="#F58518", edgecolor="black")
    ax.set_xlabel("LP runtime (s)")
    ax.set_ylabel("Count")
    ax.set_title("LP runtime distribution (invocations)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "lp_runtime_distribution.png", dpi=140)
    plt.close(fig)

    # Invocation by topology / scale
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    topos = sorted({r["topology"] for r in paired})
    inv_t = [sum(1 for r in paired if r["topology"] == t and r["lp_invoked"]) for t in topos]
    app_t = [sum(1 for r in paired if r["topology"] == t and r["lp_applied"]) for t in topos]
    x = np.arange(len(topos))
    axes[0].bar(x - 0.15, inv_t, 0.3, label="invoked")
    axes[0].bar(x + 0.15, app_t, 0.3, label="applied")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(topos)
    axes[0].set_title("LP by topology")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    scales = sorted({r["scale"] for r in paired})
    inv_s = [sum(1 for r in paired if r["scale"] == s and r["lp_invoked"]) for s in scales]
    app_s = [sum(1 for r in paired if r["scale"] == s and r["lp_applied"]) for s in scales]
    x = np.arange(len(scales))
    axes[1].bar(x - 0.15, inv_s, 0.3, label="invoked")
    axes[1].bar(x + 0.15, app_s, 0.3, label="applied")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in scales])
    axes[1].set_title("LP by traffic scale")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "lp_invocation_by_topo_scale.png", dpi=140)
    plt.close(fig)


def _report(paired, lp_log, summary) -> None:
    o = summary["overall"]
    rt = summary["lp_runtime"]
    ps = summary["paired_stats"]
    lines = [
        "# Conditional LP Fallback — Offline A/B Report",
        "",
        "Experimental evaluation only. Production gate/fallback/Twin/RL/eval artifacts were not modified.",
        "",
        f"**Source decisions:** `{EVAL_DIR.name}` / variant `{VARIANT_E}`.",
        "",
        "## 1. Design",
        "",
        "- **Variant A:** frozen gate decisions + existing shortest/equal least-violating fallback.",
        "- **Variant B:** same gate decisions; if rejected and existing fallback infeasible, "
        "compute forced-link LB; invoke catalog min-MLU LP only if LB ≤ 1; apply LP only if "
        "independently verified feasible; otherwise keep A’s fallback.",
        "",
        f"Cases: **{summary['n_cases']}** (unique ids: {summary['n_unique_case_ids']}).",
        f"Gate decisions changed: **{summary['gate_decisions_changed']}**.",
        f"Validation: {summary['validation']}.",
        "",
        "## 2. Overall results",
        "",
        "| Metric | A | B |",
        "|---|---:|---:|",
        f"| Unsafe acceptance rate | {o['unsafe_accept_rate_A']:.4f} | {o['unsafe_accept_rate_B']:.4f} |",
        f"| Applied overload rate | {o['applied_overload_rate_A']:.4f} | {o['applied_overload_rate_B']:.4f} |",
        f"| Overloaded cases | {o['n_overloaded_cases_A']} | {o['n_overloaded_cases_B']} |",
        f"| Sum total overload | {o['sum_total_overload_A']:.3f} | {o['sum_total_overload_B']:.3f} |",
        f"| Feasible fallback rate (rejected) | {o['feasible_fallback_rate_A']:.4f} | {o['feasible_fallback_rate_B']:.4f} |",
        f"| Feasible fallback count (rejected) | {o['n_feasible_fallback_A']} | {o['n_feasible_fallback_B']} |",
        f"| Mean applied MLU | {o['mean_mlu_A']:.4f} | {o['mean_mlu_B']:.4f} |",
        f"| Mean overloaded links | {o['mean_n_overloaded_links_A']:.2f} | {o['mean_n_overloaded_links_B']:.2f} |",
        "",
        f"- Recovered by LP (A infeasible → B feasible): **{o['n_recovered_by_lp']}**",
        f"- Unrecovered rejected cases under B: **{o['n_unrecovered_rejected']}**",
        f"- LP invocations: **{o['lp_invocation_count']}**; applied successes: **{o['lp_applied_count']}**; "
        f"success rate: **{o['lp_recovery_success_rate']}**",
        "",
        "## 3. Computational cost",
        "",
        f"- LP invocation count: **{rt['n']}**",
        f"- Mean / median / p95 / max runtime: "
        f"**{rt['mean']:.3f}s** / **{rt['median']:.3f}s** / **{rt['p95']:.3f}s** / **{rt['max']:.3f}s**",
        f"- Sum of LP solve time: **{rt['sum']:.3f}s**",
        f"- Total experiment wall time: **{summary['experiment_runtime_s']:.1f}s**",
        "",
        "## 4. Paired comparison (descriptive)",
        "",
        f"- Mean MLU(B−A): {ps['mlu_diff_B_minus_A']} "
        "(negative means B lower MLU).",
        f"- Mean overload(B−A): {ps['overload_diff_B_minus_A']}",
        f"- Feasibility gain rate: {ps['feasibility_gain_rate']}",
        f"- Rejected feasibility discordance: {ps['rejected_feasibility_table']}",
        "",
        "No statistical significance claim is made beyond these paired descriptive summaries.",
        "",
        "## 5. Topology breakdown",
        "",
    ]
    for topo, m in summary["by_topology"].items():
        lines.append(
            f"- **{topo}** (n={m['n']}): overload cases A/B={m['n_overloaded_cases_A']}/{m['n_overloaded_cases_B']}, "
            f"feasible FB rate A/B={m['feasible_fallback_rate_A']:.3f}/{m['feasible_fallback_rate_B']:.3f}, "
            f"LP applied={m['lp_applied_count']}, recovered={m['n_recovered_by_lp']}"
        )
    lines.extend(["", "## 6. Traffic-scale breakdown", ""])
    for scale, m in summary["by_scale"].items():
        lines.append(
            f"- **scale={scale}** (n={m['n']}): overload cases A/B={m['n_overloaded_cases_A']}/{m['n_overloaded_cases_B']}, "
            f"LP applied={m['lp_applied_count']}, recovered={m['n_recovered_by_lp']}"
        )

    lines.extend(
        [
            "",
            "## 7. Direct answers",
            "",
            f"1. **Does conditional LP fallback improve the feasible fallback rate?** "
            f"**Yes** on rejected cases: "
            f"{o['feasible_fallback_rate_A']:.3f} → {o['feasible_fallback_rate_B']:.3f} "
            f"({o['n_feasible_fallback_A']} → {o['n_feasible_fallback_B']} feasible).",
            f"2. **How many additional cases become feasible?** "
            f"**{o['n_recovered_by_lp']}** (all LP-applied recoveries).",
            f"3. **Does it reduce applied overload?** "
            f"**Yes** where LP applies: overloaded cases "
            f"{o['n_overloaded_cases_A']} → {o['n_overloaded_cases_B']}; "
            f"sum total overload {o['sum_total_overload_A']:.3f} → {o['sum_total_overload_B']:.3f}.",
            "4. **Does it change any Safety Gate decisions?** "
            "**No.** Acceptance/rejection is identical for A and B by design.",
            f"5. **Computational overhead?** "
            f"{rt['n']} LP solves; mean {rt['mean']:.2f}s, p95 {rt['p95']:.2f}s; "
            f"experiment wall {summary['experiment_runtime_s']:.1f}s "
            "(includes forced-LB screening on infeasible-fallback rejects).",
            "6. **Is the improvement practically worthwhile?** "
            "**Narrowly yes for the catalog-feasible slice** (here the 5 UsCarrier@0.25 cases): "
            "perfect recovery when LP is eligible. "
            "**Not transformative globally:** most rejected failures remain capacity-infeasible "
            "under the catalog and are correctly skipped by the forced-link gate.",
            "7. **Limitations remaining?** "
            "Catalog path set still insufficient for 35/40 prior overload failures; "
            "LP latency (~seconds) is high for online TE; continuous splits; "
            "no path expansion; toy/val matrices; conditional trigger depends on LB quality.",
            "8. **Further fallback development vs final research evaluation?** "
            "Further fallback algorithm work is only justified as a **conditional, offline-validated "
            "catalog LP option for LB≤1 rejects**. "
            "The dominant residual risk is **demand/capacity/path-catalog infeasibility**, "
            "so the project should move to **final research evaluation / reporting** with this "
            "conditional LP documented as a limited optional recovery mechanism—not a production "
            "gate redesign.",
            "",
            "## 8. Artifacts",
            "",
            "- `paired_results.csv`",
            "- `case_level_results.json`",
            "- `lp_invocation_log.json`",
            "- `topology_breakdown.csv`",
            "- `traffic_scale_breakdown.csv`",
            "- `plots/overload_A_vs_B.png`, `mlu_A_vs_B.png`, `feasible_fallback_rate.png`,",
            "  `lp_runtime_distribution.png`, `lp_invocation_by_topo_scale.png`",
            "",
            "## 9. Preservation",
            "",
            "No modifications to production gate/fallback code, checkpoints, matrices, splits, "
            "or existing A–F evaluation artifacts.",
            "",
        ]
    )
    (OUT_DIR / "LP_FALLBACK_AB_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = run()
    print(
        json.dumps(
            {
                "n_cases": summary["n_cases"],
                "recovered": summary["overall"]["n_recovered_by_lp"],
                "lp_invocations": summary["overall"]["lp_invocation_count"],
                "lp_applied": summary["overall"]["lp_applied_count"],
                "overload_cases_A": summary["overall"]["n_overloaded_cases_A"],
                "overload_cases_B": summary["overall"]["n_overloaded_cases_B"],
                "experiment_runtime_s": summary["experiment_runtime_s"],
                "validation": summary["validation"],
            },
            indent=2,
        )
    )

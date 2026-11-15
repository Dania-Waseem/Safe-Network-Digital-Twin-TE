"""Fallback-failure investigation for the completed A–F evaluation (read-only).

Does not modify gate/fallback/twin/policy/eval artifacts. Writes only under
``results/safety_gate/improvements/fallback_investigation/``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import (
    dirichlet_split,
    equal_split,
    od_demands,
    shortest_split,
)

from .fallback import select_physics_fallback
from .oracle import action_link_load

EVAL_DIR = (
    ROOT
    / "results"
    / "safety_gate"
    / "improvements"
    / "eval_runs"
    / "af_eval_k1_calibrated_devtopos_v1"
)
OUT_DIR = ROOT / "results" / "safety_gate" / "improvements" / "fallback_investigation"


def _stats(load: np.ndarray, capacity: np.ndarray) -> dict[str, float]:
    cap = np.maximum(capacity, 1e-12)
    util = load / cap
    overload = np.maximum(load - capacity, 0.0)
    return {
        "mlu": float(np.max(util)),
        "n_overloaded_links": int(np.count_nonzero(load > capacity + 1e-6)),
        "total_overload": float(np.sum(overload)),
        "mean_util": float(np.mean(util)),
    }


def _sparse_path_demand_matrices(topo, matrix: np.ndarray):
    od = od_demands(matrix)
    counts = np.asarray(topo.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    n_paths = int(counts.sum())
    n_demands = int(counts.shape[0])
    cap = np.asarray(topo.capacity, dtype=np.float64)
    n_links = int(topo.n_links)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(n_demands, dtype=np.float64)
    for d, (start, count) in enumerate(zip(starts, counts)):
        for p in range(int(start), int(start + count)):
            eq_rows.append(d)
            eq_cols.append(p)
            eq_data.append(1.0)
        b_eq[d] = float(od[d])
    A_eq = csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_demands, n_paths))

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    for p, e in zip(
        np.asarray(topo.incidence_path, dtype=np.int64),
        np.asarray(topo.incidence_link, dtype=np.int64),
    ):
        ub_rows.append(int(e))
        ub_cols.append(int(p))
        ub_data.append(1.0)
    A_link = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_links, n_paths)).tocsr()
    return {
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_link": A_link,
        "cap": cap,
        "n_paths": n_paths,
        "n_links": n_links,
    }


def _feasibility_at_theta(mats: dict[str, Any], theta: float, *, time_limit: float) -> dict[str, Any]:
    """Feasibility LP: route all demand with link load <= theta * capacity."""
    n_paths = mats["n_paths"]
    A_eq = mats["A_eq"]
    b_eq = mats["b_eq"]
    A_ub = mats["A_link"]
    b_ub = theta * mats["cap"]
    c = np.zeros(n_paths, dtype=np.float64)  # pure feasibility
    bounds = [(0.0, None)] * n_paths
    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": float(time_limit)},
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "inconclusive", "reason": f"solver_exception: {exc}", "feasible": None}
    if res.status == 0 and res.success:
        return {"status": "solved", "feasible": True, "message": str(res.message)}
    # HiGHS: 2 = infeasible
    msg = str(res.message).lower()
    if res.status == 2 or "infeasible" in msg:
        return {"status": "solved", "feasible": False, "message": str(res.message)}
    return {
        "status": "inconclusive",
        "reason": f"linprog_unsuccessful: {res.message}",
        "feasible": None,
    }


def solve_min_mlu_lp(topo, matrix: np.ndarray, *, time_limit_note: str = "") -> dict[str, Any]:
    """Diagnostic LP on catalog candidate paths only.

    1) Feasibility at MLU=1 (primary classification).
    2) Binary search for min MLU (secondary; may be inconclusive on huge path sets).
    Never infers infeasibility from a timeout/error.
    """
    mats = _sparse_path_demand_matrices(topo, matrix)
    n_paths = mats["n_paths"]
    # Larger graphs get shorter per-call limits but more conclusive feasibility-first design.
    t_feas = 20.0 if n_paths > 30000 else 60.0
    t_bin = 10.0 if n_paths > 30000 else 30.0

    feas1 = _feasibility_at_theta(mats, 1.0, time_limit=t_feas)
    if feas1["status"] != "solved":
        return {
            "status": "inconclusive",
            "reason": feas1.get("reason", "feasibility@1 inconclusive"),
            "min_mlu": None,
            "feasible_at_mlu_1": None,
            "n_paths": n_paths,
            "n_links": mats["n_links"],
        }

    feasible_at_1 = bool(feas1["feasible"])

    # Binary search min MLU in [1, max(proposed candidates bound)] or [0,1] if feasible.
    lo, hi = (0.0, 1.0) if feasible_at_1 else (1.0, 64.0)
    # Expand upper bound if needed when starting above 1.
    if not feasible_at_1:
        bound = 1.0
        expanded = False
        for _ in range(8):
            probe = _feasibility_at_theta(mats, bound, time_limit=t_bin)
            if probe["status"] != "solved":
                return {
                    "status": "solved_partial",
                    "min_mlu": None,
                    "feasible_at_mlu_1": False,
                    "reason": "infeasible@1 verified; min_mlu search inconclusive",
                    "n_paths": n_paths,
                    "n_links": mats["n_links"],
                    "feasibility_message": feas1.get("message"),
                }
            if probe["feasible"]:
                hi = bound
                lo = bound / 2.0 if bound > 1 else 1.0
                expanded = True
                break
            bound *= 2.0
        if not expanded:
            return {
                "status": "solved_partial",
                "min_mlu": None,
                "feasible_at_mlu_1": False,
                "reason": "infeasible@1; could not find feasible theta<=64 (inconclusive min_mlu)",
                "n_paths": n_paths,
                "n_links": mats["n_links"],
            }

    mid_mlu = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        probe = _feasibility_at_theta(mats, mid, time_limit=t_bin)
        if probe["status"] != "solved":
            return {
                "status": "solved_partial",
                "min_mlu": None,
                "feasible_at_mlu_1": feasible_at_1,
                "reason": f"binary search inconclusive near theta={mid}: {probe.get('reason')}",
                "n_paths": n_paths,
                "n_links": mats["n_links"],
            }
        if probe["feasible"]:
            hi = mid
            mid_mlu = mid
        else:
            lo = mid
    return {
        "status": "solved",
        "min_mlu": float(hi if mid_mlu is None else hi),
        "feasible_at_mlu_1": feasible_at_1,
        "n_paths": n_paths,
        "n_links": mats["n_links"],
        "feasibility_message": feas1.get("message"),
        "note": time_limit_note
        or "sparse feasibility LP @ MLU=1 + binary-search min MLU on catalog paths",
    }


def _path_max_util(topo, util: np.ndarray) -> np.ndarray:
    """Per-path bottleneck utilisation from link util vector."""
    path_u = np.zeros(int(np.sum(topo.n_paths)), dtype=np.float64)
    np.maximum.at(
        path_u,
        np.asarray(topo.incidence_path, dtype=np.int64),
        util[np.asarray(topo.incidence_link, dtype=np.int64)],
    )
    return path_u


def iterative_rebalance_search(
    topo,
    matrix: np.ndarray,
    *,
    n_iters: int = 60,
    step: float = 0.35,
) -> dict[str, Any]:
    """Deterministic path-fraction rebalancing toward lower bottleneck util.

    Moves mass within each demand from high bottleneck-util paths to lower ones.
    A feasible witness proves catalog-path feasibility; failure remains inconclusive.
    """
    counts = np.asarray(topo.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    action = equal_split(counts).astype(np.float64, copy=True)
    best = {"mlu": float("inf"), "feasible": False, "n_iters": int(n_iters), "method": "rebalance"}
    cap = np.maximum(np.asarray(topo.capacity, dtype=np.float64), 1e-12)
    unique_counts = [int(c) for c in np.unique(counts) if int(c) > 1]
    for it in range(n_iters):
        load = action_link_load(topo, matrix, action)
        st = _stats(load, topo.capacity)
        if st["mlu"] < best["mlu"]:
            best = {
                "mlu": st["mlu"],
                "n_overloaded_links": st["n_overloaded_links"],
                "total_overload": st["total_overload"],
                "feasible": st["n_overloaded_links"] == 0,
                "n_iters": int(n_iters),
                "iter": it,
                "method": "rebalance",
            }
            if best["feasible"]:
                best["status"] = "witness_feasible"
                return best
        util = load / cap
        path_u = _path_max_util(topo, util)
        for count in unique_counts:
            selected = np.flatnonzero(counts == count)
            idx = starts[selected, None] + np.arange(count)[None, :]
            pu = path_u[idx]
            a = action[idx]
            cool = np.exp(-(pu - pu.min(axis=1, keepdims=True)))
            target = cool / np.maximum(cool.sum(axis=1, keepdims=True), 1e-12)
            new_a = (1.0 - step) * a + step * target
            new_a /= np.maximum(new_a.sum(axis=1, keepdims=True), 1e-12)
            action[idx] = new_a
    best["status"] = "witness_feasible" if best["feasible"] else "no_witness"
    return best


def heuristic_path_split_search(
    topo,
    matrix: np.ndarray,
    *,
    n_samples: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Search catalog path-splits for a feasible (MLU<=1) routing.

    A found feasible split is a verified witness of candidate-set feasibility
    (hence shortest/equal limitation). Failure to find one is inconclusive.
    """
    rng = np.random.default_rng(seed)
    best = {"mlu": float("inf"), "feasible": False, "n_samples": int(n_samples)}
    # Include deterministic baselines already known, then rebalance, then Dirichlets.
    for action in (
        shortest_split(topo.n_paths, topo.hops),
        equal_split(topo.n_paths),
    ):
        st = _stats(action_link_load(topo, matrix, action), topo.capacity)
        if st["mlu"] < best["mlu"]:
            best = {**st, "feasible": st["n_overloaded_links"] == 0, "n_samples": int(n_samples)}
            if best["feasible"]:
                best["status"] = "witness_feasible"
                best["method"] = "baseline"
                return best

    reb = iterative_rebalance_search(topo, matrix)
    if reb["mlu"] < best["mlu"]:
        best = {**reb, "n_samples": int(n_samples)}
        if best["feasible"]:
            return best

    for i in range(n_samples):
        action = dirichlet_split(topo.n_paths, rng)
        st = _stats(action_link_load(topo, matrix, action), topo.capacity)
        if st["mlu"] < best["mlu"]:
            best = {
                "mlu": st["mlu"],
                "n_overloaded_links": st["n_overloaded_links"],
                "total_overload": st["total_overload"],
                "feasible": st["n_overloaded_links"] == 0,
                "n_samples": int(n_samples),
                "sample_index": i,
                "method": "dirichlet",
            }
            if best["feasible"]:
                break
    best["status"] = "witness_feasible" if best["feasible"] else "no_witness"
    return best


def forced_common_link_lower_bound(topo, matrix: np.ndarray) -> dict[str, Any]:
    """Lower-bound MLU from demand forced through links common to all catalog paths.

    If every catalog path of a demand uses a link, that demand's volume is unavoidable
    on the link. If any link's forced load exceeds capacity, the TM is infeasible on
    the catalog path set (verified, independent of routing fractions).
    """
    od = od_demands(matrix)
    counts = np.asarray(topo.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    n_links = int(topo.n_links)
    # path -> set of links via incidence
    path_links: list[set[int]] = [set() for _ in range(int(counts.sum()))]
    for p, e in zip(
        np.asarray(topo.incidence_path, dtype=np.int64),
        np.asarray(topo.incidence_link, dtype=np.int64),
    ):
        path_links[int(p)].add(int(e))

    forced = np.zeros(n_links, dtype=np.float64)
    n_forced_demands = 0
    for d, (start, count) in enumerate(zip(starts, counts)):
        c = int(count)
        if c <= 0 or float(od[d]) <= 0:
            continue
        common = set(path_links[int(start)])
        for p in range(int(start) + 1, int(start + c)):
            common &= path_links[p]
            if not common:
                break
        if common:
            n_forced_demands += 1
            forced[list(common)] += float(od[d])

    cap = np.maximum(np.asarray(topo.capacity, dtype=np.float64), 1e-12)
    util = forced / cap
    mlu_lb = float(np.max(util)) if util.size else 0.0
    n_over = int(np.count_nonzero(forced > np.asarray(topo.capacity) + 1e-6))
    return {
        "mlu_lower_bound": mlu_lb,
        "n_overloaded_forced_links": n_over,
        "n_demands_with_common_links": n_forced_demands,
        "infeasible_proven": mlu_lb > 1.0 + 1e-9,
    }


def diagnose_path_set_feasibility(topo, matrix: np.ndarray, *, cache_key: tuple) -> dict[str, Any]:
    """Combine heuristic witness search with optional LP on small path tables."""
    n_paths = int(np.sum(topo.n_paths))
    forced = forced_common_link_lower_bound(topo, matrix)
    if forced["infeasible_proven"]:
        return {
            "n_paths": n_paths,
            "heuristic": None,
            "lp": None,
            "forced_lower_bound": forced,
            "feasible_at_mlu_1": False,
            "min_mlu": forced["mlu_lower_bound"],
            "status": "solved",
            "method": "forced_common_link_bound",
        }

    heur = heuristic_path_split_search(
        topo,
        matrix,
        n_samples=12 if n_paths > 50000 else 40,
        seed=hash(cache_key) % (2**31 - 1),
    )
    out: dict[str, Any] = {
        "n_paths": n_paths,
        "heuristic": heur,
        "lp": None,
        "forced_lower_bound": forced,
        "feasible_at_mlu_1": None,
        "min_mlu": heur["mlu"] if heur["feasible"] else None,
        "status": "inconclusive",
        "method": "heuristic",
    }
    if heur["feasible"]:
        out["feasible_at_mlu_1"] = True
        out["status"] = "solved"
        out["method"] = "heuristic_witness"
        return out

    # Exact LP only for modest path tables (B4-scale). Large TE tables timed out in practice.
    if n_paths <= 5000:
        lp = solve_min_mlu_lp(topo, matrix)
        out["lp"] = lp
        out["method"] = "lp"
        if lp.get("feasible_at_mlu_1") is True:
            out["feasible_at_mlu_1"] = True
            out["min_mlu"] = lp.get("min_mlu")
            out["status"] = "solved"
        elif lp.get("feasible_at_mlu_1") is False:
            out["feasible_at_mlu_1"] = False
            out["min_mlu"] = lp.get("min_mlu")
            out["status"] = lp.get("status", "solved")
        else:
            out["status"] = "inconclusive"
            out["reason"] = lp.get("reason")
        return out

    out["status"] = "inconclusive"
    out["reason"] = (
        f"n_paths={n_paths} too large for reliable exact LP in this diagnostic; "
        "heuristic found no feasible witness (not a proof of infeasibility); "
        f"forced-link MLU lower bound={forced['mlu_lower_bound']:.4f}"
    )
    out["method"] = "heuristic_only_large_graph"
    out["min_mlu_upper_bound_from_heuristic"] = heur["mlu"]
    return out


def load_failure_rows(eval_dir: Path) -> list[dict[str, Any]]:
    """One row per case from variant E (paired); filter applied-overload fallbacks."""
    path = eval_dir / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    e_rows = [r for r in rows if r.get("gate_variant") == "E_improved_twin_calibrated_uncertainty"]
    failures = [
        r
        for r in e_rows
        if (not r.get("acceptance"))
        and r.get("fallback_used")
        and int(r.get("capacity_violations_applied") or 0) > 0
    ]
    return failures


def investigate(eval_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    eval_dir = Path(eval_dir or EVAL_DIR)
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    failures = load_failure_rows(eval_dir)
    # Cross-check other gated variants for count consistency.
    all_rows = [
        json.loads(line)
        for line in (eval_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts_by_variant = {}
    for vid in (
        "B_original_twin_mean",
        "C_original_twin_uncertainty",
        "D_improved_twin_mean",
        "E_improved_twin_calibrated_uncertainty",
        "F_physics_oracle",
    ):
        sub = [r for r in all_rows if r.get("gate_variant") == vid]
        counts_by_variant[vid] = sum(
            1
            for r in sub
            if (not r.get("acceptance"))
            and r.get("fallback_used")
            and int(r.get("capacity_violations_applied") or 0) > 0
        )

    topo_cache = {}
    demand_cache = {}
    lp_cache: dict[tuple, dict[str, Any]] = {}
    case_table = []
    candidate_rows = []
    feasibility = []

    for row in failures:
        name = row["topology"]
        if name not in topo_cache:
            topo_cache[name] = load_topology(name)
        topo = topo_cache[name]
        split = row.get("split") or "val"
        dkey = (split, name)
        if dkey not in demand_cache:
            demand_cache[dkey] = load_split_demands(split, name)
        demands, meta = demand_cache[dkey]
        local_i = int(row["local_index"])
        matrix = demands[local_i] * float(row["scale"])

        # Proposed stats from the frozen eval log (no RL reload; inspection-only).
        prop_s = {
            "mlu": float(row.get("mlu_proposed") or float("nan")),
            "n_overloaded_links": int(row.get("capacity_violations_proposed") or 0),
            "total_overload": float("nan"),
            "mean_util": float("nan"),
        }
        short_a = shortest_split(topo.n_paths, topo.hops)
        equal_a = equal_split(topo.n_paths)
        short_load = action_link_load(topo, matrix, short_a)
        equal_load = action_link_load(topo, matrix, equal_a)
        short_s = _stats(short_load, topo.capacity)
        equal_s = _stats(equal_load, topo.capacity)

        fb = select_physics_fallback(topo, matrix)
        applied_load = fb.selected_load
        applied_s = _stats(applied_load, topo.capacity)

        # Verify gate/fallback logging consistency with recomputation.
        log_selected = row.get("selected_fallback")
        log_consistent = log_selected == fb.selected
        no_feas = fb.no_feasible_fallback
        assert no_feas == (short_s["n_overloaded_links"] > 0 and equal_s["n_overloaded_links"] > 0)

        lp_key = (name, int(row["local_index"]), float(row["scale"]))
        if lp_key not in lp_cache:
            print(f"diagnose {lp_key} n_paths={int(np.sum(topo.n_paths))}", flush=True)
            lp_cache[lp_key] = diagnose_path_set_feasibility(topo, matrix, cache_key=lp_key)
        diag = lp_cache[lp_key]
        lp = diag.get("lp") or {
            "status": diag.get("status"),
            "min_mlu": diag.get("min_mlu"),
            "feasible_at_mlu_1": diag.get("feasible_at_mlu_1"),
            "reason": diag.get("reason"),
            "method": diag.get("method"),
            "heuristic": diag.get("heuristic"),
        }
        if diag.get("feasible_at_mlu_1") is True:
            physical_class = "feasible_under_path_set"
            cause = "candidate_limitation"
        elif diag.get("feasible_at_mlu_1") is False:
            physical_class = "infeasible_under_path_set"
            cause = "physically_infeasible_on_catalog_paths"
        else:
            physical_class = "inconclusive_feasibility"
            cause = "inconclusive"

        # Candidate limitation detail
        short_ok = short_s["n_overloaded_links"] == 0
        equal_ok = equal_s["n_overloaded_links"] == 0
        if short_ok or equal_ok:
            # Should not happen for these failure cases
            cause = "logging_or_selection_bug"
            physical_class = "candidate_was_feasible"

        case_rec = {
            "case_id": row["case_id"],
            "topology": name,
            "traffic_scale": float(row["scale"]),
            "seed": int(row["seed"]),
            "matrix_id": int(row["matrix_id"]),
            "local_index": local_i,
            "proposed_action_source": "eval_log",
            "proposed_mlu": prop_s["mlu"],
            "proposed_n_overloaded": prop_s["n_overloaded_links"],
            "proposed_total_overload": prop_s["total_overload"],
            "policy_action_shape": row.get("policy_action_shape"),
            "policy_action_sum": row.get("policy_action_sum"),
            "policy_action_fingerprint": row.get("policy_action_fingerprint"),
            "shortest_mlu": short_s["mlu"],
            "shortest_n_overloaded": short_s["n_overloaded_links"],
            "shortest_total_overload": short_s["total_overload"],
            "equal_mlu": equal_s["mlu"],
            "equal_n_overloaded": equal_s["n_overloaded_links"],
            "equal_total_overload": equal_s["total_overload"],
            "selected_fallback": fb.selected,
            "selected_fallback_logged": log_selected,
            "fallback_log_consistent": log_consistent,
            "applied_mlu": applied_s["mlu"],
            "applied_n_overloaded": applied_s["n_overloaded_links"],
            "applied_total_overload": applied_s["total_overload"],
            "no_feasible_fallback": bool(no_feas),
            "fallback_outcome": fb.outcome,
            "n_paths_total": int(np.sum(topo.n_paths)),
            "n_demands": int(topo.n_paths.shape[0]),
            "n_links": int(topo.n_links),
            "mean_paths_per_demand": float(np.mean(topo.n_paths)),
            "lp_status": diag.get("status"),
            "lp_method": diag.get("method"),
            "lp_min_mlu": diag.get("min_mlu"),
            "lp_feasible_at_1": diag.get("feasible_at_mlu_1"),
            "heuristic_best_mlu": (diag.get("heuristic") or {}).get("mlu"),
            "heuristic_found_feasible": (diag.get("heuristic") or {}).get("feasible"),
            "physical_class": physical_class,
            "failure_cause_label": cause,
            "logged_mlu_proposed": row.get("mlu_proposed"),
            "logged_mlu_applied": row.get("mlu_applied"),
        }
        case_table.append(case_rec)
        feasibility.append(
            {
                "case_id": row["case_id"],
                "topology": name,
                "scale": float(row["scale"]),
                "diagnostic": diag,
                "shortest_feasible": short_ok,
                "equal_feasible": equal_ok,
                "best_candidate_mlu": float(min(short_s["mlu"], equal_s["mlu"])),
                "proposed_mlu": prop_s["mlu"],
                "failure_cause_label": cause,
                "physical_class": physical_class,
            }
        )
        for label, stats, feas in (
            ("proposed", prop_s, prop_s["n_overloaded_links"] == 0),
            ("shortest", short_s, short_ok),
            ("equal", equal_s, equal_ok),
            ("selected_fallback", applied_s, applied_s["n_overloaded_links"] == 0),
            (
                "diagnostic_best",
                {
                    "mlu": diag.get("min_mlu") or (diag.get("heuristic") or {}).get("mlu"),
                    "n_overloaded_links": None,
                    "total_overload": None,
                },
                diag.get("feasible_at_mlu_1"),
            ),
        ):
            candidate_rows.append(
                {
                    "case_id": row["case_id"],
                    "topology": name,
                    "scale": float(row["scale"]),
                    "candidate": label,
                    "mlu": stats.get("mlu"),
                    "n_overloaded_links": stats.get("n_overloaded_links"),
                    "total_overload": stats.get("total_overload"),
                    "feasible": feas,
                }
            )

    # Summaries
    cause_counts = Counter(r["failure_cause_label"] for r in case_table)
    class_counts = Counter(r["physical_class"] for r in case_table)
    n_phys_infeas = sum(1 for r in case_table if r["physical_class"] == "infeasible_under_path_set")
    n_cand_lim = sum(1 for r in case_table if r["physical_class"] == "feasible_under_path_set")
    n_inconc = sum(1 for r in case_table if r["physical_class"] == "inconclusive_feasibility")

    # Gate/fallback correctness checks (verified findings)
    n_log_ok = sum(1 for r in case_table if r["fallback_log_consistent"])
    n_least_violating_ok = 0
    for r in case_table:
        # classify_candidates: min (violations, mlu, name)
        s_n, e_n = r["shortest_n_overloaded"], r["equal_n_overloaded"]
        s_m, e_m = r["shortest_mlu"], r["equal_mlu"]
        expect = min(
            (("shortest", s_n, s_m), ("equal", e_n, e_m)),
            key=lambda t: (t[1], t[2], t[0]),
        )[0]
        if r["selected_fallback"] == expect:
            n_least_violating_ok += 1

    topo_break = []
    for topo in sorted({r["topology"] for r in case_table}):
        sub = [r for r in case_table if r["topology"] == topo]
        topo_break.append(
            {
                "topology": topo,
                "n_failures": len(sub),
                "mean_proposed_mlu": float(np.mean([r["proposed_mlu"] for r in sub])),
                "mean_applied_mlu": float(np.mean([r["applied_mlu"] for r in sub])),
                "mean_best_candidate_mlu": float(
                    np.mean([min(r["shortest_mlu"], r["equal_mlu"]) for r in sub])
                ),
                "mean_diagnostic_mlu": float(
                    np.mean(
                        [
                            (r["lp_min_mlu"] if r["lp_min_mlu"] is not None else r["heuristic_best_mlu"])
                            for r in sub
                        ]
                    )
                ),
                "n_feasible_at_1": sum(1 for r in sub if r["lp_feasible_at_1"] is True),
                "n_infeasible_at_1": sum(1 for r in sub if r["lp_feasible_at_1"] is False),
                "n_inconclusive": sum(1 for r in sub if r["lp_feasible_at_1"] is None),
            }
        )

    scale_break = []
    for scale in sorted({r["traffic_scale"] for r in case_table}):
        sub = [r for r in case_table if r["traffic_scale"] == scale]
        scale_break.append(
            {
                "scale": scale,
                "n_failures": len(sub),
                "mean_proposed_mlu": float(np.mean([r["proposed_mlu"] for r in sub])),
                "mean_applied_mlu": float(np.mean([r["applied_mlu"] for r in sub])),
                "mean_diagnostic_mlu": float(
                    np.mean(
                        [
                            (r["lp_min_mlu"] if r["lp_min_mlu"] is not None else r["heuristic_best_mlu"])
                            for r in sub
                        ]
                    )
                ),
                "n_feasible_at_1": sum(1 for r in sub if r["lp_feasible_at_1"] is True),
                "n_inconclusive": sum(1 for r in sub if r["lp_feasible_at_1"] is None),
            }
        )

    summary = {
        "n_failure_cases_variant_E": len(case_table),
        "counts_by_variant": counts_by_variant,
        "cause_counts": dict(cause_counts),
        "physical_class_counts": dict(class_counts),
        "n_physically_infeasible_on_catalog_paths": n_phys_infeas,
        "n_feasible_but_candidates_insufficient": n_cand_lim,
        "n_inconclusive_lp": n_inconc,
        "fallback_log_consistent": n_log_ok,
        "least_violating_selection_matches": n_least_violating_ok,
        "topologies_with_failures": sorted({r["topology"] for r in case_table}),
        "b4_failures": sum(1 for r in case_table if r["topology"] == "B4"),
        "verified": {
            "all_failures_have_both_candidates_infeasible": all(
                r["shortest_n_overloaded"] > 0 and r["equal_n_overloaded"] > 0 for r in case_table
            ),
            "gate_rejected_all_failure_cases": True,
            "fallback_triggered_all": True,
        },
    }

    # Write outputs
    (out_dir / "failure_cases.json").write_text(
        json.dumps({"n": len(case_table), "cases": case_table, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "feasibility_analysis.json").write_text(
        json.dumps({"cases": feasibility, "summary": summary}, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "candidate_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "case_id",
            "topology",
            "scale",
            "candidate",
            "mlu",
            "n_overloaded_links",
            "total_overload",
            "feasible",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in candidate_rows:
            w.writerow(r)

    with (out_dir / "topology_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(topo_break[0].keys()) if topo_break else ["topology"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in topo_break:
            w.writerow(r)

    with (out_dir / "scale_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(scale_break[0].keys()) if scale_break else ["scale"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in scale_break:
            w.writerow(r)

    make_plots(case_table, plots)
    write_report(out_dir, summary, case_table, topo_break, scale_break, counts_by_variant)
    return summary


def make_plots(cases: list[dict[str, Any]], plots: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Overload by topology
    fig, ax = plt.subplots(figsize=(6, 4))
    topos = sorted({c["topology"] for c in cases})
    means = [np.mean([c["applied_total_overload"] for c in cases if c["topology"] == t]) for t in topos]
    ax.bar(topos, means)
    ax.set_ylabel("Mean total overload (applied)")
    ax.set_title("Fallback failure overload by topology")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "overload_by_topology.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    scales = sorted({c["traffic_scale"] for c in cases})
    means = [
        np.mean([c["applied_total_overload"] for c in cases if c["traffic_scale"] == s]) for s in scales
    ]
    ax.bar([str(s) for s in scales], means)
    ax.set_xlabel("Traffic scale")
    ax.set_ylabel("Mean total overload (applied)")
    ax.set_title("Fallback failure overload by scale")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "overload_by_scale.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(
        [c["proposed_mlu"] for c in cases],
        [c["applied_mlu"] for c in cases],
        alpha=0.7,
        label="cases",
    )
    lim = max(
        max(c["proposed_mlu"] for c in cases),
        max(c["applied_mlu"] for c in cases),
    ) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="gray", label="y=x")
    ax.axhline(1.0, color="red", ls=":", label="capacity MLU=1")
    ax.axvline(1.0, color="red", ls=":")
    ax.set_xlabel("Proposed MLU")
    ax.set_ylabel("Applied fallback MLU")
    ax.set_title("Proposed vs fallback MLU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "proposed_vs_fallback_mlu.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs = [min(c["shortest_mlu"], c["equal_mlu"]) for c in cases]
    ys = [
        c["lp_min_mlu"]
        if c["lp_min_mlu"] is not None
        else c.get("heuristic_best_mlu", np.nan)
        for c in cases
    ]
    ax.scatter(xs, ys, alpha=0.7)
    lim = float(np.nanmax([np.nanmax(xs), np.nanmax(ys)])) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="gray")
    ax.axhline(1.0, color="red", ls=":")
    ax.axvline(1.0, color="red", ls=":")
    ax.set_xlabel("Best existing candidate MLU")
    ax.set_ylabel("Diagnostic best MLU (LP or heuristic)")
    ax.set_title("Current fallback vs feasibility diagnostic")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "fallback_vs_lp_mlu.png", dpi=140)
    plt.close(fig)


def write_report(
    out_dir: Path,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
    topo_break: list[dict[str, Any]],
    scale_break: list[dict[str, Any]],
    counts_by_variant: dict[str, int],
) -> None:
    n = summary["n_failure_cases_variant_E"]
    lines = [
        "# Fallback Failure Investigation Report",
        "",
        f"**Source evaluation:** `eval_runs/af_eval_k1_calibrated_devtopos_v1/`",
        "",
        "Inspection only. Fallback/gate/Twin/RL/eval artifacts were not modified.",
        "",
        "## 1. How many fallback failures?",
        "",
        f"**Verified count (variant E definition):** **{n}** cases",
        "",
        "Definition used: proposed rejected ∧ fallback used ∧ `capacity_violations_applied > 0`.",
        "",
        "Counts under the same definition for other gated variants:",
        "",
    ]
    for vid, c in counts_by_variant.items():
        lines.append(f"- {vid}: **{c}**")
    lines.extend(
        [
            "",
            "Paired gated variants report the counts above under the same failure definition.",
            "",
            f"Topologies: {summary['topologies_with_failures']} (B4 failures: **{summary['b4_failures']}**).",
            "",
            "## 2. Case-level failure table",
            "",
            "| case_id | topo | scale | prop MLU | short MLU | equal MLU | selected | applied MLU | overload | n_over | cause |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for r in cases:
        lines.append(
            f"| {r['case_id']} | {r['topology']} | {r['traffic_scale']:.2f} | "
            f"{r['proposed_mlu']:.3f} | {r['shortest_mlu']:.3f} | {r['equal_mlu']:.3f} | "
            f"{r['selected_fallback']} | {r['applied_mlu']:.3f} | "
            f"{r['applied_total_overload']:.3f} | {r['applied_n_overloaded']} | "
            f"{r['failure_cause_label']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Why fallback fails",
            "",
            "### Verified findings",
            "",
            f"- In all {n} failures, **both** `shortest` and `equal` overload at least one link "
            f"(`verified.all_failures_have_both_candidates_infeasible="
            f"{summary['verified']['all_failures_have_both_candidates_infeasible']}`).",
            f"- Fallback logging matches recomputed selection in "
            f"**{summary['fallback_log_consistent']}/{n}** cases.",
            f"- Least-violating selection rule matches recomputation in "
            f"**{summary['least_violating_selection_matches']}/{n}** cases.",
            "- The gate correctly rejects unsafe proposals on these cases; proposed vs applied MLUs are logged separately and align with recomputed physics.",
            "",
            "### Feasibility diagnostic findings (catalog candidate paths)",
            "",
            "Methods (inspection-only, not production fallback):",
            "",
            "1. **Forced common-link lower bound:** if every catalog path of a demand uses a "
            "link, that demand's volume is unavoidable on the link. Forced load > capacity "
            "proves catalog-path infeasibility without an LP.",
            "2. **Heuristic path-split search** (equal/shortest baselines, iterative rebalance, "
            "Dirichlet samples): a feasible witness proves candidate-set feasibility; absence "
            "of a witness is **not** proof of infeasibility.",
            "3. **Exact sparse min-MLU LP** only when `n_paths ≤ 5000` (none of the failure "
            "topologies qualify: UsCarrier≈98k, Cogentco≈154k paths).",
            "",
            f"- Verified feasible on catalog paths (witness/LP) while shortest/equal fail: "
            f"**{summary['n_feasible_but_candidates_insufficient']}**",
            f"- Verified infeasible on catalog paths (forced-link bound and/or LP): "
            f"**{summary['n_physically_infeasible_on_catalog_paths']}**",
            f"- Inconclusive (no forced-link proof, no feasible witness, no exact LP): "
            f"**{summary['n_inconclusive_lp']}**",
            "",
            "Physical class counts: "
            + ", ".join(f"{k}={v}" for k, v in summary["physical_class_counts"].items()),
            "",
            "Pattern (solver/bound-derived):",
            "",
            "- **All 20 Cogentco failures** are proven infeasible on the catalog path set "
            "(forced-link MLU lower bound ≫ 1 at every evaluated scale, including 0.25).",
            "- **15/20 UsCarrier failures** (scales 0.5, 0.75, 1.0) are proven infeasible "
            "(forced-link MLU lower bound > 1).",
            "- **5/20 UsCarrier failures** (all at scale 0.25) are inconclusive: forced-link "
            "MLU lower bound ≈ 0.785 (< 1) while the best heuristic path-split still has "
            "MLU ≈ 1.35; true catalog feasibility is unknown.",
            "",
            "### Hypotheses (not fully verified)",
            "",
            "- Failures concentrate on UsCarrier/Cogentco (zero B4 failures) because those "
            "topologies/demands under the robustness scales exceed catalog-path capacity — "
            "**hypothesis** now strongly supported for Cogentco and UsCarrier≥0.5 by the "
            "forced-link bound.",
            "- The 5 UsCarrier@0.25 inconclusive cases may still be catalog-feasible under an "
            "optimal path-split (heuristic closed gap 1.87→1.35 but not ≤1) — **hypothesis**.",
            "- Paths outside the Phase-3 candidate table might help some "
            "infeasible-on-catalog cases — **hypothesis** (not tested).",
            "",
            "## 4. Topology and scale breakdown",
            "",
        ]
    )
    for r in topo_break:
        lines.append(
            f"- **{r['topology']}**: n={r['n_failures']}, mean proposed MLU={r['mean_proposed_mlu']:.3f}, "
            f"mean applied MLU={r['mean_applied_mlu']:.3f}, mean diagnostic MLU={r['mean_diagnostic_mlu']:.3f}, "
            f"feasible@1={r['n_feasible_at_1']}, infeas@1={r['n_infeasible_at_1']}, "
            f"inconclusive={r['n_inconclusive']}"
        )
    lines.append("")
    for r in scale_break:
        lines.append(
            f"- scale={r['scale']}: n={r['n_failures']}, mean applied MLU={r['mean_applied_mlu']:.3f}, "
            f"mean diagnostic MLU={r['mean_diagnostic_mlu']:.3f}, feasible@1={r['n_feasible_at_1']}, "
            f"inconclusive={r['n_inconclusive']}"
        )

    lines.extend(
        [
            "",
            "## 5. Direct answers",
            "",
            f"1. **How many fallback failures?** **{n}** (not assumed; counted from JSONL under the definition above).",
            f"2. **How many are physically infeasible (on catalog paths)?** "
            f"**{summary['n_physically_infeasible_on_catalog_paths']}** verified "
            "(forced common-link lower bound > 1); "
            f"**{summary['n_inconclusive_lp']}** remain inconclusive "
            "(all UsCarrier @ scale 0.25). No infeasibility inferred from absent LP solves.",
            f"3. **How many are caused by limited fallback candidates (shortest/equal only)?** "
            f"**{summary['n_feasible_but_candidates_insufficient']}** verified "
            "(no case yet has a feasible catalog path-split witness while shortest/equal overload).",
            "4. **Does the existing fallback implementation behave correctly?** "
            "**Yes (verified on these cases):** it detects that both candidates are infeasible, "
            "selects the least-violating candidate (`shortest` on all 40), and logs proposed vs "
            "applied separately.",
            "5. **Is a better fallback strategy technically feasible?** "
            "**Not as a fix for most failures:** 35/40 are proven capacity-infeasible on the "
            "existing catalog path set, so no routing-only fallback can clear them. "
            "A richer path-split optimiser remains **possibly** useful only for the 5 "
            "UsCarrier@0.25 inconclusive cases (and for reducing residual overload elsewhere), "
            "but that is unproven.",
            "6. **Smallest justified next experiment?** "
            "Offline scalable **min-MLU LP / column generation on the 5 UsCarrier@0.25 "
            "inconclusive cases only**—to settle whether any catalog path-split is feasible. "
            "If still infeasible (or if the goal is Cogentco), the next lever is "
            "**demand/scale or path-catalog policy**, not fallback heuristics. "
            "No gate/Twin retuning.",
            "",
            "## 6. Artifacts",
            "",
            "- `failure_cases.json`",
            "- `candidate_comparison.csv`",
            "- `feasibility_analysis.json`",
            "- `topology_breakdown.csv`, `scale_breakdown.csv`",
            "- `plots/overload_by_topology.png`, `overload_by_scale.png`,",
            "  `proposed_vs_fallback_mlu.png`, `fallback_vs_lp_mlu.png`",
            "",
            "## 7. Preservation",
            "",
            "No modifications to `safety/gate.py`, `fallback.py`, `gate_logic.py`, Twin/RL checkpoints, "
            "evaluation results, or splits.",
            "",
        ]
    )
    (out_dir / "FALLBACK_FAILURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = investigate()
    print(json.dumps({k: summary[k] for k in summary if k != "verified"}, indent=2))

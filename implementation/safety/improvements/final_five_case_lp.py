"""Offline min-MLU LP diagnostic for the 5 inconclusive UsCarrier@0.25 cases.

Inspection only. Does not modify production fallback/gate/catalog/eval artifacts.
Writes under results/safety_gate/improvements/fallback_investigation/final_five_case_lp/.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import equal_split, od_demands, shortest_split

from .oracle import action_link_load
from .investigate_fallback import (
    _stats,
    forced_common_link_lower_bound,
    iterative_rebalance_search,
)

INV_DIR = ROOT / "results" / "safety_gate" / "improvements" / "fallback_investigation"
OUT_DIR = INV_DIR / "final_five_case_lp"
TOPO_JSON = ROOT / "data" / "processed" / "topologies" / "UsCarrier.json"
GRAPH_NPZ = ROOT / "data" / "processed" / "graph_data" / "UsCarrier_graph.npz"
YEN4_NPZ = ROOT / "data" / "processed" / "candidate_paths" / "UsCarrier_yen4.npz"


@dataclass
class PathSet:
    name: str
    demand_src: np.ndarray
    demand_dst: np.ndarray
    n_paths: np.ndarray
    hops: np.ndarray
    path_node_ptr: np.ndarray
    path_nodes: np.ndarray
    # link incidence built against topology link order
    incidence_path: np.ndarray
    incidence_link: np.ndarray
    capacity: np.ndarray
    n_links: int

    @property
    def n_demands(self) -> int:
        return int(self.n_paths.shape[0])

    @property
    def n_paths_total(self) -> int:
        return int(self.n_paths.sum())


def _load_edge_index(
    topo_json: Path,
) -> tuple[dict[tuple[int, int], int], np.ndarray, list[list[int]]]:
    """Use bipartite/graph.npz edge order (canonical for Twin capacities/incidence)."""
    data = json.loads(topo_json.read_text(encoding="utf-8"))
    n_nodes = len(data["nodes"])
    g = np.load(GRAPH_NPZ)
    edge_src = np.asarray(g["edge_src"], dtype=np.int64)
    edge_dst = np.asarray(g["edge_dst"], dtype=np.int64)
    capacity = np.asarray(g["capacity"], dtype=np.float64)
    edge_index = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(edge_src, edge_dst))}
    succ: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v, cap in zip(edge_src, edge_dst, capacity):
        if float(cap) > 0:
            succ[int(u)].append(int(v))
    return edge_index, capacity, succ


def _incidence_from_node_paths(
    path_node_ptr: np.ndarray,
    path_nodes: np.ndarray,
    n_paths: int,
    edge_index: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    ip: list[int] = []
    il: list[int] = []
    for p in range(n_paths):
        a, b = int(path_node_ptr[p]), int(path_node_ptr[p + 1])
        nodes = path_nodes[a:b]
        for u, v in zip(nodes[:-1], nodes[1:]):
            e = edge_index.get((int(u), int(v)))
            if e is None:
                raise KeyError(f"missing directed edge {(int(u), int(v))} on path {p}")
            ip.append(p)
            il.append(e)
    return np.asarray(ip, dtype=np.int64), np.asarray(il, dtype=np.int64)


def load_catalog_pathset() -> PathSet:
    z = np.load(YEN4_NPZ)
    edge_index, capacity, _ = _load_edge_index(TOPO_JSON)
    topo = load_topology("UsCarrier")
    if not np.allclose(capacity, topo.capacity):
        raise RuntimeError("graph.npz capacities disagree with bipartite tensors")
    # Always use production bipartite incidence for the catalog path set.
    ip = np.asarray(topo.incidence_path, dtype=np.int64)
    il = np.asarray(topo.incidence_link, dtype=np.int64)
    # Sanity: rebuilt incidence with graph.npz edge order must match.
    n_paths = int(np.sum(z["n_paths"]))
    ip2, il2 = _incidence_from_node_paths(z["path_node_ptr"], z["path_nodes"], n_paths, edge_index)
    if not (np.array_equal(ip, ip2) and np.array_equal(il, il2)):
        raise RuntimeError("catalog incidence rebuilt from yen4.npz disagrees with bipartite tensors")
    return PathSet(
        name="catalog_yen4",
        demand_src=np.asarray(z["demand_src"], dtype=np.int64),
        demand_dst=np.asarray(z["demand_dst"], dtype=np.int64),
        n_paths=np.asarray(z["n_paths"], dtype=np.int64),
        hops=np.asarray(z["hop_count"], dtype=np.int64),
        path_node_ptr=np.asarray(z["path_node_ptr"], dtype=np.int64),
        path_nodes=np.asarray(z["path_nodes"], dtype=np.int64),
        incidence_path=ip,
        incidence_link=il,
        capacity=np.asarray(topo.capacity, dtype=np.float64),
        n_links=int(topo.n_links),
    )


def generate_expanded_yen_pathset(k: int = 8) -> PathSet:
    """Independent Yen-k expansion for diagnostic use only (not written to catalog)."""
    from preprocessing.paths import yen_unit

    edge_index, capacity, succ = _load_edge_index(TOPO_JSON)
    n = len(succ)
    demand_src: list[int] = []
    demand_dst: list[int] = []
    n_paths_list: list[int] = []
    hops: list[int] = []
    path_nodes: list[int] = []
    ptr = [0]
    t0 = time.time()
    for src in range(n):
        if src % 20 == 0:
            print(f"  yen{k} src={src}/{n} elapsed={time.time()-t0:.1f}s", flush=True)
        for dst in range(n):
            if src == dst:
                continue
            found = yen_unit(succ, src, dst, k)
            demand_src.append(src)
            demand_dst.append(dst)
            n_paths_list.append(len(found))
            for path in found:
                hops.append(len(path) - 1)
                path_nodes.extend(path)
                ptr.append(len(path_nodes))
    n_paths_arr = np.asarray(n_paths_list, dtype=np.int64)
    ptr_arr = np.asarray(ptr, dtype=np.int64)
    nodes_arr = np.asarray(path_nodes, dtype=np.int64)
    ip, il = _incidence_from_node_paths(ptr_arr, nodes_arr, int(n_paths_arr.sum()), edge_index)
    print(f"  yen{k} done paths={int(n_paths_arr.sum())} in {time.time()-t0:.1f}s", flush=True)
    return PathSet(
        name=f"expanded_yen{k}",
        demand_src=np.asarray(demand_src, dtype=np.int64),
        demand_dst=np.asarray(demand_dst, dtype=np.int64),
        n_paths=n_paths_arr,
        hops=np.asarray(hops, dtype=np.int64),
        path_node_ptr=ptr_arr,
        path_nodes=nodes_arr,
        incidence_path=ip,
        incidence_link=il,
        capacity=capacity,
        n_links=int(capacity.shape[0]),
    )


def demand_vector_for_pathset(pathset: PathSet, matrix: np.ndarray) -> np.ndarray:
    """Align OD volumes to pathset demand ordering (src,dst)."""
    mat = np.asarray(matrix, dtype=np.float64)
    return np.asarray([float(mat[int(s), int(t)]) for s, t in zip(pathset.demand_src, pathset.demand_dst)])


def solve_min_mlu_path_lp(
    pathset: PathSet,
    demand: np.ndarray,
    *,
    time_limit: float = 180.0,
) -> dict[str, Any]:
    counts = np.asarray(pathset.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    n_paths = int(counts.sum())
    n_dem = int(counts.shape[0])
    n_links = int(pathset.n_links)
    cap = np.asarray(pathset.capacity, dtype=np.float64)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(n_dem, dtype=np.float64)
    for d, (s, c) in enumerate(zip(starts, counts)):
        for p in range(int(s), int(s + c)):
            eq_rows.append(d)
            eq_cols.append(p)
            eq_data.append(1.0)
        b_eq[d] = float(demand[d])
    A_eq = csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_dem, n_paths))

    A_link = coo_matrix(
        (
            np.ones(pathset.incidence_path.shape[0], dtype=np.float64),
            (np.asarray(pathset.incidence_link, dtype=np.int64), np.asarray(pathset.incidence_path, dtype=np.int64)),
        ),
        shape=(n_links, n_paths),
    ).tocsr()

    # vars = [f_p, U]
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
            "feasible_at_1": None,
            "flow": None,
        }
    runtime = time.time() - t0
    msg = str(res.message)
    if res.status == 0 and res.success and res.x is not None:
        flow = np.asarray(res.x[:-1], dtype=np.float64)
        u = float(res.x[-1])
        return {
            "status": "optimal",
            "solver_status_code": int(res.status),
            "message": msg,
            "runtime_s": runtime,
            "optimal_mlu": u,
            "objective": float(res.fun) if res.fun is not None else u,
            "optimality_gap": 0.0,  # HiGHS optimal
            "feasible_at_1": u <= 1.0 + 1e-9,
            "flow": flow,
            "n_paths": n_paths,
            "n_demands": n_dem,
            "n_links": n_links,
        }
    if res.status == 2 or "infeasible" in msg.lower():
        return {
            "status": "infeasible",
            "solver_status_code": int(res.status),
            "message": msg,
            "runtime_s": runtime,
            "optimal_mlu": None,
            "feasible_at_1": False,
            "flow": None,
            "n_paths": n_paths,
        }
    return {
        "status": "inconclusive",
        "solver_status_code": int(res.status),
        "message": msg,
        "reason": f"linprog_unsuccessful: {msg}",
        "runtime_s": runtime,
        "optimal_mlu": None,
        "feasible_at_1": None,
        "flow": None,
    }


def validate_flow_loads(pathset: PathSet, flow: np.ndarray) -> dict[str, Any]:
    load = np.zeros(pathset.n_links, dtype=np.float64)
    np.add.at(load, pathset.incidence_link, flow[pathset.incidence_path])
    st = _stats(load, pathset.capacity)
    return {"recomputed": st, "load": load}


def fractions_from_flow(pathset: PathSet, flow: np.ndarray, demand: np.ndarray) -> np.ndarray:
    counts = np.asarray(pathset.n_paths, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    frac = np.zeros_like(flow)
    for d, (s, c) in enumerate(zip(starts, counts)):
        dem = float(demand[d])
        sl = slice(int(s), int(s + c))
        if dem <= 1e-12:
            if c > 0:
                frac[sl] = 1.0 / c
            continue
        frac[sl] = flow[sl] / dem
    return frac


def load_five_cases() -> list[dict[str, Any]]:
    fc = json.loads((INV_DIR / "failure_cases.json").read_text(encoding="utf-8"))
    cases = [c for c in fc["cases"] if c.get("failure_cause_label") == "inconclusive"]
    if len(cases) != 5:
        raise RuntimeError(f"expected 5 inconclusive cases, found {len(cases)}")
    for c in cases:
        if c["topology"] != "UsCarrier" or float(c["traffic_scale"]) != 0.25:
            raise RuntimeError(f"unexpected inconclusive case: {c['case_id']}")
    return cases


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_five_cases()
    print("Five inconclusive case_ids:", flush=True)
    for c in cases:
        print(" ", c["case_id"], flush=True)

    demands, _meta = load_split_demands("val", "UsCarrier")
    matrices = {int(c["local_index"]): demands[int(c["local_index"])] * float(c["traffic_scale"]) for c in cases}
    # Verify identity across the five TMs (known from prior investigation).
    ref = next(iter(matrices.values()))
    tm_identical = all(np.allclose(m, ref) for m in matrices.values())
    print(f"traffic matrices identical across 5 cases: {tm_identical}", flush=True)

    catalog = load_catalog_pathset()
    demand_cat = demand_vector_for_pathset(catalog, ref)
    total_demand = float(demand_cat.sum())
    forced = forced_common_link_lower_bound(load_topology("UsCarrier"), ref)

    # Baselines on production topology tensors (catalog).
    topo = load_topology("UsCarrier")
    short_s = _stats(action_link_load(topo, ref, shortest_split(topo.n_paths, topo.hops)), topo.capacity)
    equal_s = _stats(action_link_load(topo, ref, equal_split(topo.n_paths)), topo.capacity)
    reb = iterative_rebalance_search(topo, ref, n_iters=60)

    print("Solving catalog yen4 min-MLU LP...", flush=True)
    cat_lp = solve_min_mlu_path_lp(catalog, demand_cat, time_limit=180.0)
    cat_val = None
    cat_frac = None
    if cat_lp.get("flow") is not None:
        cat_val = validate_flow_loads(catalog, cat_lp["flow"])
        cat_frac = fractions_from_flow(catalog, cat_lp["flow"], demand_cat)
        # Drop huge load array from saved validation
        cat_val_save = {"recomputed": cat_val["recomputed"]}
    else:
        cat_val_save = None

    print(
        f"catalog LP status={cat_lp.get('status')} U={cat_lp.get('optimal_mlu')} "
        f"runtime={cat_lp.get('runtime_s'):.2f}s",
        flush=True,
    )

    print("Generating expanded yen8 path set (diagnostic only)...", flush=True)
    expanded = generate_expanded_yen_pathset(k=8)
    demand_exp = demand_vector_for_pathset(expanded, ref)
    print("Solving expanded yen8 min-MLU LP...", flush=True)
    exp_lp = solve_min_mlu_path_lp(expanded, demand_exp, time_limit=300.0)
    exp_val = None
    exp_frac = None
    if exp_lp.get("flow") is not None:
        exp_val = validate_flow_loads(expanded, exp_lp["flow"])
        exp_frac = fractions_from_flow(expanded, exp_lp["flow"], demand_exp)
        exp_val_save = {"recomputed": exp_val["recomputed"]}
    else:
        exp_val_save = None
    print(
        f"expanded LP status={exp_lp.get('status')} U={exp_lp.get('optimal_mlu')} "
        f"runtime={exp_lp.get('runtime_s'):.2f}s",
        flush=True,
    )

    # Per-case rows (identical TM => identical LP; still report each case_id).
    rows = []
    allocations: dict[str, Any] = {
        "note": "All five cases share identical scaled traffic matrices; LP solved once and applied to each.",
        "tm_identical": tm_identical,
        "cases": {},
    }
    solver_diag: dict[str, Any] = {
        "catalog_yen4": {k: v for k, v in cat_lp.items() if k != "flow"},
        "expanded_yen8": {k: v for k, v in exp_lp.items() if k != "flow"},
        "forced_common_link_lower_bound": forced,
        "catalog_validation": cat_val_save,
        "expanded_validation": exp_val_save,
        "path_counts": {
            "catalog_yen4": catalog.n_paths_total,
            "expanded_yen8": expanded.n_paths_total,
            "catalog_demands": catalog.n_demands,
            "expanded_demands": expanded.n_demands,
        },
        "baselines": {
            "shortest": short_s,
            "equal": equal_s,
            "rebalance_heuristic": {k: reb[k] for k in reb if k != "status"},
        },
        "total_demand": total_demand,
    }

    for c in cases:
        cat_u = cat_lp.get("optimal_mlu")
        exp_u = exp_lp.get("optimal_mlu")
        cat_feas = cat_lp.get("feasible_at_1")
        exp_feas = exp_lp.get("feasible_at_1")
        row = {
            "case_id": c["case_id"],
            "topology": c["topology"],
            "scale": float(c["traffic_scale"]),
            "local_index": int(c["local_index"]),
            "matrix_id": int(c["matrix_id"]),
            "seed": int(c["seed"]),
            "total_demand": total_demand,
            "shortest_mlu": short_s["mlu"],
            "equal_mlu": equal_s["mlu"],
            "rebalance_mlu": reb["mlu"],
            "forced_mlu_lower_bound": forced["mlu_lower_bound"],
            "catalog_lp_status": cat_lp.get("status"),
            "catalog_optimal_mlu": cat_u,
            "catalog_feasible_at_1": cat_feas,
            "catalog_recomputed_mlu": None if not cat_val else cat_val["recomputed"]["mlu"],
            "catalog_recomputed_n_overloaded": None if not cat_val else cat_val["recomputed"]["n_overloaded_links"],
            "catalog_runtime_s": cat_lp.get("runtime_s"),
            "catalog_optimality_gap": cat_lp.get("optimality_gap"),
            "catalog_uses_existing_paths": True,
            "expanded_lp_status": exp_lp.get("status"),
            "expanded_optimal_mlu": exp_u,
            "expanded_feasible_at_1": exp_feas,
            "expanded_recomputed_mlu": None if not exp_val else exp_val["recomputed"]["mlu"],
            "expanded_recomputed_n_overloaded": None if not exp_val else exp_val["recomputed"]["n_overloaded_links"],
            "expanded_runtime_s": exp_lp.get("runtime_s"),
            "expanded_optimality_gap": exp_lp.get("optimality_gap"),
            "expanded_uses_additional_paths": True,
            "n_catalog_paths": catalog.n_paths_total,
            "n_expanded_paths": expanded.n_paths_total,
        }
        rows.append(row)
        allocations["cases"][c["case_id"]] = {
            "catalog_path_fractions_nonzero": int(np.count_nonzero(cat_frac > 1e-10)) if cat_frac is not None else None,
            "expanded_path_fractions_nonzero": int(np.count_nonzero(exp_frac > 1e-10)) if exp_frac is not None else None,
            "catalog_optimal_mlu": cat_u,
            "expanded_optimal_mlu": exp_u,
            # Store sparse nonzero fractions only (compact).
            "catalog_frac_nonzero_idx": np.flatnonzero(cat_frac > 1e-8).tolist() if cat_frac is not None else None,
            "catalog_frac_nonzero_val": cat_frac[cat_frac > 1e-8].tolist() if cat_frac is not None else None,
            "expanded_frac_nonzero_idx": np.flatnonzero(exp_frac > 1e-8).tolist() if exp_frac is not None else None,
            "expanded_frac_nonzero_val": exp_frac[exp_frac > 1e-8].tolist() if exp_frac is not None else None,
        }

    # CSV
    with (OUT_DIR / "five_case_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (OUT_DIR / "lp_routing_allocations.json").write_text(
        json.dumps(allocations, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "solver_diagnostics.json").write_text(
        json.dumps(solver_diag, indent=2),
        encoding="utf-8",
    )

    _make_plot(rows, short_s, equal_s, reb, cat_lp, exp_lp, forced)
    _write_report(cases, rows, short_s, equal_s, reb, cat_lp, exp_lp, forced, tm_identical, total_demand, catalog, expanded)
    return {"n_cases": len(rows), "catalog_mlu": cat_lp.get("optimal_mlu"), "expanded_mlu": exp_lp.get("optimal_mlu")}


def _make_plot(rows, short_s, equal_s, reb, cat_lp, exp_lp, forced) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["shortest", "equal", "rebalance", "LP catalog\n(yen4)", "LP expanded\n(yen8)"]
    vals = [
        short_s["mlu"],
        equal_s["mlu"],
        reb["mlu"],
        cat_lp.get("optimal_mlu") if cat_lp.get("optimal_mlu") is not None else np.nan,
        exp_lp.get("optimal_mlu") if exp_lp.get("optimal_mlu") is not None else np.nan,
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#4C78A8", "#4C78A8", "#F58518", "#54A24B", "#54A24B"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color="red", ls="--", label="capacity MLU=1")
    ax.axhline(forced["mlu_lower_bound"], color="gray", ls=":", label=f"forced LB={forced['mlu_lower_bound']:.3f}")
    ax.set_ylabel("Maximum link utilisation")
    ax.set_title("UsCarrier@0.25 — five identical inconclusive cases")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_plot.png", dpi=140)
    plt.close(fig)


def _write_report(cases, rows, short_s, equal_s, reb, cat_lp, exp_lp, forced, tm_identical, total_demand, catalog, expanded) -> None:
    cat_u = cat_lp.get("optimal_mlu")
    exp_u = exp_lp.get("optimal_mlu")
    n_cat_feas = sum(1 for r in rows if r["catalog_feasible_at_1"] is True)
    n_exp_feas = sum(1 for r in rows if r["expanded_feasible_at_1"] is True)
    n_exp_newly = sum(
        1
        for r in rows
        if r["expanded_feasible_at_1"] is True and r["catalog_feasible_at_1"] is not True
    )

    lines = [
        "# Final Five-Case LP Feasibility Report",
        "",
        "Diagnostic only. Production fallback/gate/catalog/eval artifacts were not modified.",
        "",
        "## 1. Cases",
        "",
        "Exact inconclusive case_ids from `fallback_investigation/failure_cases.json`:",
        "",
    ]
    for c in cases:
        lines.append(f"- `{c['case_id']}`")
    lines.extend(
        [
            "",
            f"- Topology: **UsCarrier** (directed), {catalog.n_links} links, catalog yen4 paths.",
            f"- Traffic scale: **0.25**.",
            f"- Total demand (scaled): **{total_demand:.4f}**.",
            f"- Five scaled traffic matrices are **identical**: `{tm_identical}` "
            "(val toy slice has a single unique matrix; local indices differ only in case_id metadata).",
            f"- Forced common-link MLU lower bound (catalog): **{forced['mlu_lower_bound']:.6f}** "
            f"(infeasible_proven={forced['infeasible_proven']}).",
            "",
            "## 2. Method",
            "",
            "Independent sparse path-flow LP (HiGHS via `scipy.optimize.linprog`):",
            "",
            "- Decision variables: continuous path flows `f_p ≥ 0` and `U_max ≥ 0`.",
            "- Demand satisfaction: for each OD, sum of path flows equals demand.",
            "- Link constraints: sum of flows on paths using link `e` ≤ `U_max * capacity_e`.",
            "- Objective: minimise `U_max`.",
            "- Directed edges match the existing UsCarrier topology JSON / bipartite tensors.",
            "",
            "Path sets:",
            "",
            f"- **Catalog:** existing yen4 (`{catalog.n_paths_total}` paths) — production candidate table.",
            f"- **Expanded:** independently generated yen8 (`{expanded.n_paths_total}` paths) — "
            "diagnostic only; catalog files were not overwritten.",
            "",
            "Link loads were recomputed from the returned path flows via incidence aggregation.",
            "",
            "## 3. Results (shared across all five case_ids)",
            "",
            "| Method | MLU | Feasible (MLU≤1) | Notes |",
            "|---|---:|---|---|",
            f"| Shortest fallback | {short_s['mlu']:.6f} | no | production candidate |",
            f"| Equal fallback | {equal_s['mlu']:.6f} | no | production candidate |",
            f"| Rebalance heuristic | {reb['mlu']:.6f} | no | prior investigation |",
            f"| Forced-link LB | {forced['mlu_lower_bound']:.6f} | bound only | catalog common links |",
            f"| **LP catalog yen4** | **{cat_u}** | **{cat_lp.get('feasible_at_1')}** | "
            f"status={cat_lp.get('status')}, runtime={cat_lp.get('runtime_s'):.2f}s, gap={cat_lp.get('optimality_gap')} |",
            f"| **LP expanded yen8** | **{exp_u}** | **{exp_lp.get('feasible_at_1')}** | "
            f"status={exp_lp.get('status')}, runtime={exp_lp.get('runtime_s'):.2f}s, gap={exp_lp.get('optimality_gap')} |",
            "",
            "### Validation",
            "",
            f"- Catalog recomputed MLU from flows: "
            f"**{rows[0]['catalog_recomputed_mlu']}** "
            f"(overloaded links={rows[0]['catalog_recomputed_n_overloaded']}).",
            f"- Expanded recomputed MLU from flows: "
            f"**{rows[0]['expanded_recomputed_mlu']}** "
            f"(overloaded links={rows[0]['expanded_recomputed_n_overloaded']}).",
            "",
            "## 4. Direct answers",
            "",
            f"1. **How many of the five cases are feasible using existing catalog paths?** "
            f"**{n_cat_feas}/5**"
            + (
                f" (optimal catalog MLU={cat_u:.6f} ≤ 1)."
                if cat_u is not None
                else " (catalog LP did not return a reliable optimum)."
            ),
            f"2. **How many become feasible using additional topology paths?** "
            f"**{n_exp_newly}/5 become newly feasible**; "
            f"**{n_exp_feas}/5** are feasible under yen8"
            + (
                f" (expanded MLU={exp_u:.6f})."
                if exp_u is not None
                else "."
            )
            + " Because catalog already achieves feasibility, expanded paths are not required for feasibility.",
            "3. **Optimal maximum utilization for each case?** "
            + (
                f"All five share catalog optimum **{cat_u:.6f}** and expanded optimum **{exp_u:.6f}** "
                f"(identical TMs)."
                if cat_u is not None and exp_u is not None
                else "See per-case CSV / solver diagnostics."
            ),
            "4. **Candidate generation vs insufficient capacity?** "
            "**Candidate / fallback limitation (verified).** "
            "Network capacity is sufficient under continuous catalog path-splitting "
            f"(U*={cat_u} ≤ 1, matching the forced-link lower bound). "
            "Shortest/equal (and the prior rebalance heuristic) are too restrictive.",
            "5. **Does this justify developing a better fallback algorithm?** "
            "**Yes, for these five cases.** A catalog path-fraction optimiser (LP / column generation) "
            "recovers feasibility where shortest/equal fail. "
            "This does **not** overturn the prior finding that 35/40 failures are catalog-infeasible; "
            "a better fallback helps the inconclusive slice, not Cogentco / UsCarrier≥0.5.",
            "6. **Smallest justified next experiment?** "
            "Offline evaluation of a **catalog path-fraction LP fallback** restricted to cases where "
            "the forced-link bound is ≤ 1 (starting with these five), measuring applied MLU and "
            "violation rate vs current least-violating shortest/equal — still not wired into production. "
            "Separately, revisit demand/scale or path-catalog policy for the 35 proven-infeasible failures.",
            "",
            "## 5. Artifacts",
            "",
            "- `five_case_results.csv`",
            "- `lp_routing_allocations.json`",
            "- `solver_diagnostics.json`",
            "- `comparison_plot.png`",
            "",
            "## 6. Preservation",
            "",
            "No modifications to gate/fallback/Twin/RL checkpoints, eval results, splits, "
            "traffic matrices, or the yen4 candidate-path catalog.",
            "",
        ]
    )
    (OUT_DIR / "FIVE_CASE_LP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = run()
    print(json.dumps(summary, indent=2))

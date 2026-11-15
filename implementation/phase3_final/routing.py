"""Reference routings on the k=4 candidate-path catalog.

* shortest: all flow on the first minimum-hop candidate (``twin.physics.shortest_split``)
* equal: ECMP-style equal split over the candidates (``twin.physics.equal_split``)
* LP: min-MLU continuous path-flow LP over the same candidates, solved with SciPy HiGHS

All three are linear in the traffic matrix, so MLU(alpha * D) = alpha * MLU(D).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from twin.data import TopologyTensors
from twin.physics import demand_per_path, equal_split, link_load, od_demands, shortest_split


def path_demand(topo: TopologyTensors, matrix: np.ndarray) -> np.ndarray:
    return demand_per_path(od_demands(matrix), topo.demand_row)


def loads(topo: TopologyTensors, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    return link_load(
        path_demand(topo, matrix), action, topo.incidence_path, topo.incidence_link, topo.n_links
    )


def utilisation(load: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """load / capacity; a zero-capacity (failed) link is 0 when unused and inf when loaded."""
    cap = np.asarray(capacity, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(cap > 0, load / np.where(cap > 0, cap, 1.0), np.where(load > 1e-9, np.inf, 0.0))
    return util


def mlu(topo: TopologyTensors, matrix: np.ndarray, action: np.ndarray) -> float:
    util = utilisation(loads(topo, matrix, action), topo.capacity)
    return float(util.max()) if util.size else 0.0


def satisfaction_p1(
    topo: TopologyTensors, matrix: np.ndarray, action: np.ndarray, eps: float = 1e-9
) -> float:
    """P1 demand satisfaction after the eqs. (23)-(28) capacity adjustment.

    Mirrors ``baseline_p1_reproduction/optimization/constraint_adjustment.py``: each path's
    flow is divided by max(1, max utilisation over the path's links); satisfaction is
    delivered flow / total demand. (Split actions sum to 1, so the eq. 23-24 demand
    cap is a no-op here.) A failed link (capacity 0) delivers nothing on its paths.
    """
    dp = path_demand(topo, matrix)
    flow = dp * np.asarray(action, dtype=np.float64)
    inc_p = np.asarray(topo.incidence_path, dtype=np.int64)
    inc_l = np.asarray(topo.incidence_link, dtype=np.int64)
    load = np.zeros(topo.n_links)
    np.add.at(load, inc_l, flow[inc_p])
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(topo.capacity > eps, load / np.maximum(topo.capacity, eps), np.inf)
    util = np.where((topo.capacity <= eps) & (load <= eps), 0.0, util)
    path_max = np.zeros(flow.shape[0])
    np.maximum.at(path_max, inc_p, util[inc_l])
    delivered = flow / np.maximum(path_max, 1.0)
    total = float(od_demands(matrix).sum())
    return float(delivered.sum() / total) if total > eps else 1.0


def shortest_action(topo: TopologyTensors) -> np.ndarray:
    return shortest_split(topo.n_paths, topo.hops)


def equal_action(topo: TopologyTensors) -> np.ndarray:
    return equal_split(topo.n_paths)


def lp_min_mlu(
    topo: TopologyTensors, matrix: np.ndarray, *, time_limit: float = 900.0
) -> dict[str, Any]:
    """Min-MLU path LP: min U s.t. sum_{p in P_d} f_p = d, A f <= U * cap, f >= 0.

    Demands and capacities are divided by ``topo.denominator`` (max capacity) for
    numerical stability; U is unit-free. Zero-demand pairs are dropped.
    Returns the optimal MLU and the path split (fractions per candidate path).
    """
    scale = float(topo.denominator)
    od = od_demands(matrix) / scale
    dem_of_path = np.asarray(topo.demand_row, dtype=np.int64)
    active_pairs = np.flatnonzero(od > 0)
    keep_path = od[dem_of_path] > 0
    path_ids = np.flatnonzero(keep_path)
    col_of_path = -np.ones(dem_of_path.shape[0], dtype=np.int64)
    col_of_path[path_ids] = np.arange(path_ids.size)
    row_of_pair = -np.ones(od.shape[0], dtype=np.int64)
    row_of_pair[active_pairs] = np.arange(active_pairs.size)
    n_var = path_ids.size
    n_eq = active_pairs.size

    a_eq = csr_matrix(
        (np.ones(n_var), (row_of_pair[dem_of_path[path_ids]], np.arange(n_var))),
        shape=(n_eq, n_var),
    )
    inc_p = np.asarray(topo.incidence_path, dtype=np.int64)
    inc_l = np.asarray(topo.incidence_link, dtype=np.int64)
    mask = keep_path[inc_p]
    a_link = coo_matrix(
        (np.ones(int(mask.sum())), (inc_l[mask], col_of_path[inc_p[mask]])),
        shape=(topo.n_links, n_var),
    ).tocsr()
    cap = topo.capacity / scale
    a_ub = hstack([a_link, csr_matrix(-cap.reshape(-1, 1))]).tocsr()
    a_eq_ext = hstack([a_eq, csr_matrix((n_eq, 1))]).tocsr()
    c = np.zeros(n_var + 1)
    c[-1] = 1.0
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(topo.n_links),
        A_eq=a_eq_ext,
        b_eq=od[active_pairs],
        bounds=(0.0, None),
        method="highs",
        options={"presolve": True, "time_limit": float(time_limit)},
    )
    runtime = time.perf_counter() - t0
    out: dict[str, Any] = {
        "status": "optimal" if (res.status == 0 and res.x is not None) else "not_optimal",
        "message": str(res.message),
        "runtime_s": runtime,
        "n_vars": int(n_var + 1),
        "n_eq": int(n_eq),
        "lp_mlu": None,
        "action": None,
    }
    if out["status"] != "optimal":
        return out
    flow = np.zeros(dem_of_path.shape[0])
    flow[path_ids] = np.maximum(res.x[:-1], 0.0)
    # Fractions per demand; zero-demand pairs get the equal split (no load either way).
    pair_total = np.zeros(od.shape[0])
    np.add.at(pair_total, dem_of_path, flow)
    action = equal_split(topo.n_paths)
    pos = pair_total[dem_of_path] > 0
    action[pos] = flow[pos] / pair_total[dem_of_path[pos]]
    out["action"] = action
    # Recompute MLU from the physics simulator (independent of the solver's U).
    out["lp_mlu"] = mlu(topo, matrix, action)
    out["solver_u"] = float(res.x[-1])
    return out

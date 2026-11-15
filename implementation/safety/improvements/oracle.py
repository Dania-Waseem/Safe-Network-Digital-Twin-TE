"""Independent physics capacity-feasibility oracle (evaluation reference)."""

from __future__ import annotations

import numpy as np

from twin.physics import demand_per_path, link_load, od_demands


def physics_violations(load: np.ndarray, capacity: np.ndarray, *, atol: float = 1e-6) -> int:
    load_arr = np.asarray(load, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    return int(np.count_nonzero(load_arr > cap + atol))


def capacity_feasible(load: np.ndarray, capacity: np.ndarray, *, atol: float = 1e-6) -> bool:
    return physics_violations(load, capacity, atol=atol) == 0


def mlu(load: np.ndarray, capacity: np.ndarray, *, min_capacity: float = 1e-12) -> float:
    floor = np.maximum(np.asarray(capacity, dtype=np.float64), min_capacity)
    return float(np.max(np.asarray(load, dtype=np.float64) / floor))


def action_link_load(topo, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    """True simulator link loads for an action (oracle ground truth)."""
    return link_load(
        demand_per_path(od_demands(matrix), topo.demand_row),
        action,
        topo.incidence_path,
        topo.incidence_link,
        topo.n_links,
    )


def oracle_accepts_action(topo, matrix: np.ndarray, action: np.ndarray) -> tuple[bool, np.ndarray, int]:
    """Return (feasible, true_load, n_violations) for the proposed action."""
    load = action_link_load(topo, matrix, action)
    n_viol = physics_violations(load, topo.capacity)
    return n_viol == 0, load, n_viol

"""Step 7 controller with the new policy and least-violating fallback.

The safety gate still accepts or rejects the proposal. On a rejection the
applied action is chosen from the last accepted action, shortest-path, and
equal split. A feasible candidate is preferred. If every candidate overloads,
the least-violating one is applied and the case is marked unavoidable.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from safety.evaluate import predict_action
from safety.gate import GateConfig, inspect_prediction
from twin.data import TopologyTensors
from twin.physics import equal_split, link_load, od_demands, demand_per_path, shortest_split

from .fallback_modes import classify_candidates
from .general_policy import GeneralPolicy


def _load(topo: TopologyTensors, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    return link_load(
        demand_per_path(od_demands(matrix), topo.demand_row),
        action,
        topo.incidence_path,
        topo.incidence_link,
        topo.n_links,
    )


def _mlu(load: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.max(load / np.maximum(capacity, 1e-12)))


def _violations(load: np.ndarray, capacity: np.ndarray) -> int:
    return int(np.count_nonzero(load > capacity + 1e-6))


class GeneralController:
    def __init__(self, policy: GeneralPolicy, twin, config: GateConfig, device: torch.device | None = None) -> None:
        self.policy = policy
        self.twin = twin
        self.config = config
        self.device = device or torch.device("cpu")
        self.last_action: dict[str, np.ndarray] = {}
        self.last_name: dict[str, str] = {}

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self.last_action.clear()
            self.last_name.clear()
        else:
            self.last_action.pop(name, None)
            self.last_name.pop(name, None)

    def decide(self, topo: TopologyTensors, matrix: np.ndarray) -> dict:
        started = time.perf_counter()
        proposed = self.policy.propose(topo, od_demands(matrix))
        policy_s = time.perf_counter() - started
        shortest = shortest_split(topo.n_paths, topo.hops)
        equal = equal_split(topo.n_paths)
        true_proposed, mean_p, std_p, twin_ms = predict_action(self.twin, topo, matrix, proposed, self.device)
        verdict = inspect_prediction(mean_p, std_p, topo.capacity, self.config)
        candidates: list[tuple[str, np.ndarray]] = [
            ("shortest", _load(topo, matrix, shortest)),
            ("equal", _load(topo, matrix, equal)),
        ]
        last = self.last_action.get(topo.name)
        if last is not None:
            candidates.append((self.last_name.get(topo.name, "last_accepted"), _load(topo, matrix, last)))
        choice = classify_candidates(candidates, topo.capacity)
        chosen_name = choice["selected"]
        if verdict.accepted:
            applied_name = "p1_general"
            applied_load = true_proposed
            fallback = None
        else:
            applied_name = chosen_name
            applied_load = dict(candidates)[chosen_name]
            fallback = chosen_name
        if verdict.accepted:
            self.last_action[topo.name] = np.asarray(proposed, dtype=np.float64).copy()
            self.last_name[topo.name] = "p1_general"
        pred_mae = float(np.mean(np.abs(mean_p - true_proposed)))
        return {
            "proposed_accepted": bool(verdict.accepted),
            "unsafe_accept": bool(verdict.accepted and _violations(true_proposed, topo.capacity) > 0),
            "false_reject": bool((not verdict.accepted) and _violations(true_proposed, topo.capacity) == 0),
            "fallback": fallback,
            "fallback_used": fallback is not None,
            "applied_action": applied_name,
            "fallback_outcome": None if verdict.accepted else choice["outcome"],
            "unavoidable_overload": bool((not verdict.accepted) and choice["unavoidable_overload"]),
            "mlu_actor": _mlu(true_proposed, topo.capacity),
            "mlu_applied": _mlu(applied_load, topo.capacity),
            "mlu_shortest": _mlu(candidates[0][1], topo.capacity),
            "mlu_equal": _mlu(candidates[1][1], topo.capacity),
            "violations_actor": _violations(true_proposed, topo.capacity),
            "violations_applied": _violations(applied_load, topo.capacity),
            "violations_shortest": _violations(candidates[0][1], topo.capacity),
            "prediction_mae": pred_mae,
            "policy_s": policy_s,
            "twin_ms": twin_ms,
            "gate_ms": 0.0,
            "n_links": int(topo.n_links),
        }

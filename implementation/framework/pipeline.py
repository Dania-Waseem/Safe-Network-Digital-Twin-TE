"""End-to-end Phase 3 controller: frozen P1 actor, twin, safety gate."""

from __future__ import annotations

import time

import numpy as np
import torch

from safety.evaluate import predict_action
from safety.gate import GateConfig, inspect_prediction, select_fallback
from twin.physics import demand_per_path, link_load, od_demands, shortest_split
from twin.data import TopologyTensors

from .policy_adapter import FrozenP1Policy


def _mlu(load: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.max(load / np.maximum(capacity, 1e-12)))


def _violations(load: np.ndarray, capacity: np.ndarray) -> int:
    return int(np.count_nonzero(load > capacity + 1e-6))


def _fit_ratio(load: np.ndarray, capacity: np.ndarray) -> float:
    total = float(np.sum(load))
    if total <= 0.0:
        return 1.0
    return float(np.sum(np.minimum(load, capacity)) / total)


class Framework:
    def __init__(
        self,
        policy: FrozenP1Policy,
        twin,
        config: GateConfig,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.twin = twin
        self.config = config
        self.device = device or torch.device("cpu")
        self.last_action: dict[str, np.ndarray] = {}
        self.last_name: dict[str, str] = {}

    def reset_session(self, topology: str | None = None) -> None:
        if topology is None:
            self.last_action.clear()
            self.last_name.clear()
        else:
            self.last_action.pop(topology, None)
            self.last_name.pop(topology, None)

    def decide(self, topo: TopologyTensors, matrix: np.ndarray) -> dict:
        od = od_demands(matrix)
        started = time.perf_counter()
        proposed = self.policy.propose(topo, od)
        policy_s = time.perf_counter() - started
        shortest = shortest_split(topo.n_paths, topo.hops)

        true_proposed, mean_p, std_p, twin_ms = predict_action(
            self.twin, topo, matrix, proposed, self.device
        )
        true_short, mean_s, std_s, short_ms = predict_action(
            self.twin, topo, matrix, shortest, self.device
        )
        last = self.last_action.get(topo.name)
        last_true = last_mean = last_std = None
        last_ms = 0.0
        last_verdict = None
        if last is not None:
            last_true, last_mean, last_std, last_ms = predict_action(
                self.twin, topo, matrix, last, self.device
            )
            last_verdict = inspect_prediction(last_mean, last_std, topo.capacity, self.config)

        gate_started = time.perf_counter()
        proposed_verdict = inspect_prediction(mean_p, std_p, topo.capacity, self.config)
        short_verdict = inspect_prediction(mean_s, std_s, topo.capacity, self.config)
        applied = select_fallback(
            "p1_actor",
            proposed,
            proposed_verdict,
            last_name=self.last_name.get(topo.name),
            last_action=last,
            last_verdict=last_verdict,
            shortest_name="shortest",
            shortest_action=shortest,
            shortest_verdict=short_verdict,
            config=self.config,
        )
        gate_s = time.perf_counter() - gate_started

        if applied.source == "proposed":
            true_applied = true_proposed
            mean_applied, std_applied = mean_p, std_p
        elif applied.fallback in ("shortest", "shortest_last_resort"):
            true_applied = true_short
            mean_applied, std_applied = mean_s, std_s
        elif applied.fallback == "last_accepted" and last_true is not None:
            true_applied = last_true
            mean_applied, std_applied = last_mean, last_std
        else:
            true_applied = true_proposed
            mean_applied, std_applied = mean_p, std_p

        if applied.proposed_accepted:
            self.last_action[topo.name] = np.asarray(proposed, dtype=np.float64).copy()
            self.last_name[topo.name] = "p1_actor"

        pred_mae = float(np.mean(np.abs(mean_applied - true_applied)))
        return {
            "topology": topo.name,
            "proposed_action": "p1_actor",
            "proposed_accepted": bool(proposed_verdict.accepted),
            "reasons": list(proposed_verdict.reasons),
            "fallback": applied.fallback,
            "applied_source": applied.source,
            "mlu_mean": proposed_verdict.mlu_mean,
            "mlu_upper": proposed_verdict.mlu_upper,
            "max_relative_std": proposed_verdict.max_relative_std,
            "mlu_true_proposed": _mlu(true_proposed, topo.capacity),
            "mlu_true_shortest": _mlu(true_short, topo.capacity),
            "mlu_true_applied": _mlu(true_applied, topo.capacity),
            "violations_proposed": _violations(true_proposed, topo.capacity),
            "violations_shortest": _violations(true_short, topo.capacity),
            "violations_applied": _violations(true_applied, topo.capacity),
            "fit_ratio_proposed": _fit_ratio(true_proposed, topo.capacity),
            "fit_ratio_shortest": _fit_ratio(true_short, topo.capacity),
            "fit_ratio_applied": _fit_ratio(true_applied, topo.capacity),
            "prediction_mae_applied": pred_mae,
            "policy_s": policy_s,
            "twin_ms": twin_ms + short_ms + last_ms,
            "gate_ms": gate_s * 1000.0,
            "action_sums_to_demands": bool(
                np.allclose(
                    _demand_sums(proposed, topo.n_paths),
                    np.ones(int(np.sum(topo.n_paths > 0))),
                    atol=1e-4,
                )
            ),
            "n_links": int(topo.n_links),
            "load_checksum_proposed": float(true_proposed.sum()),
        }


def _demand_sums(action: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    positive = counts[counts > 0]
    if positive.size == 0:
        return np.zeros(0)
    # equal-length groups are not required; use reduceat
    starts = np.cumsum(counts) - counts
    ends = starts + counts
    sums = []
    for start, end, count in zip(starts, ends, counts):
        if count > 0:
            sums.append(float(np.sum(action[start:end])))
    return np.asarray(sums)


def actual_load(topo: TopologyTensors, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    od = od_demands(matrix)
    return link_load(
        demand_per_path(od, topo.demand_row),
        action,
        topo.incidence_path,
        topo.incidence_link,
        topo.n_links,
    )

"""Standardized physics-oracle fallback for improved-gate comparisons.

Reuses ``framework.fallback_modes.classify_candidates`` without modifying it.
Candidate set is fixed to shortest-path and equal-split actions for every
gated variant (B–E). Selection prefers a feasible candidate; otherwise the
least-violating candidate (fewest overloaded links, then lowest MLU).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from framework.fallback_modes import classify_candidates
from twin.physics import equal_split, shortest_split

from .oracle import action_link_load


DEFAULT_CANDIDATES = ("shortest", "equal")


@dataclass
class FallbackResult:
    selected: str
    selected_action: np.ndarray
    selected_load: np.ndarray
    outcome: str
    feasible_fallback: bool
    no_feasible_fallback: bool
    selected_violations: int
    selected_mlu: float
    candidates: list[dict[str, Any]]

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Actions/loads are large; logging keeps scalars + names. Callers that
        # need arrays keep ``selected_action`` / ``selected_load`` on the object.
        payload.pop("selected_action", None)
        payload.pop("selected_load", None)
        return payload


def build_candidate_actions(topo) -> dict[str, np.ndarray]:
    return {
        "shortest": shortest_split(topo.n_paths, topo.hops),
        "equal": equal_split(topo.n_paths),
    }


def select_physics_fallback(
    topo,
    matrix: np.ndarray,
    *,
    candidate_names: tuple[str, ...] | list[str] = DEFAULT_CANDIDATES,
) -> FallbackResult:
    """Choose fallback using independent physics feasibility on each candidate."""
    actions = build_candidate_actions(topo)
    missing = [name for name in candidate_names if name not in actions]
    if missing:
        raise KeyError(f"unsupported fallback candidates: {missing}")

    scored_pairs: list[tuple[str, np.ndarray]] = []
    loads: dict[str, np.ndarray] = {}
    for name in candidate_names:
        load = action_link_load(topo, matrix, actions[name])
        loads[name] = load
        scored_pairs.append((name, load))

    choice = classify_candidates(scored_pairs, topo.capacity)
    selected = str(choice["selected"])
    feasible = choice["outcome"] == "feasible_fallback"
    return FallbackResult(
        selected=selected,
        selected_action=actions[selected],
        selected_load=loads[selected],
        outcome=str(choice["outcome"]),
        feasible_fallback=feasible,
        no_feasible_fallback=bool(choice["unavoidable_overload"]),
        selected_violations=int(choice["selected_violations"]),
        selected_mlu=float(choice["selected_mlu"]),
        candidates=list(choice["candidates"]),
    )

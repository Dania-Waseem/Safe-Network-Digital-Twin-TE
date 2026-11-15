"""Fallback candidates that do not change the existing safety gate.

Three outcomes are reported separately:

- feasible_fallback: at least one candidate routes with no capacity violation.
- least_violating_fallback: every candidate overloads, and the one with the
  fewest overloaded links (then lowest MLU) is selected.
- unavoidable_overload: no candidate in the set is feasible. This flag is
  true exactly when the outcome is least_violating_fallback.
"""

from __future__ import annotations

import numpy as np


def _mlu(load: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.max(load / np.maximum(capacity, 1e-12)))


def _violations(load: np.ndarray, capacity: np.ndarray) -> int:
    return int(np.count_nonzero(load > capacity + 1e-6))


def classify_candidates(candidates: list[tuple[str, np.ndarray]], capacity: np.ndarray) -> dict:
    if not candidates:
        raise ValueError("at least one fallback candidate is required")
    scored = []
    for name, load in candidates:
        scored.append(
            {
                "name": name,
                "violations": _violations(load, capacity),
                "mlu": _mlu(load, capacity),
            }
        )
    feasible = [row for row in scored if row["violations"] == 0]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["mlu"], row["name"]))
        outcome = "feasible_fallback"
    else:
        chosen = min(scored, key=lambda row: (row["violations"], row["mlu"], row["name"]))
        outcome = "least_violating_fallback"
    return {
        "outcome": outcome,
        "unavoidable_overload": outcome == "least_violating_fallback",
        "selected": chosen["name"],
        "selected_violations": chosen["violations"],
        "selected_mlu": chosen["mlu"],
        "candidates": scored,
    }

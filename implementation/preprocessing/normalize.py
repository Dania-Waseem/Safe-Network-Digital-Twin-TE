"""Normalise capacities and demands with the only scale formula in the P1 code.

Frozen training does **not** apply it. ``BASELINE_FREEZE/frozen_b4_config.py``
sets ``H0_SCALING = False``, and ``build_bipartite_graph.initial_features``
documents raw ``[c(e)]`` and ``[d_p]`` as P1 equation (5).

The optional branch in that same function is::

    denom = max(capacity)
    cap = cap / denom
    demand = demand / denom

The denominator is a link capacity, not a traffic statistic, so the
train/test split cannot leak into it. Original values are kept elsewhere.
"""

from __future__ import annotations

import numpy as np

METHOD = "p1_initial_features_scale_true"
P1_TRAINING_DEFAULT = "unscaled"


def parameters(capacities: np.ndarray) -> dict:
    caps = np.asarray(capacities, dtype=np.float64)
    if caps.size == 0 or not np.isfinite(caps).all() or float(caps.max()) <= 0.0:
        raise ValueError("Cannot normalise: capacities must be positive and finite")
    denom = float(caps.max())
    return {
        "method": METHOD,
        "formula": "x_normalised = x_original / max_link_capacity",
        "denominator": denom,
        "denominator_source": "maximum directed-link capacity of this topology",
        "uses_traffic_matrices": False,
        "leakage": "none; the denominator is not computed from demands",
        "p1_training_default": P1_TRAINING_DEFAULT,
        "p1_reference": (
            "baseline_p1_reproduction/preprocessing/build_bipartite_graph.py "
            "initial_features(scale=True); frozen H0_SCALING is False"
        ),
    }


def apply_capacity(capacities: np.ndarray, denom: float) -> np.ndarray:
    return np.asarray(capacities, dtype=np.float64) / denom


def apply_demand(demand: np.ndarray, denom: float) -> np.ndarray:
    """Scale a copy. Non-finite and negative entries become NaN and are not clipped."""
    out = np.asarray(demand, dtype=np.float64) / denom
    invalid = ~np.isfinite(demand) | (np.asarray(demand, dtype=np.float64) < 0.0)
    out = out.astype(np.float64, copy=True)
    out[invalid] = np.nan
    return out

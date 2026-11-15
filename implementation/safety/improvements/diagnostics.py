"""Uncertainty / utilisation diagnostics for paired gate evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .oracle import action_link_load, mlu, physics_violations
from .twin_backend import TwinBackend, TwinPrediction


CAPACITY_BINS = (
    ("far_below", 0.0, 0.5),
    ("mid", 0.5, 0.85),
    ("near_limit", 0.85, 1.0),
    ("over_limit", 1.0, float("inf")),
)


def _floor(capacity: np.ndarray, min_capacity: float = 1e-12) -> np.ndarray:
    return np.maximum(np.asarray(capacity, dtype=np.float64), min_capacity)


def utilisation(load: np.ndarray, capacity: np.ndarray, *, min_capacity: float = 1e-12) -> np.ndarray:
    return np.asarray(load, dtype=np.float64) / _floor(capacity, min_capacity)


def capacity_proximity_bin(mlu_value: float) -> str:
    for name, lo, hi in CAPACITY_BINS:
        if lo <= mlu_value < hi:
            return name
    return "over_limit"


def scalar_stats(arr: np.ndarray | None) -> dict[str, float]:
    if arr is None or np.asarray(arr).size == 0:
        return {"mean": float("nan"), "max": float("nan"), "p95": float("nan")}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
    }


def twin_uncertainty_views(
    pred: TwinPrediction,
    *,
    conformal_std_scale: float,
) -> dict[str, np.ndarray]:
    """Split raw ensemble std vs calibrated std (absolute load units)."""
    raw = np.asarray(pred.std_load, dtype=np.float64)
    scale = float(conformal_std_scale)
    if pred.conformal_applied or (pred.conformal_std_scale_applied != 1.0):
        # pred.std_load is already calibrated; recover raw.
        calibrated = raw
        raw_std = calibrated / max(scale, 1e-30)
    else:
        raw_std = raw
        calibrated = raw_std * scale
    return {"raw_std_load": raw_std, "calibrated_std_load": calibrated}


def build_case_diagnostics(
    *,
    topo,
    matrix: np.ndarray,
    policy_action: np.ndarray,
    decision_record: dict[str, Any],
    twin_pred: TwinPrediction | None = None,
    gate_k: float = 1.0,
    min_capacity: float = 1e-12,
    conformal_std_scale: float = 1.0,
) -> dict[str, Any]:
    """Attach Twin/oracle diagnostic fields to a decision log row."""
    capacity = np.asarray(topo.capacity, dtype=np.float64)
    true_load = action_link_load(topo, matrix, policy_action)
    actual_util = utilisation(true_load, capacity, min_capacity=min_capacity)
    actual_mlu = float(np.max(actual_util)) if actual_util.size else float("nan")

    out: dict[str, Any] = {
        "actual_utilisation": scalar_stats(actual_util),
        "actual_mlu": actual_mlu,
        "capacity_proximity": capacity_proximity_bin(actual_mlu),
        "oracle_feasible_proposed": bool(physics_violations(true_load, capacity) == 0),
        "capacity_violations_proposed": int(physics_violations(true_load, capacity)),
    }

    if twin_pred is None:
        out.update(
            {
                "predicted_mean_utilisation": {"mean": float("nan"), "max": float("nan"), "p95": float("nan")},
                "raw_uncertainty_utilisation": {"mean": float("nan"), "max": float("nan"), "p95": float("nan")},
                "calibrated_uncertainty_utilisation": {
                    "mean": float("nan"),
                    "max": float("nan"),
                    "p95": float("nan"),
                },
                "absolute_prediction_error": {"mean": float("nan"), "max": float("nan"), "p95": float("nan")},
                "ucb_utilisation": {"mean": float("nan"), "max": float("nan"), "p95": float("nan")},
                "twin_mae_abs": float("nan"),
                "twin_mae_util": float("nan"),
            }
        )
        return out

    floor = _floor(capacity, min_capacity)
    mean_util = twin_pred.mean_load / floor
    views = twin_uncertainty_views(twin_pred, conformal_std_scale=conformal_std_scale)
    raw_util = views["raw_std_load"] / floor
    cal_util = views["calibrated_std_load"] / floor
    # UCB for diagnostics uses calibrated util uncertainty and the decision k.
    ucb = mean_util + float(gate_k) * cal_util
    abs_err = np.abs(twin_pred.mean_load - true_load)
    util_err = np.abs(mean_util - actual_util)

    out.update(
        {
            "predicted_mean_utilisation": scalar_stats(mean_util),
            "raw_uncertainty_utilisation": scalar_stats(raw_util),
            "calibrated_uncertainty_utilisation": scalar_stats(cal_util),
            "absolute_prediction_error": scalar_stats(abs_err),
            "utilisation_prediction_error": scalar_stats(util_err),
            "ucb_utilisation": scalar_stats(ucb),
            "twin_mae_abs": float(np.mean(abs_err)),
            "twin_mae_util": float(np.mean(util_err)),
            "conformal_std_scale": float(conformal_std_scale),
            "mlu_mean_predicted": float(np.max(mean_util)),
            "mlu_ucb_predicted": float(np.max(ucb)),
        }
    )
    # Preserve / overlay decision flags already present.
    out["acceptance"] = decision_record.get("acceptance")
    out["rejection_reason"] = decision_record.get("rejection_reason")
    out["fallback_used"] = decision_record.get("fallback_used")
    out["selected_fallback"] = decision_record.get("selected_fallback")
    out["decision_runtime_ms"] = decision_record.get("decision_runtime_ms")
    out["twin_runtime_ms"] = decision_record.get("twin_runtime_ms")
    return out


def group_key(record: dict[str, Any], *, by: str) -> str:
    if by == "topology":
        return str(record.get("topology"))
    if by == "scale":
        return str(record.get("scale"))
    if by == "capacity_proximity":
        return str(record.get("capacity_proximity"))
    if by == "topology_scale":
        return f"{record.get('topology')}@scale={record.get('scale')}"
    raise KeyError(by)


def summarize_diagnostics(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = ("topology", "scale", "capacity_proximity"),
) -> dict[str, Any]:
    """Aggregate acceptance, fallback, error and runtime by requested facets."""

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        accepts = [bool(r.get("acceptance")) for r in subset]
        fallbacks = [bool(r.get("fallback_used")) for r in subset]
        return {
            "n": len(subset),
            "acceptance_rate": float(np.mean(accepts)),
            "fallback_rate": float(np.mean(fallbacks)),
            "oracle_feasible_rate": float(
                np.mean([bool(r.get("oracle_feasible_proposed")) for r in subset])
            ),
            "mean_actual_mlu": float(
                np.nanmean([r.get("actual_mlu", float("nan")) for r in subset])
            ),
            "mean_twin_mae_util": float(
                np.nanmean([r.get("twin_mae_util", float("nan")) for r in subset])
            ),
            "mean_decision_runtime_ms": float(
                np.nanmean([r.get("decision_runtime_ms", float("nan")) for r in subset])
            ),
            "mean_raw_unc_max": float(
                np.nanmean(
                    [
                        (r.get("raw_uncertainty_utilisation") or {}).get("max", float("nan"))
                        for r in subset
                    ]
                )
            ),
            "mean_cal_unc_max": float(
                np.nanmean(
                    [
                        (r.get("calibrated_uncertainty_utilisation") or {}).get(
                            "max", float("nan")
                        )
                        for r in subset
                    ]
                )
            ),
        }

    summary: dict[str, Any] = {"overall": _agg(rows), "groups": {}}
    for facet in group_by:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(group_key(row, by=facet), []).append(row)
        summary["groups"][facet] = {k: _agg(v) for k, v in sorted(buckets.items())}
    return summary


def raw_prediction_for_diagnostics(
    backend: TwinBackend,
    topo,
    matrix: np.ndarray,
    action: np.ndarray,
) -> TwinPrediction:
    """Force conformal scale 1.0 for raw-uncertainty diagnostics without mutating backend."""
    from twin.improvements.infer import predict_loads

    backend.load()
    true_load, mean_load, std_load, twin_ms = predict_loads(
        backend.model,
        topo,
        matrix,
        action,
        backend.device,
        conformal_std_scale=1.0,
    )
    return TwinPrediction(
        true_load=np.asarray(true_load, dtype=np.float64),
        mean_load=np.asarray(mean_load, dtype=np.float64),
        std_load=np.asarray(std_load, dtype=np.float64),
        twin_ms=float(twin_ms),
        backend=backend.name,
        checkpoint_path=str(backend.path),
        conformal_std_scale_applied=1.0,
        conformal_applied=False,
    )

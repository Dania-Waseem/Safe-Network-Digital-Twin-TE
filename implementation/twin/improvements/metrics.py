"""Shared twin metrics in absolute, relative, and utilisation spaces."""

from __future__ import annotations

import numpy as np


def score_prediction(
    true_load: np.ndarray,
    mean_load: np.ndarray,
    std_load: np.ndarray,
    capacity: np.ndarray,
) -> dict[str, float]:
    true_load = np.asarray(true_load, dtype=np.float64)
    mean_load = np.asarray(mean_load, dtype=np.float64)
    std_load = np.asarray(std_load, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    err = mean_load - true_load
    abs_err = np.abs(err)
    cap = np.maximum(capacity, 1e-12)
    load_floor = np.maximum(true_load, 1.0)
    util_true = true_load / cap
    util_hat = mean_load / cap
    util_err = util_hat - util_true
    std = np.maximum(std_load, 1e-12)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_load - true_load.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    if abs_err.std() < 1e-12 or std.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(std, abs_err)[0, 1])
    return {
        "links": int(true_load.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(r2),
        "max_abs_error": float(np.max(abs_err)) if abs_err.size else float("nan"),
        "mae_relative_to_load": float(np.mean(abs_err / load_floor)),
        "mae_utilisation": float(np.mean(np.abs(util_err))),
        "rmse_utilisation": float(np.sqrt(np.mean(util_err**2))),
        "max_util_error": float(np.max(np.abs(util_err))) if util_err.size else float("nan"),
        "mean_std": float(np.mean(std_load)),
        "mean_std_utilisation": float(np.mean(std_load / cap)),
        "coverage_1std": float(np.mean(abs_err <= std)),
        "coverage_2std": float(np.mean(abs_err <= 2.0 * std)),
        "std_error_correlation": corr,
        "mean_interval_width_2std": float(np.mean(4.0 * std)),
        "mean_interval_width_2std_util": float(np.mean(4.0 * std / cap)),
        "finite": bool(np.isfinite(mean_load).all() and np.isfinite(std_load).all()),
    }


def aggregate_case_metrics(rows: list[dict]) -> dict[str, float]:
    keys = [
        "mae",
        "rmse",
        "r2",
        "max_abs_error",
        "mae_relative_to_load",
        "mae_utilisation",
        "rmse_utilisation",
        "max_util_error",
        "mean_std",
        "mean_std_utilisation",
        "coverage_1std",
        "coverage_2std",
        "std_error_correlation",
        "mean_interval_width_2std",
        "mean_interval_width_2std_util",
    ]
    out = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
    out["cases"] = float(len(rows))
    out["all_finite"] = float(all(row["finite"] for row in rows))
    return out

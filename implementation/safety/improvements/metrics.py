"""Formal safety-gate metric definitions with explicit denominators.

Every rate documents numerator, denominator, and how failed / missing
decisions are treated. Failed decisions must not vanish silently.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _as_bool(value: Any) -> bool:
    return bool(value)


def partition_rows(
    rows: Iterable[dict[str, Any]],
    *,
    require_status_field: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows into completed vs failed/incomplete.

    A row is ``failed`` if ``decision_status`` is present and not ``ok``,
    or if ``acceptance`` is missing (incomplete log).
    """
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("decision_status", "ok" if "acceptance" in row else "incomplete")
        if status != "ok" or "acceptance" not in row:
            failed.append(row)
        else:
            completed.append(row)
    return {"completed": completed, "failed": failed}


def compute_safety_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute protocol metrics with explicit denominators.

    Definitions
    -----------
    Let N = number of logged decision attempts for the variant/slice
    (including failed attempts when ``decision_status`` is recorded).

    Let C = completed decisions with a definitive accept/reject
    (``decision_status == "ok"`` and ``acceptance`` present).

    Let A = accepted completed decisions (``acceptance`` is True).
    Let R = rejected completed decisions (``acceptance`` is False).
    Note A + R = C.

    Let U = completed decisions where the **proposed** policy action is
    physically unsafe (``capacity_violations_proposed > 0`` or
    ``oracle_feasible_proposed`` is False).

    Let S = completed decisions where the proposed action is physically
    safe (``oracle_feasible_proposed`` is True).  U + S = C when oracle
    labels exist; otherwise rows missing oracle labels are counted in
    ``missing_oracle`` and excluded from U/S-conditional rates' denominators
    only when documented below.

    Metrics
    -------
    - **acceptance_rate** = |A| / |C|
      Denominator excludes failed/incomplete rows. Failed count reported
      separately as ``n_failed``; they are **not** treated as rejects.

    - **unsafe_acceptance_rate** = |{accepted ∧ unsafe proposed}| / |C|
      Primary protocol rate (per completed decision). Secondary:
      ``unsafe_acceptance_rate_among_accepts`` = same numerator / |A|
      (undefined / NaN if A=0).

    - **false_rejection_rate** = |{rejected ∧ safe proposed}| / |C|
      Secondary: ``false_rejection_rate_among_safe`` = numerator / |S|
      (NaN if S=0).

    - **fallback_frequency** = |{fallback_used}| / |C|

    - **no_feasible_fallback_frequency** =
      |{fallback_used ∧ no_feasible_fallback}| / |C|
      (Also reported among rejects: numerator / |R|, NaN if R=0.)

    - **capacity_violations** (proposed): mean/sum of
      ``capacity_violations_proposed`` over C.
      Applied-action violations use ``capacity_violations_applied``.

    - **error_detection_rate** (recall on unsafe proposals) =
      |{rejected ∧ unsafe}| / |U|   (NaN if U=0)
      i.e. fraction of physically unsafe proposals the gate rejected.

    - **false_alarm_rate** (false positive rate on safe proposals) =
      |{rejected ∧ safe}| / |S|   (NaN if S=0)
      Identical to ``false_rejection_rate_among_safe``.

    - **runtime_overhead_ms** =
      mean(``decision_runtime_ms``) over C, with optional
      ``twin_runtime_ms`` mean. Failed rows' runtimes are summarised
      separately under ``failed_runtime_ms_mean`` when present.
    """
    parts = partition_rows(rows)
    completed = parts["completed"]
    failed = parts["failed"]
    n = len(rows)
    c = len(completed)
    n_failed = len(failed)

    if c == 0:
        return {
            "n_logged": n,
            "n_completed": 0,
            "n_failed": n_failed,
            "acceptance_rate": float("nan"),
            "unsafe_acceptance_rate": float("nan"),
            "false_rejection_rate": float("nan"),
            "fallback_frequency": float("nan"),
            "no_feasible_fallback_frequency": float("nan"),
            "error_detection_rate": float("nan"),
            "false_alarm_rate": float("nan"),
            "runtime_overhead_ms": float("nan"),
            "notes": "no completed decisions; failed/incomplete rows retained in n_failed",
        }

    accepted = [_as_bool(r.get("acceptance")) for r in completed]
    a = int(np.sum(accepted))
    r = c - a

    def _unsafe(row: dict[str, Any]) -> bool | None:
        if "oracle_feasible_proposed" in row and row["oracle_feasible_proposed"] is not None:
            return not bool(row["oracle_feasible_proposed"])
        if "capacity_violations_proposed" in row and row["capacity_violations_proposed"] is not None:
            return int(row["capacity_violations_proposed"]) > 0
        return None

    unsafe_flags = [_unsafe(row) for row in completed]
    missing_oracle = sum(1 for flag in unsafe_flags if flag is None)
    labeled = [row for row, flag in zip(completed, unsafe_flags) if flag is not None]
    labeled_accept = [_as_bool(row.get("acceptance")) for row in labeled]
    labeled_unsafe = [bool(flag) for flag in unsafe_flags if flag is not None]
    u = int(np.sum(labeled_unsafe))
    s = len(labeled) - u

    unsafe_accept = int(
        np.sum(
            [
                acc and uns
                for acc, uns in zip(labeled_accept, labeled_unsafe)
            ]
        )
    )
    false_reject = int(
        np.sum(
            [
                (not acc) and (not uns)
                for acc, uns in zip(labeled_accept, labeled_unsafe)
            ]
        )
    )
    true_reject_unsafe = int(
        np.sum([(not acc) and uns for acc, uns in zip(labeled_accept, labeled_unsafe)])
    )

    fallback_used = int(np.sum([_as_bool(row.get("fallback_used")) for row in completed]))
    no_feas_fb = int(
        np.sum(
            [
                _as_bool(row.get("fallback_used")) and _as_bool(row.get("no_feasible_fallback"))
                for row in completed
            ]
        )
    )

    def _float_or_nan(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    viol_prop = [_float_or_nan(row.get("capacity_violations_proposed", 0)) or 0.0 for row in completed]
    viol_appl = [_float_or_nan(row.get("capacity_violations_applied", 0)) or 0.0 for row in completed]
    runtimes = [_float_or_nan(row.get("decision_runtime_ms")) for row in completed]
    twin_rt = [_float_or_nan(row.get("twin_runtime_ms")) for row in completed]

    def _rate(num: int, den: int) -> float:
        return float(num) / float(den) if den else float("nan")

    return {
        "n_logged": n,
        "n_completed": c,
        "n_failed": n_failed,
        "n_accepted": a,
        "n_rejected": r,
        "n_unsafe_proposed": u,
        "n_safe_proposed": s,
        "n_missing_oracle_label": missing_oracle,
        "acceptance_rate": _rate(a, c),
        "unsafe_acceptance_rate": _rate(unsafe_accept, c),
        "unsafe_acceptance_rate_among_accepts": _rate(unsafe_accept, a),
        "false_rejection_rate": _rate(false_reject, c),
        "false_rejection_rate_among_safe": _rate(false_reject, s),
        "fallback_frequency": _rate(fallback_used, c),
        "no_feasible_fallback_frequency": _rate(no_feas_fb, c),
        "no_feasible_fallback_among_rejects": _rate(no_feas_fb, r),
        "capacity_violations_proposed_mean": float(np.mean(viol_prop)),
        "capacity_violations_proposed_sum": float(np.sum(viol_prop)),
        "capacity_violations_applied_mean": float(np.mean(viol_appl)),
        "capacity_violations_applied_sum": float(np.sum(viol_appl)),
        "error_detection_rate": _rate(true_reject_unsafe, u),
        "false_alarm_rate": _rate(false_reject, s),
        "runtime_overhead_ms": float(np.nanmean(runtimes)),
        "twin_runtime_ms_mean": float(np.nanmean(twin_rt)),
        "failed_runtime_ms_mean": float(
            np.nanmean(
                [float(row.get("decision_runtime_ms", float("nan"))) for row in failed]
            )
        )
        if failed
        else float("nan"),
        "denominator_notes": {
            "acceptance_rate": "|accepted| / |completed|",
            "unsafe_acceptance_rate": "|accepted ∧ unsafe| / |completed|",
            "false_rejection_rate": "|rejected ∧ safe| / |completed|",
            "fallback_frequency": "|fallback_used| / |completed|",
            "no_feasible_fallback_frequency": "|fallback_used ∧ no_feasible_fallback| / |completed|",
            "error_detection_rate": "|rejected ∧ unsafe| / |unsafe proposed|",
            "false_alarm_rate": "|rejected ∧ safe| / |safe proposed|",
            "failed_rows": "counted in n_failed; excluded from rate denominators; never dropped from n_logged",
        },
    }

"""Decision-log helpers for the improved Safety Gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_LOG_FIELDS = (
    "seed",
    "topology",
    "matrix_id",
    "scale",
    "policy_checkpoint",
    "gate_variant",
    "twin_backend",
    "predicted_mean_utilisation",
    "predicted_uncertainty_utilisation",
    "ucb_utilisation",
    "acceptance",
    "rejection_reason",
    "selected_fallback",
    "oracle_feasible_proposed",
    "capacity_violations_proposed",
    "mlu_applied",
    "decision_runtime_ms",
    "proposed_action_source",
    "applied_action_source",
)


def validate_decision_record(record: dict[str, Any]) -> list[str]:
    """Return missing required field names (empty list if complete)."""
    return [name for name in REQUIRED_LOG_FIELDS if name not in record]


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

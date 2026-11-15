"""Resume / deduplication helpers for paired evaluation runs.

Case identity is independent of variant ordering. Completed cases are
skipped on resume so interrupted runs can continue without duplicating
A–F blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def case_id(case: dict[str, Any], *, checkpoint_id: str | None = None) -> str:
    """Stable identity for one (seed, topology, matrix, scale[, policy]) case."""
    parts = [
        f"seed={case.get('seed')}",
        f"topo={case.get('topology')}",
        f"split={case.get('split')}",
        f"mid={case.get('matrix_id')}",
        f"li={case.get('local_index')}",
        f"scale={case.get('scale')}",
    ]
    if checkpoint_id is not None:
        parts.append(f"policy={checkpoint_id}")
    return "|".join(parts)


def load_completed_case_ids(
    decisions_path: Path,
    *,
    expected_variants: int | None = 6,
) -> set[str]:
    """Return case_ids that already have a complete successful A–F block.

    Failed case rows (``decision_status != ok``) do **not** mark a case complete,
    so resume can retry them. A case is complete only when it has at least
    ``expected_variants`` successful decision rows (default 6 = A–F).
    """
    if not decisions_path.exists():
        return set()
    ok_counts: dict[str, int] = {}
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("case_id")
        if not cid:
            continue
        status = row.get("decision_status", "ok" if "acceptance" in row else "incomplete")
        if status == "ok" and "acceptance" in row:
            ok_counts[str(cid)] = ok_counts.get(str(cid), 0) + 1
    if expected_variants is None:
        return {cid for cid, n in ok_counts.items() if n > 0}
    return {cid for cid, n in ok_counts.items() if n >= int(expected_variants)}


def append_decision_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def read_decision_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

"""Shared helpers for the Phase 3 final experiments.

Every experiment writes only under ``results/phase3_final/<experiment>/``. Long
runs append one JSON line per finished unit of work, so a restarted run skips
the keys that are already on disk (same pattern as
``safety/improvements/resume.py``).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from twin.data import ROOT

FINAL_ROOT = ROOT / "results" / "phase3_final"
LP_SOLVER_NOTE = (
    "All LPs use SciPy HiGHS (scipy.optimize.linprog, method='highs'). "
    "No Gurobi licence was available, so Teal/FIGRET-style Gurobi LPs are replaced by HiGHS."
)
TOY_TM_NOTE = (
    "UsCarrier and Kdl traffic matrices are synthetic all-ones toy templates; every index is an "
    "identical copy, so additional TM indices or seeds add no traffic diversity on these topologies."
)
GEANT_SUBSAMPLE_SEED = 2026
GEANT_SUBSAMPLE_SIZE = 200


def out_dir(name: str) -> Path:
    path = FINAL_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def environment() -> dict[str, Any]:
    import scipy
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(row)) + "\n")
        fh.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially written last line from an interrupted run is ignored.
                continue
    return rows


def done_keys(path: Path, key_fields: Iterable[str]) -> set[tuple]:
    fields = tuple(key_fields)
    return {tuple(row.get(f) for f in fields) for row in read_jsonl(path)}


def matrix_hash(matrix: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(matrix, dtype=np.float64).tobytes()).hexdigest()


def geant_subsample(n_available: int) -> np.ndarray:
    """Fixed, seeded subsample of GEANT test positions (positions within the test split)."""
    rng = np.random.default_rng(GEANT_SUBSAMPLE_SEED)
    pick = rng.choice(n_available, size=min(GEANT_SUBSAMPLE_SIZE, n_available), replace=False)
    return np.sort(pick)


def fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"

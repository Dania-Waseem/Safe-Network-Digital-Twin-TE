"""Load traffic matrices and the train/val/test time split (P1 Section 5.1)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from config import B4_SPLITS, TM_DIR


def _tm_sort_key(path: Path) -> int:
    """B4.json_real_{seed}_1.0_traffic-matrix.pkl -> seed."""
    parts = path.stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def load_traffic_matrices(name: str = "B4", tm_dir: Path | None = None) -> list[np.ndarray]:
    """Return TMs ordered by seed so the time split is consecutive."""
    folder = Path(tm_dir or TM_DIR) / name
    files = sorted(folder.glob("*traffic-matrix.pkl"), key=_tm_sort_key)
    if not files:
        raise FileNotFoundError(f"No traffic matrices in {folder}")

    matrices = []
    for file in files:
        with open(file, "rb") as handle:
            tm = pickle.load(handle)
        tm = np.asarray(tm, dtype=np.float64)
        np.fill_diagonal(tm, 0.0)
        tm = np.maximum(tm, 0.0)
        matrices.append(tm)
    return matrices


def split_indices(n_tms: int, name: str = "B4") -> dict[str, tuple[int, int]]:
    """Return half-open [start, stop) index ranges."""
    if name == "B4":
        splits = dict(B4_SPLITS)
    else:
        n_train = max(1, int(0.55 * n_tms))
        n_val = max(1, int(0.25 * n_tms))
        splits = {
            "train": (0, n_train),
            "val": (n_train, min(n_tms, n_train + n_val)),
            "test": (min(n_tms, n_train + n_val), n_tms),
        }
    for key, (start, stop) in splits.items():
        splits[key] = (max(0, start), min(n_tms, stop))
    return splits

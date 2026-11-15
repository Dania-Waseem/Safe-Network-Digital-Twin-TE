"""Load traffic matrices and validate them without clipping or dropping rows.

Kinds:
- real_mapped: TEAL B4 matrices (Yates/Abilene demands mapped onto B4)
- real_figret: FIGRET GÉANT histories
- synthetic_toy: TEAL all-ones placeholders (not Microsoft WAN traffic)
- reconstructed_gravity: degree-gravity series (not a measured Cogentco trace)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

# Same reconstruction as the Step 1 catalog and the frozen P1 gravity.py.
# P1 does not publish the author constants. Seed and sigma are fixed.
GRAVITY_SEED = 2025
GRAVITY_SIGMA = 0.15
GRAVITY_COUNT = 36

# Frozen P1 config.B4_SPLITS. The paper does not publish this cut.
B4_SPLIT = {"train": [0, 20], "val": [20, 28], "test": [28, 36]}


def validate_matrix(matrix: np.ndarray, n: int) -> dict:
    """Count problems. The matrix itself is not modified."""
    matrix = np.asarray(matrix)
    record = {
        "shape": list(matrix.shape),
        "valid": True,
        "reasons": [],
        "negative_entries": 0,
        "non_finite_entries": 0,
        "self_demand_entries": 0,
        "self_demand_nonzero": 0,
        "zero_off_diagonal": 0,
        "positive_off_diagonal": 0,
    }
    if matrix.ndim != 2 or matrix.shape != (n, n):
        record["valid"] = False
        record["reasons"].append(f"shape {matrix.shape} is not {(n, n)}")
        return record
    values = matrix.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    record["non_finite_entries"] = int((~finite).sum())
    negative = finite & (values < 0.0)
    record["negative_entries"] = int(negative.sum())
    eye = np.eye(n, dtype=bool)
    record["self_demand_entries"] = n
    record["self_demand_nonzero"] = int(np.count_nonzero(values[eye]))
    off = ~eye
    record["zero_off_diagonal"] = int(np.count_nonzero((values == 0.0) & off & finite))
    record["positive_off_diagonal"] = int(np.count_nonzero((values > 0.0) & off & finite))
    if record["non_finite_entries"]:
        record["valid"] = False
        record["reasons"].append("non-finite demand")
    if record["negative_entries"]:
        record["valid"] = False
        record["reasons"].append("negative demand")
    return record


def tm_index(path: Path) -> int:
    parts = path.name.split("_")
    for key in ("real", "toy"):
        if key in parts:
            return int(parts[parts.index(key) + 1])
    raise ValueError(f"No traffic index in {path.name}")


def load_pickle_series(paths: list[Path]) -> list[np.ndarray]:
    matrices = []
    for path in paths:
        with path.open("rb") as handle:
            matrices.append(np.asarray(pickle.load(handle)))
    return matrices


def load_b4(folder: Path, n: int) -> dict:
    files = sorted(folder.glob("B4.json_real_*_traffic-matrix.pkl"), key=tm_index)
    matrices = [np.asarray(m, dtype=np.float64) for m in load_pickle_series(files)]
    stack = np.stack(matrices) if matrices else np.zeros((0, n, n))
    checks = [validate_matrix(m, n) for m in stack]
    return {
        "kind": "real_mapped",
        "label": "TEAL public real matrices: Yates/Abilene demands mapped onto B4. Not Microsoft WAN traffic.",
        "files": [p.name for p in files],
        "demand_original": stack,
        "checks": checks,
        "split": {
            "train": B4_SPLIT["train"],
            "val": B4_SPLIT["val"],
            "test": B4_SPLIT["test"],
            "status": "Frozen P1 config.B4_SPLITS (20/8/8). Lu et al. do not publish the cut.",
        },
    }


def load_toy(folder: Path, prefix: str, n: int) -> dict:
    files = sorted(folder.glob(f"{prefix}.json_toy_*_traffic-matrix.pkl"), key=tm_index)
    checks = []
    first = None
    identical = True
    for path in files:
        with path.open("rb") as handle:
            matrix = np.asarray(pickle.load(handle))
        checks.append(validate_matrix(matrix, n))
        if first is None:
            first = np.asarray(matrix, dtype=np.float64)
        elif matrix.shape != first.shape or not np.array_equal(matrix, first.astype(matrix.dtype, copy=False)):
            identical = False
    return {
        "kind": "synthetic_toy",
        "label": (
            "TEAL toy placeholder. The public repository does not include "
            "Microsoft WAN traces for this topology. Not real traffic."
        ),
        "files": [p.name for p in files],
        "demand_original": None,
        "template": first,
        "identical_files": identical,
        "checks": checks,
        "split": {
            "train": B4_SPLIT["train"],
            "val": B4_SPLIT["val"],
            "test": B4_SPLIT["test"],
            "status": (
                "Index convention only. The matrices are identical, so this is "
                "not a time split and is not used to fit normalisation."
            ),
        },
    }


def _hist_matrix(line: str, n: int) -> np.ndarray:
    values = np.fromstring(line, sep=" ", dtype=np.float64)
    if values.size != n * n:
        raise ValueError(f"GÉANT history row has {values.size} values, expected {n * n}")
    return values.reshape(n, n)


def load_geant(root: Path, n: int) -> dict:
    def load_split(folder: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        matrices = []
        opts = []
        parts = []
        for hist in sorted(folder.glob("*.hist"), key=lambda p: int(p.stem)):
            rows = []
            with hist.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(_hist_matrix(line, n))
            opt_vals = []
            with hist.with_suffix(".opt").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        opt_vals.append(float(line))
            if len(opt_vals) != len(rows):
                raise ValueError(f"{hist.name}: {len(rows)} matrices and {len(opt_vals)} opt values")
            parts.append({"file": hist.name, "rows": len(rows)})
            matrices.extend(rows)
            opts.extend(opt_vals)
        stack = np.stack(matrices) if matrices else np.zeros((0, n, n))
        return stack, np.asarray(opts, dtype=np.float64), parts

    train, train_opt, train_parts = load_split(root / "train")
    test, test_opt, test_parts = load_split(root / "test")
    checks = [validate_matrix(m, n) for m in train] + [validate_matrix(m, n) for m in test]
    return {
        "kind": "real_figret",
        "label": "FIGRET published GÉANT demand histories. Not synthesised in this project.",
        "demand_train": train,
        "demand_test": test,
        "opt_mlu_train": train_opt,
        "opt_mlu_test": test_opt,
        "train_files": train_parts,
        "test_files": test_parts,
        "checks": checks,
        "split": {
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "val_rows": 0,
            "order": "train files 1..5 in numeric order, then test files 6..7. No validation split exists.",
            "status": "FIGRET's published train/ and test/ directories. A validation cut was not invented.",
        },
    }


def gravity_matrices(graph, n_tms: int = GRAVITY_COUNT) -> np.ndarray:
    """Degree gravity. Matches the documented frozen reconstruction, including seed."""
    import networkx as nx

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("gravity expects the directed topology")
    nodes = list(range(graph.number_of_nodes()))
    attractiveness = np.array([max(graph.degree(u), 1) for u in nodes], dtype=np.float64)
    base = np.outer(attractiveness, attractiveness)
    np.fill_diagonal(base, 0.0)
    target = float(sum(data.get("capacity", 0.0) for *_, data in graph.edges(data=True)))
    base *= target / max(float(base.sum()), 1e-9)
    rng = np.random.default_rng(GRAVITY_SEED)
    matrices = []
    for _ in range(n_tms):
        noise = rng.lognormal(0.0, GRAVITY_SIGMA, size=base.shape)
        matrix = np.maximum(base * noise, 0.0)
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    return np.stack(matrices)


def load_gravity(graph, n: int) -> dict:
    stack = gravity_matrices(graph)
    checks = [validate_matrix(m, n) for m in stack]
    return {
        "kind": "reconstructed_gravity",
        "label": (
            "Reconstructed degree-gravity traffic. Not a measured Cogentco trace. "
            f"T_ij = A_i A_j, A_i = max(degree(i), 1), lognormal noise seed {GRAVITY_SEED}, "
            f"sigma {GRAVITY_SIGMA}. P1 does not publish the author constants."
        ),
        "demand_original": stack,
        "checks": checks,
        "split": {
            "train": B4_SPLIT["train"],
            "val": B4_SPLIT["val"],
            "test": B4_SPLIT["test"],
            "status": "There is no time axis. The 20/8/8 index cut is a convention, not an observed split.",
        },
    }


def summarise_checks(checks: list[dict]) -> dict:
    valid = sum(1 for c in checks if c["valid"])
    return {
        "matrices": len(checks),
        "valid_matrices": valid,
        "invalid_matrices": len(checks) - valid,
        "negative_entries": int(sum(c["negative_entries"] for c in checks)),
        "non_finite_entries": int(sum(c["non_finite_entries"] for c in checks)),
        "zero_off_diagonal_entries": int(sum(c["zero_off_diagonal"] for c in checks)),
        "positive_off_diagonal_entries": int(sum(c["positive_off_diagonal"] for c in checks)),
        "self_demand_nonzero_entries": int(sum(c["self_demand_nonzero"] for c in checks)),
    }


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

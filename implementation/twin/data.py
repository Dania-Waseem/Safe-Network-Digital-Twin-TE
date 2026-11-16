"""Load a ready topology and the Step 4 traffic slices.

ASN is not a ready topology. This module never reads ``ASN_shards``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from repo_paths import BASELINE, IMPL, MODELS, REPO as ROOT  # noqa: F401  (ROOT = repository root)
PROCESSED = ROOT / "data" / "processed"
SPLIT_ROOT = PROCESSED / "splits"

READY = ("B4", "UsCarrier", "Cogentco", "GEANT", "Kdl")
# Additional unseen test topologies with real SNDlib traffic. They are evaluation-only:
# no model, threshold or calibration constant is fitted on them, and they are kept out of
# READY so that no training or calibration loop iterates over them.
SNDLIB_TEST = ("Abilene", "Germany50")


@dataclass
class TopologyTensors:
    name: str
    n_links: int
    denominator: float
    capacity: np.ndarray
    n_paths: np.ndarray
    hops: np.ndarray
    demand_row: np.ndarray
    incidence_path: np.ndarray
    incidence_link: np.ndarray

    @property
    def n_path_nodes(self) -> int:
        return int(self.hops.shape[0])


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_topology(name: str) -> TopologyTensors:
    if name not in READY and name not in SNDLIB_TEST:
        raise KeyError(f"{name} is not a completed topology for the twin")
    graph = torch.load(
        PROCESSED / "graph_data" / f"{name}_bipartite.pt",
        map_location="cpu",
        weights_only=False,
    )
    meta = json.loads((PROCESSED / "normalization" / f"{name}.json").read_text(encoding="utf-8"))
    capacity = _as_numpy(graph["capacity_original"]).astype(np.float64)
    return TopologyTensors(
        name=name,
        n_links=int(capacity.shape[0]),
        denominator=float(meta["denominator"]),
        capacity=capacity,
        n_paths=_as_numpy(graph["n_paths"]).astype(np.int64),
        hops=_as_numpy(graph["hop_count"]).astype(np.int64),
        demand_row=_as_numpy(graph["demand_row"]).astype(np.int64),
        incidence_path=_as_numpy(graph["incidence_path"]).astype(np.int64),
        incidence_link=_as_numpy(graph["incidence_link"]).astype(np.int64),
    )


def load_split_demands(split: str, name: str) -> tuple[np.ndarray, dict]:
    path = SPLIT_ROOT / split / f"{name}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    arrays = np.load(ROOT / meta["demand_file"])
    return np.asarray(arrays["demand"], dtype=np.float64), meta

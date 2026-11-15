"""Routing actions and the link-load simulator used as twin labels.

The label is the flow that the candidate-path routing places on each directed
link. It is not an SNMP counter. Given a traffic matrix and a split of each
demand over its candidate paths, the load on link e is

    load(e) = sum_{paths p through e} demand(p) * action(p).

That identity is the ground truth the ensemble is trained to reproduce.
"""

from __future__ import annotations

import numpy as np


def od_demands(matrix: np.ndarray) -> np.ndarray:
    """Off-diagonal demands in source-major order, matching path tables."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.asarray(matrix, dtype=np.float64)[mask]


def demand_per_path(od: np.ndarray, demand_row: np.ndarray) -> np.ndarray:
    return np.asarray(od, dtype=np.float64)[np.asarray(demand_row, dtype=np.int64)]


def _group_slices(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    return counts, starts


def equal_split(n_paths: np.ndarray) -> np.ndarray:
    counts, _starts = _group_slices(n_paths)
    positive = counts[counts > 0]
    if positive.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.repeat(1.0 / positive, positive)


def shortest_split(n_paths: np.ndarray, hops: np.ndarray) -> np.ndarray:
    """Put all flow on the first minimum-hop path of each demand."""
    counts, starts = _group_slices(n_paths)
    hops = np.asarray(hops, dtype=np.int64)
    frac = np.zeros(int(counts.sum()), dtype=np.float64)
    for count in np.unique(counts):
        if count <= 0:
            continue
        selected = np.flatnonzero(counts == count)
        block = hops[starts[selected, None] + np.arange(count)[None, :]]
        chosen = starts[selected] + np.argmin(block, axis=1)
        frac[chosen] = 1.0
    return frac


def dirichlet_split(n_paths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts, starts = _group_slices(n_paths)
    frac = np.zeros(int(counts.sum()), dtype=np.float64)
    for count in np.unique(counts):
        if count <= 0:
            continue
        selected = np.flatnonzero(counts == count)
        if count == 1:
            frac[starts[selected]] = 1.0
            continue
        draws = rng.dirichlet(np.ones(count, dtype=np.float64), size=selected.size)
        index = starts[selected, None] + np.arange(count)[None, :]
        frac[index.ravel()] = draws.ravel()
    return frac


def link_load(
    demand_path: np.ndarray,
    action: np.ndarray,
    incidence_path: np.ndarray,
    incidence_link: np.ndarray,
    n_links: int,
) -> np.ndarray:
    flow = np.asarray(demand_path, dtype=np.float64) * np.asarray(action, dtype=np.float64)
    load = np.zeros(int(n_links), dtype=np.float64)
    np.add.at(
        load,
        np.asarray(incidence_link, dtype=np.int64),
        flow[np.asarray(incidence_path, dtype=np.int64)],
    )
    return load


def path_flow_checksum(demand_path: np.ndarray, action: np.ndarray, hops: np.ndarray) -> float:
    """Sum of link loads must equal sum of flow times hop count."""
    return float(np.sum(np.asarray(demand_path) * np.asarray(action) * np.asarray(hops)))

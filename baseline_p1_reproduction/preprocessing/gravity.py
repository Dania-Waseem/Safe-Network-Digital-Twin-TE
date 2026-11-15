"""Gravity-model synthetic TMs for Cogentco (P1 §5.1).

P1 cites Applegate & Cohen (2003) and Roughan et al. (2002) but does **not**
publish attractiveness, distance exponent, or a demand scale. This module
implements a standard, fully specified reconstruction:

    T_{ij} = A_i A_j    (i ≠ j),   A_i = max(deg(i), 1)

then scales the matrix so that the mean total demand equals ``target_total``.
``target_total`` is an **adaptation** (default: sum of link capacities) because
P1 does not give Cogentco's offered-load number.

Seeded RNG adds lognormal multiplicative noise per TM so 36 matrices differ
while remaining gravity-structured.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def gravity_matrices(
    graph: nx.DiGraph,
    n_tms: int = 36,
    seed: int = 2025,
    target_total: float | None = None,
    noise_sigma: float = 0.15,
) -> list[np.ndarray]:
    nodes = list(range(graph.number_of_nodes()))
    if set(graph.nodes()) != set(nodes):
        nodes = sorted(graph.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    A = np.array([max(graph.degree(u), 1) for u in nodes], dtype=np.float64)
    base = np.outer(A, A)
    np.fill_diagonal(base, 0.0)
    if target_total is None:
        caps = [float(d.get("capacity", 1.0) or 1.0) for *_, d in graph.edges(data=True)]
        target_total = float(sum(caps))
    scale = target_total / max(base.sum(), 1e-9)
    base = base * scale
    rng = np.random.default_rng(seed)
    tms = []
    for _ in range(n_tms):
        noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=base.shape)
        tm = base * noise
        np.fill_diagonal(tm, 0.0)
        tm = np.maximum(tm, 0.0)
        tms.append(tm.astype(np.float64))
    return tms

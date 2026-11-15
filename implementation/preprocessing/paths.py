"""Four hop-shortest simple paths per ordered pair.

Cost matches frozen P1 ``generate_paths._weighted_copy`` with metric
``min-hop``: an edge of positive capacity has weight 1. Edges that are not
positive are not given an invented finite cost; on these topologies every
kept edge has positive capacity, which the caller checks.

P1's ``_pad_paths`` repeats the shortest path until four paths exist.
This module does not. A pair keeps however many simple paths exist, at most 4.

Small and mid-size graphs use ``networkx.shortest_simple_paths``, the same
call P1 uses. Kdl and ASN use a unit-weight Yen search because that call does
not finish at their pair counts. Equal-length ties in the unit-weight search
follow adjacency order, which can differ from NetworkX's heap order. The hop
cost is the same.
"""

from __future__ import annotations

from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np

from .topology import Topology

K = 4
# NetworkX finishes in minutes at this size (Cogentco is about 39k pairs).
NETWORKX_PAIR_LIMIT = 60_000


def _successors(topology: Topology) -> list[list[int]]:
    succ = [[] for _ in range(topology.num_nodes)]
    for u, v in topology.edges:
        succ[u].append(v)
    return succ


def _bfs_path(succ, src, dst, banned_nodes, banned_edges, n: int) -> list[int] | None:
    """Unit-weight BFS. ``banned_edges`` holds encoded endpoints ``u * n + v``."""
    seen = bytearray(n)
    previous = array("i", [-1]) * n
    for node in banned_nodes:
        seen[node] = 1
    seen[src] = 1
    previous[src] = -2
    queue = deque([src])
    reached = False
    while queue:
        node = queue.popleft()
        if node == dst:
            reached = True
            break
        base = node * n
        for nxt in succ[node]:
            if seen[nxt] or (base + nxt) in banned_edges:
                continue
            seen[nxt] = 1
            previous[nxt] = node
            queue.append(nxt)
    if not reached:
        return None
    path = []
    cursor = dst
    while cursor != -2:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path


def yen_unit(succ, src: int, dst: int, k: int = K) -> list[list[int]]:
    """Yen's algorithm. Costs are hop counts. No padding."""
    n = len(succ)
    first = _bfs_path(succ, src, dst, (), set(), n)
    if first is None:
        return []
    accepted = [first]
    pool: list[tuple[int, ...]] = []
    pool_keys: set[tuple[int, ...]] = set()
    accepted_keys = {tuple(first)}
    for _ in range(1, k):
        previous = accepted[-1]
        for spur_at in range(len(previous) - 1):
            root = previous[: spur_at + 1]
            banned_edges: set[int] = set()
            root_key_len = spur_at + 1
            for path in accepted:
                if len(path) > root_key_len and path[:root_key_len] == root:
                    banned_edges.add(path[spur_at] * n + path[spur_at + 1])
            spur = _bfs_path(succ, root[-1], dst, root[:-1], banned_edges, n)
            if spur is None:
                continue
            candidate = tuple(root[:-1] + spur)
            if candidate in pool_keys or candidate in accepted_keys:
                continue
            pool_keys.add(candidate)
            pool.append(candidate)
        if not pool:
            break
        pool.sort(key=lambda path: (len(path), path))
        chosen = pool.pop(0)
        pool_keys.discard(chosen)
        accepted.append(list(chosen))
        accepted_keys.add(chosen)
    return accepted


_WORKER_SUCC: list[list[int]] | None = None


def _init_worker(succ: list[list[int]]) -> None:
    global _WORKER_SUCC
    _WORKER_SUCC = succ


def _yen_source_job(task: tuple) -> tuple:
    src, n, k = task
    if _WORKER_SUCC is None:
        raise RuntimeError("Path worker was started without a successor table")
    return _yen_source((src, _WORKER_SUCC, n, k))


def _yen_source(task: tuple) -> tuple:
    src, succ, n, k = task
    dests = []
    npaths = []
    hops = []
    nodes: list[int] = []
    ptr = [0]
    for dst in range(n):
        if dst == src:
            continue
        found = yen_unit(succ, src, dst, k)
        dests.append(dst)
        npaths.append(len(found))
        for path in found:
            hops.append(len(path) - 1)
            nodes.extend(path)
            ptr.append(len(nodes))
    return (
        np.asarray(dests, dtype=np.int32),
        np.asarray(npaths, dtype=np.int8),
        np.asarray(hops, dtype=np.int16),
        np.asarray(ptr, dtype=np.int64),
        np.asarray(nodes, dtype=np.int32),
    )


def networkx_pairs(topology: Topology, k: int = K) -> dict[tuple[int, int], list[list[int]]]:
    """P1's search: NetworkX Yen, weight 1 on every positive-capacity edge."""
    graph = topology.graph
    for u, v in graph.edges():
        cap = float(graph[u][v].get("capacity", 0.0))
        graph[u][v]["weight"] = 1.0 if cap > 0.0 else np.inf
    found: dict[tuple[int, int], list[list[int]]] = {}
    nodes = topology.node_ids
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            try:
                paths = list(islice(nx.shortest_simple_paths(graph, src, dst, weight="weight"), k))
            except nx.NetworkXNoPath:
                paths = []
            found[(src, dst)] = [list(map(int, path)) for path in paths]
    return found


def _pairs_to_arrays(topology: Topology, path_dict: dict) -> dict[str, np.ndarray]:
    dem_src = []
    dem_dst = []
    npaths = []
    hops = []
    ptr = [0]
    nodes: list[int] = []
    for src in topology.node_ids:
        for dst in topology.node_ids:
            if src == dst:
                continue
            paths = path_dict.get((src, dst), [])
            dem_src.append(src)
            dem_dst.append(dst)
            npaths.append(len(paths))
            for path in paths:
                hops.append(len(path) - 1)
                nodes.extend(int(v) for v in path)
                ptr.append(len(nodes))
    return {
        "demand_src": np.asarray(dem_src, dtype=np.int32),
        "demand_dst": np.asarray(dem_dst, dtype=np.int32),
        "n_paths": np.asarray(npaths, dtype=np.int8),
        "hop_count": np.asarray(hops, dtype=np.int16),
        "path_node_ptr": np.asarray(ptr, dtype=np.int64),
        "path_nodes": np.asarray(nodes, dtype=np.int32),
    }


def _merge_source_results(n: int, parts: list[tuple]) -> dict[str, np.ndarray]:
    dem_src = []
    dem_dst = []
    npaths = []
    hops = []
    ptr = [0]
    nodes: list[np.ndarray] = []
    node_count = 0
    for src, part in enumerate(parts):
        dests, counts, hop, local_ptr, local_nodes = part
        dem_src.append(np.full(dests.shape, src, dtype=np.int32))
        dem_dst.append(dests)
        npaths.append(counts)
        hops.append(hop)
        # local_ptr is per source, restarting at 0. Shift it into the global node buffer.
        shifted = local_ptr[1:] + node_count
        ptr.extend(int(v) for v in shifted)
        nodes.append(local_nodes)
        node_count += int(local_nodes.size)
    return {
        "demand_src": np.concatenate(dem_src),
        "demand_dst": np.concatenate(dem_dst),
        "n_paths": np.concatenate(npaths),
        "hop_count": np.concatenate(hops) if hops and hops[0].size else np.zeros(0, dtype=np.int16),
        "path_node_ptr": np.asarray(ptr, dtype=np.int64),
        "path_nodes": np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int32),
    }


def _shard_dir(topology: Topology) -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / "candidate_paths" / f"{topology.name}_shards"


def _save_shard(folder: Path, src: int, part: tuple) -> None:
    dests, counts, hop, local_ptr, local_nodes = part
    np.savez_compressed(
        folder / f"src_{src:05d}.npz",
        dests=dests,
        counts=counts,
        hop=hop,
        local_ptr=local_ptr,
        local_nodes=local_nodes,
    )


def _load_shard(folder: Path, src: int) -> tuple | None:
    path = folder / f"src_{src:05d}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return (
        data["dests"],
        data["counts"],
        data["hop"],
        data["local_ptr"],
        data["local_nodes"],
    )


def load_saved(path: Path | str) -> dict:
    """Reload a previously written ``*_yen4.npz`` plus its JSON meta sidecar."""
    import json

    path = Path(path)
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        fewer = int(np.count_nonzero(arrays["n_paths"] < K))
        missing = int(np.count_nonzero(arrays["n_paths"] == 0))
        meta = {
            "topology": path.stem.replace("_yen4", ""),
            "k_requested": K,
            "padded": False,
            "backend": "reloaded",
            "demands": int(arrays["n_paths"].size),
            "paths_generated": int(arrays["hop_count"].size),
            "demands_with_fewer_than_4": fewer,
            "demands_with_no_path": missing,
        }
    return {**arrays, "meta": meta}


def generate(topology: Topology, jobs: int = 1, checkpoint: bool = True) -> dict:
    if topology.num_edges and float(np.min(topology.capacities)) <= 0.0:
        raise ValueError(f"{topology.name}: a non-positive capacity reached path generation")
    pairs = topology.num_nodes * max(topology.num_nodes - 1, 0)
    if pairs <= NETWORKX_PAIR_LIMIT:
        arrays = _pairs_to_arrays(topology, networkx_pairs(topology, K))
        backend = "networkx.shortest_simple_paths"
        tie = "NetworkX Yen heap order. This is the call used by frozen P1 generate_paths.py."
    else:
        succ = _successors(topology)
        shard_folder = _shard_dir(topology) if checkpoint else None
        if shard_folder is not None:
            shard_folder.mkdir(parents=True, exist_ok=True)
        parts: list[tuple | None] = [None] * topology.num_nodes
        pending = []
        for src in range(topology.num_nodes):
            loaded = _load_shard(shard_folder, src) if shard_folder is not None else None
            if loaded is not None:
                parts[src] = loaded
            else:
                pending.append(src)
        print(
            f"  {topology.name}: {topology.num_nodes - len(pending)} shards reused, "
            f"{len(pending)} sources remaining",
            flush=True,
        )
        if pending:
            if jobs <= 1:
                for src in pending:
                    part = _yen_source((src, succ, topology.num_nodes, K))
                    parts[src] = part
                    if shard_folder is not None:
                        _save_shard(shard_folder, src, part)
                    if src % 50 == 0 or src + 1 == topology.num_nodes:
                        done = sum(1 for item in parts if item is not None)
                        print(f"  {topology.name}: sources {done}/{topology.num_nodes}", flush=True)
            else:
                pool = ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(succ,))
                try:
                    futures = {
                        pool.submit(_yen_source_job, (src, topology.num_nodes, K)): src for src in pending
                    }
                    finished = 0
                    for future in as_completed(futures):
                        src = futures[future]
                        part = future.result()
                        parts[src] = part
                        if shard_folder is not None:
                            _save_shard(shard_folder, src, part)
                        finished += 1
                        done = sum(1 for item in parts if item is not None)
                        if finished % 50 == 0 or done == topology.num_nodes:
                            print(f"  {topology.name}: sources {done}/{topology.num_nodes}", flush=True)
                finally:
                    pool.shutdown(wait=True)
        if any(part is None for part in parts):
            raise RuntimeError(f"{topology.name}: path shards are incomplete")
        arrays = _merge_source_results(topology.num_nodes, parts)  # type: ignore[arg-type]
        backend = "yen_bfs_unit_weight"
        tie = (
            "Equal-length paths follow adjacency order in the processed edge list. "
            "This can differ from NetworkX when several paths have the same hop count. "
            "The cost is still hop count, which is P1's min-hop weight."
        )
    fewer = int(np.count_nonzero(arrays["n_paths"] < K))
    missing = int(np.count_nonzero(arrays["n_paths"] == 0))
    return {
        **arrays,
        "meta": {
            "topology": topology.name,
            "k_requested": K,
            "padded": False,
            "backend": backend,
            "cost": "min-hop; positive-capacity edge weight 1 (P1 generate_paths._weighted_copy)",
            "tie_break": tie,
            "difference_from_p1": (
                "P1 preprocessing/generate_paths.py _pad_paths repeats the shortest "
                "path when fewer than 4 simple paths exist. Paths are not repeated here."
            ),
            "demands": int(arrays["n_paths"].size),
            "paths_generated": int(arrays["hop_count"].size),
            "demands_with_fewer_than_4": fewer,
            "demands_with_no_path": missing,
        },
    }


def path_list(arrays: dict, index: int) -> list[list[int]]:
    """Materialise the paths of demand ``index``. For tests and small graphs."""
    counts = arrays["n_paths"]
    start = int(np.sum(counts[:index]))
    stop = start + int(counts[index])
    ptr = arrays["path_node_ptr"]
    nodes = arrays["path_nodes"]
    paths = []
    for path_i in range(start, stop):
        paths.append(nodes[int(ptr[path_i]) : int(ptr[path_i + 1])].astype(int).tolist())
    return paths


def as_dict(arrays: dict) -> dict[tuple[int, int], list[list[int]]]:
    if arrays["n_paths"].size > 20_000:
        raise ValueError("Refusing to expand a large path set into Python lists")
    out = {}
    for index, (src, dst) in enumerate(zip(arrays["demand_src"], arrays["demand_dst"])):
        out[(int(src), int(dst))] = path_list(arrays, index)
    return out


def validate(topology: Topology, arrays: dict, sample_limit: int | None = None) -> list[str]:
    """Check endpoints, real links, simple paths, and that path 0 is shortest."""
    problems = []
    succ = _successors(topology)
    edge_ok = {(u, v) for u, v in topology.edges}
    counts = arrays["n_paths"]
    ptr = arrays["path_node_ptr"]
    nodes = arrays["path_nodes"]
    path_cursor = 0
    # First-path hop count must equal BFS distance. Checked for every source.
    offset = 0
    for src in range(topology.num_nodes):
        dist = _distances(succ, src)
        row = counts[offset : offset + topology.num_nodes - 1]
        for local, dst in enumerate(d for d in range(topology.num_nodes) if d != src):
            n_found = int(row[local])
            if dist[dst] < 0 and n_found != 0:
                problems.append(f"{src}->{dst} has {n_found} paths but BFS finds none")
            if dist[dst] >= 0 and n_found == 0:
                problems.append(f"{src}->{dst} is reachable in {dist[dst]} hops but has no path")
            if n_found and dist[dst] >= 0:
                first_hops = int(arrays["hop_count"][path_cursor])
                if first_hops != dist[dst]:
                    problems.append(f"{src}->{dst} first path has {first_hops} hops, BFS distance {dist[dst]}")
            path_cursor += n_found
        offset += topology.num_nodes - 1
    limit = arrays["hop_count"].size if sample_limit is None else min(sample_limit, arrays["hop_count"].size)
    for path_i in range(limit):
        seq = nodes[int(ptr[path_i]) : int(ptr[path_i + 1])]
        if seq.size == 0 or int(seq[0]) == int(seq[-1]) and seq.size == 1:
            problems.append(f"path {path_i} is empty or a self walk")
            continue
        if len(set(seq.tolist())) != seq.size:
            problems.append(f"path {path_i} repeats a node")
        for a, b in zip(seq[:-1].tolist(), seq[1:].tolist()):
            if (int(a), int(b)) not in edge_ok:
                problems.append(f"path {path_i} uses missing edge {a}->{b}")
                break
        if len(problems) > 20:
            break
    return problems


def _distances(succ, src: int) -> np.ndarray:
    dist = np.full(len(succ), -1, dtype=np.int32)
    dist[src] = 0
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for nxt in succ[node]:
            if dist[nxt] >= 0:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist

"""Collect and organise Phase 3 datasets.

Reads the TEAL, FIGRET, and Topology Zoo dumps already placed under
``data/`` and writes the PHASE3_NDT layout:

    data/raw/{teal,figret,topology_zoo}
    data/processed/{topologies,traffic_matrices,candidate_paths,graph_data}
    data/scenarios/{link_failures,twin_errors}

Real traffic is stored separately from toy placeholders and from the
reconstructed Cogentco gravity series. Missing Microsoft WAN traces are
not fabricated.

The frozen P1 tree (baseline_p1_reproduction) is not read or written.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
import time
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np

from repo_paths import IMPL, REPO as ROOT
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SCENARIOS = DATA / "scenarios"

TOPO_DIR = PROCESSED / "topologies"
TM_DIR = PROCESSED / "traffic_matrices"
PATH_DIR = PROCESSED / "candidate_paths"
GRAPH_DIR = PROCESSED / "graph_data"
FAIL_DIR = SCENARIOS / "link_failures"
TWIN_DIR = SCENARIOS / "twin_errors"

# P1 Table 5, as used by the frozen baseline audit. Edges are directed counts.
TABLE5 = {
    "B4": {"nodes": 12, "directed_edges": 38, "avg_sp": 2.3, "diameter": 5},
    "GEANT": {"nodes": 22, "directed_edges": 72, "avg_sp": 2.53, "diameter": 5},
    "UsCarrier": {"nodes": 158, "directed_edges": 378, "avg_sp": 12.1, "diameter": 35},
    "Cogentco": {"nodes": 197, "directed_edges": 486, "avg_sp": 10.51, "diameter": 28},
    "Kdl": {"nodes": 754, "directed_edges": 1790, "avg_sp": 22.7, "diameter": 58},
    "ASN": {"nodes": 1739, "directed_edges": 8558, "avg_sp": 3.2, "diameter": 8},
}

# Same seeds as the frozen B4 protocol card. Used only to draw scenarios.
SEEDS = list(range(100, 1100, 100))
TWIN_SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.40]
TWIN_NOISE_SEED = {
    "B4": 301,
    "GEANT": 302,
    "UsCarrier": 303,
    "Cogentco": 304,
    "Kdl": 305,
    "ASN": 306,
}
PATH_TOPOLOGIES = ("B4", "GEANT", "UsCarrier", "Cogentco")
# Kdl (~0.57M pairs) and ASN (~3.0M pairs) are deferred: Yen-4 does not
# finish in a data-collection run, and P1 already recorded that scale.
DEFERRED_PATHS = ("Kdl", "ASN")

# Frozen P1 protocol split for the 36-matrix TEAL-style series.
# P1 does not publish this cut. It is recorded, not treated as author data.
TEAL_INDEX_SPLIT = {"train": [0, 20], "val": [20, 28], "test": [28, 36]}

COGENTCO_PLACEHOLDER_CAPACITY = 10000.0
GRAVITY_SEED = 2025
GRAVITY_NOISE_SIGMA = 0.15
GRAVITY_N = 36


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_layout() -> None:
    for path in (
        RAW / "teal",
        RAW / "figret",
        RAW / "topology_zoo",
        TOPO_DIR,
        TM_DIR / "real",
        TM_DIR / "toy",
        TM_DIR / "reconstructed",
        PATH_DIR,
        GRAPH_DIR,
        FAIL_DIR,
        TWIN_DIR,
        IMPL / "digital_twin",
        IMPL / "safety_gate",
        IMPL / "policy",
        IMPL / "baselines",
        IMPL / "experiments",
        IMPL / "evaluation",
        ROOT / "results" / "logs",
        ROOT / "results" / "graphs",
        ROOT / "results" / "tables",
    ):
        path.mkdir(parents=True, exist_ok=True)


def organise_raw() -> dict:
    """Move the dropped corpora into data/raw. Idempotent."""
    moves = {
        DATA / "teal-main": RAW / "teal",
        DATA / "figret-main": RAW / "figret",
        DATA / "InternetTopologyZoo-main": RAW / "topology_zoo",
    }
    done = {}
    for src, dst in moves.items():
        if _raw_ready(dst):
            done[dst.name] = {"path": str(dst), "action": "already_organised"}
            continue
        if not src.exists():
            raise FileNotFoundError(f"Missing raw corpus: {src} (and {dst} is not organised)")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        done[dst.name] = {"path": str(dst), "action": "moved"}
    return done


def _raw_ready(dst: Path) -> bool:
    if dst.name == "teal":
        return (dst / "topologies" / "B4.json").exists()
    if dst.name == "figret":
        return (dst / "Data" / "GEANT" / "GEANT.json").exists()
    if dst.name == "topology_zoo":
        return (dst / "gml" / "Cogentco.gml").exists()
    return False


def load_node_link(path: Path) -> tuple[nx.DiGraph, list[tuple[int, int]], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(int(node["id"]))
    edges = []
    capacities = []
    for link in data["links"]:
        u, v = int(link["source"]), int(link["target"])
        cap = float(link["capacity"])
        graph.add_edge(u, v, capacity=cap)
        edges.append((u, v))
        capacities.append(cap)
    return graph, edges, np.asarray(capacities, dtype=np.float64)


def undirected_metrics(graph: nx.DiGraph) -> dict:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    undirected.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    n = undirected.number_of_nodes()
    components = nx.number_connected_components(undirected) if n else 0
    out = {
        "undirected_edges": undirected.number_of_edges(),
        "undirected_connected": components == 1,
        "undirected_components": components,
        "avg_shortest_path_undirected": None,
        "diameter_undirected": None,
    }
    if components != 1 or n < 2:
        return out
    total = 0.0
    diameter = 0
    for source in undirected.nodes():
        dist = nx.single_source_shortest_path_length(undirected, source)
        total += sum(dist.values())
        diameter = max(diameter, max(dist.values()))
    out["avg_shortest_path_undirected"] = total / (n * (n - 1))
    out["diameter_undirected"] = diameter
    return out


def directed_reachability(graph: nx.DiGraph) -> dict:
    n = graph.number_of_nodes()
    missing = 0
    for source in graph.nodes():
        seen = nx.descendants(graph, source)
        missing += (n - 1) - len(seen)
    return {
        "strongly_connected": nx.is_strongly_connected(graph) if n else False,
        "ordered_pairs_without_a_path": missing,
    }


def topology_record(name: str, graph: nx.DiGraph, edges, capacities, source: str) -> dict:
    metrics = undirected_metrics(graph)
    reach = directed_reachability(graph)
    expected = TABLE5[name]
    size_match = (
        graph.number_of_nodes() == expected["nodes"]
        and len(edges) == expected["directed_edges"]
    )
    return {
        "name": name,
        "source": source,
        "nodes": graph.number_of_nodes(),
        "directed_edges": len(edges),
        "capacity_min": float(capacities.min()) if len(capacities) else None,
        "capacity_max": float(capacities.max()) if len(capacities) else None,
        "capacity_sum": float(capacities.sum()) if len(capacities) else None,
        "table5": expected,
        "size_matches_table5": size_match,
        **metrics,
        **reach,
    }


def copy_teal_topology(src_name: str, dest_name: str) -> dict:
    src = RAW / "teal" / "topologies" / f"{src_name}.json"
    dest = TOPO_DIR / f"{dest_name}.json"
    shutil.copy2(src, dest)
    graph, edges, capacities = load_node_link(dest)
    rec = topology_record(dest_name, graph, edges, capacities, str(src))
    rec["sha256"] = sha256(dest)
    rec["capacity_note"] = "Capacities copied from the TEAL topology file."
    (GRAPH_DIR / f"{dest_name}_stats.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    _save_graph_arrays(dest_name, edges, capacities)
    return rec


def copy_geant() -> dict:
    src = RAW / "figret" / "Data" / "GEANT" / "GEANT.json"
    dest = TOPO_DIR / "GEANT.json"
    shutil.copy2(src, dest)
    graph, edges, capacities = load_node_link(dest)
    rec = topology_record("GEANT", graph, edges, capacities, str(src))
    rec["sha256"] = sha256(dest)
    rec["capacity_unit"] = (
        "Values in the FIGRET file (10e9, 2.4e9, 155e6). "
        "FIGRET's normalize_size divides capacities and demands by 1e9 at load time. "
        "This collection stores the published numbers unscaled."
    )
    rec["rejected_alternative"] = (
        "Topology Zoo Geant / Geant2001 / Geant2009 / Geant2010 / Geant2011 / Geant2012 "
        "are a different network family. They are kept under data/raw/topology_zoo and "
        "are not copied into processed topologies."
    )
    (GRAPH_DIR / "GEANT_stats.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    _save_graph_arrays("GEANT", edges, capacities)
    return rec


def _save_graph_arrays(name: str, edges, capacities: np.ndarray) -> None:
    src = np.asarray([u for u, _ in edges], dtype=np.int32)
    dst = np.asarray([v for _, v in edges], dtype=np.int32)
    np.savez_compressed(
        GRAPH_DIR / f"{name}_graph.npz",
        edge_src=src,
        edge_dst=dst,
        capacity=capacities.astype(np.float64),
    )


def parse_zoo_gml(path: Path) -> nx.Graph:
    """Topology Zoo GML reader. NetworkX rejects some of these files."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    graph = nx.Graph()
    for block in text.split("node ["):
        node_id = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("id "):
                node_id = int(line.split()[1])
                break
        if node_id is not None:
            graph.add_node(node_id)
    for block in text.split("edge ["):
        src = tgt = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("source "):
                src = int(line.split()[1])
            elif line.startswith("target "):
                tgt = int(line.split()[1])
        if src is not None and tgt is not None and src != tgt and not graph.has_edge(src, tgt):
            graph.add_edge(src, tgt)
    return graph


def build_cogentco() -> dict:
    src = RAW / "topology_zoo" / "gml" / "Cogentco.gml"
    raw = parse_zoo_gml(src)
    numeric_ids = sorted(raw.nodes())
    string_ids = sorted(raw.nodes(), key=str)
    if numeric_ids != list(range(len(numeric_ids))):
        mapping = {node: i for i, node in enumerate(numeric_ids)}
        order_note = "Zoo ids are not 0..n-1; remapped in increasing id order."
    else:
        mapping = {node: node for node in numeric_ids}
        order_note = "Zoo node ids are already 0..n-1 and are kept as matrix indices."
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(mapping)))
    edges = []
    capacities = []
    for u, v in raw.edges():
        iu, iv = mapping[u], mapping[v]
        if iu == iv:
            continue
        for a, b in ((iu, iv), (iv, iu)):
            if graph.has_edge(a, b):
                continue
            graph.add_edge(a, b, capacity=COGENTCO_PLACEHOLDER_CAPACITY)
            edges.append((a, b))
            capacities.append(COGENTCO_PLACEHOLDER_CAPACITY)
    cap_arr = np.asarray(capacities, dtype=np.float64)
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "source": "Internet Topology Zoo Cogentco.gml",
            "capacity_source": "placeholder_uniform_10000",
            "capacity_note": (
                "Cogentco.gml and Cogentco.graphml contain no link speed. "
                "Every directed edge is assigned 10000, the same placeholder "
                "the frozen P1 loader used when LinkSpeedRaw was absent. "
                "These are not measured Cogentco capacities."
            ),
        },
        "nodes": [{"id": i} for i in range(graph.number_of_nodes())],
        "links": [
            {"source": u, "target": v, "capacity": COGENTCO_PLACEHOLDER_CAPACITY}
            for u, v in edges
        ],
    }
    dest = TOPO_DIR / "Cogentco.json"
    dest.write_text(json.dumps(payload), encoding="utf-8")
    rec = topology_record("Cogentco", graph, edges, cap_arr, str(src))
    rec["sha256"] = sha256(dest)
    rec["node_order"] = order_note
    rec["string_sort_matches_numeric"] = numeric_ids == string_ids
    rec["capacity_source"] = "placeholder_uniform_10000_not_measured"
    rec["undirected_zoo_edges"] = raw.number_of_edges()
    (GRAPH_DIR / "Cogentco_stats.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    _save_graph_arrays("Cogentco", edges, cap_arr)
    return rec


def zoo_geant_inventory() -> list[dict]:
    """Record Topology Zoo GÉANT files so they are not mistaken for FIGRET."""
    rows = []
    gml_dir = RAW / "topology_zoo" / "gml"
    for path in sorted(gml_dir.glob("Geant*.gml")):
        raw = parse_zoo_gml(path)
        rows.append(
            {
                "file": path.name,
                "nodes": raw.number_of_nodes(),
                "undirected_edges": raw.number_of_edges(),
                "used": False,
                "reason": "Not the FIGRET 22-node, 72-directed-link topology required by P1 Table 5.",
            }
        )
    return rows


def _tm_index(path: Path) -> int:
    """Index embedded in TEAL names ``*_real_{k}_*`` or ``*_toy_{k}_*``."""
    parts = path.name.split("_")
    for key in ("real", "toy"):
        if key in parts:
            return int(parts[parts.index(key) + 1])
    raise ValueError(f"Cannot read TM index from {path.name}")


def collect_b4_traffic() -> dict:
    folder = RAW / "teal" / "traffic-matrices" / "real"
    files = sorted(folder.glob("B4.json_real_*_traffic-matrix.pkl"), key=_tm_index)
    matrices = []
    totals = []
    for path in files:
        matrix = np.asarray(pickle.load(path.open("rb")), dtype=np.float64)
        matrices.append(matrix)
        totals.append(float(matrix.sum()))
    stack = np.stack(matrices, axis=0)
    if stack.shape != (36, 12, 12):
        raise RuntimeError(f"Unexpected B4 TM stack {stack.shape}")
    if not np.allclose(np.diagonal(stack, axis1=1, axis2=2), 0.0):
        raise RuntimeError("B4 real TMs have a non-zero diagonal")
    out = TM_DIR / "real" / "B4.npz"
    np.savez_compressed(out, demand=stack, total_demand=np.asarray(totals))
    test = totals[28:36]
    rec = {
        "name": "B4",
        "kind": "real",
        "label": "TEAL public real matrices, Yates/Abilene demands mapped onto B4",
        "not": "Not Microsoft WAN traffic. Not gravity-synthetic.",
        "n": int(stack.shape[0]),
        "shape": [12, 12],
        "dtype": "float64",
        "diagonal": "already 0 in the TEAL files",
        "mean_total_demand_all": float(np.mean(totals)),
        "mean_total_demand_test_28_36": float(np.mean(test)),
        "p1_cited_test_mean": 27980,
        "index_split": TEAL_INDEX_SPLIT,
        "index_split_status": (
            "Protocol assumption copied from the frozen B4 card "
            "(TEAL-style 20/8/8). Lu et al. do not publish the split."
        ),
        "source_dir": str(folder),
        "file": str(out),
    }
    (TM_DIR / "real" / "B4.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def collect_toy_traffic(topo: str, file_prefix: str) -> dict:
    folder = RAW / "teal" / "traffic-matrices" / "toy"
    files = sorted(folder.glob(f"{file_prefix}.json_toy_*_traffic-matrix.pkl"), key=_tm_index)
    if len(files) != 36:
        raise RuntimeError(f"{topo}: expected 36 toy TMs, found {len(files)}")
    identical = True
    all_ones = True
    shape = None
    dtype = None
    first = None
    for path in files:
        matrix = np.asarray(pickle.load(path.open("rb")))
        shape = list(matrix.shape)
        dtype = str(matrix.dtype)
        if first is None:
            first = matrix
        elif matrix.shape != first.shape or not np.array_equal(matrix, first):
            identical = False
        if not np.all(matrix.astype(np.float64) == 1.0):
            all_ones = False
    n = shape[0]
    rec = {
        "name": topo,
        "kind": "toy",
        "label": "TEAL synthetic toy placeholder",
        "real_ms_wan_traffic": False,
        "reason": (
            "The public TEAL repository states that Microsoft WAN traces cannot be "
            "shared. UsCarrier, Kdl, and ASN ship only toy matrices."
        ),
        "n": 36,
        "shape": shape,
        "source_dtype": dtype,
        "all_entries_equal_1_including_diagonal": all_ones,
        "all_36_files_identical": identical,
        "source_total_if_ones": n * n,
        "processed_demand_rule": (
            "Off-diagonal demand is 1.0 and the diagonal is 0. "
            "No other values were invented. Matrices are not materialised, "
            "because every published file is the same all-ones array."
        ),
        "index_split": TEAL_INDEX_SPLIT,
        "index_split_status": (
            "Index convention only. These matrices are identical, so the cut "
            "is not a time split and cannot support a Table 6 claim."
        ),
        "source_dir": str(folder),
    }
    (TM_DIR / "toy" / f"{topo}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def _read_hist_matrix(line: str, n: int) -> np.ndarray:
    values = np.fromstring(line, sep=" ", dtype=np.float64)
    if values.size != n * n:
        raise ValueError(f"Expected {n*n} entries, found {values.size}")
    return values.reshape(n, n)


def collect_geant_traffic() -> dict:
    root = RAW / "figret" / "Data" / "GEANT"
    n = 22

    def load_split(folder: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        matrices = []
        opts = []
        parts = []
        for hist in sorted(folder.glob("*.hist"), key=lambda p: int(p.stem)):
            rows = []
            with hist.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(_read_hist_matrix(line, n))
            opt_path = hist.with_suffix(".opt")
            opt_vals = []
            with opt_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        opt_vals.append(float(line.strip()))
            if len(opt_vals) != len(rows):
                raise RuntimeError(f"{hist.name}: {len(rows)} matrices but {len(opt_vals)} opt values")
            parts.append({"file": hist.name, "rows": len(rows)})
            matrices.extend(rows)
            opts.extend(opt_vals)
        return np.stack(matrices), np.asarray(opts, dtype=np.float64), parts

    train, train_opt, train_parts = load_split(root / "train")
    test, test_opt, test_parts = load_split(root / "test")
    out = TM_DIR / "real" / "GEANT.npz"
    np.savez_compressed(
        out,
        demand_train=train,
        demand_test=test,
        opt_mlu_train=train_opt,
        opt_mlu_test=test_opt,
    )
    rec = {
        "name": "GEANT",
        "kind": "real",
        "label": "FIGRET published GÉANT demand histories",
        "not": (
            "Not generated by this project. Not the gravity matrices previously "
            "built on Topology Zoo Geant2012."
        ),
        "unit": (
            "Raw file units, same scale as GEANT.json capacities. "
            "FIGRET normalize_size divides by 1e9 (decimal gigabit) when training. "
            "That scale is not applied in the stored arrays."
        ),
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "shape": [22, 22],
        "dtype": "float64",
        "train_files": train_parts,
        "test_files": test_parts,
        "split": (
            "FIGRET's own train/ and test/ directories. No validation cut is "
            "published, so none is invented here. File 5.hist is kept even though "
            "it contains only 16 rows."
        ),
        "mean_total_demand_train": float(train.sum(axis=(1, 2)).mean()),
        "mean_total_demand_test": float(test.sum(axis=(1, 2)).mean()),
        "diagonal_sum_train": float(np.trace(train, axis1=1, axis2=2).sum()),
        "diagonal_sum_test": float(np.trace(test, axis1=1, axis2=2).sum()),
        "p1_cited_mean_total": "43.94 billion (Lu et al. text). Compare with the measured means; do not rescale to force a match.",
        "opt_mlu": "Per-matrix optimal MLU shipped in the sibling .opt files.",
        "source_dir": str(root),
        "file": str(out),
    }
    (TM_DIR / "real" / "GEANT.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def gravity_matrices(graph: nx.DiGraph) -> tuple[np.ndarray, dict]:
    """Degree-gravity reconstruction.

    Matches the formula already documented for the frozen P1 Cogentco series:
    T_ij = A_i A_j for i != j, A_i = max(degree(i), 1), then scale so the mean
    total equals the sum of (placeholder) link capacities, then multiply by
    lognormal noise. P1 does not publish the real attractiveness or scale.
    """
    nodes = list(range(graph.number_of_nodes()))
    attractiveness = np.array([max(graph.degree(u), 1) for u in nodes], dtype=np.float64)
    base = np.outer(attractiveness, attractiveness)
    np.fill_diagonal(base, 0.0)
    target_total = float(sum(data.get("capacity", 0.0) for *_, data in graph.edges(data=True)))
    base *= target_total / max(base.sum(), 1e-9)
    rng = np.random.default_rng(GRAVITY_SEED)
    matrices = []
    for _ in range(GRAVITY_N):
        noise = rng.lognormal(mean=0.0, sigma=GRAVITY_NOISE_SIGMA, size=base.shape)
        matrix = np.maximum(base * noise, 0.0)
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    stack = np.stack(matrices)
    meta = {
        "formula": "T_ij = A_i A_j (i != j), A_i = max(degree(i), 1), degree = in+out",
        "scale": "Each noisy matrix is not re-scaled; the noiseless base sums to the sum of placeholder capacities.",
        "target_total_of_base": target_total,
        "n": GRAVITY_N,
        "seed": GRAVITY_SEED,
        "noise": f"lognormal(0, {GRAVITY_NOISE_SIGMA}) per entry",
        "numpy": np.__version__,
        "status": "reconstructed_synthetic",
        "not_p1_dump": (
            "Lu et al. do not publish Cogentco gravity constants. "
            "This series follows the frozen project's documented reconstruction "
            "so the method is explicit. It is not an author traffic dump."
        ),
    }
    return stack, meta


def collect_cogentco_traffic() -> dict:
    graph, _, _ = load_node_link(TOPO_DIR / "Cogentco.json")
    stack, meta = gravity_matrices(graph)
    out = TM_DIR / "reconstructed" / "Cogentco_gravity.npz"
    np.savez_compressed(out, demand=stack)
    totals = stack.sum(axis=(1, 2))
    rec = {
        "name": "Cogentco",
        "kind": "reconstructed_gravity",
        "real_traffic": False,
        **meta,
        "mean_total_demand": float(totals.mean()),
        "min_total_demand": float(totals.min()),
        "max_total_demand": float(totals.max()),
        "index_split": TEAL_INDEX_SPLIT,
        "index_split_status": (
            "There is no time axis. The 20/8/8 index cut is the same convention "
            "as the 36-matrix TEAL series, not an observed split."
        ),
        "file": str(out),
    }
    (TM_DIR / "reconstructed" / "Cogentco_gravity.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )
    return rec


def _bfs_path(
    successors: dict[int, list[int]],
    src: int,
    dst: int,
    banned_nodes: set[int],
    banned_edges: set[tuple[int, int]],
) -> list[int] | None:
    """Unit-weight shortest path. ``banned_nodes`` must not include ``src``."""
    previous = {src: -1}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for nxt in successors[node]:
            if nxt in previous or nxt in banned_nodes or (node, nxt) in banned_edges:
                continue
            previous[nxt] = node
            queue.append(nxt)
    if dst not in previous:
        return None
    path = []
    cursor = dst
    while cursor != -1:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path


def _yen_k(successors: dict[int, list[int]], src: int, dst: int, k: int) -> list[list[int]]:
    """Yen's algorithm with BFS. Ties follow adjacency order in the topology file."""
    first = _bfs_path(successors, src, dst, set(), set())
    if first is None:
        return []
    accepted = [first]
    pool: list[list[int]] = []
    pool_keys: set[tuple[int, ...]] = set()
    for _ in range(1, k):
        previous = accepted[-1]
        for spur_at in range(len(previous) - 1):
            root = previous[: spur_at + 1]
            banned_edges = set()
            for path in accepted:
                if len(path) > spur_at + 1 and path[: spur_at + 1] == root:
                    banned_edges.add((path[spur_at], path[spur_at + 1]))
            spur = _bfs_path(
                successors,
                root[-1],
                dst,
                set(root[:-1]),
                banned_edges,
            )
            if spur is None:
                continue
            candidate = root[:-1] + spur
            key = tuple(candidate)
            if key in pool_keys or any(candidate == have for have in accepted):
                continue
            pool_keys.add(key)
            pool.append(candidate)
        if not pool:
            break
        pool.sort(key=len)
        accepted.append(pool.pop(0))
        pool_keys.discard(tuple(accepted[-1]))
    return accepted


def generate_paths(name: str) -> dict:
    graph, _, _ = load_node_link(TOPO_DIR / f"{name}.json")
    successors = {node: list(graph.successors(node)) for node in graph.nodes()}
    path_dict: dict[tuple[int, int], list[list[int]]] = {}
    padded = 0
    missing = 0
    started = time.perf_counter()
    nodes = list(graph.nodes())
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            found = _yen_k(successors, src, dst, 4)
            if not found:
                missing += 1
                continue
            distance = len(found[0]) - 1
            lengths = []
            for path in found:
                if len(path) != len(set(path)):
                    raise RuntimeError(f"{name} {src}->{dst} repeats a node: {path}")
                for left, right in zip(path, path[1:]):
                    if right not in successors[left]:
                        raise RuntimeError(f"{name} {src}->{dst} uses a missing edge {left}->{right}")
                lengths.append(len(path) - 1)
            if lengths != sorted(lengths):
                raise RuntimeError(f"{name} {src}->{dst} hop lengths are not shortest-first: {lengths}")
            if name in {"B4", "GEANT"} and lengths[0] != nx.shortest_path_length(graph, src, dst):
                raise RuntimeError(f"{name} {src}->{dst} first path is not a shortest path")
            if len(found) < 4:
                padded += 1
                found = found + [found[0]] * (4 - len(found))
            path_dict[(src, dst)] = found[:4]
    elapsed = time.perf_counter() - started
    dest = PATH_DIR / f"{name}_yen4_minhop.pkl"
    payload = {
        "topology": name,
        "algorithm": "yen_bfs_unit_weight",
        "tie_break": "Equal-length paths follow the adjacency order of the processed topology.",
        "k": 4,
        "metric": "min-hop",
        "edge_disjoint": False,
        "padding": "If fewer than 4 simple paths exist, the shortest is repeated.",
        "paths": path_dict,
    }
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    hops = [len(paths[0]) - 1 for paths in path_dict.values()]
    rec = {
        "topology": name,
        "file": str(dest),
        "pairs": len(path_dict),
        "pairs_missing": missing,
        "pairs_padded_to_4": padded,
        "mean_hops_shortest": float(np.mean(hops)) if hops else None,
        "seconds": round(elapsed, 3),
    }
    (PATH_DIR / f"{name}_yen4_minhop.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"  paths {name}: {rec['pairs']} pairs in {rec['seconds']}s, padded {padded}, missing {missing}", flush=True)
    return rec


def copy_figret_tunnels() -> dict:
    src = RAW / "figret" / "Data" / "GEANT" / "tunnels.txt"
    dest = PATH_DIR / "GEANT_figret_tunnels.txt"
    shutil.copy2(src, dest)
    n_pairs = 0
    n_paths = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            n_pairs += 1
            _, spec = line.split(":", 1)
            n_paths.append(len([p for p in spec.split(",") if p]))
    rec = {
        "topology": "GEANT",
        "kind": "figret_published_tunnels",
        "file": str(dest),
        "pairs": n_pairs,
        "paths_per_pair_min": int(min(n_paths)) if n_paths else 0,
        "paths_per_pair_max": int(max(n_paths)) if n_paths else 0,
        "note": (
            "FIGRET ships these tunnels (usually 3 paths). "
            "They are kept beside the Yen-4 paths and are not rewritten into 4 paths."
        ),
    }
    (PATH_DIR / "GEANT_figret_tunnels.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def link_failure_cases(name: str) -> dict:
    _, edges, _ = load_node_link(TOPO_DIR / f"{name}.json")
    cases = []
    for seed in SEEDS:
        rng = random.Random(seed)
        for n_fail in (0, 1, 2, 3):
            chosen = [] if n_fail == 0 else rng.sample(range(len(edges)), n_fail)
            cases.append(
                {
                    "seed": seed,
                    "n_failed": n_fail,
                    "edge_indices": chosen,
                    "edges": [list(edges[i]) for i in chosen],
                }
            )
    rec = {
        "topology": name,
        "protocol": (
            "Directed-edge failures, matching the frozen P1 Section 5.5 sampler: "
            "random.Random(seed) then, for n_failed in 0,1,2,3, sample n_failed "
            "edge indices without replacement. n_failed=0 draws nothing. "
            "Later draws in the same seed advance the RNG, so the 1-edge set is "
            "not a subset of the 2-edge set. A failed edge has its capacity set "
            "to 0; the opposite direction is removed only when it is also drawn."
        ),
        "num_edges": len(edges),
        "edge_order": "Order of links in data/processed/topologies/{name}.json",
        "seeds": SEEDS,
        "cases": cases,
    }
    (FAIL_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return {"topology": name, "cases": len(cases), "edges": len(edges)}


def twin_error_cases(name: str) -> dict:
    _, edges, _ = load_node_link(TOPO_DIR / f"{name}.json")
    rng = np.random.default_rng(TWIN_NOISE_SEED[name])
    noise = rng.standard_normal(size=(len(SEEDS), len(TWIN_SIGMAS), len(edges))).astype(np.float32)
    dest = TWIN_DIR / f"{name}_noise.npz"
    np.savez_compressed(dest, z=noise, seeds=np.asarray(SEEDS, dtype=np.int32), sigma=np.asarray(TWIN_SIGMAS))
    return {
        "topology": name,
        "file": str(dest),
        "shape": list(noise.shape),
        "generator_seed": TWIN_NOISE_SEED[name],
    }


def write_twin_spec(per_topology: list[dict]) -> None:
    spec = {
        "purpose": (
            "Synthetic twin-error scenarios for Phase 2 experiment E4. "
            "These are perturbations of a predicted link load. They are not "
            "traffic matrices and they do not fill in missing Microsoft WAN traces."
        ),
        "formula": "load_hat[e] = max(0, load_true[e] * (1 + sigma * z[seed_index, level_index, e]))",
        "sigma": TWIN_SIGMAS,
        "seeds": SEEDS,
        "seed_axis": "Axis 0 of z follows `seeds` in this order, not the numeric seed value.",
        "z": "Standard normal, float32, stored per topology so the draw does not depend on regenerating NumPy.",
        "generator_seeds": TWIN_NOISE_SEED,
        "numpy": np.__version__,
        "topologies": per_topology,
    }
    (TWIN_DIR / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def write_catalog(report: dict) -> None:
    topo_rows = []
    for rec in report["topologies"]:
        avg = rec["avg_shortest_path_undirected"]
        avg_s = "—" if avg is None else f"{avg:.3f}"
        dia = rec["diameter_undirected"]
        dia_s = "—" if dia is None else str(dia)
        flag = "yes" if rec["size_matches_table5"] else "NO"
        topo_rows.append(
            f"| {rec['name']} | {rec['nodes']} | {rec['directed_edges']} | {avg_s} | {dia_s} | {flag} |"
        )
    traffic_rows = []
    for rec in report["traffic"]:
        traffic_rows.append(
            f"| {rec['name']} | {rec['kind']} | {rec.get('label', rec.get('formula', ''))} |"
        )
    zoo_rows = [
        f"| {row['file']} | {row['nodes']} | {row['undirected_edges']} | not used |"
        for row in report["zoo_geant_rejected"]
    ]
    path_rows = [
        f"| {rec['topology']} | {rec['pairs']} | {rec['pairs_padded_to_4']} | {rec['pairs_missing']} | {rec['seconds']} |"
        for rec in report["paths"]
    ]
    text = f"""# Phase 3 dataset catalog

This file records what was collected for the Phase 3 digital-twin study.
The directory `implementation/` holds the Phase 3 code. The frozen P1
implementation under `baseline_p1_reproduction` was not modified.

Real demand, TEAL toy placeholders, and the reconstructed Cogentco gravity
series live in three different folders. Nothing in `traffic_matrices/toy` or
`traffic_matrices/reconstructed` is measured operator traffic.

## Layout

```
data/raw/teal              TEAL repository (topologies, real B4 TMs, toy TMs)
data/raw/figret            FIGRET repository (GÉANT graph, histories, tunnels)
data/raw/topology_zoo      Internet Topology Zoo (Cogentco is the one we use)
data/processed/topologies
data/processed/traffic_matrices/real
data/processed/traffic_matrices/toy
data/processed/traffic_matrices/reconstructed
data/processed/candidate_paths
data/processed/graph_data
data/scenarios/link_failures
data/scenarios/twin_errors
```

## Topologies

Undirected average shortest-path length and diameter are computed on the
simple undirected skeleton. P1 Table 5 prints one decimal place; the flag
below is the node and directed-edge check.

| Topology | Nodes | Directed edges | Avg SP | Diameter | Table 5 size |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(topo_rows)}

GÉANT is the FIGRET graph: 22 nodes and 72 directed links. That is the
P1 Table 5 topology. The earlier 40-node Zoo Geant2012 graph is not used.

### Topology Zoo GÉANT files left in raw only

| File | Nodes | Undirected edges | Decision |
| --- | ---: | ---: | --- |
{chr(10).join(zoo_rows)}

Cogentco's Zoo file has no link speed. Processed capacities are the uniform
placeholder 10000 on each directed edge. That placeholder is the same number
the frozen loader used when a speed field was absent. It is not a measured
capacity.

## Traffic

| Topology | Kind | What it is |
| --- | --- | --- |
{chr(10).join(traffic_rows)}

### Real

- **B4.** 36 TEAL `real_*_1.0` matrices, 12×12, diagonal already 0. TEAL documents these as Yates/Abilene demands mapped onto B4. Mean total demand over all 36 matrices is {report['b4']['mean_total_demand_all']:.2f}. Mean over the protocol test slice (indices 28:36) is {report['b4']['mean_total_demand_test_28_36']:.2f}, next to the 27980 figure cited for B4. The 20/8/8 cut is the frozen protocol assumption. Lu et al. do not publish it.
- **GÉANT.** FIGRET `train/` and `test/` histories, stored unscaled (float64). Train rows: {report['geant_tm']['train_rows']}. Test rows: {report['geant_tm']['test_rows']}. Mean total demand, train {report['geant_tm']['mean_total_demand_train']:.3e}, test {report['geant_tm']['mean_total_demand_test']:.3e}. P1's text cites about 43.94 billion for GÉANT; the stored matrices are not rescaled to force that number. FIGRET divides capacities and demands by 1e9 inside `normalize_size`. File `5.hist` has 16 rows and is kept. No validation split is added.

### Toy (not Microsoft WAN traffic)

UsCarrier, Kdl, and ASN have no public real TEAL traffic. Each of the 36 files per topology is an all-ones matrix, including the diagonal (float16). The processed rule, written in the sidecar JSON and not expanded into another matrix, is off-diagonal 1 and diagonal 0. These series cannot be used as P1 Table 6 traffic.

### Reconstructed (not a measured Cogentco trace)

Cogentco demands are a degree-gravity series: `T_ij = A_i A_j` with `A_i = max(degree(i), 1)`, base scaled to the sum of the placeholder capacities, then lognormal noise with seed {GRAVITY_SEED} and sigma {GRAVITY_NOISE_SIGMA}. Thirty-six matrices. P1 does not publish the gravity constants, so this is an explicit reconstruction, not the authors' dump. The 20/8/8 index cut has no time meaning here.

## Candidate paths

Yen k-shortest paths, k=4, hop length, not edge-disjoint. If a pair has fewer than four simple paths, the shortest is repeated and the pair is counted as padded.

| Topology | Pairs | Padded | Missing | Seconds |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(path_rows)}

Kdl and ASN path sets are not generated in this step (about 0.57 million and 3.0 million ordered pairs). `candidate_paths/Kdl_ASN_DEFERRED.json` records that. FIGRET's own GÉANT tunnels (usually three paths) are copied to `GEANT_figret_tunnels.txt` and are not rewritten as four paths.

## Generated scenarios

Link failures: for every topology and for seeds 100, 200, …, 1000, the cases with 0, 1, 2, and 3 failed directed edges are stored in `scenarios/link_failures`. Sampling follows the frozen Section 5.5 loop (`random.Random(seed)`, then draws of size 0, 1, 2, 3). Edge indices refer to the link order in the processed topology JSON.

Twin error: `scenarios/twin_errors/spec.json`. The stored tensor `z` is standard normal with shape `(n_seeds, n_levels, n_edges)`. Levels are sigma = 0, 0.05, 0.10, 0.20, 0.40. The evaluation formula is `load_hat = max(0, load_true * (1 + sigma * z))`. This perturbs a predicted link load. It is not a traffic matrix.

## Not decided here

Phase 2 trains on some topologies and holds others out, but it does not name the held-out set. This collection does not assign one.

## What was not invented

- No Microsoft WAN matrices for UsCarrier, Kdl, or ASN.
- No second GÉANT traffic series on a Zoo graph.
- No measured Cogentco link speeds and no author gravity constants.
- No validation split for the FIGRET histories.
"""
    (DATA / "DATASET_CATALOG.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("Organising raw corpora", flush=True)
    ensure_layout()
    raw_info = organise_raw()
    print("Topologies", flush=True)
    topologies = [
        copy_teal_topology("B4", "B4"),
        copy_teal_topology("UsCarrier", "UsCarrier"),
        copy_teal_topology("Kdl", "Kdl"),
        copy_teal_topology("ASN2k", "ASN"),
        copy_geant(),
        build_cogentco(),
    ]
    for rec in topologies:
        print(
            f"  {rec['name']}: {rec['nodes']} nodes, {rec['directed_edges']} directed edges, "
            f"table5={rec['size_matches_table5']}, avgSP={rec['avg_shortest_path_undirected']}, "
            f"D={rec['diameter_undirected']}",
            flush=True,
        )
    print("Traffic", flush=True)
    b4 = collect_b4_traffic()
    geant_tm = collect_geant_traffic()
    toys = [
        collect_toy_traffic("UsCarrier", "UsCarrier"),
        collect_toy_traffic("Kdl", "Kdl"),
        collect_toy_traffic("ASN", "ASN2k"),
    ]
    cogent = collect_cogentco_traffic()
    print(
        f"  B4 test mean {b4['mean_total_demand_test_28_36']:.2f}; "
        f"GEANT train/test rows {geant_tm['train_rows']}/{geant_tm['test_rows']}",
        flush=True,
    )
    print("Scenarios", flush=True)
    failures = [link_failure_cases(name) for name in TABLE5]
    twin = [twin_error_cases(name) for name in TABLE5]
    write_twin_spec(twin)
    print("Paths", flush=True)
    tunnels = copy_figret_tunnels()
    paths = [generate_paths(name) for name in PATH_TOPOLOGIES]
    deferred = {
        "topologies": list(DEFERRED_PATHS),
        "reason": (
            "Yen-4 over every ordered pair is about 0.57 million pairs on Kdl "
            "and 3.0 million pairs on ASN. Path generation is a later preprocessing "
            "job, not part of publishing the collected matrices. Reachability is "
            "already recorded in graph_data/*_stats.json."
        ),
        "algorithm_when_run": "Same Yen-4 min-hop generator as preprocessing/collect_datasets.py generate_paths",
    }
    (PATH_DIR / "Kdl_ASN_DEFERRED.json").write_text(json.dumps(deferred, indent=2), encoding="utf-8")

    report = {
        "raw": raw_info,
        "topologies": topologies,
        "zoo_geant_rejected": zoo_geant_inventory(),
        "traffic": [b4, geant_tm, *toys, cogent],
        "b4": b4,
        "geant_tm": geant_tm,
        "paths": paths,
        "figret_tunnels": tunnels,
        "failures": failures,
        "twin": twin,
        "p1_tree": "not modified",
    }
    (DATA / "collection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_catalog(report)
    print("Wrote", DATA / "DATASET_CATALOG.md", flush=True)


if __name__ == "__main__":
    main()

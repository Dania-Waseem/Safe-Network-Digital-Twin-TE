"""SNDlib Abilene and Germany50: quality checks, sampling, overlap check, preprocessing, calibration.

The two topologies are additional unseen test topologies with real (measured) traffic. They go
through the GEANT pipeline unchanged:

* topology validation (``preprocessing.topology.from_node_link``), every undirected SNDlib link
  stored as two directed links of equal capacity;
* Yen k = 4 hop-count candidate paths (``preprocessing.paths.generate``; NetworkX backend, the
  call used for GEANT and by frozen P1), no padding;
* normalisation by the maximum link capacity (``preprocessing.normalize``);
* link-path bipartite graph (``preprocessing.bipartite``);
* calibrated load levels: alpha(t) = t / LP_MLU(D) for t in {0.5, 0.7, 0.9, 1.1}, with the k = 4
  min-MLU path LP (``phase3_final.routing.lp_min_mlu``, SciPy HiGHS), as for GEANT.

Raw files under ``data/raw/sndlib`` are read only. Outputs:

* ``data/processed/{topologies,graph_data,normalization,candidate_paths}/<name>*``
* ``data/processed/splits/test/<name>.json`` and ``<name>_demand.npz`` (the 200 sampled matrices)
* ``data/processed/sndlib/<name>_sample.json`` (selected indices and timestamps, E3 subset)
* ``data/processed/sndlib/<name>_quality.json``, ``<name>_total_demand.csv``, ``abilene_b4_overlap.json``
* ``data/processed/calibrated_load/sndlib_calibration_rows.jsonl``, ``sndlib_scaled_tm_index.json``
* ``data/scenarios/link_failures/<name>.json`` (same P1 Section 5.5 sampler as the other topologies)
* ``data/raw/sndlib/QUALITY_REPORT.md``

Usage (from ``implementation/``):  python -m preprocessing.sndlib_prepare
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from repo_paths import REPO

from . import normalize
from .collect_datasets import link_failure_cases
from .paths import generate, validate
from .preprocess import GRAPH_DIR, NORM_DIR, PATH_DIR, TOPO_DIR, _save_bipartite, _save_paths, _save_topology
from .sndlib import parse_network, read_demand_archive
from .topology import Topology, connectivity_report, from_node_link
from .traffic import dump_json, summarise_checks, validate_matrix

RAW = REPO / "data" / "raw" / "sndlib"
PROCESSED = REPO / "data" / "processed"
SND_DIR = PROCESSED / "sndlib"
SPLIT_TEST = PROCESSED / "splits" / "test"
CAL_DIR = PROCESSED / "calibrated_load"

SAMPLE_SIZE = 200
SAMPLE_SEED = 2026          # same seed as the GEANT test subsample
E3_SIZE = 50
E3_SEED = 2027              # same seed as the GEANT E3 subset
TARGETS = (0.5, 0.7, 0.9, 1.1)
SPIKE_RATIO = 100.0         # entry / pair median
SPIKE_NEIGHBOUR_RATIO = 10.0
SPIKE_MATERIAL_SHARE = 0.01
TOTAL_RATIO = 100.0         # matrix total vs median total

# Germany50: the SNDlib file has pre-installed capacity 0 on every link and a single installable
# module of 40 capacity units. One module per link is used, read as 40 Gbit/s = 40,000 Mbit/s
# (the dynamic DFN matrices are in Mbit/s).
GERMANY50_MODULE_MBPS = 40_000.0

SPECS = {
    "Abilene": {
        "raw": "abilene",
        "network": "abilene.txt",
        "archive": "directed-abilene-zhang-5min-over-6months-ALL-native.tgz",
        "capacity_rule": "pre_installed",
        "traffic_label": "SNDlib dynamic Abilene matrices (Yin Zhang, measured NetFlow OD traffic), 5-minute bins, Mbit/s.",
    },
    "Germany50": {
        "raw": "germany50",
        "network": "germany50.txt",
        "archive": "directed-germany50-DFN-aggregated-5min-over-1day-native.tgz",
        "other_archives": [
            "directed-germany50-DFN-aggregated-1day-over-1month-native.tgz",
            "directed-germany50-DFN-aggregated-1month-over-1year-native.tgz",
        ],
        "capacity_rule": "one_module",
        "traffic_label": "SNDlib dynamic Germany50 matrices (DFN-Verein accounting data aggregated to the 50 germany50 "
                         "nodes), 5-minute bins over one day, Mbit/s.",
    },
}


# ------------------------------------------------------------------ topology
def build_topology(name: str, spec: dict) -> tuple[Topology, dict, list[str]]:
    net = parse_network((RAW / spec["raw"] / spec["network"]).read_text(encoding="utf-8"))
    index = {n: i for i, n in enumerate(net.nodes)}
    links = []
    cap_records = []
    for link in net.links:
        if spec["capacity_rule"] == "pre_installed":
            cap = link.pre_installed_capacity
            rule = "pre-installed capacity (Mbit/s)"
        else:
            if link.pre_installed_capacity > 0:
                raise ValueError(f"{name}: unexpected pre-installed capacity on {link.link_id}")
            if len(link.modules) != 1 or link.modules[0][0] != 40.0:
                raise ValueError(f"{name}: unexpected module list on {link.link_id}: {link.modules}")
            cap = GERMANY50_MODULE_MBPS
            rule = "one installable module of 40 units, read as 40 Gbit/s"
        cap_records.append({"link_id": link.link_id, "source": link.source, "target": link.target,
                            "pre_installed": link.pre_installed_capacity, "modules": link.modules,
                            "capacity_used_mbps": cap, "rule": rule})
        u, v = index[link.source], index[link.target]
        links.append({"source": u, "target": v, "capacity": cap})
        links.append({"source": v, "target": u, "capacity": cap})
    data = {"nodes": [{"id": i, "name": n} for i, n in enumerate(net.nodes)], "links": links}
    note = ("Capacities in Mbit/s. " + ("SNDlib pre-installed capacity per undirected link, applied to both directions."
            if spec["capacity_rule"] == "pre_installed" else
            "SNDlib gives no pre-installed capacity; one 40-unit module per undirected link, read as 40 Gbit/s, "
            "applied to both directions."))
    topo = from_node_link(name, data, str((RAW / spec["raw"] / spec["network"]).relative_to(REPO).as_posix()), note)
    for node in topo.graph.nodes:
        topo.graph.nodes[node]["name"] = net.nodes[node]
    info = {"network_meta": net.meta, "undirected_links": len(net.links), "static_demands": len(net.demands),
            "static_demand_total": float(sum(v for _, _, v in net.demands)), "unparsed_lines": net.unparsed_lines,
            "capacity_records": cap_records}
    return topo, info, net.nodes


# ------------------------------------------------------------------ quality checks
def quality(series, n: int) -> dict:
    """Validity per matrix (no clipping); returns masks and reasons."""
    m = series.matrices
    t = m.shape[0]
    off = ~np.eye(n, dtype=bool)
    od = m[:, off]
    checks = [validate_matrix(x, n) for x in m]
    reasons: list[list[str]] = [list(c["reasons"]) for c in checks]
    total = np.nansum(np.where(np.isfinite(od), od, 0.0), axis=1)
    med_total = float(np.median(total))
    for i in range(t):
        if total[i] == 0.0:
            reasons[i].append("all-zero matrix")
        elif total[i] > TOTAL_RATIO * med_total:
            reasons[i].append(f"total demand > {TOTAL_RATIO:g}x median total")
        elif total[i] < med_total / TOTAL_RATIO:
            reasons[i].append(f"total demand < median total / {TOTAL_RATIO:g}")
        if series.unknown_endpoints[i]:
            reasons[i].append("demand endpoint not in topology")
        if series.duplicate_pairs[i]:
            reasons[i].append("duplicate demand record")
        if series.unparsed[i]:
            reasons[i].append("unparsed lines")
        if not series.node_sets_match[i]:
            reasons[i].append("node section differs from network file")
    # isolated single-entry spikes
    pair_median = np.median(od, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pair_median > 0, od / np.where(pair_median > 0, pair_median, 1.0), 0.0)
    spike = ratio > SPIKE_RATIO
    prev = np.vstack([np.full((1, od.shape[1]), np.inf), ratio[:-1]])
    nxt = np.vstack([ratio[1:], np.full((1, od.shape[1]), np.inf)])
    isolated = spike & (prev <= SPIKE_NEIGHBOUR_RATIO) & (nxt <= SPIKE_NEIGHBOUR_RATIO)
    material = od >= SPIKE_MATERIAL_SHARE * np.maximum(total, 1e-12)[:, None]
    corrupt_spike = isolated & material
    for i in np.flatnonzero(corrupt_spike.any(axis=1)):
        reasons[int(i)].append("isolated single-interval spike (>100x pair median, neighbours <=10x, >=1% of total)")
    valid = np.array([not r for r in reasons])
    # sustained high-ratio episodes that are kept
    sustained = spike & ~isolated & (od >= 0.25 * np.maximum(total, 1e-12)[:, None])
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    # run lengths of consecutive bins above 100x the pair median
    runs: list[int] = []
    for k in range(spike.shape[1]):
        col = spike[:, k].astype(np.int8)
        if not col.any():
            continue
        edges = np.diff(np.concatenate([[0], col, [0]]))
        runs.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
    runs_arr = np.asarray(runs) if runs else np.zeros(1, dtype=int)
    return {
        "spike_runs": {"n_runs": int(len(runs)), "single_bin_runs": int(np.sum(runs_arr == 1)),
                       "median_run_bins": float(np.median(runs_arr)), "longest_run_bins": int(runs_arr.max()),
                       "share_of_spike_entries_in_runs_ge_12_bins": float(runs_arr[runs_arr >= 12].sum() / max(runs_arr.sum(), 1))},
        "valid": valid, "reasons": reasons, "checks": checks, "total": total,
        "spike_entries_100x": int(spike.sum()), "spike_matrices_100x": int(spike.any(axis=1).sum()),
        "isolated_spike_entries": int(isolated.sum()), "corrupt_spike_entries": int(corrupt_spike.sum()),
        "sustained_dominant_entries": int(sustained.sum()),
        "sustained_dominant_matrices": int(sustained.any(axis=1).sum()),
        "sustained_dominant_pairs": {f"{pairs[k][0]}->{pairs[k][1]}": int(sustained[:, k].sum())
                                     for k in np.flatnonzero(sustained.any(axis=0))},
        "pairs_with_zero_median": int((pair_median == 0).sum()),
        "pairs_never_positive": int((od.max(axis=0) == 0).sum()),
        "zero_entry_share": float((od == 0).mean()),
    }


def traffic_stats(m: np.ndarray, times: list[str], n: int, node_names: list[str]) -> dict:
    off = ~np.eye(n, dtype=bool)
    od = m[:, off]
    total = od.sum(axis=1)
    mean_pair = od.mean(axis=0)
    order = np.sort(mean_pair)[::-1]
    share = order / max(order.sum(), 1e-12)
    cum = np.cumsum(share)
    k10 = max(1, int(round(0.1 * share.size)))
    sorted_asc = np.sort(mean_pair)
    gini = float(1 - 2 * np.sum(np.cumsum(sorted_asc) / sorted_asc.sum()) / sorted_asc.size + 1 / sorted_asc.size)
    pairs = [(node_names[i], node_names[j]) for i in range(n) for j in range(n) if i != j]
    top = np.argsort(mean_pair)[::-1][:5]
    hours = np.array([int(t.split("-")[1][:2]) if "-" in t else -1 for t in times])
    by_hour = {int(h): float(total[hours == h].mean()) for h in sorted(set(hours.tolist())) if h >= 0}
    days = [t.split("-")[0] for t in times]
    by_day = defaultdict(list)
    for d, v in zip(days, total):
        by_day[d].append(v)
    return {
        "n_matrices": int(m.shape[0]),
        "total_demand_mbps": {"min": float(total.min()), "p5": float(np.percentile(total, 5)),
                              "median": float(np.median(total)), "mean": float(total.mean()),
                              "p95": float(np.percentile(total, 95)), "max": float(total.max())},
        "mean_total_by_hour_of_day": by_hour,
        "n_days": len(by_day),
        "daily_mean_total": {"min": float(min(np.mean(v) for v in by_day.values())),
                             "max": float(max(np.mean(v) for v in by_day.values()))},
        "pair_distribution": {"n_pairs": int(share.size), "top1_share": float(share[0]),
                              "top10pct_pairs_share": float(cum[k10 - 1]), "pairs_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
                              "gini_mean_demand": gini,
                              "mean_pair_demand_mbps": {"median": float(np.median(mean_pair)), "max": float(mean_pair.max())},
                              "top5_pairs": [{"pair": f"{pairs[k][0]}->{pairs[k][1]}", "mean_mbps": float(mean_pair[k]),
                                              "share": float(mean_pair[k] / mean_pair.sum())} for k in top]},
    }


# ------------------------------------------------------------------ overlap
def b4_overlap(abilene: np.ndarray, times: list[str], node_names: list[str]) -> dict:
    """Compare the B4 traffic (TEAL real matrices) with the SNDlib Abilene series."""
    teal = sorted(glob.glob(str(REPO / "data" / "raw" / "teal" / "traffic-matrices" / "real" / "B4.json_real_*_1.0_traffic-matrix.pkl")),
                  key=lambda p: int(re.search(r"real_(\d+)_", p).group(1)))
    b4 = np.stack([np.asarray(pickle.load(open(p, "rb")), dtype=np.float64) for p in teal])
    yates_path = REPO / "data" / "raw" / "yates" / "abilene.txt"
    yates = np.loadtxt(yates_path).reshape(-1, 12, 12)
    off = ~np.eye(12, dtype=bool)
    b4_is_yates = bool(np.allclose(b4[:, off], yates[:, off] / 1e5, rtol=1e-6) and np.all(b4[:, ~off] == 0))
    # rank-1 test: simple gravity p(s,d) = p(s) p(d)
    grav_err = []
    for y in yates:
        g = np.outer(y.sum(1), y.sum(0)) / y.sum()
        grav_err.append(float(np.abs(g - y).sum() / y.sum()))
    # exact / near-exact match of OD entries (permutation-free test on sorted, sum-normalised vectors)
    def sorted_key(x):
        s = np.sort(x, axis=-1)[..., ::-1]
        return s / np.maximum(s.sum(axis=-1, keepdims=True), 1e-300)
    ka, kb = sorted_key(abilene[:, off]), sorted_key(b4[:, off])
    nearest_sorted = [float(np.abs(ka - kb[i]).sum(axis=1).min()) for i in range(kb.shape[0])]
    consecutive = float(np.median(np.abs(np.diff(sorted_key(abilene[:2016, off]), axis=0)).sum(axis=1)))
    # time alignment through node marginals (ingress + egress shares), node order = SNDlib order
    def marg(x):
        v = np.concatenate([x.sum(-1), x.sum(-2)], axis=-1)
        return v / v.sum(axis=-1, keepdims=True)
    ms, my = marg(abilene), marg(yates)
    t0 = times.index("20040301-0000")
    ranks, best_times, same_slot_d, median_d = [], [], [], []
    for i in range(yates.shape[0]):
        d = np.abs(ms - my[i]).sum(axis=1)
        order = np.argsort(d)
        ranks.append(int(np.flatnonzero(order == t0 + i)[0]))
        best_times.append(times[int(order[0])])
        same_slot_d.append(float(d[t0 + i]))
        median_d.append(float(np.median(d)))
    yates_tot = yates.sum(axis=(1, 2))
    snd_tot = abilene[t0:t0 + yates.shape[0]].sum(axis=(1, 2))
    corr = float(np.corrcoef(yates_tot, snd_tot)[0, 1])
    overlap_slots = times[t0:t0 + yates.shape[0]]
    return {
        "b4_source": "data/raw/teal/traffic-matrices/real/B4.json_real_<i>_1.0_traffic-matrix.pkl (36 matrices)",
        "yates_file": "data/raw/yates/abilene.txt (cornell-netlab/yates, data/demands/actual/abilene.txt)",
        "b4_equals_yates_div_1e5_offdiagonal": b4_is_yates,
        "yates_rank1_gravity_rel_error_max": float(max(grav_err)),
        "nearest_sndlib_sorted_l1_per_b4_tm": {"min": float(min(nearest_sorted)), "median": float(np.median(nearest_sorted)),
                                               "max": float(max(nearest_sorted))},
        "sndlib_consecutive_5min_sorted_l1_median": consecutive,
        "marginal_alignment": {
            "hypothesis": "Yates matrix i corresponds to the 5-minute slot 2004-03-01 00:00 + 5 i min",
            "rank_of_hypothesised_slot_among_all_sndlib_matrices": ranks,
            "median_rank": float(np.median(ranks)), "max_rank": int(max(ranks)), "n_candidates": int(abilene.shape[0]),
            "best_matching_slots": sorted(Counter(best_times).items()),
            "mean_marginal_l1_hypothesised_slot": float(np.mean(same_slot_d)),
            "mean_marginal_l1_median_over_all_slots": float(np.mean(median_d)),
            "total_demand_correlation_hypothesised_slots": corr,
            "yates_total_over_sndlib_total_mean": float(np.mean(yates_tot / 1e6 / snd_tot)),
        },
        "overlapping_slots": overlap_slots,
        "b4_split_slots": {"train (0-19)": [overlap_slots[0], overlap_slots[19]],
                           "val (20-27)": [overlap_slots[20], overlap_slots[27]],
                           "held aside (28-35)": [overlap_slots[28], overlap_slots[35]]},
    }


# ------------------------------------------------------------------ sampling
def stratified_sample(valid_idx: np.ndarray, k: int, seed: int) -> np.ndarray:
    """One seeded draw from each of k equal consecutive time strata of the valid matrices."""
    if valid_idx.size <= k:
        return np.sort(valid_idx)
    rng = np.random.default_rng(seed)
    strata = np.array_split(valid_idx, k)
    return np.array(sorted(int(s[rng.integers(s.size)]) for s in strata), dtype=np.int64)


# ------------------------------------------------------------------ calibration
def calibrate(name: str, demands: np.ndarray) -> list[dict]:
    from phase3_final import routing as R
    from phase3_final.common import matrix_hash
    from twin.data import load_topology

    topo = load_topology(name)
    sp, eq = R.shortest_action(topo), R.equal_action(topo)
    rows = []
    for pos in range(demands.shape[0]):
        m = demands[pos]
        sol = R.lp_min_mlu(topo, m)
        if sol["status"] != "optimal":
            raise RuntimeError(f"LP failed for {name}/{pos}: {sol['message']}")
        ref = sol["lp_mlu"]
        rows.append({"topology": name, "split": "test", "position": pos, "tm_hash": matrix_hash(m),
                     "mlu_shortest": R.mlu(topo, m, sp), "mlu_equal": R.mlu(topo, m, eq), "mlu_lp": ref,
                     "lp_runtime_s": sol["runtime_s"], "lp_solver_u": sol["solver_u"], "reference": "lp",
                     "reference_mlu": ref, "alpha": {f"{t:.1f}": t / ref for t in TARGETS}})
    return rows


# ------------------------------------------------------------------ main pipeline
def process(name: str, spec: dict, overlap: dict | None, jobs: int) -> dict:
    topo, net_info, node_names = build_topology(name, spec)
    n = topo.num_nodes
    conn = connectivity_report(topo)
    _save_topology(topo)
    dump_json(GRAPH_DIR / f"{name}_validation.json", {**conn, "capacity_note": topo.capacity_note, "source": topo.source})

    series = read_demand_archive(RAW / spec["raw"] / spec["archive"], node_names)
    q = quality(series, n)
    valid_idx = np.flatnonzero(q["valid"])
    removed = [{"index": int(i), "time": series.times[i], "reasons": q["reasons"][i]} for i in np.flatnonzero(~q["valid"])]
    excluded_overlap: list[int] = []
    if overlap is not None:
        slots = set(overlap["overlapping_slots"])
        excluded_overlap = [int(i) for i in valid_idx if series.times[i] in slots]
        valid_idx = np.array([i for i in valid_idx if series.times[i] not in slots], dtype=np.int64)
    sample = stratified_sample(valid_idx, SAMPLE_SIZE, SAMPLE_SEED)
    rng = np.random.default_rng(E3_SEED)
    e3_pos = sorted(int(x) for x in rng.choice(sample.size, size=min(E3_SIZE, sample.size), replace=False))
    demands = series.matrices[sample]

    # every demand has a path: computed below from the path table
    params = normalize.parameters(topo.capacities)
    denom = params["denominator"]
    params.update({"topology": name, "traffic_kind": "real_sndlib",
                   "split": {"test_rows": int(sample.size), "status": "SNDlib dynamic matrices, seeded stratified sample; "
                             "test only (unseen topology)."}})
    dump_json(NORM_DIR / f"{name}.json", params)
    np.savez_compressed(NORM_DIR / f"{name}.npz", capacity_normalised=normalize.apply_capacity(topo.capacities, denom),
                        demand_normalised=normalize.apply_demand(demands, denom).astype(np.float32))
    params["capacity_normalised"] = normalize.apply_capacity(topo.capacities, denom)

    paths = generate(topo, jobs=jobs, checkpoint=False)
    problems = validate(topo, paths)
    _save_paths(name, paths)
    bundle = {"kind": "real_sndlib", "demand_original": demands, "summary": summarise_checks([validate_matrix(x, n) for x in demands])}
    bip = _save_bipartite(name, topo, paths, params, bundle)

    off = ~np.eye(n, dtype=bool)
    positive_pairs = (series.matrices[valid_idx][:, off] > 0).any(axis=0)
    no_path = paths["n_paths"].astype(np.int64) == 0
    demand_without_path = int((positive_pairs & no_path).sum())

    # files
    SND_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_TEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLIT_TEST / f"{name}_demand.npz", demand=demands)
    times_sel = [series.times[i] for i in sample]
    dump_json(SPLIT_TEST / f"{name}.json", {
        "topology": name, "traffic_kind": "real_sndlib",
        "topology_file": f"data/processed/topologies/{name}.json",
        "paths_file": f"data/processed/candidate_paths/{name}_yen4.npz",
        "paths_meta_file": f"data/processed/candidate_paths/{name}_yen4.json",
        "bipartite_file": f"data/processed/graph_data/{name}_bipartite.pt",
        "normalization_file": f"data/processed/normalization/{name}.json",
        "traffic_source_file": f"data/raw/sndlib/{spec['raw']}/{spec['archive']}",
        "present": True, "entry_name": name, "split_role": "test_unseen_topology",
        "matrix_count": int(sample.size), "matrix_shape": list(demands.shape),
        "matrix_indices": [int(i) for i in sample], "matrix_times": times_sel,
        "matrix_index_meaning": "position in the time-ordered list of all matrices in the SNDlib archive",
        "demand_file": f"data/processed/splits/test/{name}_demand.npz", "unit": "Mbit/s",
    })
    dump_json(SND_DIR / f"{name}_sample.json", {
        "topology": name, "archive": spec["archive"], "n_matrices_in_archive": int(series.matrices.shape[0]),
        "n_valid": int(q["valid"].sum()), "n_removed_invalid": len(removed),
        "n_excluded_b4_overlap": len(excluded_overlap), "n_eligible": int(valid_idx.size),
        "method": f"valid matrices in time order split into {SAMPLE_SIZE} equal consecutive strata; one matrix drawn "
                  f"per stratum with numpy default_rng({SAMPLE_SEED}). E3 link-failure subset: {E3_SIZE} positions "
                  f"drawn without replacement with default_rng({E3_SEED}).",
        "sample_size": int(sample.size), "sample_seed": SAMPLE_SEED,
        "positions": list(range(int(sample.size))), "archive_indices": [int(i) for i in sample], "times": times_sel,
        "e3_seed": E3_SEED, "e3_positions_50": e3_pos, "e3_times": [times_sel[p] for p in e3_pos],
        "excluded_b4_overlap_indices": excluded_overlap,
    })
    with (SND_DIR / f"{name}_total_demand.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archive_index", "time", "total_demand_mbps", "valid", "sampled"])
        sel = set(int(i) for i in sample)
        for i, (t, v) in enumerate(zip(series.times, q["total"])):
            w.writerow([i, t, f"{v:.6f}", bool(q["valid"][i]), i in sel])

    other = {}
    for arc in spec.get("other_archives", []):
        s2 = read_demand_archive(RAW / spec["raw"] / arc, node_names)
        q2 = quality(s2, n)
        other[arc] = {"n_matrices": int(s2.matrices.shape[0]), "first": s2.times[0], "last": s2.times[-1],
                      "granularity": sorted(set(s2.granularity)), "units": sorted(set(s2.units)),
                      "n_valid": int(q2["valid"].sum()),
                      "total_demand_median": float(np.median(q2["total"]))}

    hours_sel = Counter(int(t.split("-")[1][:2]) for t in times_sel)
    days_sel = Counter(t.split("-")[0] for t in times_sel)
    rec = {
        "topology": name,
        "network_file": f"data/raw/sndlib/{spec['raw']}/{spec['network']}",
        "archive": f"data/raw/sndlib/{spec['raw']}/{spec['archive']}",
        "nodes": n, "undirected_links": net_info["undirected_links"], "directed_links": topo.num_edges,
        "connectivity": conn, "topology_warnings": topo.warnings, "excluded_links": topo.excluded,
        "capacities_mbps": sorted(Counter(topo.capacities.tolist()).items()),
        "capacity_records": net_info["capacity_records"], "network_meta": net_info["network_meta"],
        "static_demands_in_network_file": net_info["static_demands"],
        "n_matrices": int(series.matrices.shape[0]), "first_time": series.times[0], "last_time": series.times[-1],
        "granularity": sorted(set(series.granularity)), "units": sorted(set(series.units)),
        "node_sections_match_network": bool(all(series.node_sets_match)),
        "unknown_endpoint_records": int(sum(series.unknown_endpoints)),
        "self_demand_records": int(sum(series.self_demands)), "duplicate_records": int(sum(series.duplicate_pairs)),
        "unparsed_lines": int(sum(series.unparsed)),
        "non_finite_entries": int(sum(c["non_finite_entries"] for c in q["checks"])),
        "negative_entries": int(sum(c["negative_entries"] for c in q["checks"])),
        "all_zero_matrices": int((q["total"] == 0).sum()),
        "removed": removed, "n_removed": len(removed), "n_valid": int(q["valid"].sum()),
        "spike_runs": q["spike_runs"],
        "spike_summary": {k: q[k] for k in ("spike_entries_100x", "spike_matrices_100x", "isolated_spike_entries",
                                            "corrupt_spike_entries", "sustained_dominant_entries",
                                            "sustained_dominant_matrices", "pairs_with_zero_median",
                                            "pairs_never_positive", "zero_entry_share")},
        "sustained_dominant_pairs": {f"{node_names[int(k.split('->')[0])]}->{node_names[int(k.split('->')[1])]}": v
                                     for k, v in q["sustained_dominant_pairs"].items()},
        "excluded_b4_overlap": len(excluded_overlap), "n_eligible": int(valid_idx.size),
        "traffic_stats_all_valid": traffic_stats(series.matrices[q["valid"]], [series.times[i] for i in np.flatnonzero(q["valid"])], n, node_names),
        "traffic_stats_sample": traffic_stats(demands, times_sel, n, node_names),
        "sample": {"size": int(sample.size), "first": times_sel[0], "last": times_sel[-1], "n_days": len(days_sel),
                   "hours_covered": len(hours_sel), "per_hour_min": min(hours_sel.values()), "per_hour_max": max(hours_sel.values()),
                   "e3_size": len(e3_pos)},
        "paths": {**paths["meta"], "validation_problems": len(problems), "positive_demand_pairs_without_path": demand_without_path},
        "bipartite": bip, "normalisation_denominator": denom,
        "other_archives": other,
    }
    dump_json(SND_DIR / f"{name}_quality.json", json.loads(json.dumps(rec, default=float)))
    link_failure_cases(name)
    return rec


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def write_quality_report(recs: dict[str, dict], overlap: dict, cal: dict[str, list[dict]]) -> None:
    ab, ge = recs["Abilene"], recs["Germany50"]
    L = [
        "# SNDlib Abilene and Germany50: data quality report",
        "",
        "Every number below is read from the files under `data/raw/sndlib/` by "
        "`implementation/preprocessing/sndlib_prepare.py`; the machine-readable versions are "
        "`data/processed/sndlib/<topology>_quality.json`, `<topology>_sample.json`, `<topology>_total_demand.csv`, "
        "`abilene_b4_overlap.json` and `data/processed/calibrated_load/sndlib_calibration_rows.jsonl`.",
        "",
        "## 1. Size, time resolution and date range",
        "",
        "| | Abilene | Germany50 |",
        "|---|---|---|",
        f"| Nodes | {ab['nodes']} | {ge['nodes']} |",
        f"| Undirected links in the network file | {ab['undirected_links']} | {ge['undirected_links']} |",
        f"| Directed links used (each undirected link in both directions) | {ab['directed_links']} | {ge['directed_links']} |",
        f"| Connected (strongly, directed) | {ab['connectivity']['strongly_connected']} | {ge['connectivity']['strongly_connected']} |",
        f"| Dynamic archive | `{Path(ab['archive']).name}` | `{Path(ge['archive']).name}` |",
        f"| Traffic matrices in the archive | {ab['n_matrices']:,} | {ge['n_matrices']:,} |",
        f"| Time resolution (META `granularity`) | {', '.join(ab['granularity'])} | {', '.join(ge['granularity'])} |",
        f"| First / last matrix (META `time`) | {ab['first_time']} / {ab['last_time']} | {ge['first_time']} / {ge['last_time']} |",
        f"| Calendar days with data | {ab['traffic_stats_all_valid']['n_days']} | {ge['traffic_stats_all_valid']['n_days']} |",
        f"| Demand unit (META `unit`) | {', '.join(ab['units'])} | {', '.join(ge['units'])} |",
        "",
        f"Abilene covers 1 March to 10 September 2004 with the gaps listed in the SNDlib readme "
        f"(15.03–01.04, 16.04–21.04, 29.04–30.04 and 20.08.2004); {ab['traffic_stats_all_valid']['n_days']} days × 288 "
        f"five-minute bins = {ab['traffic_stats_all_valid']['n_days'] * 288:,}, which equals the number of files. "
        "Germany50's five-minute archive covers a single day (15 February 2005). It is the only Germany50 archive "
        "with at least 200 matrices; the other two archives were downloaded and checked but are not used, because "
        "they have a different aggregation period (daily and monthly averages):",
        "",
        "| Germany50 archive | Matrices | First | Last | Granularity | Valid | Median total (Mbit/s) |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for arc, o in ge["other_archives"].items():
        L.append(f"| `{arc}` | {o['n_matrices']} | {o['first']} | {o['last']} | {', '.join(o['granularity'])} | "
                 f"{o['n_valid']} | {_fmt(o['total_demand_median'])} |")
    cab = dict(ab["capacities_mbps"])
    L += [
        "",
        "## 2. Units and capacities",
        "",
        "All demand values are in Mbit/s (`unit = MBITPERSEC` in the META section of every matrix). SNDlib converted "
        "the original Abilene counts (100 bytes per 5 minutes, packet sampling rate 100) with ×8/3/10⁶; the DFN byte "
        "counts were also converted to Mbit/s by SNDlib. No further conversion is applied to demands. Capacities are "
        "expressed in the same unit, Mbit/s.",
        "",
        "| Topology | Capacity rule | Values used (Mbit/s, directed links) |",
        "|---|---|---|",
        f"| Abilene | SNDlib pre-installed capacity of each link, applied to both directions | "
        + ", ".join(f"{_fmt(c, 0)} × {k}" for c, k in sorted(cab.items())) + " |",
        f"| Germany50 | no pre-installed capacity in the file (0.00 on all {ge['undirected_links']} links); one installable "
        f"module of 40 units per link | {_fmt(GERMANY50_MODULE_MBPS, 0)} × {ge['directed_links']} |",
        "",
        "- **Abilene.** 9,920 Mbit/s is the OC-192 payload rate and 2,480 Mbit/s the OC-48 payload rate; the single "
        "OC-48 link is ATLAng–IPLSng. The files state the demand unit as Mbit/s, and the capacities are read in the "
        "same unit. The module list in the file (40,000 units per link) describes expansion options for network "
        "design and is not used.",
        "- **Germany50.** The network file offers only installable modules (capacity 40, cost 3,290–4,150) and no "
        "installed capacity, and it states no unit for the module. The values used are therefore one module on every "
        "link, read as 40 Gbit/s = 40,000 Mbit/s. This is a modelling choice, not a measured capacity. All links get "
        "the same capacity, which is the only choice that adds no information beyond the file.",
        "- **Why the absolute Germany50 value has no effect on the reported results.** Every experiment uses "
        "calibrated load: each matrix is multiplied by alpha = t / LP-MLU, and LP-MLU is proportional to "
        "1/capacity, so a different capacity constant c rescales the routed traffic by the same c. The learned "
        "policy and the twins divide capacities and demands by the maximum capacity, so their inputs, the "
        "utilisations, the gate decisions, the fallback, shortest path, ECMP, the LP and the satisfaction metric are all "
        "unchanged. The only component that sees absolute values is the frozen P1 actor, whose features are raw "
        "capacities and demands (P1 eq. 5, unscaled).",
        "",
        "## 3. Invalid and corrupted matrices",
        "",
        "Rules (applied to every matrix, in this order; nothing is clipped or repaired):",
        "",
        "1. NaN or infinite entry; 2. negative entry; 3. all-zero matrix; "
        f"4. total demand above {TOTAL_RATIO:g}× or below 1/{TOTAL_RATIO:g} of the median total; "
        "5. demand record with an endpoint that is not a topology node, duplicate record, unparsed line or node "
        "section that differs from the network file; "
        f"6. an isolated single-interval spike: an entry above {SPIKE_RATIO:g}× the median of its node pair, "
        f"whose previous and next five-minute values are both at most {SPIKE_NEIGHBOUR_RATIO:g}× that median, "
        f"and which carries at least {SPIKE_MATERIAL_SHARE:.0%} of the matrix total.",
        "",
        "| Check | Abilene | Germany50 |",
        "|---|---:|---:|",
        f"| NaN / infinite entries | {ab['non_finite_entries']} | {ge['non_finite_entries']} |",
        f"| Negative entries | {ab['negative_entries']} | {ge['negative_entries']} |",
        f"| All-zero matrices | {ab['all_zero_matrices']} | {ge['all_zero_matrices']} |",
        f"| Demand endpoints not in the topology | {ab['unknown_endpoint_records']} | {ge['unknown_endpoint_records']} |",
        f"| Node sections identical to the network file | {ab['node_sections_match_network']} | {ge['node_sections_match_network']} |",
        f"| Self-demand records | {ab['self_demand_records']} | {ge['self_demand_records']} |",
        f"| Duplicate records / unparsed lines | {ab['duplicate_records']} / {ab['unparsed_lines']} | {ge['duplicate_records']} / {ge['unparsed_lines']} |",
        f"| Total demand, max / median | {ab['traffic_stats_all_valid']['total_demand_mbps']['max'] / ab['traffic_stats_all_valid']['total_demand_mbps']['median']:.2f} | "
        f"{ge['traffic_stats_all_valid']['total_demand_mbps']['max'] / ge['traffic_stats_all_valid']['total_demand_mbps']['median']:.2f} |",
        f"| Total demand, min / median | {ab['traffic_stats_all_valid']['total_demand_mbps']['min'] / ab['traffic_stats_all_valid']['total_demand_mbps']['median']:.3f} | "
        f"{ge['traffic_stats_all_valid']['total_demand_mbps']['min'] / ge['traffic_stats_all_valid']['total_demand_mbps']['median']:.3f} |",
        f"| Entries above 100× their pair median | {ab['spike_summary']['spike_entries_100x']:,} (in {ab['spike_summary']['spike_matrices_100x']:,} matrices) | "
        f"{ge['spike_summary']['spike_entries_100x']:,} (in {ge['spike_summary']['spike_matrices_100x']:,} matrices) |",
        f"| … of which isolated in time | {ab['spike_summary']['isolated_spike_entries']} | {ge['spike_summary']['isolated_spike_entries']} |",
        f"| … of which isolated and ≥ 1% of the total (rule 6) | {ab['spike_summary']['corrupt_spike_entries']} | {ge['spike_summary']['corrupt_spike_entries']} |",
        f"| **Matrices removed** | **{ab['n_removed']}** | **{ge['n_removed']}** |",
        f"| Valid matrices | {ab['n_valid']:,} | {ge['n_valid']:,} |",
        "",
    ]
    for rec in (ab, ge):
        for r in rec["removed"]:
            L.append(f"- {rec['topology']} removed: index {r['index']} ({r['time']}): {'; '.join(r['reasons'])}.")
    sd = ab["sustained_dominant_pairs"]
    L += [
        "",
        "Most entries above 100× their pair median are not errors. On Germany50 they sit on sparse pairs whose median "
        f"is close to zero ({ge['spike_summary']['pairs_with_zero_median']} of 2,450 pairs have median 0; "
        f"{ge['spike_summary']['pairs_never_positive']} pairs are never positive), and they carry well under 1% of "
        "the matrix total. On Abilene, "
        f"{ab['spike_summary']['sustained_dominant_matrices']} matrices contain an entry that is above 100× its "
        "pair median and carries at least 25% of the matrix total, on three pairs ("
        + ", ".join(f"{k}: {v}" for k, v in sd.items()) +
        " five-minute bins). Only "
        f"{ab['spike_summary']['isolated_spike_entries']} of the {ab['spike_summary']['spike_entries_100x']:,} Abilene "
        "entries above 100× their pair median are isolated (both neighbouring bins at most 10× the median); the others "
        f"sit inside elevated periods. Counting only bins above 100×, they form {ab['spike_runs']['n_runs']:,} runs "
        f"(longest {ab['spike_runs']['longest_run_bins']:,} bins = {ab['spike_runs']['longest_run_bins'] * 5 / 60:.0f} "
        f"hours), and {ab['spike_runs']['share_of_spike_entries_in_runs_ge_12_bins']:.0%} of the entries lie in runs "
        "of at least one hour; the dominant-pair matrices cluster in 8–15 April 2004. Persistent elevated flows of this kind "
        "are the signature of long bulk transfers rather than of a counting error, so they are kept: removing them "
        "would drop real high-load periods, which are the situations a safety gate is meant to handle.",
        "",
        "## 4. Node coverage and paths",
        "",
        "| | Abilene | Germany50 |",
        "|---|---:|---:|",
        f"| Demand records whose node is missing from the topology | {ab['unknown_endpoint_records']} | {ge['unknown_endpoint_records']} |",
        f"| Ordered pairs without a directed path | {ab['connectivity']['ordered_pairs_without_a_directed_path']} | {ge['connectivity']['ordered_pairs_without_a_directed_path']} |",
        f"| Candidate paths (Yen k = 4, hop count) | {ab['paths']['paths_generated']:,} | {ge['paths']['paths_generated']:,} |",
        f"| Pairs with fewer than 4 paths | {ab['paths']['demands_with_fewer_than_4']} | {ge['paths']['demands_with_fewer_than_4']} |",
        f"| Pairs with no path | {ab['paths']['demands_with_no_path']} | {ge['paths']['demands_with_no_path']} |",
        f"| Positive-demand pairs without a path | {ab['paths']['positive_demand_pairs_without_path']} | {ge['paths']['positive_demand_pairs_without_path']} |",
        f"| Path validation problems | {ab['paths']['validation_problems']} | {ge['paths']['validation_problems']} |",
        f"| Path backend | {ab['paths']['backend']} | {ge['paths']['backend']} |",
        "",
        "Every node named in any matrix is a node of the network file (the node section of every matrix is identical to "
        "the network's), and every demand has at least one candidate path.",
        "",
        "## 5. Traffic statistics (all valid matrices)",
        "",
        "| | Abilene | Germany50 |",
        "|---|---:|---:|",
    ]
    for key, label in (("min", "Total demand, min (Mbit/s)"), ("p5", "Total demand, 5th percentile"),
                       ("median", "Total demand, median"), ("mean", "Total demand, mean"),
                       ("p95", "Total demand, 95th percentile"), ("max", "Total demand, max")):
        L.append(f"| {label} | {_fmt(ab['traffic_stats_all_valid']['total_demand_mbps'][key])} | "
                 f"{_fmt(ge['traffic_stats_all_valid']['total_demand_mbps'][key])} |")
    pa, pg = ab["traffic_stats_all_valid"]["pair_distribution"], ge["traffic_stats_all_valid"]["pair_distribution"]
    L += [
        f"| Daily mean total, lowest / highest day | {_fmt(ab['traffic_stats_all_valid']['daily_mean_total']['min'])} / "
        f"{_fmt(ab['traffic_stats_all_valid']['daily_mean_total']['max'])} | {_fmt(ge['traffic_stats_all_valid']['daily_mean_total']['min'])} / "
        f"{_fmt(ge['traffic_stats_all_valid']['daily_mean_total']['max'])} |",
        f"| Node pairs | {pa['n_pairs']} | {pg['n_pairs']} |",
        f"| Share of traffic on the largest pair | {pa['top1_share']:.1%} | {pg['top1_share']:.1%} |",
        f"| Share on the largest 10% of pairs | {pa['top10pct_pairs_share']:.1%} | {pg['top10pct_pairs_share']:.1%} |",
        f"| Pairs carrying 80% of traffic | {pa['pairs_for_80pct']} | {pg['pairs_for_80pct']} |",
        f"| Gini coefficient of mean pair demand | {pa['gini_mean_demand']:.2f} | {pg['gini_mean_demand']:.2f} |",
        f"| Zero entries (share of all OD entries) | {ab['spike_summary']['zero_entry_share']:.1%} | {ge['spike_summary']['zero_entry_share']:.1%} |",
        "",
        "Largest pairs by mean demand:",
        "",
    ]
    for rec in (ab, ge):
        tops = rec["traffic_stats_all_valid"]["pair_distribution"]["top5_pairs"]
        L.append(f"- {rec['topology']}: " + "; ".join(f"{p['pair']} {_fmt(p['mean_mbps'])} Mbit/s ({p['share']:.1%})" for p in tops))
    L += ["", "Mean total demand by hour of day (Mbit/s, all valid matrices; hour as given in the META time stamp):", "",
          "| Hour | " + " | ".join(str(h) for h in range(24)) + " |", "|---|" + "---:|" * 24]
    for rec in (ab, ge):
        bh = rec["traffic_stats_all_valid"]["mean_total_by_hour_of_day"]
        L.append(f"| {rec['topology']} | " + " | ".join(_fmt(bh.get(str(h), bh.get(h, float('nan'))), 0) for h in range(24)) + " |")
    ov = overlap["marginal_alignment"]
    L += [
        "",
        "The complete total-demand time series is in `data/processed/sndlib/<topology>_total_demand.csv`.",
        "",
        "## 6. Overlap between Abilene and the B4 traffic",
        "",
        "The B4 traffic in this repository is TEAL's `real` set, which TEAL documents as the Yates `abilene.txt` "
        "demands mapped onto B4. The Yates file (`data/raw/yates/abilene.txt`) was compared with the SNDlib series:",
        "",
        f"- B4 matrix *i* equals Yates matrix *i* divided by 10⁵ with the diagonal set to 0 "
        f"(`{overlap['b4_equals_yates_div_1e5_offdiagonal']}` for all 36), with the node order unchanged (the only "
        "isomorphism between the Yates Abilene graph and the SNDlib Abilene graph is the identity).",
        f"- Every Yates matrix is exactly rank one (largest relative deviation from the gravity product of its own "
        f"row and column sums: {overlap['yates_rank1_gravity_rel_error_max']:.1e}). Yates therefore uses the "
        "*simple gravity* estimate p(s,d) = p(s)·p(d) that Zhang's archive ships alongside the measured OD values, "
        "not the measured OD matrix that SNDlib uses.",
        f"- No SNDlib matrix equals a B4 matrix, even allowing any node permutation and any scale: the sorted, "
        f"sum-normalised OD vectors differ by L1 = {overlap['nearest_sndlib_sorted_l1_per_b4_tm']['min']:.3f}–"
        f"{overlap['nearest_sndlib_sorted_l1_per_b4_tm']['max']:.3f} at best.",
        f"- The **time range overlaps**. Matching the ingress and egress shares of each Yates matrix against all "
        f"{ov['n_candidates']:,} SNDlib matrices, the SNDlib slot 2004-03-01 00:00 + 5·*i* min ranks "
        f"{int(np.min(ov['rank_of_hypothesised_slot_among_all_sndlib_matrices']))}–{ov['max_rank']} "
        f"(median {ov['median_rank']:.0f}) for Yates matrix *i*; every best match falls between 00:00 and 05:10 on "
        f"1 March 2004; the total demand of the 36 Yates matrices tracks the SNDlib totals of those 36 slots "
        f"(correlation {ov['total_demand_correlation_hypothesised_slots']:.2f}; Yates is "
        f"{ov['yates_total_over_sndlib_total_mean']:.2f}× larger because the gravity estimate includes the diagonal).",
        "",
        "B4's traffic is thus a gravity estimate of the Abilene network during 2004-03-01 00:00–02:55 "
        "(B4 train indices 0–19 = 00:00–01:35, validation 20–27 = 01:40–02:15, held aside 28–35 = 02:20–02:55). "
        f"These {len(overlap['overlapping_slots'])} slots are excluded before sampling, so no Abilene test matrix "
        "comes from an interval the model was trained or validated on. The Abilene topology itself is new to the "
        "model (B4 is a different graph with 38 directed links), but its traffic comes from the same measurement "
        "campaign as the B4 training traffic. Abilene is therefore an **unseen topology with related traffic**. "
        "Germany50 (DFN, 2005) has no relation to any training data.",
        "",
        "## 7. Sampling",
        "",
        f"The eligible matrices (valid, and for Abilene outside the B4 slots) are put in time order and split into "
        f"{SAMPLE_SIZE} equal consecutive strata; one matrix is drawn from each stratum with numpy "
        f"`default_rng({SAMPLE_SEED})`. The link-failure experiment uses {E3_SIZE} of these, drawn without "
        f"replacement with `default_rng({E3_SEED})`. Indices and timestamps are in "
        "`data/processed/sndlib/<topology>_sample.json`.",
        "",
        "| | Abilene | Germany50 |",
        "|---|---:|---:|",
        f"| Eligible matrices | {ab['n_eligible']:,} | {ge['n_eligible']:,} |",
        f"| Sampled | {ab['sample']['size']} | {ge['sample']['size']} |",
        f"| First / last sampled | {ab['sample']['first']} / {ab['sample']['last']} | {ge['sample']['first']} / {ge['sample']['last']} |",
        f"| Distinct days in the sample | {ab['sample']['n_days']} | {ge['sample']['n_days']} |",
        f"| Hours of day covered | {ab['sample']['hours_covered']} / 24 (min {ab['sample']['per_hour_min']}, max {ab['sample']['per_hour_max']} per hour) | "
        f"{ge['sample']['hours_covered']} / 24 (min {ge['sample']['per_hour_min']}, max {ge['sample']['per_hour_max']} per hour) |",
        f"| Median total demand, sample / all valid (Mbit/s) | {_fmt(ab['traffic_stats_sample']['total_demand_mbps']['median'])} / "
        f"{_fmt(ab['traffic_stats_all_valid']['total_demand_mbps']['median'])} | {_fmt(ge['traffic_stats_sample']['total_demand_mbps']['median'])} / "
        f"{_fmt(ge['traffic_stats_all_valid']['total_demand_mbps']['median'])} |",
        f"| Link-failure subset | {ab['sample']['e3_size']} | {ge['sample']['e3_size']} |",
        "",
        "Germany50 has 288 valid matrices, so 200 can be drawn, but all of them come from one day: the sample spreads "
        "over the times of that day and not over different days.",
        "",
        "## 8. Calibrated load",
        "",
        "Each sampled matrix is rescaled to the reference MLU targets 0.5, 0.7, 0.9 and 1.1, with the k = 4 min-MLU "
        "path LP (SciPy HiGHS) as the reference, exactly as for GEANT. Native-scale MLUs of the sampled matrices:",
        "",
        "| Topology | TMs | Shortest MLU | ECMP MLU | LP MLU | LP s/TM | alpha(0.9), median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in cal.items():
        L.append(f"| {name} | {len(rows)} | {np.mean([r['mlu_shortest'] for r in rows]):.3f} | "
                 f"{np.mean([r['mlu_equal'] for r in rows]):.3f} | {np.mean([r['mlu_lp'] for r in rows]):.3f} | "
                 f"{np.mean([r['lp_runtime_s'] for r in rows]):.2f} | {np.median([r['alpha']['0.9'] for r in rows]):.2f} |")
    L += [
        "",
        "At native scale both networks are lightly loaded (Abilene, with its real capacities; Germany50's native "
        "MLU depends on the chosen capacity and is not meaningful). This is why only the calibrated load levels are "
        "used for these topologies.",
        "",
        "## 9. Preprocessed files",
        "",
        "- `data/processed/topologies/<topology>.json`, `graph_data/<topology>_{graph.npz,bipartite.pt,validation.json}`",
        "- `data/processed/candidate_paths/<topology>_yen4.{npz,json}`, `normalization/<topology>.{json,npz}` "
        "(denominator = maximum link capacity)",
        "- `data/processed/splits/test/<topology>.json`, `<topology>_demand.npz` (the 200 sampled matrices, Mbit/s)",
        "- `data/processed/calibrated_load/sndlib_scaled_tm_index.json` (alpha per matrix and target)",
        "- `data/scenarios/link_failures/<topology>.json` (seeds 100–1000, 0–3 failed directed links, P1 §5.5 sampler)",
    ]
    (RAW / "QUALITY_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    SND_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    # Abilene first: the overlap check defines the excluded slots.
    ab_spec = SPECS["Abilene"]
    nodes = parse_network((RAW / ab_spec["raw"] / ab_spec["network"]).read_text(encoding="utf-8")).nodes
    ab_series = read_demand_archive(RAW / ab_spec["raw"] / ab_spec["archive"], nodes)
    overlap = b4_overlap(ab_series.matrices, ab_series.times, nodes)
    dump_json(SND_DIR / "abilene_b4_overlap.json", overlap)
    del ab_series
    recs = {"Abilene": process("Abilene", ab_spec, overlap, args.jobs),
            "Germany50": process("Germany50", SPECS["Germany50"], None, args.jobs)}
    cal = {}
    rows_path = CAL_DIR / "sndlib_calibration_rows.jsonl"
    if rows_path.exists():
        rows_path.unlink()
    index = {"description": "Calibrated TM for (topology, test, position, target) is alpha * D[position]; D from "
                            "data/processed/splits/test/<topology>_demand.npz. Reference = k=4 min-MLU path LP (HiGHS).",
             "targets": TARGETS, "alpha": {}, "reference": {}, "positions_200": {}, "e3_positions_50": {}}
    for name in ("Abilene", "Germany50"):
        demands = np.load(SPLIT_TEST / f"{name}_demand.npz")["demand"]
        rows = calibrate(name, demands)
        cal[name] = rows
        with rows_path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        sample = json.loads((SND_DIR / f"{name}_sample.json").read_text(encoding="utf-8"))
        index["positions_200"][name] = sample["positions"]
        index["e3_positions_50"][name] = sample["e3_positions_50"]
        for r in rows:
            index["alpha"][f"{name}/test/{r['position']}"] = r["alpha"]
            index["reference"][f"{name}/test/{r['position']}"] = r["reference"]
    dump_json(CAL_DIR / "sndlib_scaled_tm_index.json", index)
    write_quality_report(recs, overlap, cal)
    print("wrote", RAW / "QUALITY_REPORT.md")


if __name__ == "__main__":
    main()

"""Step 3: validate topologies and traffic, normalise, build paths and bipartite graphs.

Run from the phase 3 directory:

    python -m preprocessing.preprocess

Raw files are only read. The frozen P1 tree is not imported and not written.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from networkx.readwrite import json_graph

from . import normalize
from .bipartite import FEATURE_COLUMNS, incidence
from .paths import generate, load_saved, validate
from .topology import Topology, connectivity_report, from_node_link, from_zoo_gml
from .traffic import dump_json, load_b4, load_geant, load_gravity, load_toy, summarise_checks

from repo_paths import REPO as ROOT
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TOPO_DIR = PROCESSED / "topologies"
TM_DIR = PROCESSED / "traffic_matrices"
PATH_DIR = PROCESSED / "candidate_paths"
GRAPH_DIR = PROCESSED / "graph_data"
NORM_DIR = PROCESSED / "normalization"
LOG_PATH = ROOT / "results" / "logs" / "step3_preprocess.log"

# Same placeholder the frozen Zoo loader uses when LinkSpeed is absent.
COGENTCO_PLACEHOLDER = 10000.0

SPECS = [
    {"name": "B4", "family": "teal", "file": "B4.json", "traffic": "real"},
    {"name": "UsCarrier", "family": "teal", "file": "UsCarrier.json", "traffic": "toy", "prefix": "UsCarrier"},
    {"name": "Kdl", "family": "teal", "file": "Kdl.json", "traffic": "toy", "prefix": "Kdl"},
    {"name": "ASN", "family": "teal", "file": "ASN2k.json", "traffic": "toy", "prefix": "ASN2k"},
    {"name": "GEANT", "family": "figret", "traffic": "real"},
    {"name": "Cogentco", "family": "zoo", "traffic": "reconstructed"},
]


def _logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase3.preprocess")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream)
    return log


def _save_topology(topology: Topology) -> None:
    data = json_graph.node_link_data(topology.graph, edges="links")
    for link, cap in zip(data["links"], topology.capacities.tolist()):
        link["capacity"] = cap
    data["graph"] = {
        "name": topology.name,
        "source": topology.source,
        "capacity_note": topology.capacity_note,
    }
    (TOPO_DIR / f"{topology.name}.json").write_text(json.dumps(data), encoding="utf-8")
    src = np.asarray([u for u, _ in topology.edges], dtype=np.int32)
    dst = np.asarray([v for _, v in topology.edges], dtype=np.int32)
    np.savez_compressed(
        GRAPH_DIR / f"{topology.name}_graph.npz",
        edge_src=src,
        edge_dst=dst,
        capacity=topology.capacities,
    )


def _load_spec(spec: dict) -> Topology:
    name = spec["name"]
    if spec["family"] == "teal":
        path = RAW / "teal" / "topologies" / spec["file"]
        data = json.loads(path.read_text(encoding="utf-8"))
        return from_node_link(name, data, str(path), "Capacities copied from the TEAL topology file.")
    if spec["family"] == "figret":
        path = RAW / "figret" / "Data" / "GEANT" / "GEANT.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        note = (
            "FIGRET file units (capacities such as 1e10, 2.4e9, 1.55e8). "
            "FIGRET divides by 1e9 at training time. That conversion is not applied here; "
            "P1's scale, when used, divides by the maximum capacity of this graph."
        )
        return from_node_link(name, data, str(path), note)
    if spec["family"] == "zoo":
        path = RAW / "topology_zoo" / "gml" / "Cogentco.gml"
        return from_zoo_gml(name, path.read_text(encoding="utf-8", errors="ignore"), str(path), COGENTCO_PLACEHOLDER)
    raise ValueError(f"Unknown dataset family for {name}")


def _load_traffic(spec: dict, topology: Topology) -> dict:
    name = spec["name"]
    n = topology.num_nodes
    if spec["traffic"] == "real" and spec["family"] == "teal":
        bundle = load_b4(RAW / "teal" / "traffic-matrices" / "real", n)
    elif spec["traffic"] == "toy":
        bundle = load_toy(RAW / "teal" / "traffic-matrices" / "toy", spec["prefix"], n)
    elif spec["family"] == "figret":
        bundle = load_geant(RAW / "figret" / "Data" / "GEANT", n)
    elif spec["traffic"] == "reconstructed":
        bundle = load_gravity(topology.graph, n)
    else:
        raise ValueError(f"{name}: no traffic loader")
    bundle["summary"] = summarise_checks(bundle["checks"])
    return bundle


def _original_stack(bundle: dict) -> np.ndarray | None:
    if bundle.get("demand_original") is not None:
        return bundle["demand_original"]
    if "demand_train" in bundle:
        return np.concatenate([bundle["demand_train"], bundle["demand_test"]], axis=0)
    return None


def _save_traffic(name: str, bundle: dict) -> None:
    skip = {"demand_original", "demand_train", "demand_test", "template", "checks", "opt_mlu_train", "opt_mlu_test"}
    summary = {k: v for k, v in bundle.items() if k not in skip and not isinstance(v, np.ndarray)}
    summary["summary"] = bundle["summary"]
    summary["invalid_matrix_reasons"] = [
        {"index": i, "reasons": c["reasons"]} for i, c in enumerate(bundle["checks"]) if not c["valid"]
    ]
    if bundle["kind"] == "real_mapped":
        out = TM_DIR / "real" / f"{name}.npz"
        np.savez_compressed(out, demand_original=bundle["demand_original"])
        summary["file"] = str(out)
        dump_json(TM_DIR / "real" / f"{name}.json", summary)
    elif bundle["kind"] == "real_figret":
        out = TM_DIR / "real" / f"{name}.npz"
        np.savez_compressed(
            out,
            demand_train=bundle["demand_train"],
            demand_test=bundle["demand_test"],
            opt_mlu_train=bundle["opt_mlu_train"],
            opt_mlu_test=bundle["opt_mlu_test"],
        )
        summary["file"] = str(out)
        dump_json(TM_DIR / "real" / f"{name}.json", summary)
    elif bundle["kind"] == "synthetic_toy":
        out = TM_DIR / "toy" / f"{name}_template.npz"
        np.savez_compressed(out, demand_template=bundle["template"])
        summary["file"] = str(out)
        summary["preserved_as"] = "one template; every source file matched this array" if bundle["identical_files"] else "files differ; see warnings"
        dump_json(TM_DIR / "toy" / f"{name}.json", summary)
    else:
        out = TM_DIR / "reconstructed" / f"{name}_gravity.npz"
        np.savez_compressed(out, demand_original=bundle["demand_original"])
        summary["file"] = str(out)
        dump_json(TM_DIR / "reconstructed" / f"{name}_gravity.json", summary)


def _od(stack: np.ndarray) -> np.ndarray:
    n = stack.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return np.asarray(stack[:, mask] if stack.ndim == 3 else stack[mask], dtype=np.float64)


def _save_normalisation(name: str, topology: Topology, bundle: dict) -> dict:
    params = normalize.parameters(topology.capacities)
    denom = params["denominator"]
    cap_norm = normalize.apply_capacity(topology.capacities, denom)
    params["topology"] = name
    params["split"] = bundle["split"]
    params["traffic_kind"] = bundle["kind"]
    stack = _original_stack(bundle)
    payload = {"capacity_normalised": cap_norm}
    if stack is not None:
        norm = normalize.apply_demand(stack, denom)
        payload["demand_normalised"] = norm.astype(np.float32)
        payload["demand_original_od"] = _od(stack)
        payload["demand_normalised_od"] = _od(norm).astype(np.float32)
    elif bundle["kind"] == "synthetic_toy":
        template = bundle["template"]
        od = _od(template[None, ...])[0]
        params["toy_original_off_diagonal"] = float(od.flat[0]) if od.size else None
        params["toy_normalised_off_diagonal"] = float(od.flat[0]) / denom if od.size else None
        params["toy_files_identical"] = bool(bundle["identical_files"])
        if not bundle["identical_files"]:
            raise ValueError(f"{name}: toy files are not identical, so a single template would hide differences")
    dump_json(NORM_DIR / f"{name}.json", params)
    np.savez_compressed(NORM_DIR / f"{name}.npz", **payload)
    params["capacity_normalised"] = cap_norm
    return params


def _save_paths(name: str, paths: dict) -> None:
    arrays = {key: value for key, value in paths.items() if key != "meta"}
    np.savez_compressed(PATH_DIR / f"{name}_yen4.npz", **arrays)
    dump_json(PATH_DIR / f"{name}_yen4.json", paths["meta"])
    for stale in (PATH_DIR / f"{name}_yen4_minhop.pkl", PATH_DIR / f"{name}_yen4_minhop.json"):
        if stale.exists():
            stale.unlink()


def _save_bipartite(name: str, topology: Topology, paths: dict, params: dict, bundle: dict) -> dict:
    links = incidence(topology, paths)
    stack = _original_stack(bundle)
    record = {
        "format": 1,
        "topology": name,
        "feature_columns": FEATURE_COLUMNS,
        "traffic_kind": bundle["kind"],
        "normalisation_method": params["method"],
        "denominator": params["denominator"],
        "link_src": torch.from_numpy(np.asarray([u for u, _ in topology.edges], dtype=np.int32)),
        "link_dst": torch.from_numpy(np.asarray([v for _, v in topology.edges], dtype=np.int32)),
        "capacity_original": torch.from_numpy(topology.capacities.astype(np.float64)),
        "capacity_normalised": torch.from_numpy(params["capacity_normalised"].astype(np.float32)),
        "demand_src": torch.from_numpy(paths["demand_src"]),
        "demand_dst": torch.from_numpy(paths["demand_dst"]),
        "n_paths": torch.from_numpy(paths["n_paths"]),
        "hop_count": torch.from_numpy(paths["hop_count"]),
        "path_node_ptr": torch.from_numpy(paths["path_node_ptr"]),
        "path_nodes": torch.from_numpy(paths["path_nodes"]),
        "incidence_path": torch.from_numpy(links["path_index"]),
        "incidence_link": torch.from_numpy(links["link_index"]),
        "demand_row": torch.from_numpy(links["demand_row"]),
        "edge_index_note": (
            "incidence_path is the path-node index and incidence_link is the link-node index. "
            "Link nodes occupy 0..E-1 and path nodes occupy E..E+P-1, as in P1. "
            "The undirected edge_index is [path+E, link] and [link, path+E]."
        ),
    }
    if stack is not None:
        record["demand_original_od"] = torch.from_numpy(_od(stack))
        record["demand_normalised_od"] = torch.from_numpy(_od(normalize.apply_demand(stack, params["denominator"])).astype(np.float32))
    else:
        record["toy_original_off_diagonal"] = params["toy_original_off_diagonal"]
        record["toy_normalised_off_diagonal"] = params["toy_normalised_off_diagonal"]
        record["n_matrices"] = bundle["summary"]["matrices"]
    dest = GRAPH_DIR / f"{name}_bipartite.pt"
    torch.save(record, dest)
    return {
        "file": str(dest),
        "link_nodes": topology.num_edges,
        "path_nodes": int(paths["n_paths"].astype(np.int64).sum()),
        "incidence_edges_one_direction": int(links["path_index"].size),
        "matrices": bundle["summary"]["matrices"],
    }


def _row(topology: Topology, conn: dict, bundle: dict, paths: dict, bip: dict, params: dict, path_problems: list[str]) -> dict:
    checks = bundle["checks"]
    # A valid demand here is an ordered pair with a finite, non-negative volume.
    # Self-demand is not an OD demand. Zero volume is valid and counted separately.
    zero = int(sum(c["zero_off_diagonal"] for c in checks))
    positive = int(sum(c["positive_off_diagonal"] for c in checks))
    invalid_entries = int(sum(c["negative_entries"] + c["non_finite_entries"] for c in checks))
    return {
        "topology": topology.name,
        "nodes": conn["nodes"],
        "directed_links": conn["directed_edges"],
        "undirected_links": conn["undirected_edges"],
        "connectivity": {
            "weakly_connected": conn["weakly_connected"],
            "strongly_connected": conn["strongly_connected"],
            "undirected_connected": conn["undirected_connected"],
            "weak_component_sizes": conn["weak_component_sizes"],
            "undirected_component_sizes": conn["undirected_component_sizes"],
            "ordered_pairs_without_a_directed_path": conn["ordered_pairs_without_a_directed_path"],
        },
        "traffic_kind": bundle["kind"],
        "traffic_label": bundle["label"],
        "traffic_matrices": bundle["summary"]["matrices"],
        "valid_traffic_matrices": bundle["summary"]["valid_matrices"],
        "invalid_traffic_matrices": bundle["summary"]["invalid_matrices"],
        "valid_od_demands": positive + zero,
        "positive_od_demands": positive,
        "zero_od_demands": zero,
        "invalid_od_entries": invalid_entries,
        "self_demand_nonzero_entries": bundle["summary"]["self_demand_nonzero_entries"],
        "candidate_paths": paths["meta"]["paths_generated"],
        "demands_with_fewer_than_4_paths": paths["meta"]["demands_with_fewer_than_4"],
        "demands_with_no_path": paths["meta"]["demands_with_no_path"],
        "path_backend": paths["meta"]["backend"],
        "paths_padded": False,
        "path_validation_problems": path_problems[:20],
        "path_validation_problem_count": len(path_problems),
        "normalisation": {
            "method": params["method"],
            "denominator": params["denominator"],
            "uses_traffic_matrices": params["uses_traffic_matrices"],
            "p1_training_default": params["p1_training_default"],
        },
        "bipartite": bip,
        "excluded_links": topology.excluded,
        "warnings": topology.warnings + ([] if bundle.get("identical_files", True) else [f"{topology.name} toy files are not identical"]),
    }


def process_one(spec: dict, jobs: int, log: logging.Logger, reuse_paths: bool = False) -> dict:
    name = spec["name"]
    log.info("Loading %s", name)
    topology = _load_spec(spec)
    conn = connectivity_report(topology)
    _save_topology(topology)
    dump_json(GRAPH_DIR / f"{name}_validation.json", {**conn, "capacity_note": topology.capacity_note, "source": topology.source})
    log.info(
        "%s: %s nodes, %s directed links, undirected connected=%s",
        name,
        conn["nodes"],
        conn["directed_edges"],
        conn["undirected_connected"],
    )
    bundle = _load_traffic(spec, topology)
    _save_traffic(name, bundle)
    log.info(
        "%s traffic %s: %s matrices, %s invalid",
        name,
        bundle["kind"],
        bundle["summary"]["matrices"],
        bundle["summary"]["invalid_matrices"],
    )
    params = _save_normalisation(name, topology, bundle)
    path_file = PATH_DIR / f"{name}_yen4.npz"
    bip_file = GRAPH_DIR / f"{name}_bipartite.pt"
    if reuse_paths and path_file.exists():
        log.info("%s reusing existing paths from %s", name, path_file)
        paths = load_saved(path_file)
        problems = []
        if bip_file.exists() and (PATH_DIR / f"{name}_yen4.json").exists():
            log.info("%s reusing existing bipartite graph from %s", name, bip_file)
            bip = {
                "file": str(bip_file),
                "link_nodes": topology.num_edges,
                "path_nodes": int(paths["n_paths"].astype(np.int64).sum()),
                "incidence_edges_one_direction": int(paths["hop_count"].astype(np.int64).sum()),
                "matrices": bundle["summary"]["matrices"],
                "reused": True,
            }
            return _row(topology, conn, bundle, paths, bip, params, problems)
    else:
        log.info("%s paths (%s workers)", name, jobs)
        paths = generate(topology, jobs=jobs, checkpoint=True)
        problems = validate(topology, paths)
        if problems:
            log.warning("%s path validation reported %s problems; first: %s", name, len(problems), problems[0])
        _save_paths(name, paths)
    log.info(
        "%s paths=%s fewer_than_4=%s",
        name,
        paths["meta"]["paths_generated"],
        paths["meta"]["demands_with_fewer_than_4"],
    )
    bip = _save_bipartite(name, topology, paths, params, bundle)
    return _row(topology, conn, bundle, paths, bip, params, problems)


def _markdown(rows: list[dict], failures: list[dict]) -> str:
    lines = [
        "# Phase 3 preprocessing report",
        "",
        "Raw datasets were read only. Normalisation uses P1's optional formula",
        "`x / max(link capacity)`. The frozen B4 baseline does not apply that scale",
        "(`H0_SCALING = False`); both the raw value and the scaled value are stored.",
        "Paths are hop-shortest and are not repeated to force a count of four.",
        "",
        "| Topology | Nodes | Directed links | Connected | Matrices | Invalid matrices | Paths | Fewer than 4 paths | Denominator |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        conn = "yes" if row["connectivity"]["undirected_connected"] else "NO"
        lines.append(
            f"| {row['topology']} | {row['nodes']} | {row['directed_links']} | {conn} | "
            f"{row['traffic_matrices']} | {row['invalid_traffic_matrices']} | "
            f"{row['candidate_paths']} | {row['demands_with_fewer_than_4_paths']} | "
            f"{row['normalisation']['denominator']:g} |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"## {row['topology']}")
        lines.append("")
        lines.append(f"Traffic: {row['traffic_kind']}. {row['traffic_label']}")
        lines.append(
            f"Valid OD entries (finite and non-negative, including zeros): {row['valid_od_demands']}. "
            f"Of those, {row['zero_od_demands']} are zero and {row['positive_od_demands']} are positive. "
            f"Invalid OD entries: {row['invalid_od_entries']}. "
            f"Non-zero self-demand entries: {row['self_demand_nonzero_entries']}."
        )
        lines.append(
            f"Path backend: {row['path_backend']}. "
            f"Demands with no path: {row['demands_with_no_path']}. "
            f"Path checks failed: {row['path_validation_problem_count']}."
        )
        lines.append(
            f"Bipartite file: link nodes {row['bipartite']['link_nodes']}, "
            f"path nodes {row['bipartite']['path_nodes']}, "
            f"one-direction incidences {row['bipartite']['incidence_edges_one_direction']}."
        )
        if row["warnings"]:
            lines.append("Warnings:")
            for warning in row["warnings"]:
                lines.append(f"- {warning}")
        if row["excluded_links"]:
            lines.append(f"Excluded link records: {len(row['excluded_links'])} (listed in the JSON report).")
        lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for fail in failures:
            lines.append(f"- {fail['topology']}: {fail['error']}")
        lines.append("")
    lines.append("## Not processed")
    lines.append("")
    lines.append("- FIGRET Facebook pod topologies are in the raw dump and are not part of the Phase 2 dataset list.")
    lines.append("- Topology Zoo GÉANT graphs are not the 22-node FIGRET graph and are not processed.")
    lines.append("- Kdl and ASN have no real traffic. Their toy matrices are labelled synthetic.")
    lines.append("")
    lines.append("## Difference from P1 path padding")
    lines.append("")
    lines.append(
        "P1 `preprocessing/generate_paths.py` repeats the shortest path when a pair has fewer than four simple paths. "
        "This pipeline records the actual number and does not repeat a path."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 Step 3 preprocessing")
    parser.add_argument("--only", nargs="*", default=[], help="Topology names to process. Default: all six.")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--reuse-paths",
        action="store_true",
        help="Reuse existing *_yen4.npz / bipartite.pt when present instead of regenerating paths.",
    )
    args = parser.parse_args()
    for path in (TOPO_DIR, TM_DIR / "real", TM_DIR / "toy", TM_DIR / "reconstructed", PATH_DIR, GRAPH_DIR, NORM_DIR):
        path.mkdir(parents=True, exist_ok=True)
    log = _logger()
    selected = set(args.only) or {spec["name"] for spec in SPECS}
    unknown = selected - {spec["name"] for spec in SPECS}
    if unknown:
        raise SystemExit(f"Unknown topology names: {sorted(unknown)}")
    rows = []
    failures = []
    for spec in SPECS:
        if spec["name"] not in selected:
            continue
        try:
            rows.append(process_one(spec, args.jobs, log, reuse_paths=args.reuse_paths))
        except Exception as exc:
            log.error("%s failed: %s", spec["name"], exc)
            log.error(traceback.format_exc())
            failures.append({"topology": spec["name"], "error": str(exc)})
    report_path = PROCESSED / "preprocessing_report.json"
    if args.only and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        by_name = {row["topology"]: row for row in previous.get("rows", [])}
        for row in rows:
            by_name[row["topology"]] = row
        done = {row["topology"] for row in rows}
        fail_by = {item["topology"]: item for item in previous.get("failures", []) if item["topology"] not in done}
        for item in failures:
            fail_by[item["topology"]] = item
        order = [spec["name"] for spec in SPECS]
        rows = [by_name[name] for name in order if name in by_name]
        failures = [fail_by[name] for name in order if name in fail_by]
    finished = {row["topology"] for row in rows}
    if {"Kdl", "ASN"} <= finished and not any(item["topology"] in {"Kdl", "ASN"} for item in failures):
        deferred = PATH_DIR / "Kdl_ASN_DEFERRED.json"
        if deferred.exists():
            deferred.unlink()
    report = {"rows": rows, "failures": failures}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (PROCESSED / "PREPROCESSING_REPORT.md").write_text(_markdown(rows, failures), encoding="utf-8")
    log.info("Wrote %s", PROCESSED / "PREPROCESSING_REPORT.md")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

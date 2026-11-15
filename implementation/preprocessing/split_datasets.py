"""Step 4: topology hold-out plus time-ordered traffic splits.

Ready topologies only (ASN excluded until path merge finishes).

Roles
- train: B4, UsCarrier, Cogentco with early traffic indices [0, 20)
- val:   same topologies with mid indices [20, 28) for tuning
- test:  unseen topologies GEANT + Kdl (ASN pending)

GEANT uses FIGRET's published test histories only. FIGRET train rows are
written under held_aside/ so they cannot leak into Phase 3 training.
Train-topology late indices [28, 36) are also held aside as future traffic
on seen graphs (not mixed into train/val).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .traffic import B4_SPLIT, dump_json

from repo_paths import REPO as ROOT
PROCESSED = ROOT / "data" / "processed"
SPLIT_ROOT = PROCESSED / "splits"

# Topologies with complete paths + bipartite artifacts.
READY = ("B4", "UsCarrier", "GEANT", "Cogentco", "Kdl")
PENDING = ("ASN",)

TRAIN_TOPOLOGIES = ("B4", "UsCarrier", "Cogentco")
VAL_TOPOLOGIES = ("B4", "UsCarrier", "Cogentco")
TEST_TOPOLOGIES = ("GEANT", "Kdl")

TEAL_TRAIN = tuple(B4_SPLIT["train"])
TEAL_VAL = tuple(B4_SPLIT["val"])
TEAL_FUTURE = tuple(B4_SPLIT["test"])


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _artifact_map(name: str) -> dict:
    topo = PROCESSED / "topologies" / f"{name}.json"
    paths = PROCESSED / "candidate_paths" / f"{name}_yen4.npz"
    paths_meta = PROCESSED / "candidate_paths" / f"{name}_yen4.json"
    bip = PROCESSED / "graph_data" / f"{name}_bipartite.pt"
    norm = PROCESSED / "normalization" / f"{name}.json"
    if name == "B4":
        tm = PROCESSED / "traffic_matrices" / "real" / "B4.npz"
        kind = "real_mapped"
    elif name == "GEANT":
        tm = PROCESSED / "traffic_matrices" / "real" / "GEANT.npz"
        kind = "real_figret"
    elif name == "Cogentco":
        tm = PROCESSED / "traffic_matrices" / "reconstructed" / "Cogentco_gravity.npz"
        kind = "reconstructed_gravity"
    elif name in ("UsCarrier", "Kdl", "ASN"):
        tm = PROCESSED / "traffic_matrices" / "toy" / f"{name}_template.npz"
        kind = "synthetic_toy"
    else:
        raise KeyError(name)
    return {
        "topology": name,
        "traffic_kind": kind,
        "topology_file": _rel(topo),
        "paths_file": _rel(paths),
        "paths_meta_file": _rel(paths_meta),
        "bipartite_file": _rel(bip),
        "normalization_file": _rel(norm),
        "traffic_source_file": _rel(tm),
        "present": all(p.exists() for p in (topo, paths, bip, tm)),
    }


def _load_series(name: str) -> tuple[np.ndarray, dict]:
    info = _artifact_map(name)
    path = ROOT / info["traffic_source_file"]
    data = np.load(path)
    if info["traffic_kind"] == "real_figret":
        return data["demand_test"], {
            "source_keys": ["demand_test"],
            "n_available": int(data["demand_test"].shape[0]),
            "figret_train_rows": int(data["demand_train"].shape[0]),
            "opt_mlu_test": data["opt_mlu_test"],
            "demand_train_held_aside": data["demand_train"],
            "opt_mlu_train_held_aside": data["opt_mlu_train"],
        }
    if info["traffic_kind"] == "synthetic_toy":
        template = np.asarray(data["demand_template"], dtype=np.float64)
        stack = np.stack([template for _ in range(36)])
        return stack, {
            "source_keys": ["demand_template"],
            "n_available": 36,
            "identical_matrices": True,
            "note": "Toy files are identical; index cuts are protocol only.",
        }
    stack = np.asarray(data["demand_original"], dtype=np.float64)
    return stack, {"source_keys": ["demand_original"], "n_available": int(stack.shape[0])}


def _slice_indices(start: int, stop: int) -> list[int]:
    return list(range(start, stop))


def _write_topology_bundle(
    out_dir: Path,
    topology: str,
    demands: np.ndarray,
    meta: dict,
    *,
    stem: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or topology
    demand_path = out_dir / f"{stem}_demand.npz"
    payload = {"demand": np.asarray(demands, dtype=np.float64)}
    if "opt_mlu" in meta:
        payload["opt_mlu"] = np.asarray(meta["opt_mlu"], dtype=np.float64)
    np.savez_compressed(demand_path, **payload)
    arts = _artifact_map(topology)
    record = {
        **arts,
        "entry_name": stem,
        "split_role": meta["split_role"],
        "matrix_count": int(demands.shape[0]),
        "matrix_shape": list(demands.shape),
        "matrix_indices": meta.get("matrix_indices"),
        "index_range": meta.get("index_range"),
        "time_split_note": meta.get("time_split_note"),
        "demand_file": _rel(demand_path),
        "mean_total_demand": float(demands.sum(axis=(1, 2)).mean()) if demands.size else None,
    }
    dump_json(out_dir / f"{stem}.json", record)
    return record


def build_splits() -> dict:
    for name in READY:
        arts = _artifact_map(name)
        if not arts["present"]:
            raise FileNotFoundError(f"{name} is marked ready but artifacts are incomplete: {arts}")

    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)
    for part in ("train", "val", "test", "held_aside"):
        (SPLIT_ROOT / part).mkdir(parents=True)

    train_rows = []
    val_rows = []
    test_rows = []
    held_rows = []

    for name in TRAIN_TOPOLOGIES:
        stack, extra = _load_series(name)
        train = stack[TEAL_TRAIN[0] : TEAL_TRAIN[1]]
        val = stack[TEAL_VAL[0] : TEAL_VAL[1]]
        future = stack[TEAL_FUTURE[0] : TEAL_FUTURE[1]]
        train_rows.append(
            _write_topology_bundle(
                SPLIT_ROOT / "train",
                name,
                train,
                {
                    "split_role": "train",
                    "matrix_indices": _slice_indices(*TEAL_TRAIN),
                    "index_range": list(TEAL_TRAIN),
                    "time_split_note": (
                        "Frozen P1/TEAL 20/8/8 early slice. Future indices are excluded."
                        if name == "B4"
                        else extra.get("note", "TEAL-style early slice; see traffic status notes.")
                    ),
                },
            )
        )
        val_rows.append(
            _write_topology_bundle(
                SPLIT_ROOT / "val",
                name,
                val,
                {
                    "split_role": "val",
                    "matrix_indices": _slice_indices(*TEAL_VAL),
                    "index_range": list(TEAL_VAL),
                    "time_split_note": "Mid slice for hyperparameter tuning only; never used as final test.",
                },
            )
        )
        held_rows.append(
            _write_topology_bundle(
                SPLIT_ROOT / "held_aside",
                name,
                future,
                {
                    "split_role": "held_aside_seen_topology_future_traffic",
                    "matrix_indices": _slice_indices(*TEAL_FUTURE),
                    "index_range": list(TEAL_FUTURE),
                    "time_split_note": (
                        "Late indices on train topologies. Kept out of train/val to prevent "
                        "future-traffic leakage. Not a topology-generalization test."
                    ),
                },
                stem=f"{name}_future_traffic",
            )
        )

    # Unseen test topologies.
    geant_stack, geant_extra = _load_series("GEANT")
    test_rows.append(
        _write_topology_bundle(
            SPLIT_ROOT / "test",
            "GEANT",
            geant_stack,
            {
                "split_role": "test_unseen_topology",
                "matrix_indices": list(range(int(geant_stack.shape[0]))),
                "index_range": [0, int(geant_stack.shape[0])],
                "time_split_note": (
                    "FIGRET published test histories only. FIGRET train rows are held aside."
                ),
                "opt_mlu": geant_extra["opt_mlu_test"],
            },
        )
    )
    held_rows.append(
        _write_topology_bundle(
            SPLIT_ROOT / "held_aside",
            "GEANT",
            geant_extra["demand_train_held_aside"],
            {
                "split_role": "held_aside_figret_train",
                "matrix_indices": list(range(int(geant_extra["figret_train_rows"]))),
                "index_range": [0, int(geant_extra["figret_train_rows"])],
                "time_split_note": (
                    "FIGRET train histories on the unseen GEANT topology. Not used for "
                    "Phase 3 train/val/test so the topology remains held out."
                ),
                "opt_mlu": geant_extra["opt_mlu_train_held_aside"],
            },
            stem="GEANT_figret_train",
        )
    )

    kdl_stack, kdl_extra = _load_series("Kdl")
    kdl_test = kdl_stack[TEAL_FUTURE[0] : TEAL_FUTURE[1]]
    test_rows.append(
        _write_topology_bundle(
            SPLIT_ROOT / "test",
            "Kdl",
            kdl_test,
            {
                "split_role": "test_unseen_topology",
                "matrix_indices": _slice_indices(*TEAL_FUTURE),
                "index_range": list(TEAL_FUTURE),
                "time_split_note": (
                    "Unseen topology. Uses late index convention only; toy matrices are identical."
                ),
            },
        )
    )
    # Keep early/mid Kdl indices out of any train/val path.
    held_rows.append(
        _write_topology_bundle(
            SPLIT_ROOT / "held_aside",
            "Kdl",
            kdl_stack[: TEAL_FUTURE[0]],
            {
                "split_role": "held_aside_unseen_topology_unused_indices",
                "matrix_indices": _slice_indices(0, TEAL_FUTURE[0]),
                "index_range": [0, TEAL_FUTURE[0]],
                "time_split_note": (
                    "Kdl is test-only. Early/mid indices are unused so the topology never "
                    "appears in train or val. Toy matrices are identical."
                ),
            },
            stem="Kdl_early_mid_unused",
        )
    )

    asn_shards = PROCESSED / "candidate_paths" / "ASN_shards"
    shard_count = len(list(asn_shards.glob("src_*.npz"))) if asn_shards.exists() else 0
    pending = {
        "ASN": {
            "status": "pending_incomplete",
            "reason": "Path merge and bipartite graph were not finished. Preprocessing paused.",
            "nodes_expected": 1739,
            "shards_present": shard_count,
            "shards_missing": max(0, 1739 - shard_count),
            "merged_paths": False,
            "bipartite": False,
            "shard_dir": _rel(asn_shards),
            "note": "Existing ASN_shards checkpoints must be preserved; do not delete.",
        }
    }
    dump_json(SPLIT_ROOT / "pending.json", pending)

    manifest = {
        "step": 4,
        "policy": {
            "train_topologies": list(TRAIN_TOPOLOGIES),
            "val_topologies": list(VAL_TOPOLOGIES),
            "test_topologies": list(TEST_TOPOLOGIES),
            "pending_topologies": list(PENDING),
            "ready_topologies": list(READY),
            "teal_index_split": {"train": list(TEAL_TRAIN), "val": list(TEAL_VAL), "future_held_aside": list(TEAL_FUTURE)},
            "rules": [
                "Training uses only selected topologies.",
                "Validation uses separate later traffic indices on those same topologies.",
                "Testing uses completely unseen topologies (GEANT, Kdl).",
                "Future traffic on train topologies is held aside and not used in train/val.",
                "ASN is excluded until preprocessing completes.",
            ],
        },
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "held_aside": held_rows,
        "pending": pending,
        "counts": {
            "train_topology_entries": len(train_rows),
            "val_topology_entries": len(val_rows),
            "test_topology_entries": len(test_rows),
            "train_matrices": int(sum(r["matrix_count"] for r in train_rows)),
            "val_matrices": int(sum(r["matrix_count"] for r in val_rows)),
            "test_matrices": int(sum(r["matrix_count"] for r in test_rows)),
        },
    }
    dump_json(SPLIT_ROOT / "SPLIT_MANIFEST.json", manifest)
    (SPLIT_ROOT / "SPLIT_REPORT.md").write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _markdown(manifest: dict) -> str:
    p = manifest["policy"]
    c = manifest["counts"]
    lines = [
        "# Phase 3 Step 4 dataset splits",
        "",
        "ASN is pending and is not included in train, validation, or test.",
        "",
        "## Topology roles",
        "",
        f"- **Train topologies:** {', '.join(p['train_topologies'])}",
        f"- **Validation topologies (tuning scenarios):** {', '.join(p['val_topologies'])} with later traffic indices",
        f"- **Test topologies (unseen):** {', '.join(p['test_topologies'])}",
        f"- **Pending:** {', '.join(p['pending_topologies'])}",
        "",
        "## Time / index cuts",
        "",
        f"- TEAL-style series: train `{p['teal_index_split']['train']}`, "
        f"val `{p['teal_index_split']['val']}`, future held aside `{p['teal_index_split']['future_held_aside']}`",
        "- GEANT test uses FIGRET published test histories only; FIGRET train is held aside",
        "- Kdl is test-only; early/mid toy indices are unused",
        "",
        "## Matrix counts",
        "",
        f"- Train matrices: {c['train_matrices']}",
        f"- Validation matrices: {c['val_matrices']}",
        f"- Test matrices: {c['test_matrices']}",
        "",
        "## Output layout",
        "",
        "```",
        "data/processed/splits/train/",
        "data/processed/splits/val/",
        "data/processed/splits/test/",
        "data/processed/splits/held_aside/",
        "data/processed/splits/SPLIT_MANIFEST.json",
        "```",
        "",
        "Each topology folder entry stores a demand `.npz` slice plus a JSON pointer to",
        "the shared topology, path, bipartite, and normalisation artifacts.",
        "",
    ]
    asn = manifest["pending"]["ASN"]
    lines.extend(
        [
            "## ASN pending",
            "",
            f"- Shards present: {asn['shards_present']} / {asn['nodes_expected']}",
            f"- Missing sources: {asn['shards_missing']}",
            "- Merged `ASN_yen4.npz` / `ASN_bipartite.pt`: not created",
            "- Checkpoints under `candidate_paths/ASN_shards/` are preserved",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = build_splits()
    print(json.dumps(manifest["counts"], indent=2))
    print("Wrote", SPLIT_ROOT / "SPLIT_MANIFEST.json")


if __name__ == "__main__":
    main()

"""Multi-seed robustness evaluation of Phase 3 routing policies.

Compares frozen P1, the original 3-epoch generalized actor, both improved
checkpoints, shortest-path, equal-split, and the full twin+gate framework
under identical traffic matrices per seed.

GEANT and Kdl are read only from their test splits. No training or tuning
happens here. Artifacts write under ``results/evaluation/robustness_v1/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safety.evaluate import load_config, load_trained_ensemble
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import demand_per_path, equal_split, link_load, od_demands, shortest_split

from .general_controller import GeneralController
from .general_policy import CHECKPOINT, GeneralPolicy, IMPROVED_CHECKPOINT
from .pipeline import Framework
from .policy_adapter import DEFAULT_CHECKPOINT, FrozenP1Policy, PolicyGraph, build_policy_graph

# Built once per topology; rebuilding Kdl's edge table (~1.6 GiB) repeatedly OOMs.
_GRAPH_CACHE: dict[str, PolicyGraph] = {}


def _shared_graph(topo, device: torch.device) -> PolicyGraph:
    cached = _GRAPH_CACHE.get(topo.name)
    if cached is None:
        gc.collect()
        cached = build_policy_graph(topo)
        cached.edge_index = cached.edge_index.to(device)
        _GRAPH_CACHE[topo.name] = cached
        gc.collect()
    return cached


def _install_shared_graph(policy, topo, device: torch.device) -> None:
    policy._graphs[topo.name] = _shared_graph(topo, device)

RESULT_DIR = ROOT / "results" / "evaluation" / "robustness_v1"
SEEDS = (101, 202, 303, 404, 505)
SCALES = (0.25, 0.5, 0.75, 1.0)

# Per-topology draw size. Matches the prior extended-eval budget so Kdl stays tractable.
TOPOLOGY_SPEC = (
    ("B4", "val", 2),
    ("UsCarrier", "val", 1),
    ("Cogentco", "val", 1),
    ("GEANT", "test", 2),
    ("Kdl", "test", 1),
)

IMPROVED_BEST = ROOT / "models" / "policy" / "improved" / "best_checkpoint.pt"
IMPROVED_STABLE = ROOT / "models" / "policy" / "improved" / "recommended_stable.pt"
BASELINE_GENERAL = CHECKPOINT

LEARNED_ACTORS = (
    ("frozen_p1", "frozen"),
    ("general_3epoch", "general"),
    ("improved_best", "general"),
    ("improved_stable", "general"),
)


def _mlu(load: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.max(load / np.maximum(capacity, 1e-12)))


def _violations(load: np.ndarray, capacity: np.ndarray) -> int:
    return int(np.count_nonzero(load > capacity + 1e-6))


def _load_action(topo, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    return link_load(
        demand_per_path(od_demands(matrix), topo.demand_row),
        action,
        topo.incidence_path,
        topo.incidence_link,
        topo.n_links,
    )


def sample_matrix_indices(n_available: int, count: int, seed: int, topology: str) -> list[int]:
    """Seeded sample without replacement; deterministic across methods."""
    if count > n_available:
        raise ValueError(f"{topology}: requested {count} matrices but only {n_available} available")
    rng = np.random.default_rng(seed + 17 * (sum(map(ord, topology)) % 997))
    chosen = rng.choice(n_available, size=count, replace=False)
    return sorted(int(i) for i in chosen)


def build_catalog(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    catalog: dict[str, Any] = {"seeds": list(seeds), "topologies": {}}
    for name, split, count in TOPOLOGY_SPEC:
        demands, meta = load_split_demands(split, name)
        per_seed = {}
        for seed in seeds:
            local = sample_matrix_indices(demands.shape[0], count, seed, name)
            per_seed[str(seed)] = {
                "local_indices": local,
                "matrix_ids": [int(meta["matrix_indices"][i]) for i in local],
            }
        catalog["topologies"][name] = {
            "split": split,
            "count": count,
            "n_available": int(demands.shape[0]),
            "unseen": name in ("GEANT", "Kdl"),
            "by_seed": per_seed,
        }
    return catalog


def _physics_stats(topo, matrix: np.ndarray, action: np.ndarray) -> dict[str, float | int]:
    load = _load_action(topo, matrix, action)
    return {
        "mlu": _mlu(load, topo.capacity),
        "violations": _violations(load, topo.capacity),
        "runtime_s": 0.0,
    }


def _feasibility(topo, matrix: np.ndarray, short_action: np.ndarray, equal_action: np.ndarray) -> dict:
    short_load = _load_action(topo, matrix, short_action)
    equal_load = _load_action(topo, matrix, equal_action)
    short_v = _violations(short_load, topo.capacity)
    equal_v = _violations(equal_load, topo.capacity)
    feasible_any = short_v == 0 or equal_v == 0
    return {
        "mlu_shortest": _mlu(short_load, topo.capacity),
        "mlu_equal": _mlu(equal_load, topo.capacity),
        "violations_shortest": short_v,
        "violations_equal": equal_v,
        "feasible_shortest": short_v == 0,
        "feasible_equal": equal_v == 0,
        "feasible_any_baseline": feasible_any,
        "scenario": "feasible" if feasible_any else "infeasible",
    }


def _actor_paths() -> dict[str, Path]:
    return {
        "frozen_p1": Path(DEFAULT_CHECKPOINT),
        "general_3epoch": Path(BASELINE_GENERAL),
        "improved_best": Path(IMPROVED_BEST),
        "improved_stable": Path(IMPROVED_STABLE),
    }


def _load_general(path: Path, device: torch.device) -> GeneralPolicy:
    policy = GeneralPolicy(device)
    policy.load(path)
    policy.model.eval()
    return policy


def _case_bases(seed: int, catalog: dict) -> list[dict]:
    """Physics baselines and shared case metadata; no learned policies loaded."""
    rows: list[dict] = []
    for name, split, _count in TOPOLOGY_SPEC:
        topo = load_topology(name)
        demands, meta = load_split_demands(split, name)
        entry = catalog["topologies"][name]["by_seed"][str(seed)]
        short_action = shortest_split(topo.n_paths, topo.hops)
        equal_action = equal_split(topo.n_paths)
        for local_i in entry["local_indices"]:
            matrix = demands[local_i]
            matrix_id = int(meta["matrix_indices"][local_i])
            for scale in SCALES:
                scaled = matrix * float(scale)
                feas = _feasibility(topo, scaled, short_action, equal_action)
                base = {
                    "seed": seed,
                    "topology": name,
                    "split": split,
                    "unseen": name in ("GEANT", "Kdl"),
                    "local_index": local_i,
                    "matrix_id": matrix_id,
                    "scale": float(scale),
                    **feas,
                }
                for method, mlu_key, viol_key in (
                    ("shortest", "mlu_shortest", "violations_shortest"),
                    ("equal", "mlu_equal", "violations_equal"),
                ):
                    rows.append(
                        {
                            **base,
                            "method": method,
                            "family": "baseline",
                            "mlu_actor": feas[mlu_key],
                            "violations_actor": feas[viol_key],
                            "mlu_applied": feas[mlu_key],
                            "violations_applied": feas[viol_key],
                            "acceptance": None,
                            "unsafe_accept": None,
                            "false_reject": None,
                            "fallback_used": None,
                            "fallback": None,
                            "unavoidable_overload": None,
                            "policy_s": 0.0,
                            "twin_ms": 0.0,
                            "gate_ms": 0.0,
                            "runtime_s": 0.0,
                        }
                    )
                print(
                    f"seed {seed} {name} tm {matrix_id} x{scale:.2f} "
                    f"scenario={feas['scenario']} sp={feas['mlu_shortest']:.3f}",
                    flush=True,
                )
        del topo
    return rows


def _iter_cases(seed: int, catalog: dict):
    for name, split, _count in TOPOLOGY_SPEC:
        topo = load_topology(name)
        demands, meta = load_split_demands(split, name)
        entry = catalog["topologies"][name]["by_seed"][str(seed)]
        short_action = shortest_split(topo.n_paths, topo.hops)
        equal_action = equal_split(topo.n_paths)
        try:
            for local_i in entry["local_indices"]:
                matrix = demands[local_i]
                matrix_id = int(meta["matrix_indices"][local_i])
                for scale in SCALES:
                    scaled = matrix * float(scale)
                    feas = _feasibility(topo, scaled, short_action, equal_action)
                    base = {
                        "seed": seed,
                        "topology": name,
                        "split": split,
                        "unseen": name in ("GEANT", "Kdl"),
                        "local_index": local_i,
                        "matrix_id": matrix_id,
                        "scale": float(scale),
                        **feas,
                    }
                    yield topo, scaled, base
        finally:
            del topo


def run_seed(
    seed: int,
    catalog: dict,
    twin,
    gate_config,
    device: torch.device,
    actors_to_run: tuple[str, ...] | None = None,
) -> list[dict]:
    """Evaluate one seed. Learned actors are loaded one at a time to keep Kdl memory bounded."""
    paths = _actor_paths()
    wanted = actors_to_run or tuple(paths)
    for name in wanted:
        if name != "frozen_p1" and not paths[name].exists():
            raise FileNotFoundError(f"missing checkpoint for {name}: {paths[name]}")

    rows = _case_bases(seed, catalog)

    if "frozen_p1" in wanted:
        print(f"seed {seed} method frozen_p1", flush=True)
        frozen = FrozenP1Policy(paths["frozen_p1"], device)
        framework = Framework(frozen, twin, gate_config, device)
        last_topo = None
        for topo, scaled, base in _iter_cases(seed, catalog):
            if last_topo != topo.name:
                framework.reset_session(last_topo)
                last_topo = topo.name
                framework.reset_session(topo.name)
            _install_shared_graph(frozen, topo, device)
            started = time.perf_counter()
            decision = framework.decide(topo, scaled)
            runtime_s = time.perf_counter() - started
            rows.append(
                {
                    **base,
                    "method": "frozen_p1",
                    "family": "framework",
                    "mlu_actor": decision["mlu_true_proposed"],
                    "violations_actor": decision["violations_proposed"],
                    "mlu_applied": decision["mlu_true_applied"],
                    "violations_applied": decision["violations_applied"],
                    "acceptance": bool(decision["proposed_accepted"]),
                    "unsafe_accept": bool(
                        decision["proposed_accepted"] and decision["violations_proposed"] > 0
                    ),
                    "false_reject": bool(
                        (not decision["proposed_accepted"]) and decision["violations_proposed"] == 0
                    ),
                    "fallback_used": decision["fallback"] is not None,
                    "fallback": decision["fallback"],
                    "unavoidable_overload": bool(
                        (not decision["proposed_accepted"])
                        and decision["violations_applied"] > 0
                        and decision["violations_shortest"] > 0
                    ),
                    "policy_s": decision["policy_s"],
                    "twin_ms": decision["twin_ms"],
                    "gate_ms": decision["gate_ms"],
                    "runtime_s": runtime_s,
                }
            )
            print(
                f"  frozen_p1 {base['topology']} tm {base['matrix_id']} x{base['scale']:.2f} "
                f"mlu={decision['mlu_true_proposed']:.3f} accept={decision['proposed_accepted']}",
                flush=True,
            )
        del frozen, framework
        gc.collect()

    for method in wanted:
        if method == "frozen_p1":
            continue
        print(f"seed {seed} method {method}", flush=True)
        policy = _load_general(paths[method], device)
        controller = GeneralController(policy, twin, gate_config, device)
        last_topo = None
        for topo, scaled, base in _iter_cases(seed, catalog):
            if last_topo != topo.name:
                if last_topo is not None:
                    controller.reset(last_topo)
                last_topo = topo.name
                controller.reset(topo.name)
            _install_shared_graph(policy, topo, device)
            started = time.perf_counter()
            decision = controller.decide(topo, scaled)
            runtime_s = time.perf_counter() - started
            rows.append(
                {
                    **base,
                    "method": method,
                    "family": "framework",
                    "mlu_actor": decision["mlu_actor"],
                    "violations_actor": decision["violations_actor"],
                    "mlu_applied": decision["mlu_applied"],
                    "violations_applied": decision["violations_applied"],
                    "acceptance": bool(decision["proposed_accepted"]),
                    "unsafe_accept": bool(decision["unsafe_accept"]),
                    "false_reject": bool(decision["false_reject"]),
                    "fallback_used": bool(decision["fallback_used"]),
                    "fallback": decision["fallback"],
                    "unavoidable_overload": bool(decision["unavoidable_overload"]),
                    "policy_s": decision["policy_s"],
                    "twin_ms": decision["twin_ms"],
                    "gate_ms": decision["gate_ms"],
                    "runtime_s": runtime_s,
                }
            )
            print(
                f"  {method} {base['topology']} tm {base['matrix_id']} x{base['scale']:.2f} "
                f"mlu={decision['mlu_actor']:.3f} accept={decision['proposed_accepted']}",
                flush=True,
            )
        del policy, controller
        gc.collect()

    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Per-topology (and scenario) mean±std across seeds for each method."""
    methods = sorted({row["method"] for row in rows})
    topologies = [name for name, _split, _n in TOPOLOGY_SPEC]
    scales = list(SCALES)
    metric_keys = (
        "mlu_actor",
        "mlu_applied",
        "violations_actor",
        "violations_applied",
        "acceptance",
        "unsafe_accept",
        "false_reject",
        "fallback_used",
        "unavoidable_overload",
        "runtime_s",
        "policy_s",
        "twin_ms",
    )

    def _summarize(bundle: list[dict], method: str) -> dict[str, Any]:
        items = [row for row in bundle if row["method"] == method]
        seeds = sorted({row["seed"] for row in items})
        out: dict[str, Any] = {"cases": len(items), "seeds": len(seeds)}
        for key in metric_keys:
            per_seed = []
            for seed in seeds:
                vals = [row[key] for row in items if row["seed"] == seed and row.get(key) is not None]
                if vals:
                    per_seed.append(float(np.mean(vals)))
            mean, std = _mean_std(per_seed)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        return out

    by_topo_scale = []
    for topology in topologies:
        for scale in scales:
            bundle = [row for row in rows if row["topology"] == topology and row["scale"] == scale]
            if not bundle:
                continue
            entry = {
                "topology": topology,
                "scale": scale,
                "unseen": topology in ("GEANT", "Kdl"),
                "feasible_rate": float(np.mean([row["feasible_any_baseline"] for row in bundle])),
                "methods": {method: _summarize(bundle, method) for method in methods},
            }
            by_topo_scale.append(entry)

    by_topo_scenario = []
    for topology in topologies:
        for scenario in ("feasible", "infeasible"):
            bundle = [row for row in rows if row["topology"] == topology and row["scenario"] == scenario]
            if not bundle:
                continue
            by_topo_scenario.append(
                {
                    "topology": topology,
                    "scenario": scenario,
                    "cases_per_method": len([r for r in bundle if r["method"] == methods[0]]),
                    "methods": {method: _summarize(bundle, method) for method in methods},
                }
            )

    # Aggregate with and without Cogentco to quantify distortion.
    distortion = {}
    for label, predicate in (
        ("all_topologies", lambda r: True),
        ("without_cogentco", lambda r: r["topology"] != "Cogentco"),
        ("cogentco_only", lambda r: r["topology"] == "Cogentco"),
        ("seen_train_topologies", lambda r: r["topology"] in ("B4", "UsCarrier", "Cogentco")),
        ("unseen_test_topologies", lambda r: r["topology"] in ("GEANT", "Kdl")),
        ("feasible_only", lambda r: r["scenario"] == "feasible"),
        ("infeasible_only", lambda r: r["scenario"] == "infeasible"),
    ):
        bundle = [row for row in rows if predicate(row)]
        distortion[label] = {method: _summarize(bundle, method) for method in methods}

    return {
        "by_topology_scale": by_topo_scale,
        "by_topology_scenario": by_topo_scenario,
        "distortion": distortion,
        "methods": methods,
    }


def _plots(rows: list[dict], summary: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (folder / "PLOT_STATUS.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    methods_actor = ["frozen_p1", "general_3epoch", "improved_best", "improved_stable", "shortest", "equal"]
    colors = {
        "frozen_p1": "#4C78A8",
        "general_3epoch": "#F58518",
        "improved_best": "#E45756",
        "improved_stable": "#54A24B",
        "shortest": "#B279A2",
        "equal": "#72B7B2",
    }
    topologies = [name for name, *_ in TOPOLOGY_SPEC]

    # Per-topology actor MLU vs scale with error bars across seeds.
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, topology in zip(axes, topologies):
        for method in methods_actor:
            xs, ys, es = [], [], []
            for scale in SCALES:
                block = next(
                    (
                        item["methods"].get(method)
                        for item in summary["by_topology_scale"]
                        if item["topology"] == topology and item["scale"] == scale
                    ),
                    None,
                )
                if not block or block.get("cases", 0) == 0:
                    continue
                xs.append(scale)
                ys.append(block["mlu_actor_mean"])
                es.append(block["mlu_actor_std"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", label=method, color=colors.get(method), capsize=3)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(topology + (" (unseen)" if topology in ("GEANT", "Kdl") else ""))
        ax.set_xlabel("Traffic scale")
        ax.set_ylabel("Actor MLU")
        ax.set_xticks(list(SCALES))
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=9)
    fig.suptitle("Per-topology actor MLU (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(folder / "mlu_actor_by_topology.png", dpi=140)
    plt.close(fig)

    # Acceptance for framework methods.
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    fw_methods = ["frozen_p1", "general_3epoch", "improved_best", "improved_stable"]
    for ax, topology in zip(axes, topologies):
        for method in fw_methods:
            xs, ys, es = [], [], []
            for scale in SCALES:
                block = next(
                    (
                        item["methods"].get(method)
                        for item in summary["by_topology_scale"]
                        if item["topology"] == topology and item["scale"] == scale
                    ),
                    None,
                )
                if not block or block.get("acceptance_mean") is None or not np.isfinite(block["acceptance_mean"]):
                    continue
                xs.append(scale)
                ys.append(block["acceptance_mean"])
                es.append(block["acceptance_std"])
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", label=method, color=colors.get(method), capsize=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(topology)
        ax.set_xlabel("Traffic scale")
        ax.set_ylabel("Acceptance rate")
        ax.set_xticks(list(SCALES))
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=9)
    fig.suptitle("Framework acceptance (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(folder / "acceptance_by_topology.png", dpi=140)
    plt.close(fig)

    # Cogentco distortion bar chart on applied/actor MLU.
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels_x = ["all", "w/o Cogentco", "Cogentco only"]
    keys = ["all_topologies", "without_cogentco", "cogentco_only"]
    x = np.arange(len(labels_x))
    width = 0.18
    for i, method in enumerate(fw_methods):
        means = [summary["distortion"][k][method]["mlu_actor_mean"] for k in keys]
        stds = [summary["distortion"][k][method]["mlu_actor_std"] for k in keys]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=method, color=colors[method], capsize=3)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Actor MLU")
    ax.set_title("Cogentco distortion of aggregate actor MLU")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "cogentco_distortion.png", dpi=140)
    plt.close(fig)

    # Feasible vs infeasible applied MLU for improved_stable.
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for scenario, marker in (("feasible", "o"), ("infeasible", "x")):
        xs, ys = [], []
        for topology in topologies:
            block = next(
                (
                    item
                    for item in summary["by_topology_scenario"]
                    if item["topology"] == topology and item["scenario"] == scenario
                ),
                None,
            )
            if not block:
                continue
            stats = block["methods"].get("improved_stable")
            if not stats or stats.get("cases", 0) == 0:
                continue
            xs.append(topology)
            ys.append(stats["mlu_applied_mean"])
        if xs:
            ax.plot(xs, ys, marker=marker, label=scenario)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_ylabel("Applied MLU (improved_stable)")
    ax.set_title("Feasible vs infeasible scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "feasible_vs_infeasible.png", dpi=140)
    plt.close(fig)


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g}±{std:.{digits}g}"


def write_report(summary: dict, config: dict, out_path: Path) -> str:
    lines = [
        "# Phase 3 multi-seed robustness evaluation",
        "",
        "All methods saw the **same** seeded traffic matrices within each seed.",
        "GEANT and Kdl were evaluated only on their **test** splits (unseen).",
        "No training or hyperparameter tuning was performed in this step.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{config['seeds']}`",
        f"- Scales: `{config['scales']}`",
        f"- Matrix draws: " + ", ".join(f"{n}×{c} ({s})" for n, s, c in TOPOLOGY_SPEC),
        f"- Checkpoints: `{json.dumps(config['checkpoints'], indent=2)}`",
        "- Feasible scenario: shortest **or** equal split has zero capacity violations.",
        "",
        "## Per-topology actor MLU (mean ± std across seeds)",
        "",
        "| Topology | Scale | Feasible rate | Frozen P1 | General 3-epoch | Improved best | Improved stable | Shortest | Equal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["by_topology_scale"]:
        m = item["methods"]
        lines.append(
            "| {topo} | {scale:.2f} | {fr:.2f} | {fz} | {g3} | {ib} | {is_} | {sp} | {eq} |".format(
                topo=item["topology"],
                scale=item["scale"],
                fr=item["feasible_rate"],
                fz=_fmt(m["frozen_p1"]["mlu_actor_mean"], m["frozen_p1"]["mlu_actor_std"]),
                g3=_fmt(m["general_3epoch"]["mlu_actor_mean"], m["general_3epoch"]["mlu_actor_std"]),
                ib=_fmt(m["improved_best"]["mlu_actor_mean"], m["improved_best"]["mlu_actor_std"]),
                is_=_fmt(m["improved_stable"]["mlu_actor_mean"], m["improved_stable"]["mlu_actor_std"]),
                sp=_fmt(m["shortest"]["mlu_actor_mean"], m["shortest"]["mlu_actor_std"]),
                eq=_fmt(m["equal"]["mlu_actor_mean"], m["equal"]["mlu_actor_std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Framework safety and applied routing",
            "",
            "| Topology | Scale | Method | Accept | Unsafe | False reject | Fallback | Applied MLU | Actor viol. | Runtime s |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["by_topology_scale"]:
        for method in ("frozen_p1", "general_3epoch", "improved_best", "improved_stable"):
            m = item["methods"][method]
            lines.append(
                "| {topo} | {scale:.2f} | {method} | {acc} | {ua} | {fr} | {fb} | {ap} | {vv} | {rt} |".format(
                    topo=item["topology"],
                    scale=item["scale"],
                    method=method,
                    acc=_fmt(m["acceptance_mean"], m["acceptance_std"], 2),
                    ua=_fmt(m["unsafe_accept_mean"], m["unsafe_accept_std"], 2),
                    fr=_fmt(m["false_reject_mean"], m["false_reject_std"], 2),
                    fb=_fmt(m["fallback_used_mean"], m["fallback_used_std"], 2),
                    ap=_fmt(m["mlu_applied_mean"], m["mlu_applied_std"]),
                    vv=_fmt(m["violations_actor_mean"], m["violations_actor_std"], 2),
                    rt=_fmt(m["runtime_s_mean"], m["runtime_s_std"], 2),
                )
            )

    lines.extend(
        [
            "",
            "## Feasible vs infeasible scenarios",
            "",
            "Feasibility is determined from capacity constraints under shortest/equal,",
            "not from model predictions.",
            "",
            "| Topology | Scenario | Improved stable applied MLU | Improved best applied MLU | Shortest MLU |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summary["by_topology_scenario"]:
        m = item["methods"]
        lines.append(
            "| {topo} | {sc} | {st} | {be} | {sp} |".format(
                topo=item["topology"],
                sc=item["scenario"],
                st=_fmt(m["improved_stable"]["mlu_applied_mean"], m["improved_stable"]["mlu_applied_std"]),
                be=_fmt(m["improved_best"]["mlu_applied_mean"], m["improved_best"]["mlu_applied_std"]),
                sp=_fmt(m["shortest"]["mlu_actor_mean"], m["shortest"]["mlu_actor_std"]),
            )
        )

    d = summary["distortion"]
    lines.extend(
        [
            "",
            "## Does Cogentco distort aggregate performance?",
            "",
            "| Slice | Frozen actor MLU | General 3-epoch | Improved best | Improved stable | Shortest |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, title in (
        ("all_topologies", "All topologies"),
        ("without_cogentco", "Without Cogentco"),
        ("cogentco_only", "Cogentco only"),
        ("unseen_test_topologies", "Unseen GEANT+Kdl"),
        ("feasible_only", "Feasible cases only"),
        ("infeasible_only", "Infeasible cases only"),
    ):
        block = d[label]
        lines.append(
            "| {title} | {a} | {b} | {c} | {d_} | {e} |".format(
                title=title,
                a=_fmt(block["frozen_p1"]["mlu_actor_mean"], block["frozen_p1"]["mlu_actor_std"]),
                b=_fmt(block["general_3epoch"]["mlu_actor_mean"], block["general_3epoch"]["mlu_actor_std"]),
                c=_fmt(block["improved_best"]["mlu_actor_mean"], block["improved_best"]["mlu_actor_std"]),
                d_=_fmt(block["improved_stable"]["mlu_actor_mean"], block["improved_stable"]["mlu_actor_std"]),
                e=_fmt(block["shortest"]["mlu_actor_mean"], block["shortest"]["mlu_actor_std"]),
            )
        )

    all_m = d["all_topologies"]["improved_stable"]["mlu_actor_mean"]
    no_c = d["without_cogentco"]["improved_stable"]["mlu_actor_mean"]
    only_c = d["cogentco_only"]["improved_stable"]["mlu_actor_mean"]
    lines.extend(
        [
            "",
            f"On this run, improved_stable actor MLU is **{_fmt(all_m, d['all_topologies']['improved_stable']['mlu_actor_std'])}** "
            f"across all topologies, **{_fmt(no_c, d['without_cogentco']['improved_stable']['mlu_actor_std'])}** without Cogentco, "
            f"and **{_fmt(only_c, d['cogentco_only']['improved_stable']['mlu_actor_std'])}** on Cogentco alone. "
            "Cogentco therefore dominates any unweighted aggregate mean whenever it is included.",
            "",
            "## Method notes",
            "",
            "- `frozen_p1` uses the original Step-7 `Framework` fallback (last accepted → shortest).",
            "- Generalized actors use `GeneralController` (least-violating among last/shortest/equal).",
            "- Actor MLU columns are physics loads from the proposal before fallback.",
            "- Applied MLU is after the twin+gate decision.",
            "- Both `improved_best` (`best_checkpoint.pt`) and `improved_stable` (`recommended_stable.pt`) are evaluated; the validation-mean winner is not assumed best overall.",
            "",
            "## Limitations / incomplete items",
            "",
            "- ASN shards were not read or evaluated.",
            "- Matrix budgets stay small on large graphs (1 UsCarrier / Cogentco / Kdl draw per seed) for runtime.",
            "- Policies are deterministic; seed variation comes from traffic-matrix sampling, not stochastic actions.",
            "- Frozen and generalized frameworks differ in fallback policy; compare actor-only MLU for a pure policy ranking.",
            "- No retraining or gate/twin modification was performed.",
            "",
        ]
    )
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


def run(
    seeds: tuple[int, ...] = SEEDS,
    result_dir: Path | None = None,
    actors: tuple[str, ...] | None = None,
) -> dict:
    out = Path(result_dir or RESULT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    twin = load_trained_ensemble(device)
    gate_config = load_config()
    catalog = build_catalog(seeds)
    paths = {k: str(v) for k, v in _actor_paths().items()}
    config = {
        "seeds": list(seeds),
        "scales": list(SCALES),
        "topology_spec": [
            {"topology": n, "split": s, "count": c, "unseen": n in ("GEANT", "Kdl")}
            for n, s, c in TOPOLOGY_SPEC
        ],
        "checkpoints": paths,
        "improved_checkpoint_aliases": {
            "improved_best": str(IMPROVED_BEST),
            "improved_stable": str(IMPROVED_STABLE),
            "general_3epoch": str(BASELINE_GENERAL),
            "also_referenced": str(IMPROVED_CHECKPOINT),
        },
        "feasibility_rule": "shortest or equal has zero capacity violations",
        "preserved": {
            "frozen_p1": str(DEFAULT_CHECKPOINT),
            "twin": "models/twin/ensemble.pt",
            "splits": "data/processed/splits",
            "asn": "not read",
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    for seed in seeds:
        print(f"=== robustness seed {seed} ===", flush=True)
        seed_rows = run_seed(seed, catalog, twin, gate_config, device, actors_to_run=actors)
        seed_path = out / "raw" / f"seed_{seed}.jsonl"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text("\n".join(json.dumps(row) for row in seed_rows) + "\n", encoding="utf-8")
        all_rows.extend(seed_rows)

    summary = aggregate(all_rows)
    (out / "metrics.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    (out / "decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in all_rows) + "\n", encoding="utf-8"
    )
    _plots(all_rows, summary, out / "plots")
    write_report(summary, config, out / "ROBUSTNESS_REPORT.md")
    print("Wrote", out / "ROBUSTNESS_REPORT.md", flush=True)
    return {"rows": all_rows, "summary": summary, "config": config, "result_dir": str(out)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Phase 3 multi-seed robustness evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--actors",
        nargs="+",
        default=None,
        help="subset of actors: frozen_p1 general_3epoch improved_best improved_stable",
    )
    args = parser.parse_args(argv)
    return run(seeds=tuple(args.seeds), result_dir=args.result_dir, actors=tuple(args.actors) if args.actors else None)


if __name__ == "__main__":
    main()

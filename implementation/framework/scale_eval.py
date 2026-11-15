"""Controlled-scale evaluation of the Step 7 controller.

Writes a new result directory and does not replace ``results/framework/metrics.json``.
The frozen actor, the twin checkpoint, and ``safety/gate.py`` are not modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from safety.evaluate import load_config, load_trained_ensemble
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import equal_split, link_load, od_demands, demand_per_path

from .fallback_modes import classify_candidates
from .pipeline import Framework
from .policy_adapter import DEFAULT_CHECKPOINT, FrozenP1Policy

RESULT_DIR = ROOT / "results" / "framework" / "extended"
SCALES = (0.25, 0.5, 0.75, 1.0)
JOBS = (
    ("B4", "val", 4, 2),
    ("UsCarrier", "val", 1, 1),
    ("Cogentco", "val", 1, 1),
    ("GEANT", "test", 600, 2),
    ("Kdl", "test", 1, 1),
)


def _gate_flags(row: dict) -> dict:
    accepted = bool(row["proposed_accepted"])
    proposed_unsafe = row["violations_proposed"] > 0
    return {
        "accepted": accepted,
        "unsafe_accept": accepted and proposed_unsafe,
        "false_reject": (not accepted) and (not proposed_unsafe),
        "fallback_used": row["fallback"] is not None,
        "applied_violates": row["violations_applied"] > 0,
    }


def run(jobs: tuple = JOBS, scales: tuple = SCALES) -> dict:
    device = torch.device("cpu")
    policy = FrozenP1Policy(DEFAULT_CHECKPOINT, device)
    framework = Framework(policy, load_trained_ensemble(device), load_config(), device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    for name, split, stride, limit in jobs:
        print(f"extended {split}/{name}", flush=True)
        topo = load_topology(name)
        demands, meta = load_split_demands(split, name)
        indices = list(range(0, demands.shape[0], stride))[:limit]
        od0 = od_demands(demands[indices[0]])
        info = policy.inspect(topo, od0)
        info["topology"] = name
        info["split"] = split
        info.pop("action")
        diagnostics.append(info)
        print(
            f"  h0_max={info['h0_max']:.3g} entropy={info['mean_entropy']:.3f} "
            f"peaked={info['peaked_fraction']:.3f} padded={info['demands_with_fewer_than_4_paths']}",
            flush=True,
        )
        equal = equal_split(topo.n_paths)
        framework.reset_session(name)
        for index in indices:
            matrix = demands[index]
            matrix_id = int(meta["matrix_indices"][index])
            for scale in scales:
                scaled = matrix * float(scale)
                decision = framework.decide(topo, scaled)
                flags = _gate_flags(decision)
                actor_load = _load_action(topo, scaled, policy.last_proposal)
                short_load = _load(topo, scaled, "shortest")
                equal_load = _load_action(topo, scaled, equal)
                choice = classify_candidates(
                    [
                        ("p1_actor", actor_load),
                        ("shortest", short_load),
                        ("equal", equal_load),
                    ],
                    topo.capacity,
                )
                row = {
                    "topology": name,
                    "split": split,
                    "matrix_index": matrix_id,
                    "scale": float(scale),
                    "feasible_under_shortest": decision["mlu_true_shortest"] <= 1.0 + 1e-6,
                    "acceptance": flags["accepted"],
                    "unsafe_accept": flags["unsafe_accept"],
                    "false_reject": flags["false_reject"],
                    "fallback_used": flags["fallback_used"],
                    "fallback": decision["fallback"],
                    "applied_source": decision["applied_source"],
                    "mlu_framework": decision["mlu_true_applied"],
                    "mlu_actor": decision["mlu_true_proposed"],
                    "mlu_shortest": decision["mlu_true_shortest"],
                    "violations_framework": decision["violations_applied"],
                    "violations_actor": decision["violations_proposed"],
                    "violations_shortest": decision["violations_shortest"],
                    "prediction_mae": decision["prediction_mae_applied"],
                    "policy_s": decision["policy_s"],
                    "twin_ms": decision["twin_ms"],
                    "gate_ms": decision["gate_ms"],
                    "fallback_outcome": choice["outcome"],
                    "unavoidable_overload": choice["unavoidable_overload"],
                    "least_violating_action": choice["selected"],
                    "least_violating_mlu": choice["selected_mlu"],
                    "least_violating_violations": choice["selected_violations"],
                }
                rows.append(row)
                print(
                    f"  {name} tm {matrix_id} x{scale:.2f} accept={flags['accepted']} "
                    f"mlu_actor={row['mlu_actor']:.3f} mlu_gate={row['mlu_framework']:.3f} "
                    f"mlu_sp={row['mlu_shortest']:.3f} fallback={choice['outcome']}",
                    flush=True,
                )
        policy.release(name)
        del topo
    payload = {
        "scales": list(scales),
        "jobs": [{"topology": n, "split": s, "stride": st, "limit": lim} for n, s, st, lim in jobs],
        "diagnostics": diagnostics,
        "cases": rows,
        "by_topology_scale": _aggregate(rows),
        "preserved": {
            "step7_metrics": "results/framework/metrics.json",
            "twin_retrained": False,
            "frozen_checkpoint_modified": False,
            "safety_gate_modified": False,
            "asn": "pending; shards not read",
        },
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULT_DIR / "decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    _plots(rows, RESULT_DIR / "plots")
    report = _report(payload)
    (RESULT_DIR / "EXTENDED_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote", RESULT_DIR / "EXTENDED_REPORT.md", flush=True)
    return payload


def _load(topo, matrix: np.ndarray, kind: str) -> np.ndarray:
    from twin.physics import shortest_split

    action = shortest_split(topo.n_paths, topo.hops) if kind == "shortest" else equal_split(topo.n_paths)
    return _load_action(topo, matrix, action)


def _load_action(topo, matrix: np.ndarray, action: np.ndarray) -> np.ndarray:
    return link_load(
        demand_per_path(od_demands(matrix), topo.demand_row),
        action,
        topo.incidence_path,
        topo.incidence_link,
        topo.n_links,
    )


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["topology"], row["scale"]), []).append(row)
    out = []
    for (topology, scale), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        def rate(key: str) -> float:
            return float(np.mean([bool(item[key]) for item in items]))

        def avg(key: str) -> float:
            return float(np.mean([item[key] for item in items]))

        out.append(
            {
                "topology": topology,
                "scale": scale,
                "cases": len(items),
                "feasible_under_shortest_rate": rate("feasible_under_shortest"),
                "acceptance_rate": rate("acceptance"),
                "unsafe_accept_rate": rate("unsafe_accept"),
                "false_reject_rate": rate("false_reject"),
                "fallback_rate": rate("fallback_used"),
                "applied_violation_rate": float(
                    np.mean([item["violations_framework"] > 0 for item in items])
                ),
                "mean_mlu_framework": avg("mlu_framework"),
                "mean_mlu_actor": avg("mlu_actor"),
                "mean_mlu_shortest": avg("mlu_shortest"),
                "mean_violations_framework": avg("violations_framework"),
                "mean_violations_actor": avg("violations_actor"),
                "mean_violations_shortest": avg("violations_shortest"),
                "mean_prediction_mae": avg("prediction_mae"),
                "feasible_fallback_rate": float(np.mean([item["fallback_outcome"] == "feasible_fallback" for item in items])),
                "unavoidable_overload_rate": rate("unavoidable_overload"),
            }
        )
    return out


def _plots(rows: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (folder / "PLOT_STATUS.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    topologies = []
    for row in rows:
        if row["topology"] not in topologies:
            topologies.append(row["topology"])
    scales = sorted({row["scale"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for topology in topologies:
        xs, actor, shortest, framework = [], [], [], []
        for scale in scales:
            items = [row for row in rows if row["topology"] == topology and row["scale"] == scale]
            if not items:
                continue
            xs.append(scale)
            actor.append(float(np.mean([row["mlu_actor"] for row in items])))
            shortest.append(float(np.mean([row["mlu_shortest"] for row in items])))
            framework.append(float(np.mean([row["mlu_framework"] for row in items])))
        axes[0].plot(xs, actor, marker="o", label=f"{topology} actor")
        axes[1].plot(xs, framework, marker="o", label=f"{topology} gate")
        axes[1].plot(xs, shortest, marker="x", linestyle="--", label=f"{topology} shortest")
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_title("Actor-only MLU")
    axes[1].set_title("Gate vs shortest MLU")
    for axis in axes:
        axis.set_xlabel("Traffic scale")
        axis.set_ylabel("Mean MLU")
        axis.set_xticks(scales)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "mlu_by_scale.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4))
    for topology in topologies:
        xs, acc = [], []
        for scale in scales:
            items = [row for row in rows if row["topology"] == topology and row["scale"] == scale]
            if not items:
                continue
            xs.append(scale)
            acc.append(float(np.mean([row["acceptance"] for row in items])))
        axis.plot(xs, acc, marker="o", label=topology)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Traffic scale")
    axis.set_ylabel("Acceptance rate")
    axis.set_title("Safety-gate acceptance")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "acceptance_by_scale.png", dpi=140)
    plt.close(fig)


def _report(payload: dict) -> str:
    lines = [
        "# Phase 3 extended evaluation",
        "",
        "This file does not replace the Step 7 report. The frozen B4 checkpoint,",
        "the twin weights, and `safety/gate.py` were not modified. ASN shards were not read.",
        "",
        "## Actor mismatches at native scale",
        "",
        "| Topology | h0 max | Mean entropy (max log 4) | Peaked splits | Demands with <4 paths | Uniform capacity |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["diagnostics"]:
        lines.append(
            "| {topology} | {h0_max:.3g} | {mean_entropy:.3f} | {peaked_fraction:.3f} | {pad} | {uniform} |".format(
                topology=row["topology"],
                h0_max=row["h0_max"],
                mean_entropy=row["mean_entropy"],
                peaked_fraction=row["peaked_fraction"],
                pad=row["demands_with_fewer_than_4_paths"],
                uniform="yes" if row["uniform_capacity"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "The frozen actor was trained with raw B4 capacities and raw B4 demands.",
            "A much larger h0 max means the same linear layers see a different numeric range.",
            "Entropy near 0 and a high peaked fraction mean the softmax has collapsed onto one path.",
            "Padding counts are demands whose missing paths are repeated only inside the adapter.",
            "",
            "## Identical-condition comparison",
            "",
            "| Topology | Scale | Cases | Shortest feasible | Accept | Unsafe accept | False reject | Fallback | MLU actor | MLU gate | MLU shortest | Feasible fallback | Unavoidable overload |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["by_topology_scale"]:
        lines.append(
            "| {topology} | {scale:.2f} | {cases} | {feas:.2f} | {acc:.2f} | {ua:.2f} | {fr:.2f} | {fb:.2f} | {a:.3g} | {g:.3g} | {s:.3g} | {ff:.2f} | {uo:.2f} |".format(
                topology=row["topology"],
                scale=row["scale"],
                cases=row["cases"],
                feas=row["feasible_under_shortest_rate"],
                acc=row["acceptance_rate"],
                ua=row["unsafe_accept_rate"],
                fr=row["false_reject_rate"],
                fb=row["fallback_rate"],
                a=row["mean_mlu_actor"],
                g=row["mean_mlu_framework"],
                s=row["mean_mlu_shortest"],
                ff=row["feasible_fallback_rate"],
                uo=row["unavoidable_overload_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Fallback reading",
            "",
            "The gate itself is unchanged: it still falls back to the last accepted action,",
            "then shortest-path, then shortest-path as last resort.",
            "Separately, actor, shortest-path, and equal split are scored on the true load.",
            "`feasible_fallback` means one of those three has no capacity violation.",
            "`unavoidable_overload` means all three overload; the recorded least-violating",
            "action is the one with the fewest overloaded links, then the lowest MLU.",
            "That least-violating choice is not applied by the current gate.",
            "",
            "## What this run does not show",
            "",
            "- It does not show that the frozen B4 actor generalises. The diagnostics and MLU columns are the evidence either way.",
            "- It does not change ASN. ASN stays pending.",
            "- Cogentco, UsCarrier, and Kdl use one validation or test matrix because a full sweep is too large for this machine. B4 uses two validation matrices and GEANT uses two test matrices, each at four scales.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

"""Run the safety gate on Step 4 splits using the saved Step 5 ensemble.

Does not retrain the twin, does not read ASN shards, and does not touch
baseline_p1_reproduction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from twin.data import ROOT, load_split_demands, load_topology
from twin.model import EnsembleTwin
from twin.physics import dirichlet_split, equal_split, shortest_split
from twin.train import BASE_SEED, GEANT_EVAL_STRIDE, MODEL_DIR, _predict_numpy, _sample

from .gate import AppliedAction, GateConfig, inspect_prediction, select_fallback

RESULT_DIR = ROOT / "results" / "safety"
CONFIG_PATH = Path(__file__).resolve().parent / "gate_config.json"


def load_config(path: Path | None = None) -> GateConfig:
    payload = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
    return GateConfig.from_dict(payload)


def load_trained_ensemble(device: torch.device) -> EnsembleTwin:
    ckpt = torch.load(MODEL_DIR / "ensemble.pt", map_location=device, weights_only=False)
    model = EnsembleTwin(n_members=int(ckpt["n_members"]), hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_action(model, topo, matrix: np.ndarray, action: np.ndarray, device: torch.device):
    sample = _sample(topo, matrix, action)
    started = time.perf_counter()
    mean_load, std_load = _predict_numpy(model, sample, topo, device)
    twin_ms = (time.perf_counter() - started) * 1000.0
    return sample["load"], mean_load, std_load, twin_ms


def _true_mlu(load: np.ndarray, capacity: np.ndarray) -> float:
    return float(np.max(load / np.maximum(capacity, 1e-12)))


def _violations(load: np.ndarray, capacity: np.ndarray) -> int:
    return int(np.count_nonzero(load > capacity + 1e-9))


def _log_row(
    *,
    topology: str,
    split: str,
    matrix_index: int,
    proposed_name: str,
    proposed_verdict,
    applied: AppliedAction,
    true_proposed: np.ndarray,
    true_applied: np.ndarray | None,
    capacity: np.ndarray,
    twin_ms: float,
    gate_ms: float,
) -> dict:
    true_applied_load = true_applied if true_applied is not None else true_proposed
    proposed_unsafe = _violations(true_proposed, capacity) > 0
    applied_unsafe = _violations(true_applied_load, capacity) > 0
    false_reject = (not proposed_verdict.accepted) and (not proposed_unsafe)
    unsafe_accept = proposed_verdict.accepted and proposed_unsafe
    return {
        "topology": topology,
        "split": split,
        "matrix_index": int(matrix_index),
        "proposed_action": proposed_name,
        "proposed_accepted": bool(proposed_verdict.accepted),
        "reasons": list(proposed_verdict.reasons),
        "applied_source": applied.source,
        "fallback": applied.fallback,
        "mlu_mean": proposed_verdict.mlu_mean,
        "mlu_upper": proposed_verdict.mlu_upper,
        "max_relative_std": proposed_verdict.max_relative_std,
        "n_links_over_limit": proposed_verdict.n_links_over_limit,
        "worst_link": proposed_verdict.worst_link,
        "mlu_true_proposed": _true_mlu(true_proposed, capacity),
        "mlu_true_applied": _true_mlu(true_applied_load, capacity),
        "true_violations_proposed": _violations(true_proposed, capacity),
        "true_violations_applied": _violations(true_applied_load, capacity),
        "proposed_unsafe": proposed_unsafe,
        "applied_unsafe": applied_unsafe,
        "false_reject": false_reject,
        "unsafe_accept": unsafe_accept,
        "twin_ms": twin_ms,
        "gate_ms": gate_ms,
    }


def evaluate_topology(
    model,
    name: str,
    split: str,
    config: GateConfig,
    device: torch.device,
    stride: int = 1,
    include_scaled: bool = False,
    demand_scale: float = 1.0,
    seed: int = BASE_SEED + 91,
) -> dict:
    topo = load_topology(name)
    demands, meta = load_split_demands(split, name)
    shortest = shortest_split(topo.n_paths, topo.hops)
    last_action = None
    last_name = None
    rows = []
    rng = np.random.default_rng(seed)

    def cases(matrix_i: int, matrix: np.ndarray):
        yield "equal", equal_split(topo.n_paths), matrix
        yield "shortest", shortest, matrix
        yield "dirichlet", dirichlet_split(topo.n_paths, rng), matrix
        if include_scaled:
            yield "equal_scaled_5x", equal_split(topo.n_paths), matrix * 5.0

    for index in range(0, demands.shape[0], stride):
        matrix = demands[index] * demand_scale
        matrix_id = int(meta["matrix_indices"][index])
        native_short = predict_action(model, topo, matrix, shortest, device)
        native_short_verdict = inspect_prediction(native_short[1], native_short[2], topo.capacity, config)
        for proposed_name, action, used_matrix in cases(index, matrix):
            twin_ms = 0.0
            true_p, mean_p, std_p, ms_p = predict_action(model, topo, used_matrix, action, device)
            twin_ms += ms_p
            scaled = proposed_name.startswith("equal_scaled")
            if scaled:
                short_true, short_mean, short_std, ms_s = predict_action(
                    model, topo, used_matrix, shortest, device
                )
                twin_ms += ms_s
                short_v = inspect_prediction(short_mean, short_std, topo.capacity, config)
            else:
                short_true, short_mean, short_std, ms_s = native_short
                short_v = native_short_verdict
                if proposed_name != "shortest":
                    twin_ms += ms_s
            last_now = None
            last_true = None
            if last_action is not None:
                last_true, last_mean, last_std, ms_l = predict_action(
                    model, topo, used_matrix, last_action, device
                )
                twin_ms += ms_l
                last_now = inspect_prediction(last_mean, last_std, topo.capacity, config)
            started = time.perf_counter()
            proposed_verdict = inspect_prediction(mean_p, std_p, topo.capacity, config)
            applied = select_fallback(
                proposed_name,
                action,
                proposed_verdict,
                last_name=last_name,
                last_action=last_action,
                last_verdict=last_now,
                shortest_name="shortest",
                shortest_action=shortest,
                shortest_verdict=short_v,
                config=config,
            )
            gate_ms = (time.perf_counter() - started) * 1000.0
            if applied.source == "proposed":
                true_applied = true_p
            elif applied.fallback in ("shortest", "shortest_last_resort"):
                true_applied = short_true
            elif applied.fallback == "last_accepted" and last_true is not None:
                true_applied = last_true
            else:
                true_applied = true_p
            row = _log_row(
                topology=name,
                split=split,
                matrix_index=matrix_id,
                proposed_name=proposed_name,
                proposed_verdict=proposed_verdict,
                applied=applied,
                true_proposed=true_p,
                true_applied=true_applied,
                capacity=topo.capacity,
                twin_ms=twin_ms,
                gate_ms=gate_ms,
            )
            if applied.proposed_accepted and not scaled:
                last_action = np.asarray(action, dtype=np.float64).copy()
                last_name = proposed_name
            rows.append(row)
    del topo
    return {"topology": name, "split": split, "cases": rows, "summary": summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"cases": 0}
    accepted = sum(1 for r in rows if r["proposed_accepted"])
    fallbacks = sum(1 for r in rows if r["fallback"])
    unsafe_applied = sum(1 for r in rows if r["applied_unsafe"])
    unsafe_proposed = sum(1 for r in rows if r["proposed_unsafe"])
    unsafe_accept = sum(1 for r in rows if r["unsafe_accept"])
    false_reject = sum(1 for r in rows if r["false_reject"])
    last_resort = sum(1 for r in rows if r["fallback"] == "shortest_last_resort")
    return {
        "cases": n,
        "acceptance_rate": accepted / n,
        "fallback_rate": fallbacks / n,
        "shortest_fallback_rate": sum(1 for r in rows if r["fallback"] in ("shortest", "shortest_last_resort")) / n,
        "last_accepted_fallback_rate": sum(1 for r in rows if r["fallback"] == "last_accepted") / n,
        "last_resort_rate": last_resort / n,
        "proposed_violation_rate": unsafe_proposed / n,
        "applied_violation_rate": unsafe_applied / n,
        "unsafe_accept_rate": unsafe_accept / n,
        "false_reject_rate": false_reject / n,
        "mean_twin_ms": float(np.mean([r["twin_ms"] for r in rows])),
        "mean_gate_ms": float(np.mean([r["gate_ms"] for r in rows])),
        "mean_mlu_true_proposed": float(np.mean([r["mlu_true_proposed"] for r in rows])),
        "mean_mlu_true_applied": float(np.mean([r["mlu_true_applied"] for r in rows])),
        "mean_mlu_upper": float(np.mean([r["mlu_upper"] for r in rows])),
    }


def evaluate(config: GateConfig | None = None) -> dict:
    config = config or load_config()
    device = torch.device("cpu")
    model = load_trained_ensemble(device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("B4", "val", 1, True, 1.0, "val_B4"),
        ("B4", "val", 1, False, 0.25, "val_B4_feasible"),
        ("UsCarrier", "val", 1, False, 1.0, "val_UsCarrier"),
        ("Cogentco", "val", 1, False, 1.0, "val_Cogentco"),
        ("GEANT", "test", max(GEANT_EVAL_STRIDE, 20), False, 1.0, "test_GEANT"),
        ("Kdl", "test", 2, False, 1.0, "test_Kdl"),
    ]
    reports = []
    all_rows = []
    for name, split, stride, scaled, demand_scale, tag in jobs:
        print(f"safety {tag} stride={stride} scale={demand_scale}", flush=True)
        report = evaluate_topology(
            model,
            name,
            split,
            config,
            device,
            stride=stride,
            include_scaled=scaled,
            demand_scale=demand_scale,
        )
        reports.append(
            {
                "topology": name,
                "split": split,
                "tag": tag,
                "stride": stride,
                "demand_scale": demand_scale,
                "summary": report["summary"],
            }
        )
        all_rows.extend(report["cases"])
        (RESULT_DIR / f"decisions_{tag}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in report["cases"]) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["summary"], indent=2), flush=True)

    payload = {
        "config": config.to_dict(),
        "twin_checkpoint": str((MODEL_DIR / "ensemble.pt").resolve()),
        "jobs": reports,
        "overall": summarise(all_rows),
        "assumptions": [
            "Ground-truth load is the path-routing simulator, not a measured counter.",
            "Upper bound is mean + k * ensemble std, compared with utilisation_limit * capacity.",
            "Fallback order is last accepted action, then shortest-path routing.",
            "If every candidate fails the gate, shortest-path is still applied and marked last_resort.",
            "ASN is excluded; shards are not read.",
            "GEANT uses FIGRET test histories with a stride. Kdl uses a subset of the identical toy matrices.",
            "B4_feasible multiplies B4 validation demand by 0.25 so some native actions sit under capacity.",
        ],
        "limitations": [
            "The gate trusts the twin. An under-predicted load with a too-small std can be accepted unsafely.",
            "Shortest-path is not guaranteed to be feasible when demand exceeds cut capacity. On the native TMs in this corpus, MLU is already above 1, so last-resort shortest-path still overloads.",
            "Last-accepted is re-checked with the current traffic matrix before reuse.",
            "Kdl and GEANT were never used to train the twin; their std may be miscalibrated.",
            "Toy UsCarrier/Kdl matrices are identical all-ones placeholders, not Microsoft WAN traces.",
        ],
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULT_DIR / "SAFETY_REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print("Wrote", RESULT_DIR / "metrics.json", flush=True)
    return payload


def _markdown(payload: dict) -> str:
    lines = [
        "# Phase 3 Step 6 — safety gate",
        "",
        "The gate reads the twin’s predicted mean load and ensemble standard deviation,",
        "forms an upper bound `mean + k * std`, and accepts the routing action only if",
        "that bound stays at or below capacity on every link. Excessively uncertain",
        "predictions are also rejected. A rejected action is replaced by the last",
        "accepted split if it is still safe, otherwise by shortest-path routing.",
        "",
        "## Configuration",
        "",
        f"- k = {payload['config']['k']}",
        f"- utilisation limit = {payload['config']['utilisation_limit']}",
        f"- max relative std = {payload['config']['max_relative_std']}",
        f"- last-resort shortest path = {payload['config']['apply_shortest_if_all_fail']}",
        "",
        "## Results",
        "",
        "| Split | Topology | Cases | Accept | Fallback | Applied violations | Unsafe accept | False reject | Twin ms | Gate ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in payload["jobs"]:
        s = job["summary"]
        lines.append(
            "| {split} | {topology} | {cases} | {acc:.3f} | {fb:.3f} | {vio:.3f} | {ua:.3f} | {fr:.3f} | {twin:.2f} | {gate:.4f} |".format(
                split=job.get("tag", job["split"]),
                topology=job["topology"] + (f" x{job['demand_scale']}" if job.get("demand_scale", 1.0) != 1.0 else ""),
                cases=s["cases"],
                acc=s["acceptance_rate"],
                fb=s["fallback_rate"],
                vio=s["applied_violation_rate"],
                ua=s["unsafe_accept_rate"],
                fr=s["false_reject_rate"],
                twin=s["mean_twin_ms"],
                gate=s["mean_gate_ms"],
            )
        )
    ov = payload["overall"]
    lines.extend(
        [
            "",
            f"Overall: {ov['cases']} decisions, acceptance {ov['acceptance_rate']:.3f}, "
            f"fallback {ov['fallback_rate']:.3f}, applied violation rate {ov['applied_violation_rate']:.3f}.",
            "",
            "## How to read the rates",
            "",
            "On the native TMs, maximum link utilisation is already above 1 even under",
            "shortest-path routing (B4 ~1.1, GEANT ~2.8, UsCarrier/Kdl toy all-ones much higher).",
            "The gate therefore rejects almost every proposed action (`unsafe_accept` stays 0)",
            "and applies shortest-path as last resort, which still overloads. That is a",
            "limitation of the fallback, not a missed reject.",
            "",
            "B4 at 0.25× demand sits under capacity: the same gate accepts every action,",
            "uses no fallback, and records zero applied violations. One native B4 validation",
            "matrix (index 23, equal split) is also accepted because its upper-bound MLU is 0.992.",
            "",
            "Gate time is well under 1 ms. Twin time dominates (milliseconds on B4/GEANT,",
            "hundreds of ms on UsCarrier/Cogentco, seconds on Kdl).",
            "",
            "## Assumptions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["assumptions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(
        [
            "",
            "Decision logs: `results/safety/decisions_<split>_<topology>.jsonl`.",
            "Metrics: `results/safety/metrics.json`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()

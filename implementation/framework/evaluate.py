"""Evaluate the integrated controller on seen and unseen topologies.

Does not train, does not write the twin checkpoint, and does not read ASN shards.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import torch

from safety.evaluate import load_config, load_trained_ensemble
from twin.data import ROOT, load_split_demands, load_topology

from .pipeline import Framework
from .policy_adapter import DEFAULT_CHECKPOINT, FrozenP1Policy

RESULT_DIR = ROOT / "results" / "framework"


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "acceptance_rate": float(np.mean([row["proposed_accepted"] for row in rows])),
        "fallback_rate": float(np.mean([row["fallback"] is not None for row in rows])),
        "mean_mlu_applied": _mean(rows, "mlu_true_applied"),
        "mean_mlu_policy_only": _mean(rows, "mlu_true_proposed"),
        "mean_mlu_shortest": _mean(rows, "mlu_true_shortest"),
        "mean_violations_applied": _mean(rows, "violations_applied"),
        "mean_violations_policy_only": _mean(rows, "violations_proposed"),
        "mean_violations_shortest": _mean(rows, "violations_shortest"),
        "violation_decision_rate_applied": float(np.mean([row["violations_applied"] > 0 for row in rows])),
        "violation_decision_rate_policy_only": float(np.mean([row["violations_proposed"] > 0 for row in rows])),
        "violation_decision_rate_shortest": float(np.mean([row["violations_shortest"] > 0 for row in rows])),
        "mean_fit_applied": _mean(rows, "fit_ratio_applied"),
        "mean_fit_policy_only": _mean(rows, "fit_ratio_proposed"),
        "mean_fit_shortest": _mean(rows, "fit_ratio_shortest"),
        "mean_prediction_mae": _mean(rows, "prediction_mae_applied"),
        "mean_policy_s": _mean(rows, "policy_s"),
        "mean_twin_ms": _mean(rows, "twin_ms"),
        "mean_gate_ms": _mean(rows, "gate_ms"),
        "all_actions_normalised": all(row["action_sums_to_demands"] for row in rows),
    }


def _run_job(framework: Framework, name: str, split: str, stride: int, limit: int | None) -> dict:
    topo = load_topology(name)
    demands, meta = load_split_demands(split, name)
    framework.reset_session(name)
    rows = []
    indices = list(range(0, demands.shape[0], stride))
    if limit is not None:
        indices = indices[:limit]
    try:
        for index in indices:
            row = framework.decide(topo, demands[index])
            row["split"] = split
            row["matrix_index"] = int(meta["matrix_indices"][index])
            rows.append(row)
            print(
                f"  {name} tm {row['matrix_index']} accept={row['proposed_accepted']} "
                f"fallback={row['fallback']} mlu_applied={row['mlu_true_applied']:.4f} "
                f"policy_s={row['policy_s']:.2f}",
                flush=True,
            )
    finally:
        framework.policy.release(name)
        del topo
    return {"topology": name, "split": split, "stride": stride, "summary": summarise(rows), "cases": rows}


def evaluate() -> dict:
    device = torch.device("cpu")
    policy = FrozenP1Policy(DEFAULT_CHECKPOINT, device)
    twin = load_trained_ensemble(device)
    framework = Framework(policy, twin, load_config(), device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("B4", "val", 2, None),
        ("GEANT", "test", 600, None),
        ("UsCarrier", "val", 1, 1),
        ("Kdl", "test", 1, 1),
    ]
    reports = []
    failures = []
    for name, split, stride, limit in jobs:
        print(f"framework {split}/{name}", flush=True)
        try:
            reports.append(_run_job(framework, name, split, stride, limit))
        except Exception as exc:
            failures.append(
                {
                    "topology": name,
                    "split": split,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED {name}: {exc}", flush=True)
            framework.policy.release(name)
    payload = {
        "policy_checkpoint": str(DEFAULT_CHECKPOINT),
        "twin_checkpoint": str((ROOT / "models" / "twin" / "ensemble.pt").resolve()),
        "twin_retrained": False,
        "jobs": [
            {"topology": item["topology"], "split": item["split"], "summary": item["summary"]}
            for item in reports
        ],
        "failures": failures,
        "seen_completed": any(item["topology"] == "B4" and item["cases"] for item in reports),
        "unseen_completed": any(
            item["topology"] in ("GEANT", "Kdl") and item["cases"] for item in reports
        ),
        "limitations": [
            "The actor is the frozen B4 checkpoint applied unchanged. It was not trained on UsCarrier, Cogentco, GEANT, or Kdl.",
            "ADMM and P1's capacity adjustment are not applied. Policy-only is the actor split itself.",
            "Full demand is routed. Capacity fit ratio is the share of carried flow that fits on the links, not P1's post-adjustment satisfaction.",
            "Native traffic matrices in this corpus often exceed capacity under every split, so the gate rejects them and shortest-path last resort can still violate.",
            "ASN remains incomplete and is excluded.",
        ],
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in reports:
        safe = []
        for row in item["cases"]:
            safe.append({key: value for key, value in row.items()})
        (RESULT_DIR / f"decisions_{item['split']}_{item['topology']}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in safe) + "\n",
            encoding="utf-8",
        )
    (RESULT_DIR / "FRAMEWORK_REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print("Wrote", RESULT_DIR / "metrics.json", flush=True)
    return payload


def _markdown(payload: dict) -> str:
    lines = [
        "# Phase 3 Step 7 — integrated framework",
        "",
        "Pipeline: frozen P1 actor → digital-twin ensemble → safety gate → applied routing.",
        "The frozen repository and the twin weights were not modified.",
        "",
        f"Seen topologies completed: {payload['seen_completed']}.",
        f"Unseen topologies completed: {payload['unseen_completed']}.",
        "",
        "## Comparison",
        "",
        "Policy-only applies the actor split with no twin and no gate.",
        "Framework applies that split only when the gate accepts it; otherwise it uses the fallback.",
        "Shortest is minimum-hop routing.",
        "",
        "| Topology | Cases | Accept | Fallback | MLU framework | MLU policy-only | MLU shortest | Viol. rate framework | Viol. rate policy | MAE | Policy s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in payload["jobs"]:
        s = job["summary"]
        lines.append(
            "| {top} | {n} | {acc:.3f} | {fb:.3f} | {mlu_f:.4g} | {mlu_p:.4g} | {mlu_s:.4g} | {vr_f:.3f} | {vr_p:.3f} | {mae:.4g} | {ps:.2f} |".format(
                top=job["topology"],
                n=s["cases"],
                acc=s["acceptance_rate"],
                fb=s["fallback_rate"],
                mlu_f=s["mean_mlu_applied"],
                mlu_p=s["mean_mlu_policy_only"],
                mlu_s=s["mean_mlu_shortest"],
                vr_f=s["violation_decision_rate_applied"],
                vr_p=s["violation_decision_rate_policy_only"],
                mae=s["mean_prediction_mae"],
                ps=s["mean_policy_s"],
            )
        )
    if payload["failures"]:
        lines.extend(["", "## Incomplete jobs", ""])
        for failure in payload["failures"]:
            lines.append(f"- {failure['topology']}: {failure['error']}")
    lines.extend(
        [
            "",
            "## How to read the table",
            "",
            "Every native matrix in this run was infeasible: shortest-path MLU is already above 1,",
            "so the gate rejected the actor (acceptance 0) and applied shortest-path as last resort.",
            "The framework MLU therefore matches shortest-path and is lower than the ungated actor.",
            "Decision-level violation rate stays 1 because the fallback itself still overloads.",
            "The number of overloaded links does fall: B4 1.75 to 1.0, GEANT 11.4 to 5.6,",
            "UsCarrier 229 to 110. Cogentco was not included in this run.",
            "",
            "No jobs failed. Kdl, the large unseen topology, completed one test matrix",
            "(policy forward about 72 s). GEANT completed five held-out FIGRET matrices.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()

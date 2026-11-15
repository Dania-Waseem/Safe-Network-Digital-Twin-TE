"""Paired A–F evaluation runner (infrastructure).

Default CLI is smoke-only. Full A–F experiments must be requested explicitly
via ``--full`` and are blocked from threshold tuning on GEANT/Kdl here.

Does not modify original gate, twin, RL, or prior result trees. Writes only
under ``results/safety_gate/improvements/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twin.data import ROOT, load_split_demands, load_topology
from .config import DEFAULT_CONFIG_PATH, load_improvements_config
from .decision import VARIANT_BACKEND, VARIANT_IDS, decide_variant
from .diagnostics import build_case_diagnostics, summarize_diagnostics, twin_uncertainty_views
from .graph_cache import TopologyGraphCache
from .logging_util import validate_decision_record
from .policy_backend import PolicyBackend, policy_backend_from_config
from .resume import append_decision_rows, case_id, load_completed_case_ids, read_decision_rows
from .twin_backend import TwinBackend, backends_from_config
from .metrics import compute_safety_metrics


RESULT_DIR = ROOT / "results" / "safety_gate" / "improvements"
SMOKE_DIR = RESULT_DIR / "smoke"


def _action_fingerprint(action: np.ndarray) -> str:
    arr = np.asarray(action, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _resolve_cases(cfg: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    """Build a list of evaluation cases. Smoke uses a tiny B4-only slice."""
    evaluation = cfg.get("evaluation", {})
    if smoke:
        smoke_cfg = evaluation.get("smoke", {})
        topologies = smoke_cfg.get("topologies", ["B4"])
        split = smoke_cfg.get("split", "val")
        n_matrices = int(smoke_cfg.get("n_matrices", 1))
        scales = smoke_cfg.get("scales", [0.5, 1.0])
        seeds = smoke_cfg.get("seeds", [101])
        cases = []
        for seed in seeds:
            for name in topologies:
                topo_split = split
                if name in evaluation.get("held_out_topologies", ["GEANT", "Kdl"]):
                    topo_split = smoke_cfg.get("held_out_split", "test")
                demands, meta = load_split_demands(topo_split, name)
                ids = meta.get("matrix_indices", list(range(demands.shape[0])))
                for local_i in range(min(n_matrices, demands.shape[0])):
                    for scale in scales:
                        cases.append(
                            {
                                "seed": int(seed),
                                "topology": name,
                                "split": topo_split,
                                "local_index": int(local_i),
                                "matrix_id": int(ids[local_i]),
                                "scale": float(scale),
                                "unseen": name in evaluation.get("held_out_topologies", []),
                            }
                        )
        return cases

    # Full evaluation: prefer frozen robustness_v1 catalog indices when configured.
    seeds = evaluation.get("seeds", [101])
    scales = evaluation.get("scales", [1.0])
    held_out = set(evaluation.get("held_out_topologies", ["GEANT", "Kdl"]))
    catalog_path = evaluation.get("robustness_catalog")
    if catalog_path:
        path = Path(catalog_path)
        if not path.is_absolute():
            path = ROOT / catalog_path
        catalog = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(evaluation.get("authorized_topologies") or evaluation.get("seen_topologies") or [])
        cases = []
        for name, spec in catalog["topologies"].items():
            if allowed and name not in allowed:
                continue
            if name in held_out:
                # Explicit exclusion: never silently evaluate held-out topos in this runner path.
                continue
            split = spec["split"]
            for seed in seeds:
                entry = spec["by_seed"][str(seed)]
                for local_i, matrix_id in zip(entry["local_indices"], entry["matrix_ids"]):
                    for scale in scales:
                        cases.append(
                            {
                                "seed": int(seed),
                                "topology": name,
                                "split": split,
                                "local_index": int(local_i),
                                "matrix_id": int(matrix_id),
                                "scale": float(scale),
                                "unseen": False,
                                "catalog": "robustness_v1",
                            }
                        )
        return cases

    topo_spec = evaluation.get(
        "topology_spec",
        [
            {"name": "B4", "split": "val", "count": 2},
            {"name": "UsCarrier", "split": "val", "count": 1},
            {"name": "Cogentco", "split": "val", "count": 1},
        ],
    )
    cases = []
    for seed in seeds:
        for spec in topo_spec:
            name = spec["name"]
            if name in held_out:
                continue
            split = spec["split"]
            count = int(spec["count"])
            demands, meta = load_split_demands(split, name)
            ids = meta.get("matrix_indices", list(range(demands.shape[0])))
            from framework.robustness_eval import sample_matrix_indices

            locals_idx = sample_matrix_indices(demands.shape[0], count, int(seed), name)
            for local_i in locals_idx:
                for scale in scales:
                    cases.append(
                        {
                            "seed": int(seed),
                            "topology": name,
                            "split": split,
                            "local_index": int(local_i),
                            "matrix_id": int(ids[local_i]),
                            "scale": float(scale),
                            "unseen": False,
                        }
                    )
    return cases


def _assert_no_unseen_tuning(cfg: dict[str, Any]) -> None:
    evaluation = cfg.get("evaluation", {})
    tune_on = set(evaluation.get("tune_on", []))
    never = set(evaluation.get("never_tune_on", ["GEANT", "Kdl"]))
    overlap = tune_on & never
    if overlap:
        raise RuntimeError(f"config illegally tunes on held-out topologies: {sorted(overlap)}")


def evaluate_paired_case(
    *,
    case: dict[str, Any],
    topo,
    matrix: np.ndarray,
    policy_action: np.ndarray,
    policy_meta: dict[str, Any],
    backends: dict[str, TwinBackend],
    cfg: dict[str, Any],
    variants: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    """Run all variants on one shared (matrix, policy_action)."""
    action_fp = _action_fingerprint(policy_action)
    rows: list[dict[str, Any]] = []

    # Cache twin predictions per backend for this case (fair + efficient).
    twin_cache: dict[str, Any] = {}

    for variant_id in variants:
        record = decide_variant(
            variant_id,
            topo=topo,
            matrix=matrix,
            policy_action=policy_action,
            backends=backends,
            cfg=cfg,
            seed=case["seed"],
            matrix_id=case["matrix_id"],
            scale=case["scale"],
            policy_checkpoint=policy_meta.get("checkpoint_path"),
            policy_checkpoint_id=policy_meta.get("checkpoint_id"),
        )
        record["split"] = case.get("split")
        record["local_index"] = case.get("local_index")
        record["unseen"] = bool(case.get("unseen"))
        record["policy_action_fingerprint"] = action_fp
        record["proposed_action_source"] = "policy"
        record.update({f"policy_{k}": v for k, v in policy_meta.items() if k != "action"})

        backend_name = VARIANT_BACKEND.get(variant_id)
        twin_pred = None
        conformal_scale = 1.0
        if backend_name and backend_name in backends:
            if backend_name not in twin_cache:
                twin_cache[backend_name] = backends[backend_name].predict(topo, matrix, policy_action)
            twin_pred = twin_cache[backend_name]
            # Diagnostic scale: use calibrated twin's factor for improved models;
            # original twin has no conformal factor (identity).
            if backend_name in ("improved", "improved_calibrated"):
                cal_backend = backends.get("improved_calibrated", backends[backend_name])
                conformal_scale = float(cal_backend.resolved_conformal_scale())
            else:
                conformal_scale = 1.0
            floor = np.maximum(
                topo.capacity, float(cfg.get("gate", {}).get("min_capacity", 1e-12))
            )
            views = twin_uncertainty_views(twin_pred, conformal_std_scale=conformal_scale)
            raw_util = views["raw_std_load"] / floor
            cal_util = views["calibrated_std_load"] / floor
            record["raw_uncertainty_utilisation"] = {
                "mean": float(np.mean(raw_util)),
                "max": float(np.max(raw_util)),
                "p95": float(np.percentile(raw_util, 95)),
            }
            record["calibrated_uncertainty_utilisation"] = {
                "mean": float(np.mean(cal_util)),
                "max": float(np.max(cal_util)),
                "p95": float(np.percentile(cal_util, 95)),
            }

        diag = build_case_diagnostics(
            topo=topo,
            matrix=matrix,
            policy_action=policy_action,
            decision_record=record,
            twin_pred=twin_pred,
            gate_k=float(record.get("gate_k") or 0.0),
            min_capacity=float(cfg.get("gate", {}).get("min_capacity", 1e-12)),
            conformal_std_scale=conformal_scale if twin_pred is not None else 1.0,
        )
        # Decision-record fields win for acceptance; diagnostics fill the rest.
        for key, value in diag.items():
            if key not in record or record[key] is None:
                record[key] = value
            elif key in (
                "actual_utilisation",
                "actual_mlu",
                "capacity_proximity",
                "absolute_prediction_error",
                "utilisation_prediction_error",
                "twin_mae_abs",
                "twin_mae_util",
                "raw_uncertainty_utilisation",
                "calibrated_uncertainty_utilisation",
            ):
                record[key] = value

        missing = validate_decision_record(record)
        record["log_missing_fields"] = missing
        rows.append(record)

    # Fairness: identical policy action fingerprint across the paired block.
    fps = {r["policy_action_fingerprint"] for r in rows}
    if len(fps) != 1:
        raise RuntimeError(f"paired fairness violated: fingerprints={fps}")
    return rows


def run_evaluation(
    cfg: dict[str, Any] | None = None,
    *,
    smoke: bool = True,
    allow_full: bool = False,
    output_dir: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute smoke (default) or full paired evaluation infrastructure run.

    When ``resume`` is True (default), completed ``case_id`` blocks already
    present in ``decisions.jsonl`` are skipped and new rows are appended.
    """
    cfg = cfg or load_improvements_config()
    _assert_no_unseen_tuning(cfg)

    if not smoke and not allow_full:
        raise RuntimeError(
            "Full A–F evaluation is disabled in this stage. "
            "Pass allow_full=True / --full only when intentionally starting experiments."
        )

    evaluation = cfg.get("evaluation", {})
    device = torch.device(str(evaluation.get("device", "cpu")))
    variants = tuple(evaluation.get("variants", VARIANT_IDS))
    out_dir = Path(output_dir or (SMOKE_DIR if smoke else RESULT_DIR / "eval_runs" / "manual"))
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "decisions.jsonl"

    graph_cache = TopologyGraphCache(device)
    policy = policy_backend_from_config(cfg, device=device, graph_cache=graph_cache)
    policy.load()
    backends = backends_from_config(cfg, device)

    cases = _resolve_cases(cfg, smoke=smoke)
    started = time.perf_counter()
    completed_ids = (
        load_completed_case_ids(decisions_path, expected_variants=len(variants))
        if resume
        else set()
    )
    if not resume and decisions_path.exists():
        decisions_path.unlink()

    new_rows: list[dict[str, Any]] = []
    skipped_cases = 0
    failed_cases: list[dict[str, Any]] = []
    topo_cache: dict[str, Any] = {}
    demand_cache: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}

    for case in cases:
        cid = case_id(case, checkpoint_id=policy.checkpoint_id)
        if resume and cid in completed_ids:
            skipped_cases += 1
            continue

        name = case["topology"]
        try:
            if name not in topo_cache:
                topo_cache[name] = load_topology(name)
            topo = topo_cache[name]
            dkey = (case["split"], name)
            if dkey not in demand_cache:
                demand_cache[dkey] = load_split_demands(case["split"], name)
            demands, _meta = demand_cache[dkey]
            if case["local_index"] >= demands.shape[0]:
                raise IndexError(
                    f"missing traffic matrix local_index={case['local_index']} "
                    f"for {name}/{case['split']} (n={demands.shape[0]})"
                )
            matrix = demands[case["local_index"]] * float(case["scale"])

            proposal = policy.propose(topo, matrix)
            policy_meta = proposal.to_log_dict()
            rows = evaluate_paired_case(
                case=case,
                topo=topo,
                matrix=matrix,
                policy_action=proposal.action,
                policy_meta=policy_meta,
                backends=backends,
                cfg=cfg,
                variants=variants,
            )
            for row in rows:
                row["case_id"] = cid
                row["decision_status"] = "ok"
            append_decision_rows(decisions_path, rows)
            completed_ids.add(cid)
            new_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001 — record and continue for resume safety
            fail_row = {
                "case_id": cid,
                "decision_status": "failed",
                "failure_stage": "case_execution",
                "failure_message": str(exc),
                "seed": case.get("seed"),
                "topology": case.get("topology"),
                "matrix_id": case.get("matrix_id"),
                "scale": case.get("scale"),
                "gate_variant": None,
                "acceptance": None,
                "policy_checkpoint_id": policy.checkpoint_id,
            }
            append_decision_rows(decisions_path, [fail_row])
            failed_cases.append(fail_row)

    all_rows = read_decision_rows(decisions_path)
    ok_rows = [r for r in all_rows if r.get("decision_status", "ok") == "ok" and "acceptance" in r]
    elapsed_s = time.perf_counter() - started
    summary = summarize_diagnostics(ok_rows)
    summary["metrics_by_variant"] = {}
    for vid in variants:
        subset = [r for r in all_rows if r.get("gate_variant") == vid]
        summary["metrics_by_variant"][vid] = compute_safety_metrics(subset)
    summary["n_cases"] = len(cases)
    summary["n_rows"] = len(ok_rows)
    summary["n_logged_lines"] = len(all_rows)
    summary["n_variants"] = len(variants)
    summary["n_cases_skipped_resume"] = skipped_cases
    summary["n_cases_failed"] = len(failed_cases)
    summary["elapsed_s"] = elapsed_s
    summary["smoke"] = smoke
    summary["resume"] = resume
    summary["policy_checkpoint"] = str(policy.checkpoint_path)
    summary["policy_checkpoint_id"] = policy.checkpoint_id
    summary["graph_cache"] = graph_cache.metadata()
    summary["fairness"] = {
        "paired_action_fingerprint_unique_per_case": True,
        "note": "Each case block shares one policy_action_fingerprint across A–F.",
    }
    summary["protocol"] = {
        "never_tune_on": evaluation.get("never_tune_on", ["GEANT", "Kdl"]),
        "tune_on": evaluation.get("tune_on", []),
        "calibration_topologies": "B4/UsCarrier/Cogentco only (frozen twin conformal; not refit here)",
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    meta = {
        "config_path": cfg.get("_config_path"),
        "output_dir": str(out_dir),
        "smoke": smoke,
        "allow_full": allow_full,
        "resume": resume,
        "n_cases": len(cases),
        "n_rows": len(ok_rows),
        "n_logged_lines": len(all_rows),
        "n_cases_skipped_resume": skipped_cases,
        "n_cases_failed": len(failed_cases),
        "variants": list(variants),
        "preservation": {
            "original_twin_sha256_expected": "7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685",
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "rows": ok_rows,
        "all_logged_rows": all_rows,
        "summary": summary,
        "meta": meta,
        "output_dir": str(out_dir),
        "decisions_path": str(decisions_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safety Gate improvements evaluation runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Run limited smoke cases only (default).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly allow full catalog evaluation (not for this infrastructure stage).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing decisions.jsonl and rewrite from scratch.",
    )
    args = parser.parse_args(argv)

    smoke = not args.full
    if args.full:
        # Full catalogue evaluation runs only with --full;
        # infrastructure stage stops after smoke. Keep --full available but warn.
        print(
            "WARNING: --full requested. This stage's protocol prefers smoke only; "
            "proceeding because --full was explicit."
        )
        smoke = False

    cfg = load_improvements_config(args.config)
    result = run_evaluation(
        cfg,
        smoke=smoke,
        allow_full=bool(args.full),
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "n_rows": result["meta"]["n_rows"],
                "n_cases": result["meta"]["n_cases"],
                "smoke": result["meta"]["smoke"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

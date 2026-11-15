"""Paired A–F decision variants for the improved Safety Gate stack."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import gate_params_from_config
from .fallback import DEFAULT_CANDIDATES, select_physics_fallback
from .gate_logic import ImprovedGateConfig, ImprovedGateVerdict, inspect_utilisation_ucb
from .oracle import action_link_load, mlu, oracle_accepts_action, physics_violations
from .twin_backend import TwinBackend, TwinPrediction

VARIANT_IDS = (
    "A_policy_only",
    "B_original_twin_mean",
    "C_original_twin_uncertainty",
    "D_improved_twin_mean",
    "E_improved_twin_calibrated_uncertainty",
    "F_physics_oracle",
)

# Maps variant → twin backend name in config (None = no twin).
VARIANT_BACKEND = {
    "A_policy_only": None,
    "B_original_twin_mean": "original",
    "C_original_twin_uncertainty": "original",
    "D_improved_twin_mean": "improved",
    "E_improved_twin_calibrated_uncertainty": "improved_calibrated",
    "F_physics_oracle": None,
}

MEAN_ONLY_VARIANTS = {"B_original_twin_mean", "D_improved_twin_mean"}
UNCERTAINTY_VARIANTS = {
    "C_original_twin_uncertainty",
    "E_improved_twin_calibrated_uncertainty",
}


def _scalar_stats(arr: np.ndarray | None) -> dict[str, float]:
    if arr is None or arr.size == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _gate_for_variant(cfg: dict[str, Any], variant_id: str) -> ImprovedGateConfig:
    params = gate_params_from_config(cfg, variant_id)
    gate = ImprovedGateConfig.from_dict(params)
    if variant_id in MEAN_ONLY_VARIANTS:
        # Isolate uncertainty contribution: force k=0 and disable relative-std veto.
        gate.k = 0.0
        gate.max_relative_std = None
    return gate


def decide_variant(
    variant_id: str,
    *,
    topo,
    matrix: np.ndarray,
    policy_action: np.ndarray,
    backends: dict[str, TwinBackend],
    cfg: dict[str, Any],
    seed: int | None = None,
    matrix_id: str | int | None = None,
    scale: float | None = None,
    policy_checkpoint: str | None = None,
    policy_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run one A–F variant on a shared (topology, matrix, policy action) triple."""
    if variant_id not in VARIANT_IDS:
        raise KeyError(f"unknown variant {variant_id!r}; expected one of {VARIANT_IDS}")

    started = time.perf_counter()
    capacity = np.asarray(topo.capacity, dtype=np.float64)
    actor_load = action_link_load(topo, matrix, policy_action)
    actor_viol = physics_violations(actor_load, capacity)
    oracle_feasible, _, _ = oracle_accepts_action(topo, matrix, policy_action)

    twin_pred: TwinPrediction | None = None
    verdict: ImprovedGateVerdict | None = None
    twin_backend_name = VARIANT_BACKEND[variant_id]
    gate_cfg = _gate_for_variant(cfg, variant_id)

    accepted: bool
    reasons: list[str]
    proposed_accepted: bool

    if variant_id == "A_policy_only":
        accepted = True
        proposed_accepted = True
        reasons = []
    elif variant_id == "F_physics_oracle":
        accepted = bool(oracle_feasible)
        proposed_accepted = accepted
        reasons = [] if accepted else ["oracle_capacity_violation"]
    else:
        if twin_backend_name is None or twin_backend_name not in backends:
            raise KeyError(f"variant {variant_id} requires twin backend {twin_backend_name!r}")
        twin_pred = backends[twin_backend_name].predict(topo, matrix, policy_action)
        std_for_gate = (
            np.zeros_like(twin_pred.mean_load)
            if variant_id in MEAN_ONLY_VARIANTS
            else twin_pred.std_load
        )
        verdict = inspect_utilisation_ucb(
            twin_pred.mean_load,
            std_for_gate,
            capacity,
            gate_cfg,
        )
        accepted = bool(verdict.accepted)
        proposed_accepted = accepted
        reasons = list(verdict.reasons)

    fallback_info = None
    applied_name = "policy"
    applied_action = policy_action
    applied_load = actor_load
    fallback_used = False

    candidate_names = tuple(
        cfg.get("fallback", {}).get("candidates", list(DEFAULT_CANDIDATES))
    )

    if variant_id != "A_policy_only" and not accepted:
        fb = select_physics_fallback(topo, matrix, candidate_names=candidate_names)
        fallback_info = fb
        fallback_used = True
        applied_name = fb.selected
        applied_action = fb.selected_action
        applied_load = fb.selected_load

    applied_viol = physics_violations(applied_load, capacity)
    decision_ms = (time.perf_counter() - started) * 1000.0

    mean_util = None
    std_util = None
    ucb = None
    twin_std_util = None
    if twin_pred is not None:
        floor = np.maximum(capacity, float(gate_cfg.min_capacity))
        twin_mean_util = twin_pred.mean_load / floor
        twin_std_util = twin_pred.std_load / floor
        # Log Twin predictive uncertainty even for mean-only (gate may zero std).
        mean_util = twin_mean_util
        std_util = twin_std_util
        if verdict is not None:
            ucb = verdict.ucb
        else:
            ucb = twin_mean_util + float(gate_cfg.k) * twin_std_util
    elif verdict is not None:
        mean_util = verdict.mean_util
        std_util = verdict.std_util
        ucb = verdict.ucb

    record: dict[str, Any] = {
        "seed": seed,
        "topology": getattr(topo, "name", None),
        "matrix_id": matrix_id,
        "scale": scale,
        "policy_checkpoint": policy_checkpoint
        or cfg.get("policy", {}).get("default_checkpoint"),
        "policy_checkpoint_id": policy_checkpoint_id
        or cfg.get("policy", {}).get("checkpoint_id"),
        "gate_variant": variant_id,
        "twin_backend": twin_backend_name or "none",
        "proposed_action_source": "policy",
        "proposed_accepted": proposed_accepted,
        "acceptance": accepted,
        "rejection_reasons": list(reasons),
        "rejection_reason": reasons[0] if reasons else None,
        "fallback_used": fallback_used,
        "selected_fallback": None if not fallback_used else applied_name,
        "fallback_feasible": None
        if fallback_info is None
        else bool(fallback_info.feasible_fallback),
        "no_feasible_fallback": None
        if fallback_info is None
        else bool(fallback_info.no_feasible_fallback),
        "fallback_outcome": None if fallback_info is None else fallback_info.outcome,
        "applied_action_source": applied_name,
        "oracle_feasible_proposed": bool(oracle_feasible),
        "capacity_violations_proposed": int(actor_viol),
        "capacity_violations_applied": int(applied_viol),
        "mlu_proposed": mlu(actor_load, capacity),
        "mlu_applied": mlu(applied_load, capacity),
        "unsafe_accept": bool(accepted and actor_viol > 0),
        "false_reject": bool((not accepted) and actor_viol == 0),
        "decision_runtime_ms": float(decision_ms),
        "twin_runtime_ms": None if twin_pred is None else float(twin_pred.twin_ms),
        "conformal_std_scale_applied": None
        if twin_pred is None
        else float(twin_pred.conformal_std_scale_applied),
        "conformal_applied": None if twin_pred is None else bool(twin_pred.conformal_applied),
        "gate_k": float(gate_cfg.k),
        "gate_utilisation_limit": float(gate_cfg.utilisation_limit),
        "gate_max_relative_std": gate_cfg.max_relative_std,
        "predicted_mean_utilisation": _scalar_stats(mean_util),
        "predicted_uncertainty_utilisation": _scalar_stats(std_util),
        "ucb_utilisation": _scalar_stats(ucb),
        "gate_used_uncertainty": variant_id in UNCERTAINTY_VARIANTS,
        "mlu_mean_predicted": None if verdict is None else float(verdict.mlu_mean),
        "mlu_ucb_predicted": None if verdict is None else float(verdict.mlu_ucb),
        "worst_link": None if verdict is None else int(verdict.worst_link),
        "n_links_over_ucb_limit": None if verdict is None else int(verdict.n_links_over_limit),
        "twin_checkpoint": None if twin_pred is None else twin_pred.checkpoint_path,
    }
    if fallback_info is not None:
        record["fallback_detail"] = fallback_info.to_log_dict()
    return record


def decide_all_variants(
    *,
    topo,
    matrix: np.ndarray,
    policy_action: np.ndarray,
    backends: dict[str, TwinBackend],
    cfg: dict[str, Any],
    variants: tuple[str, ...] | list[str] = VARIANT_IDS,
    **meta: Any,
) -> list[dict[str, Any]]:
    """Run every requested variant on the same observation (paired comparison)."""
    return [
        decide_variant(
            vid,
            topo=topo,
            matrix=matrix,
            policy_action=policy_action,
            backends=backends,
            cfg=cfg,
            **meta,
        )
        for vid in variants
    ]

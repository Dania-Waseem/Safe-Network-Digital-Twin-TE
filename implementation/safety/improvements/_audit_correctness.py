"""One-shot correctness audit runner (not part of the public API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from framework.robustness_eval import IMPROVED_BEST, IMPROVED_STABLE
from safety.evaluate import load_trained_ensemble, predict_action
from safety.improvements.config import load_improvements_config
from safety.improvements.decision import decide_all_variants, decide_variant
from safety.improvements.fallback import DEFAULT_CANDIDATES, select_physics_fallback
from safety.improvements.gate_logic import (
    REASON_INVALID,
    ImprovedGateConfig,
    inspect_utilisation_ucb,
)
from safety.improvements.twin_backend import backends_from_config
from twin.data import IMPL, ROOT, load_split_demands, load_topology
from twin.improvements.infer import load_ensemble_checkpoint, predict_loads
from twin.physics import equal_split, od_demands


def main() -> dict:
    device = torch.device("cpu")
    cfg = load_improvements_config()
    backends = backends_from_config(cfg, device)
    topo = load_topology("B4")
    demands, _meta = load_split_demands("val", "B4")
    matrix = demands[0]
    action = equal_split(topo.n_paths)

    report: dict = {"checks": [], "issues": [], "warnings": []}

    def ok(name: str, detail: str, passed: bool = True) -> None:
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            report["issues"].append(f"{name}: {detail}")

    def warn(name: str, detail: str) -> None:
        report["warnings"].append({"name": name, "detail": detail})

    orig_path = Path(cfg["twins"]["original"]["path"])
    imp_path = Path(cfg["twins"]["improved"]["path"])
    cal_path = Path(cfg["twins"]["improved_calibrated"]["path"])
    ok(
        "paths_exist",
        f"orig={orig_path.exists()} imp={imp_path.exists()} cal={cal_path.exists()}",
        orig_path.exists() and imp_path.exists() and cal_path.exists(),
    )
    sha = hashlib.sha256(orig_path.read_bytes()).hexdigest()
    ok(
        "original_sha",
        sha,
        sha == "7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685",
    )

    _m_orig, p_orig = load_ensemble_checkpoint(orig_path, device)
    _m_imp, p_imp = load_ensemble_checkpoint(imp_path, device)
    m_cal, p_cal = load_ensemble_checkpoint(cal_path, device)
    report["architecture"] = {
        "original": {
            "n_members": p_orig.get("n_members"),
            "hidden": p_orig.get("hidden"),
            "n_params": int(sum(v.numel() for v in p_orig["state_dict"].values())),
        },
        "improved": {
            "n_members": p_imp.get("n_members"),
            "hidden": p_imp.get("hidden"),
            "n_params": int(sum(v.numel() for v in p_imp["state_dict"].values())),
            "conformal_std_scale": p_imp.get("conformal_std_scale", None),
        },
        "calibrated": {
            "n_members": p_cal.get("n_members"),
            "hidden": p_cal.get("hidden"),
            "n_params": int(sum(v.numel() for v in p_cal["state_dict"].values())),
            "conformal_std_scale": p_cal.get("conformal_std_scale"),
        },
    }
    ok(
        "arch_match_members",
        str(p_orig.get("n_members")),
        p_orig.get("n_members") == p_imp.get("n_members") == p_cal.get("n_members") == 5,
    )
    ok(
        "arch_match_hidden",
        str(p_orig.get("hidden")),
        p_orig.get("hidden") == p_imp.get("hidden") == p_cal.get("hidden"),
    )
    same_weights = all(
        torch.equal(p_imp["state_dict"][k], p_cal["state_dict"][k]) for k in p_imp["state_dict"]
    )
    ok("improved_vs_calibrated_weights_identical", f"same={same_weights}", same_weights)
    ok(
        "calibrated_has_scale",
        str(p_cal.get("conformal_std_scale")),
        p_cal.get("conformal_std_scale") is not None
        and abs(float(p_cal["conformal_std_scale"]) - 1.0) > 1e-9,
    )
    ok(
        "improved_anchored_scale_is_one",
        str(p_imp.get("conformal_std_scale")),
        float(p_imp.get("conformal_std_scale", 1.0)) == 1.0,
    )

    legacy = load_trained_ensemble(device)
    t1, m1, s1, _ = predict_action(legacy, topo, matrix, action, device)
    pred_o = backends["original"].predict(topo, matrix, action)
    ok(
        "original_mean_matches_legacy",
        f"maxabs={float(np.max(np.abs(m1 - pred_o.mean_load))):.3e}",
        np.allclose(m1, pred_o.mean_load, rtol=0, atol=0),
    )
    ok(
        "original_std_matches_legacy",
        f"maxabs={float(np.max(np.abs(s1 - pred_o.std_load))):.3e}",
        np.allclose(s1, pred_o.std_load, rtol=0, atol=0),
    )
    ok(
        "original_true_matches",
        f"maxabs={float(np.max(np.abs(t1 - pred_o.true_load))):.3e}",
        np.allclose(t1, pred_o.true_load),
    )
    ok(
        "shapes_n_links",
        f"mean={pred_o.mean_load.shape} cap={topo.capacity.shape}",
        pred_o.mean_load.shape == topo.capacity.shape == (topo.n_links,),
    )
    ok(
        "units_absolute_load",
        f"max_mean={float(pred_o.mean_load.max()):.4g} denom={topo.denominator}",
        float(pred_o.mean_load.max()) > 1.0,
    )

    pred_i = backends["improved"].predict(topo, matrix, action)
    pred_c = backends["improved_calibrated"].predict(topo, matrix, action)
    ok("improved_shapes", str(pred_i.mean_load.shape), pred_i.mean_load.shape == topo.capacity.shape)
    ok("calibrated_shapes", str(pred_c.mean_load.shape), pred_c.mean_load.shape == topo.capacity.shape)

    ok(
        "D_E_means_identical_backends",
        f"maxabs={float(np.max(np.abs(pred_i.mean_load - pred_c.mean_load))):.3e}",
        np.allclose(pred_i.mean_load, pred_c.mean_load, rtol=1e-7, atol=1e-9),
    )
    scale = float(pred_c.conformal_std_scale_applied)
    _raw_true, _raw_mean, raw_std, _ = predict_loads(
        m_cal, topo, matrix, action, device, conformal_std_scale=1.0
    )
    once = raw_std * scale
    twice = raw_std * scale * scale
    ok(
        "E_scale_applied_once",
        f"scale={scale} max|E-once|={float(np.max(np.abs(pred_c.std_load - once))):.3e}",
        np.allclose(pred_c.std_load, once) and not np.allclose(pred_c.std_load, twice),
    )
    ok(
        "D_std_unscaled",
        f"max|D-raw|={float(np.max(np.abs(pred_i.std_load - raw_std))):.3e}",
        np.allclose(pred_i.std_load, raw_std, rtol=1e-7, atol=1e-9),
    )
    ok(
        "E_loads_correct_ckpt",
        Path(pred_c.checkpoint_path).name,
        Path(pred_c.checkpoint_path).name == "ensemble_physics_anchored_conformal.pt",
    )
    ok(
        "D_loads_anchored_ckpt",
        Path(pred_i.checkpoint_path).name,
        Path(pred_i.checkpoint_path).name == "ensemble_physics_anchored.pt",
    )

    row_d = decide_variant(
        "D_improved_twin_mean",
        topo=topo,
        matrix=matrix,
        policy_action=action,
        backends=backends,
        cfg=cfg,
        seed=1,
        matrix_id=0,
        scale=1.0,
    )
    row_e = decide_variant(
        "E_improved_twin_calibrated_uncertainty",
        topo=topo,
        matrix=matrix,
        policy_action=action,
        backends=backends,
        cfg=cfg,
        seed=1,
        matrix_id=0,
        scale=1.0,
    )
    ok("D_gate_k_zero", str(row_d["gate_k"]), row_d["gate_k"] == 0.0)
    ok("E_gate_k_positive", str(row_e["gate_k"]), row_e["gate_k"] > 0)
    ok("D_not_gate_used_uncertainty", str(row_d["gate_used_uncertainty"]), not row_d["gate_used_uncertainty"])
    ok("E_gate_used_uncertainty", str(row_e["gate_used_uncertainty"]), row_e["gate_used_uncertainty"])
    ok(
        "E_conformal_flag",
        f"applied={row_e['conformal_applied']} scale={row_e['conformal_std_scale_applied']}",
        row_e["conformal_applied"] is True,
    )
    ok(
        "D_no_conformal_flag",
        f"applied={row_d['conformal_applied']} scale={row_d['conformal_std_scale_applied']}",
        row_d["conformal_applied"] is False and row_d["conformal_std_scale_applied"] == 1.0,
    )

    v_mean = inspect_utilisation_ucb(
        pred_i.mean_load, np.zeros_like(pred_i.mean_load), topo.capacity, ImprovedGateConfig(k=0.0)
    )
    v_unc = inspect_utilisation_ucb(
        pred_c.mean_load,
        pred_c.std_load,
        topo.capacity,
        ImprovedGateConfig(k=float(cfg["gate"]["k"])),
    )
    ok(
        "mean_only_ucb_equals_mean_util",
        f"maxdiff={float(np.max(np.abs(v_mean.ucb - v_mean.mean_util))):.3e}",
        np.allclose(v_mean.ucb, v_mean.mean_util),
    )
    ok(
        "uncertainty_ucb_uses_std",
        f"max(ucb-mean)={float(np.max(v_unc.ucb - v_unc.mean_util)):.3e}",
        float(np.max(v_unc.ucb - v_unc.mean_util)) > 0,
    )

    cap = np.array([100.0, 50.0])
    mean = np.array([10.0, 20.0])
    std = np.array([1.0, 1.0])
    ok("positive_cap_accept", "", inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=1)).accepted)
    ok(
        "zero_cap_reject",
        "",
        not inspect_utilisation_ucb(
            np.array([1.0]), np.array([0.0]), np.array([0.0]), ImprovedGateConfig(k=0)
        ).accepted,
    )
    ok(
        "tiny_cap_reject_if_load",
        "",
        not inspect_utilisation_ucb(
            np.array([1.0]), np.array([0.0]), np.array([1e-15]), ImprovedGateConfig(k=0)
        ).accepted,
    )
    ok(
        "nan_cap_invalid",
        "",
        inspect_utilisation_ucb(mean, std, np.array([100.0, np.nan]), ImprovedGateConfig()).reasons
        == [REASON_INVALID],
    )
    ok(
        "inf_cap_invalid",
        "",
        inspect_utilisation_ucb(mean, std, np.array([100.0, np.inf]), ImprovedGateConfig()).reasons
        == [REASON_INVALID],
    )
    ok(
        "zero_load_accept",
        "",
        inspect_utilisation_ucb(np.zeros(2), np.zeros(2), cap, ImprovedGateConfig(k=2)).accepted,
    )
    ok(
        "nan_mean_invalid",
        "",
        inspect_utilisation_ucb(np.array([np.nan, 1.0]), std, cap).reasons == [REASON_INVALID],
    )
    ok(
        "inf_std_invalid",
        "",
        inspect_utilisation_ucb(mean, np.array([1.0, np.inf]), cap).reasons == [REASON_INVALID],
    )
    neg_v = inspect_utilisation_ucb(
        np.array([-5.0, 10.0]), np.array([0.0, 0.0]), cap, ImprovedGateConfig(k=0)
    )
    ok(
        "negative_mean_rejected_as_invalid",
        f"accepted={neg_v.accepted} reasons={neg_v.reasons}",
        (not neg_v.accepted) and neg_v.reasons == [REASON_INVALID],
    )
    ok(
        "overload_reject",
        "",
        not inspect_utilisation_ucb(
            np.array([120.0]), np.array([0.01]), np.array([100.0]), ImprovedGateConfig(k=1)
        ).accepted,
    )
    neg_cap = inspect_utilisation_ucb(
        np.array([1.0]), np.array([0.0]), np.array([-5.0]), ImprovedGateConfig(k=0)
    )
    ok(
        "negative_capacity_rejects",
        f"reasons={neg_cap.reasons} accepted={neg_cap.accepted}",
        (not neg_cap.accepted) and neg_cap.reasons == [REASON_INVALID],
    )

    ok("fallback_candidates_default", str(DEFAULT_CANDIDATES), DEFAULT_CANDIDATES == ("shortest", "equal"))
    fb = select_physics_fallback(topo, matrix * 50.0)
    ok("fallback_selected_in_set", fb.selected, fb.selected in DEFAULT_CANDIDATES)
    ok(
        "fallback_feasibility_xor",
        f"feas={fb.feasible_fallback} none={fb.no_feasible_fallback}",
        fb.feasible_fallback != fb.no_feasible_fallback,
    )
    rows = decide_all_variants(
        topo=topo,
        matrix=matrix * 100.0,
        policy_action=action,
        backends=backends,
        cfg=cfg,
        seed=9,
        matrix_id="hi",
        scale=100.0,
    )
    fb2 = select_physics_fallback(topo, matrix * 100.0)
    for r in rows:
        if r["gate_variant"] not in (
            "B_original_twin_mean",
            "C_original_twin_uncertainty",
            "D_improved_twin_mean",
            "E_improved_twin_calibrated_uncertainty",
        ):
            continue
        if not r["acceptance"]:
            ok(
                f"fallback_distinguishes_{r['gate_variant']}",
                f"proposed={r['proposed_action_source']} applied={r['applied_action_source']}",
                r["proposed_action_source"] == "policy"
                and r["applied_action_source"] in ("shortest", "equal")
                and r["fallback_used"],
            )
            ok(
                f"fallback_consistent_{r['gate_variant']}",
                f"row={r['selected_fallback']} physics={fb2.selected}",
                r["selected_fallback"] == fb2.selected,
            )
    fb_lo = select_physics_fallback(topo, matrix * 0.1)
    ok(
        "low_scale_feasible_or_least",
        f"outcome={fb_lo.outcome}",
        fb_lo.outcome in ("feasible_fallback", "least_violating_fallback"),
    )
    if fb_lo.feasible_fallback:
        ok("feasible_has_zero_viol", str(fb_lo.selected_violations), fb_lo.selected_violations == 0)

    stable = Path(IMPROVED_STABLE)
    best = Path(IMPROVED_BEST)
    ok("rl_stable_exists", str(stable), stable.exists())
    ok("rl_best_exists", str(best), best.exists())
    policy = GeneralPolicy(device)
    payload = policy.load(stable)
    report["rl_checkpoint"] = {
        "path": str(stable),
        "keys": sorted(payload.keys()),
        "has_model": "model" in payload,
        "epoch": payload.get("epoch"),
        "cfg_policy_path": cfg["policy"]["default_checkpoint"],
    }
    ok("rl_payload_has_model", str(sorted(payload.keys())[:12]), "model" in payload)
    ok(
        "cfg_points_to_stable",
        Path(cfg["policy"]["default_checkpoint"]).resolve().as_posix(),
        Path(cfg["policy"]["default_checkpoint"]).resolve() == stable.resolve(),
    )
    demand_vec = od_demands(matrix)
    prop = policy.propose(topo, demand_vec)
    ok(
        "rl_action_shape",
        f"action={prop.shape} n_path_nodes={topo.n_path_nodes}",
        prop.shape == (topo.n_path_nodes,),
    )
    ok("rl_action_finite", str(bool(np.isfinite(prop).all())), bool(np.isfinite(prop).all()))
    ok("rl_action_nonneg", f"min={float(prop.min())}", float(prop.min()) >= -1e-6)
    row_a = decide_variant(
        "A_policy_only",
        topo=topo,
        matrix=matrix,
        policy_action=prop,
        backends=backends,
        cfg=cfg,
        seed=1,
        matrix_id=0,
        scale=1.0,
        policy_checkpoint=str(stable),
        policy_checkpoint_id="improved_stable",
    )
    ok(
        "rl_through_A",
        f"accept={row_a['acceptance']} applied={row_a['applied_action_source']}",
        row_a["acceptance"] and row_a["applied_action_source"] == "policy",
    )
    row_e2 = decide_variant(
        "E_improved_twin_calibrated_uncertainty",
        topo=topo,
        matrix=matrix,
        policy_action=prop,
        backends=backends,
        cfg=cfg,
        seed=1,
        matrix_id=0,
        scale=1.0,
        policy_checkpoint=str(stable),
        policy_checkpoint_id="improved_stable",
    )
    ok(
        "rl_through_E",
        f"backend={row_e2['twin_backend']} ckpt_id={row_e2['policy_checkpoint_id']}",
        row_e2["twin_backend"] == "improved_calibrated"
        and row_e2["policy_checkpoint_id"] == "improved_stable",
    )

    if not (IMPL / "safety" / "improvements" / "run_eval.py").exists():
        warn("no_run_eval_yet", "Evaluation runner module missing.")
    if not (IMPL / "safety" / "improvements" / "policy_backend.py").exists():
        warn("no_policy_backend_module", "Policy backend module missing.")
    # Infrastructure may exist while full A–F experiments remain unrun (correct for this stage).
    if not (ROOT / "results" / "safety_gate" / "improvements" / "eval_runs").exists():
        warn(
            "full_af_not_started",
            "No full A–F eval_runs directory yet (expected until experiment stage).",
        )

    # Double conformal trap: applying scale in backend AND manually again.
    double = TwinDoubleCheck(backends["improved_calibrated"], topo, matrix, action)
    ok(
        "no_latent_double_scale_in_backend",
        f"ratio={double['ratio']:.6f}",
        abs(double["ratio"] - scale) < 1e-9,
    )

    report["summary"] = {
        "n_checks": len(report["checks"]),
        "n_passed": sum(1 for c in report["checks"] if c["passed"]),
        "n_failed": sum(1 for c in report["checks"] if not c["passed"]),
        "n_warnings": len(report["warnings"]),
        "n_issues": len(report["issues"]),
        "conformal_std_scale": scale,
    }
    out = ROOT / "results" / "safety_gate" / "improvements" / "correctness_audit_raw.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def TwinDoubleCheck(backend, topo, matrix, action) -> dict:
    """Ratio of calibrated std to raw ensemble std from the same model."""
    pred = backend.predict(topo, matrix, action)
    raw_true, raw_mean, raw_std, _ = predict_loads(
        backend.model, topo, matrix, action, backend.device, conformal_std_scale=1.0
    )
    ratio = float(np.mean(pred.std_load / np.maximum(raw_std, 1e-30)))
    return {"ratio": ratio, "means_equal": bool(np.allclose(pred.mean_load, raw_mean))}


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["summary"], indent=2))
    print("ISSUES", result["issues"])
    print("WARNINGS", [w["name"] for w in result["warnings"]])
    for c in result["checks"]:
        if not c["passed"]:
            print("FAIL", c)

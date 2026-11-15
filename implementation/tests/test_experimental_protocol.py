"""Limited synthetic tests for the experimental protocol (no full A–F)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from safety.improvements.config import load_improvements_config
from safety.improvements.decision import VARIANT_IDS, decide_all_variants
from safety.improvements.diagnostics import (
    build_case_diagnostics,
    capacity_proximity_bin,
    twin_uncertainty_views,
)
from safety.improvements.fallback import select_physics_fallback
from safety.improvements.gate_logic import ImprovedGateConfig, inspect_utilisation_ucb
from safety.improvements.metrics import compute_safety_metrics, partition_rows
from safety.improvements.oracle import action_link_load, physics_violations
from safety.improvements.resume import (
    append_decision_rows,
    case_id,
    load_completed_case_ids,
    read_decision_rows,
)
from safety.improvements.run_eval import _action_fingerprint, evaluate_paired_case
from safety.improvements.twin_backend import TwinPrediction, backends_from_config
from twin.data import load_split_demands, load_topology
from twin.physics import equal_split


class PairingIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_improvements_config()
        cls.backends = backends_from_config(cls.cfg)
        cls.topo = load_topology("B4")
        demands, meta = load_split_demands("val", "B4")
        cls.matrix = demands[0] * 0.5
        cls.action = equal_split(cls.topo.n_paths)
        cls.case = {
            "seed": 101,
            "topology": "B4",
            "split": "val",
            "local_index": 0,
            "matrix_id": int(meta.get("matrix_indices", [0])[0]),
            "scale": 0.5,
            "unseen": False,
        }

    def test_b_through_e_share_identical_policy_action(self) -> None:
        rows = evaluate_paired_case(
            case=self.case,
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            policy_meta={"checkpoint_id": "improved_stable", "checkpoint_path": "x"},
            backends=self.backends,
            cfg=self.cfg,
            variants=VARIANT_IDS,
        )
        fps = {r["policy_action_fingerprint"] for r in rows}
        self.assertEqual(len(fps), 1)
        self.assertEqual(fps.pop(), _action_fingerprint(self.action))
        for r in rows:
            self.assertEqual(r["seed"], 101)
            self.assertEqual(r["scale"], 0.5)
            self.assertEqual(r["matrix_id"], self.case["matrix_id"])
            self.assertEqual(r["proposed_action_source"], "policy")

    def test_variant_order_does_not_change_accept_decisions(self) -> None:
        forward = decide_all_variants(
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            variants=VARIANT_IDS,
        )
        reverse = decide_all_variants(
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            variants=tuple(reversed(VARIANT_IDS)),
        )
        fwd = {r["gate_variant"]: r["acceptance"] for r in forward}
        rev = {r["gate_variant"]: r["acceptance"] for r in reverse}
        self.assertEqual(fwd, rev)

    def test_proposed_vs_applied_distinguished_on_reject(self) -> None:
        # Force reject via mean-only with tiny capacity clone is hard; use oracle F path
        # by inspecting a high-scale matrix instead.
        matrix_hi = self.matrix * 200.0
        rows = evaluate_paired_case(
            case={**self.case, "scale": 100.0},
            topo=self.topo,
            matrix=matrix_hi,
            policy_action=self.action,
            policy_meta={"checkpoint_id": "improved_stable", "checkpoint_path": "x"},
            backends=self.backends,
            cfg=self.cfg,
            variants=("D_improved_twin_mean", "E_improved_twin_calibrated_uncertainty"),
        )
        for r in rows:
            self.assertEqual(r["proposed_action_source"], "policy")
            if not r["acceptance"]:
                self.assertTrue(r["fallback_used"])
                self.assertIn(r["applied_action_source"], {"shortest", "equal"})
                self.assertNotEqual(r["applied_action_source"], "policy")


class OracleIndependenceTest(unittest.TestCase):
    def test_oracle_uses_physics_not_twin(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        matrix = demands[0]
        action = equal_split(topo.n_paths)
        true_load = action_link_load(topo, matrix, action)
        # Poisoned twin-like prediction must not affect oracle violations.
        fake_mean = true_load * 0.0
        viol = physics_violations(true_load, topo.capacity)
        self.assertEqual(
            viol,
            int(np.count_nonzero(true_load > topo.capacity + 1e-6)),
        )
        # Fallback selection ignores twin std entirely.
        fb = select_physics_fallback(topo, matrix)
        self.assertIn(fb.selected, {"shortest", "equal"})
        # Even if UCB would reject, oracle feasibility is from true load.
        verdict = inspect_utilisation_ucb(
            fake_mean,
            np.ones_like(fake_mean) * 1e6,
            topo.capacity,
            ImprovedGateConfig(k=2.0),
        )
        # Twin gate may reject; oracle label stays physics-based.
        oracle_feas = physics_violations(true_load, topo.capacity) == 0
        self.assertIsInstance(verdict.accepted, bool)
        self.assertIsInstance(oracle_feas, bool)


class DiagnosticCorrectnessTest(unittest.TestCase):
    def test_mean_uncertainty_same_link_axis_and_calibration_once(self) -> None:
        mean = np.array([10.0, 20.0, 30.0])
        raw_std = np.array([1.0, 2.0, 3.0])
        scale = 0.9545454545454546
        pred = TwinPrediction(
            true_load=mean,
            mean_load=mean,
            std_load=raw_std * scale,
            twin_ms=1.0,
            backend="improved_calibrated",
            checkpoint_path="x",
            conformal_std_scale_applied=scale,
            conformal_applied=True,
        )
        views = twin_uncertainty_views(pred, conformal_std_scale=scale)
        np.testing.assert_allclose(views["raw_std_load"], raw_std, rtol=1e-6)
        np.testing.assert_allclose(views["calibrated_std_load"], raw_std * scale, rtol=1e-6)
        # Not double-scaled.
        np.testing.assert_allclose(
            views["calibrated_std_load"],
            views["raw_std_load"] * scale,
            rtol=1e-6,
        )
        self.assertEqual(views["raw_std_load"].shape, mean.shape)

    def test_errors_use_actual_simulated_util(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        matrix = demands[0] * 0.5
        action = equal_split(topo.n_paths)
        true_load = action_link_load(topo, matrix, action)
        pred = TwinPrediction(
            true_load=true_load,
            mean_load=true_load * 1.01,
            std_load=np.ones_like(true_load) * 0.1,
            twin_ms=0.5,
            backend="improved",
            checkpoint_path="x",
            conformal_std_scale_applied=1.0,
            conformal_applied=False,
        )
        diag = build_case_diagnostics(
            topo=topo,
            matrix=matrix,
            policy_action=action,
            decision_record={"acceptance": True},
            twin_pred=pred,
            gate_k=1.0,
            conformal_std_scale=0.95,
        )
        self.assertGreater(diag["twin_mae_abs"], 0.0)
        self.assertEqual(diag["capacity_proximity"], capacity_proximity_bin(diag["actual_mlu"]))
        self.assertIn(diag["capacity_proximity"], {"far_below", "mid", "near_limit", "over_limit"})


class MetricsDenominatorTest(unittest.TestCase):
    def test_failed_rows_retained_not_as_rejects(self) -> None:
        rows = [
            {
                "decision_status": "ok",
                "acceptance": True,
                "oracle_feasible_proposed": False,
                "capacity_violations_proposed": 2,
                "capacity_violations_applied": 0,
                "fallback_used": False,
                "no_feasible_fallback": False,
                "decision_runtime_ms": 1.0,
            },
            {
                "decision_status": "ok",
                "acceptance": False,
                "oracle_feasible_proposed": True,
                "capacity_violations_proposed": 0,
                "capacity_violations_applied": 0,
                "fallback_used": True,
                "no_feasible_fallback": False,
                "decision_runtime_ms": 2.0,
            },
            {
                "decision_status": "failed",
                "acceptance": None,
                "failure_message": "boom",
                "decision_runtime_ms": 9.0,
            },
        ]
        parts = partition_rows(rows)
        self.assertEqual(len(parts["completed"]), 2)
        self.assertEqual(len(parts["failed"]), 1)
        metrics = compute_safety_metrics(rows)
        self.assertEqual(metrics["n_logged"], 3)
        self.assertEqual(metrics["n_completed"], 2)
        self.assertEqual(metrics["n_failed"], 1)
        self.assertAlmostEqual(metrics["acceptance_rate"], 0.5)
        self.assertAlmostEqual(metrics["unsafe_acceptance_rate"], 0.5)  # 1/2 completed
        self.assertAlmostEqual(metrics["false_rejection_rate"], 0.5)
        self.assertAlmostEqual(metrics["error_detection_rate"], 0.0)  # unsafe accepted
        self.assertAlmostEqual(metrics["false_alarm_rate"], 1.0)  # 1 false reject / 1 safe


class ResumeDedupTest(unittest.TestCase):
    def test_resume_skips_complete_blocks_not_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            case = {
                "seed": 1,
                "topology": "B4",
                "split": "val",
                "matrix_id": 0,
                "local_index": 0,
                "scale": 1.0,
            }
            cid = case_id(case, checkpoint_id="improved_stable")
            ok_rows = [
                {
                    "case_id": cid,
                    "decision_status": "ok",
                    "acceptance": True,
                    "gate_variant": vid,
                }
                for vid in VARIANT_IDS
            ]
            append_decision_rows(path, ok_rows)
            done = load_completed_case_ids(path, expected_variants=6)
            self.assertIn(cid, done)

            fail_case = {**case, "scale": 0.5}
            fail_cid = case_id(fail_case, checkpoint_id="improved_stable")
            append_decision_rows(
                path,
                [
                    {
                        "case_id": fail_cid,
                        "decision_status": "failed",
                        "acceptance": None,
                        "failure_message": "policy failed",
                    }
                ],
            )
            done2 = load_completed_case_ids(path, expected_variants=6)
            self.assertNotIn(fail_cid, done2)
            self.assertEqual(len(read_decision_rows(path)), 7)


class CandidatePathIdentityTest(unittest.TestCase):
    def test_fallback_candidates_identical_across_calls(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        matrix = demands[0]
        a = select_physics_fallback(topo, matrix)
        b = select_physics_fallback(topo, matrix)
        self.assertEqual(a.selected, b.selected)
        self.assertEqual(a.outcome, b.outcome)
        np.testing.assert_array_equal(a.selected_action, b.selected_action)


if __name__ == "__main__":
    unittest.main()

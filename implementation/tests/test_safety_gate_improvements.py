"""Tests for the isolated improved Safety Gate (no full A–F experiment)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from safety.gate import GateConfig, inspect_prediction
from safety.improvements.config import DEFAULT_CONFIG_PATH, load_improvements_config
from safety.improvements.decision import VARIANT_IDS, decide_all_variants, decide_variant
from safety.improvements.fallback import select_physics_fallback
from safety.improvements.gate_logic import (
    REASON_INVALID,
    REASON_UCB,
    ImprovedGateConfig,
    inspect_utilisation_ucb,
)
from safety.improvements.logging_util import REQUIRED_LOG_FIELDS, validate_decision_record, write_jsonl
from safety.improvements.oracle import capacity_feasible, oracle_accepts_action, physics_violations
from safety.improvements.twin_backend import TwinBackend, backends_from_config
from twin.data import IMPL, ROOT, load_split_demands, load_topology
from twin.physics import equal_split, shortest_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ORIGINAL_TWIN = ROOT / "models" / "twin" / "ensemble.pt"
ORIGINAL_GATE = IMPL / "safety" / "gate.py"
ORIGINAL_GATE_CFG = IMPL / "safety" / "gate_config.json"
IMPROVED_CONFORMAL = (
    ROOT / "models" / "twin_improved" / "ensemble_physics_anchored_conformal.pt"
)
IMPROVED_ANCHORED = (
    ROOT / "models" / "twin_improved" / "ensemble_physics_anchored.pt"
)

# Frozen at implementation time for preservation checks.
EXPECTED_ORIGINAL_TWIN_SHA = "7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685"


class ConfigTest(unittest.TestCase):
    def test_default_config_loads_and_resolves_paths(self) -> None:
        self.assertTrue(DEFAULT_CONFIG_PATH.exists())
        cfg = load_improvements_config()
        self.assertIn("gate", cfg)
        self.assertTrue(Path(cfg["twins"]["original"]["path"]).exists())
        self.assertTrue(Path(cfg["twins"]["improved_calibrated"]["path"]).exists())
        self.assertIsNone(cfg["gate"]["max_relative_std"])
        self.assertEqual(set(cfg["evaluation"]["variants"]), set(VARIANT_IDS))


class UtilisationGateTest(unittest.TestCase):
    def test_uncertainty_aware_accepts_safe_ucb(self) -> None:
        mean = np.array([10.0, 20.0, 30.0])
        std = np.array([1.0, 1.0, 1.0])
        cap = np.array([100.0, 100.0, 100.0])
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=2.0))
        self.assertTrue(verdict.accepted)
        np.testing.assert_allclose(verdict.ucb, mean / cap + 2.0 * (std / cap))
        self.assertEqual(verdict.reasons, [])

    def test_mean_only_ignores_std(self) -> None:
        mean = np.array([50.0, 90.0])
        std = np.array([40.0, 40.0])  # would breach if k>0
        cap = np.array([100.0, 100.0])
        mean_cfg = ImprovedGateConfig(k=0.0, max_relative_std=None)
        verdict = inspect_utilisation_ucb(mean, std, cap, mean_cfg)
        self.assertTrue(verdict.accepted)
        np.testing.assert_allclose(verdict.ucb, mean / cap)

    def test_uncertainty_rejects_when_ucb_crosses_limit(self) -> None:
        mean = np.array([50.0, 95.0])
        std = np.array([1.0, 4.0])
        cap = np.array([100.0, 100.0])
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=2.0))
        self.assertFalse(verdict.accepted)
        self.assertIn(REASON_UCB, verdict.reasons)
        self.assertGreater(verdict.mlu_ucb, 1.0)

    def test_capacity_boundary_equal_limit_accepts(self) -> None:
        mean = np.array([100.0])
        std = np.array([0.0])
        cap = np.array([100.0])
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=1.0, utilisation_limit=1.0))
        self.assertTrue(verdict.accepted)

    def test_invalid_nonfinite(self) -> None:
        verdict = inspect_utilisation_ucb(
            np.array([1.0, np.nan]),
            np.array([0.1, 0.1]),
            np.array([10.0, 10.0]),
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reasons, [REASON_INVALID])

    def test_zero_capacity_link_uses_floor_and_can_reject(self) -> None:
        mean = np.array([1.0, 0.0])
        std = np.array([0.0, 0.0])
        cap = np.array([0.0, 10.0])
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=0.0, min_capacity=1e-12))
        self.assertFalse(verdict.accepted)
        self.assertIn(REASON_UCB, verdict.reasons)
        self.assertGreater(verdict.mean_util[0], 1.0)

    def test_negative_capacity_is_invalid(self) -> None:
        verdict = inspect_utilisation_ucb(
            np.array([1.0]), np.array([0.0]), np.array([-5.0]), ImprovedGateConfig(k=0.0)
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reasons, [REASON_INVALID])

    def test_negative_mean_is_invalid(self) -> None:
        verdict = inspect_utilisation_ucb(
            np.array([-1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            ImprovedGateConfig(k=0.0),
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reasons, [REASON_INVALID])

    def test_nan_capacity_is_invalid(self) -> None:
        verdict = inspect_utilisation_ucb(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, np.nan]),
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reasons, [REASON_INVALID])

    def test_does_not_reject_solely_for_large_std_when_veto_disabled(self) -> None:
        mean = np.array([1.0, 1.0])
        std = np.array([0.1, 80.0])
        cap = np.array([100.0, 100.0])
        # With k=0 and no relative veto, high std alone must not reject.
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=0.0, max_relative_std=None))
        self.assertTrue(verdict.accepted)

    def test_util_space_matches_scaled_absolute_bound(self) -> None:
        mean = np.array([10.0, 25.0])
        std = np.array([2.0, 3.0])
        cap = np.array([50.0, 100.0])
        k = 1.5
        verdict = inspect_utilisation_ucb(mean, std, cap, ImprovedGateConfig(k=k))
        expected_ucb = (mean + k * std) / cap
        np.testing.assert_allclose(verdict.ucb, expected_ucb)


class OracleFallbackTest(unittest.TestCase):
    def test_oracle_on_b4_equal_split(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        matrix = demands[0]
        feasible, load, n_viol = oracle_accepts_action(topo, matrix, equal_split(topo.n_paths))
        self.assertEqual(n_viol, physics_violations(load, topo.capacity))
        self.assertEqual(feasible, capacity_feasible(load, topo.capacity))

    def test_fallback_consistent_and_records_feasibility(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        matrix = demands[0] * 0.25
        a = select_physics_fallback(topo, matrix)
        b = select_physics_fallback(topo, matrix)
        self.assertEqual(a.selected, b.selected)
        self.assertEqual(a.outcome, b.outcome)
        self.assertEqual(a.feasible_fallback, not a.no_feasible_fallback)
        self.assertIn(a.selected, {"shortest", "equal"})


class TwinIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.cfg = load_improvements_config()
        cls.backends = backends_from_config(cls.cfg, cls.device)
        cls.topo = load_topology("B4")
        demands, meta = load_split_demands("val", "B4")
        cls.matrix = demands[0]
        cls.matrix_id = 0
        cls.meta = meta
        cls.action = equal_split(cls.topo.n_paths)

    def test_original_twin_inference(self) -> None:
        pred = self.backends["original"].predict(self.topo, self.matrix, self.action)
        self.assertEqual(pred.mean_load.shape, self.topo.capacity.shape)
        self.assertTrue(np.all(pred.std_load >= 0))
        self.assertEqual(pred.conformal_std_scale_applied, 1.0)
        self.assertFalse(pred.conformal_applied)

    def test_improved_twin_inference(self) -> None:
        pred = self.backends["improved"].predict(self.topo, self.matrix, self.action)
        self.assertEqual(pred.mean_load.shape, self.topo.capacity.shape)
        self.assertTrue(np.isfinite(pred.mean_load).all())

    def test_conformal_calibration_applied(self) -> None:
        raw = TwinBackend(
            "raw",
            IMPROVED_CONFORMAL,
            apply_conformal=False,
            device=self.device,
        ).predict(self.topo, self.matrix, self.action)
        cal = self.backends["improved_calibrated"].predict(self.topo, self.matrix, self.action)
        scale = float(cal.conformal_std_scale_applied)
        self.assertNotEqual(scale, 1.0)
        np.testing.assert_allclose(cal.mean_load, raw.mean_load, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cal.std_load, raw.std_load * scale, rtol=1e-5, atol=1e-8)
        self.assertTrue(cal.conformal_applied)

    def test_variants_are_paired_on_same_action(self) -> None:
        rows = decide_all_variants(
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            seed=101,
            matrix_id=str(self.matrix_id),
            scale=1.0,
        )
        self.assertEqual(len(rows), len(VARIANT_IDS))
        variants = [r["gate_variant"] for r in rows]
        self.assertEqual(variants, list(VARIANT_IDS))
        # Paired: same oracle label for the proposed action across variants.
        feas = {r["oracle_feasible_proposed"] for r in rows}
        self.assertEqual(len(feas), 1)
        for row in rows:
            missing = validate_decision_record(row)
            self.assertEqual(missing, [], msg=f"{row['gate_variant']}: {missing}")

    def test_mean_only_vs_uncertainty_decision_paths(self) -> None:
        mean_row = decide_variant(
            "D_improved_twin_mean",
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            seed=101,
            matrix_id="m0",
            scale=1.0,
        )
        unc_row = decide_variant(
            "E_improved_twin_calibrated_uncertainty",
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            seed=101,
            matrix_id="m0",
            scale=1.0,
        )
        self.assertEqual(mean_row["gate_k"], 0.0)
        self.assertGreater(unc_row["gate_k"], 0.0)
        self.assertEqual(mean_row["twin_backend"], "improved")
        self.assertEqual(unc_row["twin_backend"], "improved_calibrated")
        self.assertTrue(unc_row["conformal_applied"])

    def test_oracle_variant_uses_physics(self) -> None:
        row = decide_variant(
            "F_physics_oracle",
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            seed=101,
            matrix_id="m0",
            scale=1.0,
        )
        self.assertEqual(row["twin_backend"], "none")
        self.assertEqual(row["acceptance"], row["oracle_feasible_proposed"])

    def test_policy_only_never_falls_back(self) -> None:
        row = decide_variant(
            "A_policy_only",
            topo=self.topo,
            matrix=self.matrix,
            policy_action=self.action,
            backends=self.backends,
            cfg=self.cfg,
            seed=101,
            matrix_id="m0",
            scale=1.0,
        )
        self.assertTrue(row["acceptance"])
        self.assertFalse(row["fallback_used"])
        self.assertEqual(row["applied_action_source"], "policy")
        self.assertEqual(row["proposed_action_source"], "policy")

    def test_decision_logging_roundtrip(self) -> None:
        row = decide_variant(
            "C_original_twin_uncertainty",
            topo=self.topo,
            matrix=self.matrix,
            policy_action=shortest_split(self.topo.n_paths, self.topo.hops),
            backends=self.backends,
            cfg=self.cfg,
            seed=202,
            matrix_id="m0",
            scale=0.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            write_jsonl(path, [row])
            loaded = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            for key in REQUIRED_LOG_FIELDS:
                self.assertIn(key, loaded)
            self.assertIn(loaded["applied_action_source"], {"policy", "shortest", "equal"})


class PreservationTest(unittest.TestCase):
    def test_original_artifacts_unchanged(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_TWIN), EXPECTED_ORIGINAL_TWIN_SHA)
        self.assertTrue(ORIGINAL_GATE.exists())
        self.assertTrue(ORIGINAL_GATE_CFG.exists())
        self.assertTrue(IMPROVED_CONFORMAL.exists())
        self.assertTrue(IMPROVED_ANCHORED.exists())
        # Improved stack must not shadow original gate entrypoints.
        from safety import gate as original_gate

        self.assertTrue(hasattr(original_gate, "inspect_prediction"))
        self.assertTrue(hasattr(original_gate, "select_fallback"))
        # Original gate still behaves as before on a simple case.
        verdict = inspect_prediction(
            np.array([10.0, 20.0]),
            np.array([1.0, 1.0]),
            np.array([100.0, 100.0]),
            GateConfig(k=2.0, max_relative_std=0.5),
        )
        self.assertTrue(verdict.accepted)


if __name__ == "__main__":
    unittest.main()

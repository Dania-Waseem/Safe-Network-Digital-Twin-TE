"""Safety-gate accept/reject cases. Does not start ASN preprocessing."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from safety.evaluate import load_config, load_trained_ensemble, predict_action
from safety.gate import (
    REASON_INVALID,
    REASON_OVERLOAD,
    REASON_UNCERTAIN,
    GateConfig,
    inspect_prediction,
    select_fallback,
)
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import equal_split, shortest_split


class SyntheticGateTest(unittest.TestCase):
    def test_accepts_when_upper_bound_below_capacity(self) -> None:
        mean = np.array([10.0, 20.0, 30.0])
        std = np.array([1.0, 1.0, 1.0])
        cap = np.array([100.0, 100.0, 100.0])
        verdict = inspect_prediction(mean, std, cap, GateConfig(k=2.0, max_relative_std=0.5))
        self.assertTrue(verdict.accepted)
        self.assertEqual(verdict.reasons, [])
        np.testing.assert_allclose(verdict.upper, mean + 2.0 * std)

    def test_rejects_overloaded_link(self) -> None:
        mean = np.array([50.0, 95.0])
        std = np.array([1.0, 4.0])
        cap = np.array([100.0, 100.0])
        verdict = inspect_prediction(mean, std, cap, GateConfig(k=2.0, max_relative_std=1.0))
        self.assertFalse(verdict.accepted)
        self.assertIn(REASON_OVERLOAD, verdict.reasons)
        self.assertGreater(verdict.mlu_upper, 1.0)
        self.assertEqual(verdict.worst_link, 1)

    def test_rejects_high_uncertainty_even_if_mean_is_low(self) -> None:
        mean = np.array([1.0, 1.0])
        std = np.array([0.1, 80.0])
        cap = np.array([100.0, 100.0])
        verdict = inspect_prediction(mean, std, cap, GateConfig(k=0.0, max_relative_std=0.5))
        self.assertFalse(verdict.accepted)
        self.assertIn(REASON_UNCERTAIN, verdict.reasons)
        self.assertNotIn(REASON_OVERLOAD, verdict.reasons)

    def test_rejects_invalid_nonfinite(self) -> None:
        verdict = inspect_prediction(
            np.array([1.0, np.nan]),
            np.array([0.1, 0.1]),
            np.array([10.0, 10.0]),
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual(verdict.reasons, [REASON_INVALID])

    def test_fallback_prefers_last_accepted_then_shortest(self) -> None:
        config = GateConfig()
        cap = np.ones(3) * 10.0
        bad = inspect_prediction(np.array([20.0, 1.0, 1.0]), np.zeros(3), cap, config)
        good = inspect_prediction(np.array([1.0, 1.0, 1.0]), np.zeros(3), cap, config)
        proposed = np.array([1.0, 0.0, 0.0])
        last = np.array([0.0, 1.0, 0.0])
        shortest = np.array([0.0, 0.0, 1.0])
        applied = select_fallback(
            "dirichlet",
            proposed,
            bad,
            last_name="equal",
            last_action=last,
            last_verdict=good,
            shortest_name="shortest",
            shortest_action=shortest,
            shortest_verdict=good,
            config=config,
        )
        self.assertEqual(applied.fallback, "last_accepted")
        np.testing.assert_array_equal(applied.action, last)
        applied_sp = select_fallback(
            "dirichlet",
            proposed,
            bad,
            last_name=None,
            last_action=None,
            last_verdict=None,
            shortest_name="shortest",
            shortest_action=shortest,
            shortest_verdict=good,
            config=config,
        )
        self.assertEqual(applied_sp.fallback, "shortest")


class TrainedTwinGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.model = load_trained_ensemble(cls.device)
        cls.config = load_config()

    def test_b4_equal_split_produces_a_verdict(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        action = equal_split(topo.n_paths)
        true_load, mean, std, _ms = predict_action(self.model, topo, demands[0], action, self.device)
        verdict = inspect_prediction(mean, std, topo.capacity, self.config)
        self.assertEqual(mean.shape, topo.capacity.shape)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(std).all())
        self.assertEqual(true_load.shape, mean.shape)
        self.assertIsInstance(verdict.accepted, bool)

    def test_geant_unseen_gate_runs(self) -> None:
        topo = load_topology("GEANT")
        demands, _ = load_split_demands("test", "GEANT")
        action = shortest_split(topo.n_paths, topo.hops)
        _true, mean, std, _ms = predict_action(self.model, topo, demands[0], action, self.device)
        verdict = inspect_prediction(mean, std, topo.capacity, self.config)
        self.assertEqual(mean.shape[0], 72)
        self.assertTrue(np.isfinite(verdict.upper).all())
        self.assertGreater(verdict.mlu_upper, 0.0)

    def test_kdl_uses_saved_twin_predictions(self) -> None:
        pred = np.load(ROOT / "results" / "twin" / "predictions" / "test_Kdl.npz")
        norm = np.load(ROOT / "data" / "processed" / "normalization" / "Kdl.npz")
        meta = json.loads((ROOT / "data" / "processed" / "normalization" / "Kdl.json").read_text(encoding="utf-8"))
        capacity = np.asarray(norm["capacity_normalised"], dtype=np.float64) * float(meta["denominator"])
        mean = np.asarray(pred["mean_load"][0], dtype=np.float64)
        std = np.asarray(pred["std_load"][0], dtype=np.float64)
        self.assertEqual(mean.shape, capacity.shape)
        verdict = inspect_prediction(mean, std, capacity, self.config)
        self.assertTrue(np.isfinite(verdict.upper).all())
        self.assertEqual(mean.shape[0], 1790)


if __name__ == "__main__":
    unittest.main()

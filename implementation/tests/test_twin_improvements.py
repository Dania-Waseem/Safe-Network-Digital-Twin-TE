"""Tests for Digital Twin improvement helpers (does not retrain)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from twin.data import ROOT, load_split_demands, load_topology
from twin.improvements.infer import load_ensemble_checkpoint, predict_loads
from twin.improvements.metrics import score_prediction
from twin.model import EnsembleTwin
from twin.physics import equal_split
from twin.train import MODEL_DIR


class MetricsTest(unittest.TestCase):
    def test_perfect_prediction_scores(self) -> None:
        true = np.array([1.0, 2.0, 3.0])
        cap = np.array([10.0, 10.0, 10.0])
        stats = score_prediction(true, true, np.ones(3) * 0.1, cap)
        self.assertAlmostEqual(stats["mae"], 0.0)
        self.assertAlmostEqual(stats["mae_utilisation"], 0.0)
        self.assertGreaterEqual(stats["r2"], 0.999)


class CompatibilityTest(unittest.TestCase):
    def test_original_checkpoint_loads_and_predicts(self) -> None:
        path = MODEL_DIR / "ensemble.pt"
        self.assertTrue(path.exists())
        device = torch.device("cpu")
        model, payload = load_ensemble_checkpoint(path, device)
        self.assertIn("state_dict", payload)
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        true, mean, std, _ms = predict_loads(model, topo, demands[0], equal_split(topo.n_paths), device)
        self.assertEqual(mean.shape, true.shape)
        self.assertTrue(np.all(std >= 0))
        rel = np.max(np.abs(mean - true)) / max(float(np.max(true)), 1.0)
        self.assertLess(rel, 0.01)

    def test_improvement_dir_is_separate(self) -> None:
        out = ROOT / "results" / "digital_twin" / "improvements"
        self.assertNotEqual(out.resolve(), MODEL_DIR.resolve())


class AnchorInitTest(unittest.TestCase):
    def test_fresh_model_near_physics(self) -> None:
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        model = EnsembleTwin(n_members=2, hidden=16, dropout=0.0)
        true, mean, std, _ = predict_loads(
            model, topo, demands[0], equal_split(topo.n_paths), torch.device("cpu")
        )
        rel = np.max(np.abs(mean - true)) / max(float(np.max(true)), 1.0)
        self.assertLess(rel, 1e-3)


if __name__ == "__main__":
    unittest.main()

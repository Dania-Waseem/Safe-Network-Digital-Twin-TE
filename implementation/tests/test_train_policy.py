"""Unit checks for improved policy training controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from framework.general_policy import CHECKPOINT, IMPROVED_CHECKPOINT
from framework.train_policy import (
    EXPERIMENT_ROOT,
    HELD_OUT,
    LEGACY_CHECKPOINT,
    SEARCH_GRID,
    TRAIN_TOPOLOGIES,
    EarlyStopper,
    TrainConfig,
    assert_safe_plan,
    train_plan,
    val_plan,
)


class EarlyStopperTest(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self) -> None:
        stopper = EarlyStopper(patience=2)
        best, stop = stopper.step(0, 5.0)
        self.assertTrue(best)
        self.assertFalse(stop)
        best, stop = stopper.step(1, 5.5)
        self.assertFalse(best)
        self.assertFalse(stop)
        best, stop = stopper.step(2, 5.4)
        self.assertFalse(best)
        self.assertTrue(stop)
        self.assertEqual(stopper.best_epoch, 0)
        self.assertEqual(stopper.best, 5.0)

    def test_resets_on_improvement(self) -> None:
        stopper = EarlyStopper(patience=2)
        stopper.step(0, 5.0)
        stopper.step(1, 5.2)
        best, stop = stopper.step(2, 4.5)
        self.assertTrue(best)
        self.assertFalse(stop)
        self.assertEqual(stopper.bad_epochs, 0)


class PlanSafetyTest(unittest.TestCase):
    def test_plans_exclude_held_out_topologies(self) -> None:
        cfg = TrainConfig()
        names = [n for n, *_ in train_plan(cfg)] + [n for n, *_ in val_plan(cfg)]
        assert_safe_plan(names)
        for held in HELD_OUT:
            self.assertNotIn(held, names)
        self.assertEqual(set(TRAIN_TOPOLOGIES), {"B4", "UsCarrier", "Cogentco"})

    def test_assert_safe_plan_rejects_geant(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_safe_plan(["B4", "GEANT"])

    def test_experiment_paths_are_separate_from_legacy(self) -> None:
        self.assertEqual(LEGACY_CHECKPOINT, CHECKPOINT)
        self.assertNotEqual(EXPERIMENT_ROOT.resolve(), CHECKPOINT.parent.resolve())
        self.assertEqual(IMPROVED_CHECKPOINT.name, "best_checkpoint.pt")
        self.assertEqual(IMPROVED_CHECKPOINT.parent.name, "improved")
        self.assertNotEqual(IMPROVED_CHECKPOINT.resolve(), CHECKPOINT.resolve())

    def test_search_grid_has_varied_budgets(self) -> None:
        lrs = {cfg.actor_lr for cfg in SEARCH_GRID}
        batches = {cfg.batch_size for cfg in SEARCH_GRID}
        epochs = {cfg.max_epochs for cfg in SEARCH_GRID}
        self.assertGreaterEqual(len(lrs), 2)
        self.assertGreaterEqual(len(batches), 2)
        self.assertGreaterEqual(len(epochs), 2)
        self.assertTrue(all(cfg.max_epochs > 3 for cfg in SEARCH_GRID))
        self.assertTrue(all(cfg.patience >= 1 for cfg in SEARCH_GRID))


class ConfigSerializationTest(unittest.TestCase):
    def test_config_roundtrip(self) -> None:
        cfg = TrainConfig(run_id="unit", actor_lr=3e-4, batch_size=2, max_epochs=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["run_id"], "unit")
        self.assertEqual(loaded["batch_size"], 2)
        self.assertAlmostEqual(loaded["actor_lr"], 3e-4)


if __name__ == "__main__":
    unittest.main()
